"""Environment class for MCTS."""

import os
import pkgutil
import sys
import tempfile
import time

import gym
import numpy as np
from ravens.tasks import cameras
from ravens.utils import pybullet_utils
from ravens.environments.environment import Environment
import pybullet as p

PICK  = 0
PLACE = 1

PLACE_STEP = 0.0003
PLACE_DELTA_THRESHOLD = 0.005

UR5_URDF_PATH = 'ur5/ur5.urdf'
UR5_WORKSPACE_URDF_PATH = 'ur5/workspace.urdf'
PLANE_URDF_PATH = 'plane/plane.urdf'

class EnvironmentMCTS(Environment):
  """OpenAI Gym-style environment class for training the dynamics model."""

  def __init__(self,
               assets_root,
               task=None,
               disp=False,
               shared_memory=False,
               hz=240,
               use_egl=False):
    """Creates OpenAI Gym-style environment with PyBullet.

    Args:
      assets_root: root directory of assets.
      task: the task to use. If None, the user must call set_task for the
        environment to work properly.
      disp: show environment with PyBullet's built-in display viewer.
      shared_memory: run with shared memory.
      hz: PyBullet physics simulation step speed. Set to 480 for deformables.
      use_egl: Whether to use EGL rendering. Only supported on Linux. Should get
        a significant speedup in rendering when using.

    Raises:
      RuntimeError: if pybullet cannot load fileIOPlugin.
    """
    super().__init__(assets_root, task, disp, shared_memory, hz, use_egl)
    self.position_bounds = gym.spaces.Box(
        low=np.array([0.25, -0.25, 0.], dtype=np.float32),
        high=np.array([0.75, 0.25, 0.28], dtype=np.float32),
        shape=(3,),
        dtype=np.float32)

  def step(self, action=None):
    """Execute action with specified primitive.

    Args:
      action: action to execute. ([x, y, z], [qx, qy, qz, qw])

    Returns:
      (pick_obs, place_obs, reward, done, info) tuple containing MDP step data.
      pick_obs and place_obs are the observation after the pick and place
      motion, respectively.
    """
    
    if action is not None:
      pick_pose = action['pose0']
      place_pose = action['pose1']
    
      #### Pick ####
      timeout = self.task.primitive(self.movej, self.movep, self.ee, pick_pose, PICK)

      # Exit early if action times out. We still return an observation
      # so that we don't break the Gym API contract.
      if timeout:
        pick_obs  = self._get_obs()
        place_obs = pick_obs
        return [pick_obs, place_obs], 0.0, True, self.info

      # Step simulator asynchronously until objects settle.
      while not self.is_static:
        p.stepSimulation()
      pick_obs = self._get_obs()

      #### Place ####
      timeout = self.task.primitive(self.movej, self.movep, self.ee, place_pose, PLACE)

      # Exit early if action times out. We still return an observation
      # so that we don't break the Gym API contract.
      if timeout:
        pick_obs = self._get_obs()
        place_obs = pick_obs
        return [pick_obs, place_obs], 0.0, True, self.info

      # Step simulator asynchronously until objects settle.
      while not self.is_static:
        p.stepSimulation()
      place_obs = self._get_obs()
    
    else:
      # For reset
      pick_obs = self._get_obs()
      place_obs = None

    # Get task rewards.
    reward, info = self.task.reward() if action is not None else (0, {})
    done = self.task.done()

    # Add ground truth robot state into info.
    info.update(self.info)

    return [pick_obs, place_obs], reward, done, info

  def get_total_reward(self):
    total_reward = self.task.get_total_reward()
    ravens_reward = self.task._rewards
    return total_reward, ravens_reward

  def reset(self):
    """Performs common reset functionality for all supported tasks."""
    if not self.task:
      raise ValueError('environment task must be set. Call set_task or pass '
                       'the task arg in the environment constructor.')
    self.obj_ids = {'fixed': [], 'rigid': [], 'deformable': []}
    p.resetSimulation(p.RESET_USE_DEFORMABLE_WORLD)
    p.setGravity(0, 0, -9.8)

    # Temporarily disable rendering to load scene faster.
    p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 0)

    pybullet_utils.load_urdf(p, os.path.join(self.assets_root, PLANE_URDF_PATH),
                             [0, 0, -0.001])
    pybullet_utils.load_urdf(
        p, os.path.join(self.assets_root, UR5_WORKSPACE_URDF_PATH), [0.5, 0, 0])

    # Load UR5 robot arm equipped with suction end effector.
    # TODO(andyzeng): add back parallel-jaw grippers.
    self.ur5 = pybullet_utils.load_urdf(
        p, os.path.join(self.assets_root, UR5_URDF_PATH))
    self.ee = self.task.ee(self.assets_root, self.ur5, 9, self.obj_ids)
    self.ee_tip = 10  # Link ID of suction cup.

    # Get revolute joint indices of robot (skip fixed joints).
    n_joints = p.getNumJoints(self.ur5)
    joints = [p.getJointInfo(self.ur5, i) for i in range(n_joints)]
    self.joints = [j[0] for j in joints if j[2] == p.JOINT_REVOLUTE]

    # Move robot to home joint configuration.
    for i in range(len(self.joints)):
      p.resetJointState(self.ur5, self.joints[i], self.homej[i])

    # Reset end effector.
    self.ee.release()

    # Reset task.
    base_urdf, base_size, base_id, objs_id = self.task.reset(self)

    # Re-enable rendering.
    p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 1)

    [init_obs, _], _, _, info = self.step()
    return init_obs, base_urdf, base_size, base_id, objs_id, info