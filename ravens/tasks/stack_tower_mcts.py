'''
Stacking a tower (mcts).
All blocks are of the same color.
The number of blocks is random within 2 to 4
Primitive types:
  0 - put a block on a base
  1 - stack a block on another block
  2 - stack a block on top of two blocks
@author: Hongtao Wu
Dec 14, 2021
'''


import numpy as np
from ravens.tasks.task_mcts import TaskMCTS
from ravens.utils import utils

import pybullet as p

class StackTowerMCTS(TaskMCTS):
  """Stacking tower task (mcts)."""

  def __init__(self, *args, **kwargs):
    super().__init__(*args, **kwargs)
    self.min_num_blocks = 2
    self.max_num_blocks = 4
    self.max_steps = None

    # Workspace bounds (0.5 x 0.5 m2).
    self.bounds = np.array([[0.25, 0.75], [-0.25, 0.25], [0, 0.28]])
    # self.bounds = np.array([[0.25, 0.75], [-0.5, 0.5], [0, 0.28]])

  def reset(self, env):
    super().reset(env)

    # Add base.
    base_size = (0.05, 0.05, 0.005)
    base_urdf = 'stand/stand_1.urdf'
    base_pose = self.get_random_pose(env, base_size)
    print(f"base_pose: {base_pose}")
    base_id = env.add_object(base_urdf, base_pose, 'fixed')

    # Number of blocks
    # num_blocks = np.random.choice(range(self.min_num_blocks, self.max_num_blocks + 1))
    num_blocks = 3
    self.max_steps = num_blocks + 2

    # Block colors.
    colors = [utils.COLORS['red']] * num_blocks
    
    # Add blocks.
    objs = []
    objs_id = []
    block_size = (0.04, 0.04, 0.04)
    block_urdf = 'stacking/block.urdf'
    for i in range(num_blocks):
      block_pose = self.get_random_pose(env, block_size)
      block_id = env.add_object(block_urdf, block_pose)
      p.changeVisualShape(block_id, -1, rgbaColor=colors[i] + [1])
      # object_id, (symmetry, _)
      objs.append((block_id, (np.pi / 2, None)))
      objs_id.append(block_id)

    # Place position.
    place_pos = []
    for i in range(num_blocks):
      place_pos.append((0, 0, block_size[-1] / 2 + i * block_size[-1]))
    # Target pose (pos, quat) in world frame
    targs = [(utils.apply(base_pose, i), base_pose[1]) for i in place_pos]

    # Primitive = [0, 1, 1, ...].
    primitives = [0]
    for i in range(num_blocks - 1):
      primitives.append(1)
    
    # Goal: blocks are stacked in tower
    # objs, matches, targs, replace, rotations, metric, params, max_reward, primitives.
    self.goals.append((objs, np.ones((num_blocks, num_blocks)), targs,
                       False, True, 'pose', None, 1, primitives))

    return base_urdf, base_size, base_id, objs_id