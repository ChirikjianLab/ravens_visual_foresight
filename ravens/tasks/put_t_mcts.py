'''
All blocks are of the same color.
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


class PutTMCTS(TaskMCTS):
  """Put blocks on a T base (mcts)."""

  def __init__(self, *args, **kwargs):
    super().__init__(*args, **kwargs)

    self.block_size = (0.04, 0.04, 0.04)

    self.max_steps = None
    
    # Workspace bounds (0.5 x 0.5 m2).
    self.bounds = np.array([[0.25, 0.75], [-0.25, 0.25], [0, 0.28]])
    # self.bounds = np.array([[0.25, 0.75], [-0.5, 0.5], [0, 0.28]])

  def reset(self, env):
    super().reset(env)

    # Determine the number of blocks.
    num_blocks = 3
    self.max_steps = num_blocks + 2
    
    # Add base.
    base_size = (0.10, 0.10, 0.005)
    base_urdf = 'stand/stand_t.urdf'
    base_pose = self.get_random_pose(env, base_size)
    base_id = env.add_object(base_urdf, base_pose, 'fixed')

    # Add blocks.
    colors = [utils.COLORS['red']] * num_blocks
    objs = []
    objs_id = []
    block_urdf = 'stacking/block.urdf'
    for i in range(num_blocks):
      block_pose = self.get_random_pose(env, self.block_size)
      block_id = env.add_object(block_urdf, block_pose)
      p.changeVisualShape(block_id, -1, rgbaColor=colors[i] + [1])
      # object_id, (symmetry, _)
      objs.append((block_id, (np.pi / 2, None)))
      objs_id.append(block_id)

    place_pos = [
        (   0,     0, self.block_size[-1] / 2),
        (   0,  0.05, self.block_size[-1] / 2),
        (0.05, 0.025, self.block_size[-1] / 2)]
    primitives = [0, 0, 0]

    # Target positions in the world.
    targs = [(utils.apply(base_pose, i), base_pose[1]) for i in place_pos]

    # Goal
    self.goals.append((objs, np.ones((num_blocks, num_blocks)), targs,
        False, True, 'pose', None, 1, primitives))

    return base_urdf, base_size, base_id, objs_id