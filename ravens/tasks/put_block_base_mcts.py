'''
Put blocks on the base (mcts).
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


class PutBlockBaseMCTS(TaskMCTS):
  """Make a row (mcts)."""

  def __init__(self, *args, **kwargs):
    super().__init__(*args, **kwargs)

    self.block_size = (0.04, 0.04, 0.04)

    self.num_blocks_list = [3]
    self.base_urdf_list = [
      'stand/stand_3.urdf'
    ]
    self.base_size_list = [
      (0.05, 0.15, 0.005)
    ]

    self.place_pos_list = [
      [(0,    0, self.block_size[-1] / 2),
       (0, 0.05, self.block_size[-1] / 2),
       (0, 0.10, self.block_size[-1] / 2),]
    ]

    self.max_steps = None
    
    # Workspace bounds (0.5 x 0.5 m2).
    self.bounds = np.array([[0.25, 0.75], [-0.25, 0.25], [0, 0.28]])
    # self.bounds = np.array([[0.25, 0.75], [-0.5, 0.5], [0, 0.28]])

  def reset(self, env):
    super().reset(env)

    # Determine the number of blocks.
    config_idx = np.random.choice(range(len(self.num_blocks_list)))
    num_blocks = self.num_blocks_list[config_idx]
    base_urdf = self.base_urdf_list[config_idx]
    base_size = self.base_size_list[config_idx]
    place_pos = self.place_pos_list[config_idx]
    primitives = [0] * num_blocks
    self.max_steps = num_blocks + 2
    print(f"block_num: {num_blocks}")
    
    # Add base.
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

    # Target positions in the world.
    targs = [(utils.apply(base_pose, i), base_pose[1]) for i in place_pos]

    # Goal
    self.goals.append((objs, np.ones((num_blocks, num_blocks)), targs,
        False, True, 'pose', None, 1, primitives))

    return base_urdf, base_size, base_id, objs_id