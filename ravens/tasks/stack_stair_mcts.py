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

class StackStairMCTS(TaskMCTS):
  """Stack stair task (mcts)."""

  def __init__(self, *args, **kwargs):
    super().__init__(*args, **kwargs)
    self.max_steps = 5

    # Workspace bounds (0.5 x 0.5 m2).
    self.bounds = np.array([[0.25, 0.75], [-0.25, 0.25], [0, 0.28]])
    # self.bounds = np.array([[0.25, 0.75], [-0.5, 0.5], [0, 0.28]])

  def reset(self, env):
    super().reset(env)

    # Add base.
    base_size = (0.05, 0.10, 0.005)
    base_urdf = 'stand/stand_2.urdf'
    base_pose = self.get_random_pose(env, base_size)
    base_id = env.add_object(base_urdf, base_pose, 'fixed')

    # Block colors.
    colors = [utils.COLORS['red']] * 3
    num_blocks = 3

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

    # Two different ways of stacking squares
    seq_idx_list = [0, 1]
    seq_idx = np.random.choice(seq_idx_list)

    if seq_idx == 0:
      # First way is to stack layer by layer.
      place_pos = [(0,     0,  block_size[-1] / 2 + 0 * block_size[-1]), 
                   (0,   0.05, block_size[-1] / 2 + 0 * block_size[-1]),
                   (0,     0,  block_size[-1] / 2 + 1 * block_size[-1])]
      primitives = [0, 0, 1]

      # Target positions in the world.
      targs = [(utils.apply(base_pose, i), base_pose[1]) for i in place_pos]

      if self.mode == 'train':
        # Goal: blocks are stacked in a pyramid.
        self.goals.append((objs[:2], np.ones((2, 2)), targs[:2],
            False, True, 'pose', None, 2 / 3, primitives[:2]))
        self.goals.append((objs[2:], np.ones((1, 1)), targs[2:],
            False, True, 'pose', None, 1 / 3, primitives[2:]))
      else:
        self.goals.append((objs, np.ones((num_blocks, num_blocks)), targs,
            False, True, 'pose', None, 1, primitives))
    
    elif seq_idx == 1:
      # Second way is to build tower by tower
      place_pos = [(0,     0,  block_size[-1] / 2 + 0 * block_size[-1]),
                   (0,     0,  block_size[-1] / 2 + 1 * block_size[-1]), 
                   (0,   0.05, block_size[-1] / 2 + 0 * block_size[-1])]
      primitives = [0, 1, 0]

      # Target positions in the world.
      targs = [(utils.apply(base_pose, i), base_pose[1]) for i in place_pos]

      if self.mode == 'train':
        # Goal: blocks are stacked in a pyramid.
        self.goals.append((objs[:1], np.ones((1, 1)), targs[:1],
            False, True, 'pose', None, 1 / 3, primitives[:1]))
        self.goals.append((objs[1:2], np.ones((1, 1)), targs[1:2],
            False, True, 'pose', None, 1 / 3, primitives[1:2]))
        self.goals.append((objs[2:3], np.ones((1, 1)), targs[2:3],
            False, True, 'pose', None, 1 / 3, primitives[2:3]))
      else:
        self.goals.append((objs, np.ones((num_blocks, num_blocks)), targs,
            False, True, 'pose', None, 1, primitives))

    return base_urdf, base_size, base_id, objs_id