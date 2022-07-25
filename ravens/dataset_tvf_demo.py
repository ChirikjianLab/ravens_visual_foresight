"""Image dataset for demo collection of TVF."""

import os
import pickle

import numpy as np
from ravens import tasks
from ravens.tasks import cameras
import tensorflow as tf
from ravens.dataset import Dataset

# See transporter.py, regression.py, dummy.py, task.py, etc.
PIXEL_SIZE = 0.003125
CAMERA_CONFIG = cameras.RealSenseD415.CONFIG
BOUNDS = np.array([[0.25, 0.75], [-0.25, 0.25], [0, 0.28]])

# Names as strings, REVERSE-sorted so longer (more specific) names are first.
TASK_NAMES = (tasks.names).keys()
TASK_NAMES = sorted(TASK_NAMES)[::-1]


class DatasetTVFDemo(Dataset):
  """A simple image dataset class for collecting data for TVF."""

  def __init__(self, path):
    super().__init__(path)

  def add(self, seed, episode):
    """Add an episode to the dataset.

    Args:
      seed: random seed used to initialize the episode.
      episode: list of (pick_obs, obs, act, reward, info) tuples.
    """
    color, depth = [], []
    action, reward, info = [], [], []

    for obs, act, r, i in episode:
      color.append(obs['color'])
      depth.append(obs['depth'])
      action.append(act)
      reward.append(r)
      info.append(i)

    color = np.uint8(color)
    depth = np.float32(depth)

    def dump(data, field):
      field_path = os.path.join(self.path, field)
      if not tf.io.gfile.exists(field_path):
        tf.io.gfile.makedirs(field_path)
      fname = f'{self.n_episodes:06d}-{seed}.pkl'  # -{len(episode):06d}
      with tf.io.gfile.GFile(os.path.join(field_path, fname), 'wb') as f:
        pickle.dump(data, f)

    dump(color, 'color')
    dump(depth, 'depth')
    dump(action, 'action')
    dump(reward, 'reward')
    dump(info, 'info')

    self.n_episodes += 1
    self.max_seed = max(self.max_seed, seed)
