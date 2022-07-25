"""Trainer for PP Dynamics."""

import os
import numpy as np
from ravens.tasks import cameras
from ravens.utils import utils
from ravens.models.pp_dynamics import PPDynamics
import tensorflow as tf

import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt


class PPDynamicsTrainer:
  """Class for training the PP dynamics."""

  def __init__(self, model_name, repeat_H_lambda=5, h_only=False):
    """Constructor.

    Args:
      model_name: name of the model
      repeat_H_lambda: weight for the height channel in the L2 loss
      h_only: flag of training only H
    """

    self.total_steps = 0
    self.mask_size = 44
    self.h_only = h_only

    self.repeat_H_lambda = repeat_H_lambda
    self.pix_size = 0.003125
    self.cam_config = cameras.RealSenseD415.CONFIG
    self.bounds = np.array([[0.25, 0.75], [-0.25, 0.25], [0, 0.28]])
    # self.bounds = np.array([[0.25, 0.75], [-0.5, 0.5], [0, 0.28]])
    self.model_name = model_name

    if self.h_only:
      self.in_shape = (160, 160, 3)  # H + pick_mask + place_mask
      self.out_shape = (160, 160, 1) # H
    else:
      self.in_shape = (160, 160, 9) # RGBH + pick_mask + place_mask
      self.out_shape = (160, 160, 4) # RGBH

    self.dynamics = PPDynamics(
        self.in_shape, 
        self.out_shape[-1], 
        self.mask_size,
        model_name=self.model_name)

  def get_rgbh_from_obs(self, obs):
    """Get RBGH reconctructed from observations."""

    cmap, hmap = utils.get_fused_heightmap(
        obs, self.cam_config, self.bounds, self.pix_size)

    img = np.concatenate((cmap,
                          hmap[Ellipsis, None],
                          hmap[Ellipsis, None],
                          hmap[Ellipsis, None]), axis=2)

    return img

  def get_sample_pp(self, dataset, augment=True):
    """Get a sample from ravens.DatasetPPDynamics."""

    # Get a sample from the dataset.
    # (_, init_obs, act, _, _), (_, target_obs, _, _, _) = dataset.sample(cache=True)
    (init_obs, act, _, _), (target_obs, _, _, _) = dataset.sample()
    init_img = self.get_rgbh_from_obs(init_obs)
    target_img = self.get_rgbh_from_obs(target_obs)

    # Get the action.
    p0_xyz, p0_xyzw = act['pose0']
    p1_xyz, p1_xyzw = act['pose1']
    p0 = utils.xyz_to_pix(p0_xyz, self.bounds, self.pix_size)
    p1 = utils.xyz_to_pix(p1_xyz, self.bounds, self.pix_size)
    p0_theta = -np.float32(utils.quatXYZW_to_eulerXYZ(p0_xyzw)[2])
    p1_theta = -np.float32(utils.quatXYZW_to_eulerXYZ(p1_xyzw)[2])
    assert p0_theta == 0.0
    p1_theta = p1_theta - p0_theta

    # Check the actions are in bound.
    if not self.action_in_bound(p0):
      return None, None, None, None, None
    if not self.action_in_bound(p1):
      return None, None, None, None, None

    # Data augmentation.
    if augment:
      # Make sure that the whole block is 
      # still in the scene after perturbation.
      p0_1 = (p0[0] - 20, p0[1])
      p0_2 = (p0[0] + 20, p0[1])
      p0_3 = (p0[0], p0[1] - 20)
      p0_4 = (p0[0], p0[1] + 20)
      p1_1 = (p1[0] - 20, p1[1])
      p1_2 = (p1[0] + 20, p1[1])
      p1_3 = (p1[0], p1[1] - 20)
      p1_4 = (p1[0], p1[1] + 20)

      [init_img, target_img], _, [p0, _, _, _, _, p1, _, _, _, _], _ = utils.perturb_pp(
          [init_img, target_img],
          [p0, p0_1, p0_2, p0_3, p0_4, p1, p1_1, p1_2, p1_3, p1_4])

    return init_img, target_img, p0, p1, p1_theta

  def get_sample_pp_real(self, dataset, augment=True):
    """Get a sample from ravens.DatasetPPDynamicsReal."""

    # Get a sample from the dataset.
    (init_cmap, init_hmap, act), (target_cmap, target_hmap, _) = dataset.sample()
    
    init_cmap, init_hmap = PPDynamicsTrainer.preprocess(init_cmap, init_hmap)
    target_cmap, target_hmap = PPDynamicsTrainer.preprocess(target_cmap, target_hmap)

    init_img = np.concatenate((init_cmap,
                               init_hmap[Ellipsis, None],
                               init_hmap[Ellipsis, None],
                               init_hmap[Ellipsis, None]), axis=2)
    target_img = np.concatenate((target_cmap,
                                 target_hmap[Ellipsis, None],
                                 target_hmap[Ellipsis, None],
                                 target_hmap[Ellipsis, None]), axis=2)

    # Get the action.
    p0 = [int(act['pose0'][1]), int(act['pose0'][0])]
    p0_theta = act['pose0'][2]
    p1 = [int(act['pose1'][1]), int(act['pose1'][0])]
    p1_theta = act['pose1'][2]
    assert p0_theta == 0.0
    p1_theta = (p1_theta - p0_theta)

    # Check the actions are in bound.
    if not self.action_in_bound(p0):
      return None, None, None, None, None
    if not self.action_in_bound(p1):
      return None, None, None, None, None

    # Data augmentation.
    if augment:
      # Make sure that the whole block is 
      # still in the scene after perturbation.
      p0_1 = (p0[0] - 20, p0[1])
      p0_2 = (p0[0] + 20, p0[1])
      p0_3 = (p0[0], p0[1] - 20)
      p0_4 = (p0[0], p0[1] + 20)
      p1_1 = (p1[0] - 20, p1[1])
      p1_2 = (p1[0] + 20, p1[1])
      p1_3 = (p1[0], p1[1] - 20)
      p1_4 = (p1[0], p1[1] + 20)

      [init_img, target_img], _, [p0, _, _, _, _, p1, _, _, _, _], _ = utils.perturb_pp(
          [init_img, target_img],
          [p0, p0_1, p0_2, p0_3, p0_4, p1, p1_1, p1_2, p1_3, p1_4])

    return init_img, target_img, p0, p1, p1_theta

  @staticmethod
  def preprocess(cmap, hmap):
    hmap_temp = np.copy(hmap)
    hmap_means = np.mean(hmap_temp)
    hmap_temp = hmap_temp * (hmap_temp > 1.2 * hmap_means)

    cmap_temp = np.copy(cmap)
    cmap_normalize = np.copy(cmap_temp) / 255.0
    cmap_hsv = matplotlib.colors.rgb_to_hsv(cmap_normalize)
    for i in range(3):
      cmap_temp[:, :, i] = cmap_temp[:, :, i] * (hmap_temp > 1e-4) * (cmap_hsv[:, :, 2] > 0.3)
    hmap_temp = hmap_temp * (cmap_hsv[:, :, 2] > 0.3)

    return cmap_temp, hmap_temp
    
  def train_pp(self, dataset, writer=None, debug=False, real=False):
    """Train on a data point sampled from a random dataset and random episode.
    
    Args:
      dataset: a ravens.DatasetPPDynamics.
      writer: a TF summary writer (for tensorboard).
    """

    # Set learning phase as training.
    tf.keras.backend.set_learning_phase(1)

    # Get a training sample.
    if real:
      init_img, target_img, p0, p1, p1_theta = self.get_sample_pp_real(dataset, augment=True)
    else:
      init_img, target_img, p0, p1, p1_theta = self.get_sample_pp(dataset, augment=True)
    if init_img is None:
      return

    if self.h_only:
      # Picking H from the images.
      init_img = init_img[:, :, 3:4]
      target_img = target_img[:, :, 3:4]
    else:
      # Only taking the RGBH channel for the target data.
      init_img = init_img[:, :, :4]
      target_img = target_img[:, :, :4]

    # Debug
    if debug:
      import matplotlib
      matplotlib.use('TkAgg')
      max_height = 0.14
      normalize = matplotlib.colors.Normalize(vmin=0.0, vmax=max_height)
      import matplotlib.pyplot as plt
      init_img_copy = np.copy(init_img)
      target_img_copy = np.copy(target_img)
      init_img_copy[p0[0], p0[1], :] = 10.0
      target_img_copy[p1[0], p1[1], :] = 10.0

      print(f"p1_theta: {180 * p1_theta / np.pi}")

      if self.h_only:
        f, ax = plt.subplots(2)
        ax[0].imshow(init_img_copy[:, :, 0], norm=normalize)
        ax[1].imshow(target_img_copy[:, :, 0], norm=normalize)
      else:
        f, ax = plt.subplots(4)
        ax[0].imshow(init_img_copy[:, :, :3] / 255.0)
        ax[1].imshow(init_img_copy[:, :, 3], norm=normalize)
        ax[2].imshow(target_img_copy[:, :, :3] / 255.0)
        ax[3].imshow(target_img_copy[:, :, 3], norm=normalize)
      plt.show()

    # Get training loss.
    step = self.total_steps + 1
    loss = self.dynamics.train_pp(
        init_img,
        target_img,
        p0,
        p1,
        p1_theta,
        repeat_H_lambda=self.repeat_H_lambda,
        h_only=self.h_only)

    with writer.as_default():
      sc = tf.summary.scalar
      sc('train_loss/PP', loss, step)
    print(f'PP Dynamics -- Train Iter: {step} Loss: {loss:.4f}')
    self.total_steps = step

  def validate_pp(self, dataset, episode_num=20, visualize=False, real=False):
    """Validation for PP Dynamics."""

    # Set learning phase as testing.
    tf.keras.backend.set_learning_phase(0)

    dataset_num = dataset.dataset_num

    total_rgb_loss = 0.0
    total_height_loss = 0.0
    test_transition_num = 0
    for dataset_id in range(dataset_num):
      total_dataset_rgb_loss = 0.0
      total_dataset_height_loss = 0.0
      test_dataset_transition_num = 0
      for episode_id in range(episode_num):
        if real:
          episode  = dataset.load(dataset_id, episode_id)
        else:
          episode, _ = dataset.load(dataset_id, episode_id, images=True, cache=False)

        for i in range(len(episode) - 1):
          if real:
            (init_cmap, init_hmap, act) = episode[i]
            (target_cmap, target_hmap, _) = episode[i+1]

            init_cmap, init_hmap = PPDynamicsTrainer.preprocess(init_cmap, init_hmap)
            target_cmap, target_hmap = PPDynamicsTrainer.preprocess(target_cmap, target_hmap)

            init_img = np.concatenate([init_cmap, init_hmap[Ellipsis, None]], axis=2)
            target_img = np.concatenate([target_cmap, target_hmap[Ellipsis, None]], axis=2)

            # Get the action.
            p0 = [int(act['pose0'][1]), int(act['pose0'][0])]
            p0_theta = act['pose0'][2]
            p1 = [int(act['pose1'][1]), int(act['pose1'][0])]
            p1_theta = act['pose1'][2]
            assert p0_theta == 0.0
            p1_theta = (p1_theta - p0_theta)
          else:
            (init_obs, act, _, _) = episode[i]
            (target_obs, _, _, _) = episode[i+1]

            if init_obs is None:
              continue

            init_img = self.get_rgbh_from_obs(init_obs)
            target_img = self.get_rgbh_from_obs(target_obs)

            p0_xyz, p0_xyzw = act['pose0']
            p1_xyz, p1_xyzw = act['pose1']
            p0 = utils.xyz_to_pix(p0_xyz, self.bounds, self.pix_size)
            p1 = utils.xyz_to_pix(p1_xyz, self.bounds, self.pix_size)
            p0_theta = -np.float32(utils.quatXYZW_to_eulerXYZ(p0_xyzw)[2])
            p1_theta = -np.float32(utils.quatXYZW_to_eulerXYZ(p1_xyzw)[2])
            assert p0_theta == 0.0
            p1_theta = p1_theta - p0_theta

          # Check the actions are in bound.
          if not self.action_in_bound(p0):
            continue
          if not self.action_in_bound(p1):
            continue
          
          init_img = init_img[:, :, :4]
          target_img = target_img[:, :, :4]
          
          if self.h_only:
            init_img = init_img[:, :, 3:4]
            target_img = target_img[:, :, 3:4]

          init_img_copy = np.copy(init_img)
          target_img_copy = np.copy(target_img)
          
          out_img, rgb_loss, height_loss = self.dynamics.test_pp(
              init_img,  
              target_img, 
              p0, 
              p1, 
              p1_theta, 
              repeat_H_lambda=1, 
              h_only=self.h_only)

          if visualize:
            max_height = 0.14
            normalize = matplotlib.colors.Normalize(vmin=0.0, vmax=max_height)
            if self.h_only:
              f, ax = plt.subplots(2, 2)
              init_img_copy[p0[0], p0[1]] = 0.0
              target_img_copy[p1[0], p1[1]] = 0.0 
              ax[0, 0].imshow(init_img_copy, norm=normalize)
              ax[0, 1].imshow(target_img_copy, norm=normalize)
              ax[1, 0].imshow(target_img_copy - out_img)
              ax[1, 1].imshow(out_img, norm=normalize)
            else:
              f, ax = plt.subplots(3, 2)
              init_img_copy[p0[0], p0[1], -1] = 0.0
              target_img_copy[p1[0], p1[1], -1] = 0.0
              ax[0, 0].imshow(init_img_copy[:, :, :3] / 255.0)
              ax[0, 1].imshow(init_img_copy[:, :, -1], norm=normalize)
              ax[1, 0].imshow(target_img_copy[:, :, :3] / 255.0)
              ax[1, 1].imshow(target_img_copy[:, :, -1], norm=normalize)
              ax[2, 0].imshow(out_img[:, :, :3] / 255.0)
              ax[2, 1].imshow(out_img[:, :, -1], norm=normalize)
            plt.show()

          total_dataset_rgb_loss += rgb_loss
          total_dataset_height_loss += height_loss
          test_dataset_transition_num += 1

      total_rgb_loss += total_dataset_rgb_loss
      total_height_loss += total_dataset_height_loss
      test_transition_num += test_dataset_transition_num

      avg_dataset_rgb_loss = total_dataset_rgb_loss / test_dataset_transition_num
      avg_dataset_height_loss = total_dataset_height_loss / test_dataset_transition_num
      print('------------------')
      print(f'Dataset: {dataset.path_list[dataset_id]}')
      print(f'Test transition num: {test_dataset_transition_num} Avg RGB loss: {avg_dataset_rgb_loss:.9f} Avg Height loss: {avg_dataset_height_loss:.9f}')

    return total_rgb_loss, total_height_loss, test_transition_num

  def imagine_img(self, img, act):
    """Imagine the image after taking the action.
    
    Args:
      img: RGBHHH
      act: action. It is a dictionary in the form as
        {'pose0': (pos0, quat0), 'pose1': (pos1, quat1)}
    """

    p0_xyz, p0_xyzw = act['pose0']
    p1_xyz, p1_xyzw = act['pose1']
    p0 = utils.xyz_to_pix(p0_xyz, self.bounds, self.pix_size)
    p1 = utils.xyz_to_pix(p1_xyz, self.bounds, self.pix_size)
    p0_theta = -np.float32(utils.quatXYZW_to_eulerXYZ(p0_xyzw)[2])
    p1_theta = -np.float32(utils.quatXYZW_to_eulerXYZ(p1_xyzw)[2])
    assert p0_theta == 0.0
    p1_theta = p1_theta - p0_theta

    assert self.action_in_bound(p0)
    assert self.action_in_bound(p1)

    if self.h_only:
      img = img[:, :, 3:4]
    else:
      img = img[:, :, :4]
    
    out_img = self.dynamics.imagine(
      img, p0, p1, p1_theta, self.h_only)

    return out_img

  def save_to_dir(self, model_dir, model_name):
    """Save model to path."""

    if not tf.io.gfile.exists(model_dir):
      tf.io.gfile.makedirs(model_dir)
    model_fname = f'{model_name}-ckpt-{self.total_steps}.h5'
    model_path = os.path.join(model_dir, model_fname)
    self.dynamics.save(model_path)

  def load_from_path(self, model_path, total_steps):
    """Load pre-trained model."""

    self.dynamics.load(model_path)
    self.total_steps = total_steps
    print(f'Loaded pre-trained model from: {model_path}')

  def action_in_bound(self, p):
    """Check if the action is in bound."""

    if p[0] >= self.in_shape[0]:
      print(f'p is out of bound: {p}')
      return False
    if p[0] < 0:
      print(f'p is out of bound: {p}')
      return False
    if p[1] >= self.in_shape[1]:
      print(f'p is out of bound: {p}')
      return False
    if p[1] < 0:
      print(f'p is out of bound: {p}')
      return False

    return True