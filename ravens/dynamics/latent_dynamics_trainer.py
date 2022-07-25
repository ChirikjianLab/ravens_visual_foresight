"""Ablation Dynamics Trainer."""

import os

import numpy as np
from ravens.tasks import cameras
from ravens.utils import utils
from ravens.models.latent_dynamics import LatentDynamics
import tensorflow as tf

import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

class LatentDynamicsTrainer:
  """Class that trains with the collected data."""

  def __init__(self, model_name='resnet', repeat_H_lambda=5, h_only=False):
    """Constructor.
    
    Args:
        model_name: name of the model
    """
    print(f"model_name: {model_name}")
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
      self.in_shape = (160, 160, 1)  # H
      self.out_shape = (160, 160, 1) # H
    else:
      self.in_shape = (160, 160, 4) # RGBH
      self.out_shape = (160, 160, 4) # RGBH

    self.dynamics = LatentDynamics(
        self.in_shape, 
        self.out_shape[-1])
    print(f'in_shape: {self.in_shape}')
    print(f'out_shape: {self.out_shape}')
  
  def get_rgbh_from_obs(self, obs):
    """Get RBGH reconctructed from observations."""

    cmap, hmap = utils.get_fused_heightmap(
        obs, self.cam_config, self.bounds, self.pix_size)

    img = np.concatenate((cmap,
                          hmap[Ellipsis, None],
                          hmap[Ellipsis, None],
                          hmap[Ellipsis, None]), axis=2)

    return img

  def get_sample(self, dataset, augment=True, h_only=False):
    """Get a dataset sample.

    Args:
      dataset: a ravens.Dataset.
      augment: if True, perform data augmentation.
      h_only: if True, only the height channel will be used.
    """

    (obs, act, _, _), (target_obs, _, _, _) = dataset.sample()
    init_img = self.get_rgbh_from_obs(obs) # RGBHHH
    target_img = self.get_rgbh_from_obs(target_obs) # RGBHHH

    # Only translation is considered
    p0_xyz, p0_xyzw = act['pose0']
    p1_xyz, p1_xyzw = act['pose1']

    p0_theta = -np.float32(utils.quatXYZW_to_eulerXYZ(p0_xyzw)[2])
    assert p0_theta == 0.0
    p1_theta = -np.float32(utils.quatXYZW_to_eulerXYZ(p1_xyzw)[2])
    p1_theta = p1_theta - p0_theta
    
    # Map from xyz to pixel
    p0 = utils.xyz_to_pix(p0_xyz, self.bounds, self.pix_size)
    p1 = utils.xyz_to_pix(p1_xyz, self.bounds, self.pix_size)

    if not self.action_in_bound(p0):
      print('p0 is out of range')
      return None, None, None, None, None

    if not self.action_in_bound(p1):
      print('p1 is out of range')
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

  def preprocess_data_validate(self, obs, act, target_obs):
    """Preprocess the data for validation."""

    in_img = self.get_training_image(obs)
    target_data = self.get_target_data(target_obs)

    # Only translation is considered
    p0_xyz, _ = act['pose0']
    p1_xyz, _ = act['pose1']
    
    # Map from xyz to pixel
    p0 = utils.xyz_to_pix(p0_xyz, self.bounds, self.pix_size)
    p1 = utils.xyz_to_pix(p1_xyz, self.bounds, self.pix_size)
    
    # One-hot encoding for action
    (h, w, c) = in_img.shape
    pick_action = np.zeros((h, w))
    pick_action[p0[0], p0[1]] = 1.0
    place_action = np.zeros((h, w))
    place_action[p1[0], p1[1]] = 1.0

    input_data = np.concatenate((in_img, 
                                 pick_action[Ellipsis, None], 
                                 place_action[Ellipsis, None]), axis=2)

    return input_data, target_data

  def train(self, dataset, writer=None, h_only=False, real=False):
    """Train on a dataset sample for 1 iteration.
    
    Args:
      dataset: a ravens.Dataset.
      writer: a TF summary writer (for tensorboard).
    """

    # Set learning phase as training
    tf.keras.backend.set_learning_phase(1)

    # Get a training sample
    if real:
      raise NotImplementedError
    else:
      init_img, target_img, p0, p1, p1_theta = self.get_sample(dataset, h_only=h_only)

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
    if False:
      import matplotlib
      matplotlib.use('TkAgg')
      max_height = 0.14
      normalize = matplotlib.colors.Normalize(vmin=0.0, vmax=max_height)
      import matplotlib.pyplot as plt
      init_img_copy = np.copy(init_img)
      target_img_copy = np.copy(target_img)
      init_img_copy[p0[0], p0[1], :] = 10.0
      target_img_copy[p1[0], p1[1], :] = 10.0
      print(f"p1_theta: {p1_theta}")
      print(f"init_img: {init_img.shape}")
      print(f"target_img: {target_img.shape}")

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

    # Get training loss
    step = self.total_steps + 1
    loss = self.dynamics.train(
      init_img, 
      target_img,
      p0,
      p1,
      p1_theta,
      repeat_H_lambda=self.repeat_H_lambda,
      h_only=h_only)

    with writer.as_default():
      sc = tf.summary.scalar
      sc('train_loss/LatentDynamics', loss, step)
    print(f'Latent Dynamics -- Train Iter: {step} Loss: {loss:.4f}')
    self.total_steps = step

  def test(self, dataset):
    """Test the result from the dataset.
    
    Args:
      dataset: a ravens.Dataset.
    """

    raise NotImplementedError

  def imagine(self, img, act):
    """Imagine the image after the action."""
    
    raise NotImplementedError

  def validate(self, dataset, episode_num=20, visualize=False, real=False):
    """Validation."""
    
    tf.keras.backend.set_learning_phase(0)
    total_rgb_loss = 0.0
    total_height_loss = 0.0
    test_transition_num = 0
    for dataset_id in range(dataset.dataset_num):
      total_dataset_rgb_loss = 0.0
      total_dataset_height_loss = 0.0
      test_dataset_transition_num = 0
      for episode_id in range(episode_num):
        episode, _ = dataset.load(dataset_id, episode_id, images=True, cache=False)

        for i in range(len(episode) - 1):
          if real:
            raise NotImplementedError
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
          
          out_img, rgb_loss, height_loss = self.dynamics.test(
              init_img,
              target_img,
              p0,
              p1,
              p1_theta,
              repeat_H_lambda=1, # lambda = 1
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
      avg_datset_height_loss = total_dataset_height_loss / test_dataset_transition_num
      print('------------------')
      print(f'dataset: {dataset.path_list[dataset_id]}')
      print(f'Test transition num: {test_dataset_transition_num} Avg RGB loss: {avg_dataset_rgb_loss:.9f} Avg Height loss: {avg_datset_height_loss:.9f}')

    print('***********************')
    avg_rgb_loss = total_rgb_loss / test_transition_num
    avg_height_loss = total_height_loss / test_transition_num
    print(f'Test transition num: {test_transition_num}')
    print(f'Avg RGB loss: {avg_rgb_loss:.9f}')
    print(f'Avg Height loss: {avg_height_loss:.9f}')

    return total_rgb_loss, total_height_loss, test_transition_num

  def load_from_path(self, model_path, total_steps=0):
    print(f'Loading pre-trained model from {model_path}')
    self.dynamics.load(model_path)
    self.total_steps = total_steps

  def save_to_dir(self, model_dir, model_name):
    """Save models to path."""
    if not tf.io.gfile.exists(model_dir):
      tf.io.gfile.makedirs(model_dir)
    dynamics_fname = f'{model_name}-ckpt-{self.total_steps}.h5'
    dynamics_fname= os.path.join(model_dir, dynamics_fname)
    self.dynamics.save(dynamics_fname)

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