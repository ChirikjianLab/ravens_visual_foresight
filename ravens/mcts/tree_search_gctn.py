"""Best value search for visual task planning."""

import copy
import numpy as np
from ravens.dynamics.pp_dynamics_trainer import PPDynamicsTrainer
from ravens import agents_gctn as agents
from ravens.utils import utils
from ravens.tasks import cameras

import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt


class BestFirstSearch:
  """Class to perform best first search for visual task planning."""

  def __init__(self,
               num_bootstraps,
               k,
               dynamics_model_name,
               h_only=False):
    """Constructor.
    
    Args:
      seq_len: length for searching.
      h_only: flag to use H only.
      dynamics_model: model name of the dynamics.
      model_root_dir: root directory of the model
    """

    self.pix_size = 0.003125
    self.cam_config = cameras.RealSenseD415.CONFIG
    self.bounds = np.array([[0.25, 0.75], [-0.25, 0.25], [0, 0.28]])
    self.img_shape = (160, 160, 4)

    # h_only flag
    self.h_only = h_only

    # Dynamics handler.
    self.num_bootstraps = num_bootstraps
    self.dynamics_bootstraps = []
    for i in range(self.num_bootstraps):
      self.dynamics_bootstraps.append(PPDynamicsTrainer(dynamics_model_name, h_only=h_only))

    # Primitive handler.
    name = f'GCTN-Multi-transporter-goal-n-40000'
    self.agent = agents.names['transporter-goal'](name, None, num_rotations=36)

    # Number of primitive
    self.k = k
    print(f'K-means: {k}')

    # RGBH image of the goal scene.
    self.goal_img = None 
    self.goal_img_rgbhhh = None

    # Means and variances for color and depth images.
    self.color_mean = 0.18877631
    self.depth_mean = 0.00509261
    self.color_std = 0.07276466
    self.depth_std = 0.00903967

  def load_dynamics_model(self, bootstrap_idx, model_path, total_steps):
    """Load the model for the dynamics model."""

    self.dynamics_bootstraps[bootstrap_idx].load_from_path(model_path, total_steps)
    print(f'Load dynamics model from: {model_path}')

  def load_gctn_model(self, attention_path, transport_path, n_iters):
    """Load the model for the primitive."""

    self.agent.load_from_path(attention_path, transport_path, n_iters)
    print(f'Load gctn attention model from: {attention_path}')
    print(f'Load gctn transport model from: {transport_path}')

  def set_goal_obs(self, goal_obs):
    """Set the goal of the search with observation from environment."""

    goal_img_rgbhhh = self.get_image(goal_obs)
    self.set_goal_img(goal_img_rgbhhh)

  def set_goal_img(self, goal_img_rgbhhh):
    """Set the goal of the search."""

    self.goal_img = goal_img_rgbhhh[:, :, :4]
    self.goal_img_rgbhhh = goal_img_rgbhhh

  def expand_img(self, img_rgbhhh, visualize):
    """Expand different primitives from a state.
    
    Args:
      img: RGBHHH image of the current scene.
      seq_len: length of the rollout.
    Returns:
      acts: list of actions taken by each primitive.
      next_imgs: list of images after taking the actions.
    """ 

    # Get the policy and imagined images for each primitive.
    acts = []
    next_imgs = []

    input_img = np.copy(img_rgbhhh)
    goal_img = np.copy(self.goal_img_rgbhhh)
    acts = self.agent.get_top_k_acts(self.k, input_img, goal_img)

    for i in range(len(acts)):
      next_imgs.append([])
    
    for i in range(len(acts)):
      for j in range(self.num_bootstraps):
        input_img = np.copy(img_rgbhhh)
        next_img = self.dynamics_bootstraps[j].imagine_img(input_img, acts[i])
        next_imgs[i].append(next_img)

    # Compute the means of next_imgs
    next_imgs_means = []
    for i in range(len(acts)):
      means = np.zeros(next_imgs[0][0].shape)
      for j in range(self.num_bootstraps):
        means += next_imgs[i][j]
      means /= self.num_bootstraps
      next_imgs_means.append(means)

    if visualize:
      f, ax = plt.subplots(len(acts), 2 + 2)
      max_height = 0.14
      normalize = matplotlib.colors.Normalize(vmin=0.0, vmax=max_height)

      for i in range(len(acts)):
         # Get training labels from data sample.
        p0_xyz, p0_xyzw = acts[i]['pose0']
        p1_xyz, p1_xyzw = acts[i]['pose1']
        p0 = utils.xyz_to_pix(p0_xyz, self.bounds, self.pix_size)
        p1 = utils.xyz_to_pix(p1_xyz, self.bounds, self.pix_size)

        img_copy = np.copy(img_rgbhhh)
        img_copy[p0[0], p0[1], :] = 10.0  
        img_copy[p1[0], p1[1], :] = 10.0
        ax[i, 0].imshow(img_copy[:, :, :3] / 255.0)
        ax[i, 1].imshow(img_copy[:, :, 3], norm=normalize)

        ax[i, 2].imshow(next_imgs_means[i][:, :, :3] / 255.0)
        ax[i, 3].imshow(next_imgs_means[i][:, :, 3], norm=normalize)
      
      plt.show()

    return acts, next_imgs, next_imgs_means

  def rollout_img(self, img_rgbhhh, steps, visualize):
    """Rollout the sequence from an input image. Note that
        only one bootstraps is used.
    """
    input_img = np.copy(img_rgbhhh)
    goal_img = np.copy(self.goal_img_rgbhhh)

    next_img_list = []
    act_list = []
    for i in range(steps):
      input_imagine_img = np.copy(input_img)
      # Get the action.
      act = self.agent.get_top_k_acts(1, input_img, goal_img)
      act_list.append(act[0])
      # Visual foresight.
      next_img = self.dynamics_bootstraps[0].imagine_img(input_imagine_img, act[0])
      next_img_list.append(next_img)
      input_img = self.make_rgbhhh(next_img)

    if visualize:
      f, ax = plt.subplots(2, steps + 1)
      max_height = 0.14
      normalize = matplotlib.colors.Normalize(vmin=0.0, vmax=max_height)

      ax[0, 0].imshow(img_rgbhhh[:, :, :3] / 255.0)
      ax[1, 0].imshow(img_rgbhhh[:, :, 3], norm=normalize)

      for i in range(steps):
        ax[0, i+1].imshow(next_img_list[i][:, :, :3]/255.0)
        ax[1, i+1].imshow(next_img_list[i][:, :, 3], norm=normalize)
      
      plt.show()

    return act_list, next_img_list

  def expand_obs_depth1(self, obs, visualize):
    """Imagine the result of taking different primitive 
        actions for two steps.
    
    Args:
      obs: observation of the scene.
      visualize: flag to visualize the imagination.
      depth: depth of visualization.
    """

    # Get the RGBHHH image.
    img = self.get_image(obs)

    # Imagine the first step
    acts_1, next_imgs_1, next_imgs_means_1 = self.expand_img(img, visualize)

    return acts_1, next_imgs_1, next_imgs_means_1

  def expand_obs_depth2(self, obs, visualize):
    """Imagine the result of taking different primitive 
        actions for two steps.
    
    Args:
      obs: observation of the scene.
      visualize: flag to visualize the imagination.
    """

    # Get the RGBHHH image.
    img = self.get_image(obs)

    # Imagine the first step.
    acts_1, next_imgs_1, next_imgs_means_1 = self.expand_img(img, visualize)

    # Imagine the second step.
    acts_2 = []
    next_imgs_2 = []
    next_imgs_means_2 = []
    for i in range(self.k):
      img = self.make_rgbhhh(next_imgs_means_1[i])
      acts_2_k, next_imgs_2_k, next_imgs_means_2_k = self.expand_img(img, visualize)
      acts_2.append(acts_2_k)
      next_imgs_2.append(next_imgs_2_k)
      next_imgs_means_2.append(next_imgs_means_2_k)

    return [acts_1, acts_2], [next_imgs_1, next_imgs_2], [next_imgs_means_1, next_imgs_means_2]

  def expand_obs_depth3(self, obs, visualize):
    """Imagine the result of taking different primitive 
        actions for three steps.
      
      Args:
        obs: observation of the scene.
        visualize: flag to visualize the imagination.
    """

    # Get the RGBHHH image.
    img = self.get_image(obs)

    # Imagine the first step.
    acts_1, next_imgs_1, next_imgs_means_1 = self.expand_img(img, visualize)

    # Imagine the second step.
    acts_2 = []
    next_imgs_2 = []
    next_imgs_means_2 = []
    for i in range(self.k):
      img = self.make_rgbhhh(next_imgs_means_1[i])
      acts_2_primitive, next_imgs_2_primitive, next_imgs_means_2_primitive = self.expand_img(img, visualize)
      acts_2.append(acts_2_primitive)
      next_imgs_2.append(next_imgs_2_primitive)
      next_imgs_means_2.append(next_imgs_means_2_primitive)

    # Imagine the third step.
    acts_3 = []
    next_imgs_3 = []
    next_imgs_means_3 = []
    for i_0 in range(self.k):
      acts_3_d0 = []
      next_imgs_3_d0 = []
      next_imgs_means_3_d0 = []
      for i_1 in range(self.k):
        img = self.make_rgbhhh(next_imgs_means_2[i_0][i_1])
        acts_3_d1, next_imgs_3_d1, next_imgs_means_3_d1 = self.expand_img(img, visualize)
        acts_3_d0.append(acts_3_d1)
        next_imgs_3_d0.append(next_imgs_3_d1)
        next_imgs_means_3_d0.append(next_imgs_means_3_d1)
      acts_3.append(acts_3_d0)
      next_imgs_3.append(next_imgs_3_d0)
      next_imgs_means_3.append(next_imgs_means_3_d0)

    return [acts_1, acts_2, acts_3], [next_imgs_1, next_imgs_2, next_imgs_3], [next_imgs_means_1, next_imgs_means_2, next_imgs_means_3]

  def expand_obs_depth4(self, obs, visualize):
    """Imagine the result of taking different primitive 
        actions for three steps.
      
      Args:
        obs: observation of the scene.
        visualize: flag to visualize the imagination.
    """

    # Get the RGBHHH image.
    img = self.get_image(obs)

    # Imagine the first step.
    acts_1, next_imgs_1, next_imgs_means_1 = self.expand_img(img, visualize)

    # Imagine the second step.
    acts_2 = []
    next_imgs_2 = []
    next_imgs_means_2 = []
    for i in range(self.k):
      img = self.make_rgbhhh(next_imgs_means_1[i])
      acts_2_primitive, next_imgs_2_primitive, next_imgs_means_2_primitive = self.expand_img(img, visualize)
      acts_2.append(acts_2_primitive)
      next_imgs_2.append(next_imgs_2_primitive)
      next_imgs_means_2.append(next_imgs_means_2_primitive)

    # Imagine the third step.
    acts_3 = []
    next_imgs_3 = []
    next_imgs_means_3 = []
    for i_0 in range(self.k):
      acts_3_d0 = []
      next_imgs_3_d0 = []
      next_imgs_means_3_d0 = []
      for i_1 in range(self.k):
        img = self.make_rgbhhh(next_imgs_means_2[i_0][i_1])
        acts_3_d1, next_imgs_3_d1, next_imgs_means_3_d1 = self.expand_img(img, visualize)
        acts_3_d0.append(acts_3_d1)
        next_imgs_3_d0.append(next_imgs_3_d1)
        next_imgs_means_3_d0.append(next_imgs_means_3_d1)
      acts_3.append(acts_3_d0)
      next_imgs_3.append(next_imgs_3_d0)
      next_imgs_means_3.append(next_imgs_means_3_d0)

    # Imagine the fourth step.
    acts_4 = []
    next_imgs_4 = []
    next_imgs_means_4 = []
    for i_0 in range(self.k):
      acts_4_d0 = []
      next_imgs_4_d0 = []
      next_imgs_means_4_d0 = []
      for i_1 in range(self.k):
        acts_4_d1 = []
        next_imgs_4_d1 = []
        next_imgs_means_4_d1 = []
        for i_2 in range(self.k):
          img = self.make_rgbhhh(next_imgs_means_3[i_0][i_1][i_2])
          acts_4_d2, next_imgs_4_d2, next_imgs_means_4_d2 = self.expand_img(img, visualize)
          acts_4_d1.append(acts_4_d2)
          next_imgs_4_d1.append(next_imgs_4_d2)
          next_imgs_means_4_d1.append(next_imgs_means_4_d2)
        acts_4_d0.append(acts_4_d1)
        next_imgs_4_d0.append(next_imgs_4_d1)
        next_imgs_means_4_d0.append(next_imgs_means_4_d1)
      acts_4.append(acts_4_d0)
      next_imgs_4.append(next_imgs_4_d0)
      next_imgs_means_4.append(next_imgs_means_4_d0)

    return [acts_1, acts_2, acts_3, acts_4], [next_imgs_1, next_imgs_2, next_imgs_3, next_imgs_4], [next_imgs_means_1, next_imgs_means_2, next_imgs_means_3, next_imgs_means_4]

  def expand_obs_depth4_rollout1(self, obs, visualize):
    """Imagine the result of taking different primitive 
        actions for three steps.
      
      Args:
        obs: observation of the scene.
        visualize: flag to visualize the imagination.
    """

    # Get the RGBHHH image.
    img = self.get_image(obs)

    # Imagine the first step.
    acts_1, next_imgs_1, next_imgs_means_1 = self.expand_img(img, visualize)

    # Imagine the second step.
    acts_2 = []
    next_imgs_2 = []
    next_imgs_means_2 = []
    for i in range(self.k):
      img = self.make_rgbhhh(next_imgs_means_1[i])
      acts_2_primitive, next_imgs_2_primitive, next_imgs_means_2_primitive = self.expand_img(img, visualize)
      acts_2.append(acts_2_primitive)
      next_imgs_2.append(next_imgs_2_primitive)
      next_imgs_means_2.append(next_imgs_means_2_primitive)

    # Imagine the third step.
    acts_3 = []
    next_imgs_3 = []
    next_imgs_means_3 = []
    for i_0 in range(self.k):
      acts_3_d0 = []
      next_imgs_3_d0 = []
      next_imgs_means_3_d0 = []
      for i_1 in range(self.k):
        img = self.make_rgbhhh(next_imgs_means_2[i_0][i_1])
        acts_3_d1, next_imgs_3_d1, next_imgs_means_3_d1 = self.expand_img(img, visualize)
        acts_3_d0.append(acts_3_d1)
        next_imgs_3_d0.append(next_imgs_3_d1)
        next_imgs_means_3_d0.append(next_imgs_means_3_d1)
      acts_3.append(acts_3_d0)
      next_imgs_3.append(next_imgs_3_d0)
      next_imgs_means_3.append(next_imgs_means_3_d0)

    # Imagine the fourth step.
    acts_4 = []
    next_imgs_4 = []
    next_imgs_means_4 = []
    for i_0 in range(self.k):
      acts_4_d0 = []
      next_imgs_4_d0 = []
      next_imgs_means_4_d0 = []
      for i_1 in range(self.k):
        acts_4_d1 = []
        next_imgs_4_d1 = []
        next_imgs_means_4_d1 = []
        for i_2 in range(self.k):
          img = self.make_rgbhhh(next_imgs_means_3[i_0][i_1][i_2])
          acts_4_d2, next_imgs_4_d2, next_imgs_means_4_d2 = self.expand_img(img, visualize)
          acts_4_d1.append(acts_4_d2)
          next_imgs_4_d1.append(next_imgs_4_d2)
          next_imgs_means_4_d1.append(next_imgs_means_4_d2)
        acts_4_d0.append(acts_4_d1)
        next_imgs_4_d0.append(next_imgs_4_d1)
        next_imgs_means_4_d0.append(next_imgs_means_4_d1)
      acts_4.append(acts_4_d0)
      next_imgs_4.append(next_imgs_4_d0)
      next_imgs_means_4.append(next_imgs_means_4_d0)

    # Rollout the last step.
    acts_5 = []
    next_imgs_5 = []
    for i_0 in range(self.k):
      acts_5_d0 = []
      next_imgs_5_d0 = []
      for i_1 in range(self.k):
        acts_5_d1 = []
        next_imgs_5_d1 = []
        for i_2 in range(self.k):
          acts_5_d2 = []
          next_imgs_5_d2 = []
          for i_3 in range(self.k):
            img = self.make_rgbhhh(next_imgs_means_4[i_0][i_1][i_2][i_3])
            acts_5_rollout, next_imgs_5_rollout = self.rollout_img(img, 1, visualize)
            acts_5_d2.append(acts_5_rollout)
            next_imgs_5_d2.append(next_imgs_5_rollout)
          acts_5_d1.append(acts_5_d2)
          next_imgs_5_d1.append(next_imgs_5_d2)
        acts_5_d0.append(acts_5_d1)
        next_imgs_5_d0.append(next_imgs_5_d1)
      acts_5.append(acts_5_d0)
      next_imgs_5.append(next_imgs_5_d0)

    return [acts_1, acts_2, acts_3, acts_4, acts_5], [next_imgs_1, next_imgs_2, next_imgs_3, next_imgs_4, next_imgs_5], [next_imgs_means_1, next_imgs_means_2, next_imgs_means_3, next_imgs_means_4]

  def compute_variance(self, next_img_list, next_img_means):
    """Compute the variance based on L2 loss.
    
    Args:
      acts: list of actions.
      next_imgs: list of next images from different bootstrap.
    """

    # Copy to avoid changing the images.
    img_list = copy.deepcopy(next_img_list)
    means = np.copy(next_img_means)

    # Get the shape.
    h, w, c = img_list[0].shape
    
    # Preprocess means.
    means = self.preprocess(means)

    # Preprocess images.
    for j in range(self.num_bootstraps):
      img_list[j] = self.preprocess(img_list[j])
      
    # Compute variance.
    rgb_variance = 0.0
    h_variance = 0.0
    for j in range(self.num_bootstraps):
      diff = img_list[j] - means
      rgb_variance += np.sum(diff[:, :, :3] * diff[:, :, :3]) / (h * w * (c-1))
      h_variance += np.sum(diff[:, :, 3] * diff[:, :, 3]) / (h * w * 1)

    return rgb_variance, h_variance

  def select_best_act_depth1(self, next_imgs_means_1):
    """Select the best action based on L2 loss."""
    
    goal_img = np.copy(self.goal_img)
    goal_img = self.preprocess(goal_img)

    h_diff_1_list = []
    h_diff_map_1_list = []

    # Compute the difference for layer1 leaf.
    for i_1 in range(self.k):
      h_diff_map = self.preprocess(next_imgs_means_1[i_1]) - goal_img
      h_diff_map = np.abs(h_diff_map[:, :, 3])
      h_diff = np.sum(h_diff_map) / (self.img_shape[0] * self.img_shape[1] * self.img_shape[2])
      
      h_diff_1_list.append(h_diff)
      h_diff_map_1_list.append(h_diff_map)

    h_diff_1_list = np.array(h_diff_1_list).flatten()
    reward_list = 1 - h_diff_1_list
    reward_argsort = np.argsort(reward_list)
    acts_1_idx = reward_argsort[-1]
    print(f'Reward: {reward_list} Best Depth: 0 Best Action: {acts_1_idx}')

    return acts_1_idx
    
  def select_best_act_depth2(self, next_imgs_means_1, next_imgs_means_2):
    """Select the best action based on L2 loss and model uncertainty."""

    goal_img = np.copy(self.goal_img)
    goal_img = self.preprocess(goal_img)

    h_diff_2_list = []
    h_diff_map_2_list = []

    h_diff_1_list = []
    h_diff_map_1_list = []

    # Compute the difference for layer1 leaf.
    for i_1 in range(self.k):
      # Compute diff.
      h_diff_map = self.preprocess(next_imgs_means_1[i_1]) - goal_img
      h_diff_map = np.abs(h_diff_map[:, :, 3])
      h_diff = np.sum(h_diff_map) / (self.img_shape[0] * self.img_shape[1] * self.img_shape[2])

      h_diff_1_list.append(h_diff)
      h_diff_map_1_list.append(h_diff_map)

    # Compute the difference for layer2 leaf.
    for i_1 in range(self.k):
      h_diff_2_node_list = []
      h_diff_map_2_node_list = []
      for i_2 in range(self.k):
        # Compute diff.
        h_diff_map = self.preprocess(next_imgs_means_2[i_1][i_2]) - goal_img
        h_diff_map = np.abs(h_diff_map[:, :, 3])
        h_diff = np.sum(h_diff_map) / (self.img_shape[0] * self.img_shape[1] * self.img_shape[2])

        h_diff_2_node_list.append(h_diff)
        h_diff_map_2_node_list.append(h_diff_map)
        
      h_diff_2_list.append(h_diff_2_node_list)
      h_diff_map_2_list.append(h_diff_map_2_node_list)

    h_diff_1_list = np.array(h_diff_1_list).flatten()
    h_diff_2_list = np.array(h_diff_2_list).flatten()

    reward_1_list = 1 - h_diff_1_list
    reward_2_list = 0.99 * (1 - h_diff_2_list)

    reward_list = np.concatenate([reward_1_list, reward_2_list])
    reward_argsort = np.argsort(reward_list)
    best_idx = reward_argsort[-1]

    if best_idx < self.k:
      acts_1_idx = best_idx
      print(f'Reward: {reward_list} Best Depth: 0 Best Action: {acts_1_idx}')
    else:
      argsort_2_list = np.argsort(reward_2_list)
      best_idx = argsort_2_list[-1]
      acts_1_idx = best_idx // self.k
      print(f'Reward: {reward_list} Best Depth: 1 Best Action: {acts_1_idx}')    
      
    return acts_1_idx

  def select_best_act_depth3(self, next_imgs_means_1, next_imgs_means_2, next_imgs_means_3):
    """Select the best action based on L2 loss and model uncertainty."""

    goal_img = np.copy(self.goal_img)
    goal_img = self.preprocess(goal_img)

    h_diff_1_list = []
    h_diff_map_1_list = []
    h_diff_2_list = []
    h_diff_map_2_list = []
    h_diff_3_list = []
    h_diff_map_3_list = []

    # Compute the difference and variance for layer1 leaf.
    for i_0 in range(self.k):
      # Compute diff.
      h_diff_map = self.preprocess(next_imgs_means_1[i_0]) - goal_img
      h_diff_map = np.abs(h_diff_map[:, :, 3])
      h_diff = np.sum(h_diff_map) / (self.img_shape[0] * self.img_shape[1] * self.img_shape[2])

      h_diff_1_list.append(h_diff)
      h_diff_map_1_list.append(h_diff_map)

    # Compute the difference and variance for layer2 leaf.
    for i_0 in range(self.k):
      h_diff_2_d0_list = []
      h_diff_map_2_d0_list = []
      for i_1 in range(self.k):
        # Compute diff.
        h_diff_map = self.preprocess(next_imgs_means_2[i_0][i_1]) - goal_img
        h_diff_map = np.abs(h_diff_map[:, :, 3])
        h_diff = np.sum(h_diff_map) / (self.img_shape[0] * self.img_shape[1] * self.img_shape[2])

        h_diff_2_d0_list.append(h_diff)
        h_diff_map_2_d0_list.append(h_diff_map)
        
      h_diff_2_list.append(h_diff_2_d0_list)
      h_diff_map_2_list.append(h_diff_map_2_d0_list)

    # Compute the difference and variance for layer3 leaf.
    for i_0 in range(self.k):
      h_diff_3_d0_list = []
      h_diff_map_3_d0_list = []
      for i_1 in range(self.k):
        h_diff_3_d1_list = []
        h_diff_map_3_d1_list = []
        for i_2 in range(self.k):
          # Compute diff.
          h_diff_map = self.preprocess(next_imgs_means_3[i_0][i_1][i_2]) - goal_img
          h_diff_map = np.abs(h_diff_map[:, :, 3])
          h_diff = np.sum(h_diff_map) / (self.img_shape[0] * self.img_shape[1] * self.img_shape[2])

          h_diff_3_d1_list.append(h_diff)
          h_diff_map_3_d1_list.append(h_diff_map)

        h_diff_3_d0_list.append(h_diff_3_d1_list)
        h_diff_map_3_d0_list.append(h_diff_map_3_d1_list)

      h_diff_3_list.append(h_diff_3_d0_list)
      h_diff_map_3_list.append(h_diff_map_3_d0_list)

    
    h_diff_1_list = np.array(h_diff_1_list) # (3, )
    h_diff_2_list = np.array(h_diff_2_list) # (3, 3)
    h_diff_3_list = np.array(h_diff_3_list) # (3, 3, 3)

    h_diff_1_flatten_list = h_diff_1_list.flatten()
    h_diff_2_flatten_list = h_diff_2_list.flatten()
    h_diff_3_flatten_list = h_diff_3_list.flatten()

    reward_1_list = 1 - h_diff_1_flatten_list
    reward_2_list = 0.99 * (1 - h_diff_2_flatten_list)
    reward_3_list = 0.99 * 0.99 * (1 - h_diff_3_flatten_list)
    
    max_reward_idx_1 = np.argsort(reward_1_list)[-1]
    max_reward_idx_2 = np.argsort(reward_2_list)[-1]
    max_reward_idx_3 = np.argsort(reward_3_list)[-1]

    max_reward_list = np.array([
      reward_1_list[max_reward_idx_1],
      reward_2_list[max_reward_idx_2],
      reward_3_list[max_reward_idx_3]
    ])

    best_depth = np.argsort(max_reward_list)[-1]

    if best_depth == 0:
      acts_1_idx = max_reward_idx_1
      print(f'Reward: {reward_1_list[max_reward_idx_1]} Best Depth: {best_depth} Best Action: {acts_1_idx}')
    elif best_depth == 1:
      acts_1_idx = max_reward_idx_2 // self.k
      acts_2_idx = max_reward_idx_2 % self.k
      print(f'Reward: {reward_2_list[max_reward_idx_2]} Best Depth: {best_depth} Best Action: {acts_1_idx}-{acts_2_idx}')
    elif best_depth == 2:
      acts_1_idx = max_reward_idx_3 // (self.k * self.k)
      acts_2_idx = (max_reward_idx_3 - (self.k * self.k) * acts_1_idx) // self.k
      acts_3_idx = (max_reward_idx_3 - (self.k * self.k) * acts_1_idx) % self.k
      print(f'Reward: {reward_3_list[max_reward_idx_3]} Best Depth: {best_depth} Best Action: {acts_1_idx}-{acts_2_idx}-{acts_3_idx}')
      
    return acts_1_idx

  def select_best_act_depth4(self, next_imgs_means_1, next_imgs_means_2, next_imgs_means_3, next_imgs_means_4):
    """Select the best action based on L2 loss and model uncertainty."""

    goal_img = np.copy(self.goal_img)
    goal_img = self.preprocess(goal_img)

    h_diff_1_list = []
    h_diff_map_1_list = []
    h_diff_2_list = []
    h_diff_map_2_list = []
    h_diff_3_list = []
    h_diff_map_3_list = []
    h_diff_4_list = []
    h_diff_map_4_list = []

    # Compute the difference and variance for layer1 leaf.
    for i_0 in range(self.k):
      # Compute diff.
      h_diff_map = self.preprocess(next_imgs_means_1[i_0]) - goal_img
      h_diff_map = np.abs(h_diff_map[:, :, 3])
      h_diff = np.sum(h_diff_map) / (self.img_shape[0] * self.img_shape[1] * self.img_shape[2])

      h_diff_1_list.append(h_diff)
      h_diff_map_1_list.append(h_diff_map)

    # Compute the difference and variance for layer2 leaf.
    for i_0 in range(self.k):
      h_diff_2_d0_list = []
      h_diff_map_2_d0_list = []
      for i_1 in range(self.k):
        # Compute diff.
        h_diff_map = self.preprocess(next_imgs_means_2[i_0][i_1]) - goal_img
        h_diff_map = np.abs(h_diff_map[:, :, 3])
        h_diff = np.sum(h_diff_map) / (self.img_shape[0] * self.img_shape[1] * self.img_shape[2])

        h_diff_2_d0_list.append(h_diff)
        h_diff_map_2_d0_list.append(h_diff_map)
        
      h_diff_2_list.append(h_diff_2_d0_list)
      h_diff_map_2_list.append(h_diff_map_2_d0_list)

    # Compute the difference and variance for layer3 leaf.
    for i_0 in range(self.k):
      h_diff_3_d0_list = []
      h_diff_map_3_d0_list = []
      for i_1 in range(self.k):
        h_diff_3_d1_list = []
        h_diff_map_3_d1_list = []
        for i_2 in range(self.k):
          # Compute diff.
          h_diff_map = self.preprocess(next_imgs_means_3[i_0][i_1][i_2]) - goal_img
          h_diff_map = np.abs(h_diff_map[:, :, 3])
          h_diff = np.sum(h_diff_map) / (self.img_shape[0] * self.img_shape[1] * self.img_shape[2])

          h_diff_3_d1_list.append(h_diff)
          h_diff_map_3_d1_list.append(h_diff_map)

        h_diff_3_d0_list.append(h_diff_3_d1_list)
        h_diff_map_3_d0_list.append(h_diff_map_3_d1_list)

      h_diff_3_list.append(h_diff_3_d0_list)
      h_diff_map_3_list.append(h_diff_map_3_d0_list)

    # Compute the difference and variance for layer4 leaf.
    for i_0 in range(self.k):
      h_diff_4_d0_list = []
      h_diff_map_4_d0_list = []
      for i_1 in range(self.k):
        h_diff_4_d1_list = []
        h_diff_map_4_d1_list = []
        for i_2 in range(self.k):
          h_diff_4_d2_list = []
          h_diff_map_4_d2_list= []
          for i_3 in range(self.k):
            # Compute diff.
            h_diff_map = self.preprocess(next_imgs_means_4[i_0][i_1][i_2][i_3]) - goal_img
            h_diff_map = np.abs(h_diff_map[:, :, 3])
            h_diff = np.sum(h_diff_map) / (self.img_shape[0] * self.img_shape[1] * self.img_shape[2])

            h_diff_4_d2_list.append(h_diff)
            h_diff_map_4_d2_list.append(h_diff_map)

          h_diff_4_d1_list.append(h_diff_4_d2_list)
          h_diff_map_4_d1_list.append(h_diff_4_d1_list)

        h_diff_4_d0_list.append(h_diff_4_d1_list)
        h_diff_map_4_d0_list.append(h_diff_map_4_d1_list)

      h_diff_4_list.append(h_diff_4_d0_list)
      h_diff_map_4_list.append(h_diff_map_4_d0_list)
    
    h_diff_1_list = np.array(h_diff_1_list) # (3, )
    h_diff_2_list = np.array(h_diff_2_list) # (3, 3)
    h_diff_3_list = np.array(h_diff_3_list) # (3, 3, 3)
    h_diff_4_list = np.array(h_diff_4_list) # (3, 3, 3, 3)

    h_diff_1_flatten_list = h_diff_1_list.flatten()
    h_diff_2_flatten_list = h_diff_2_list.flatten()
    h_diff_3_flatten_list = h_diff_3_list.flatten()
    h_diff_4_flatten_list = h_diff_4_list.flatten()

    reward_1_list = 1 - h_diff_1_flatten_list
    reward_2_list = 0.99 * (1 - h_diff_2_flatten_list)
    reward_3_list = 0.99 * 0.99 * (1 - h_diff_3_flatten_list)
    reward_4_list = 0.99 * 0.99 * 0.99 * (1 - h_diff_4_flatten_list)
    
    max_reward_idx_1 = np.argsort(reward_1_list)[-1]
    max_reward_idx_2 = np.argsort(reward_2_list)[-1]
    max_reward_idx_3 = np.argsort(reward_3_list)[-1]
    max_reward_idx_4 = np.argsort(reward_4_list)[-1]

    max_reward_list = np.array([
      reward_1_list[max_reward_idx_1],
      reward_2_list[max_reward_idx_2],
      reward_3_list[max_reward_idx_3],
      reward_4_list[max_reward_idx_4],
    ])

    best_depth = np.argsort(max_reward_list)[-1]

    if best_depth == 0:
      acts_1_idx = max_reward_idx_1
      print(f'Reward: {reward_1_list[max_reward_idx_1]} Best Depth: {best_depth} Best Action: {acts_1_idx}')
    elif best_depth == 1:
      acts_1_idx = max_reward_idx_2 // self.k
      acts_2_idx = max_reward_idx_2 % self.k
      print(f'Reward: {reward_2_list[max_reward_idx_2]} Best Depth: {best_depth} Best Action: {acts_1_idx}-{acts_2_idx}')
    elif best_depth == 2:
      acts_1_idx = max_reward_idx_3 // (self.k * self.k)
      acts_2_idx = (max_reward_idx_3 - (self.k * self.k) * acts_1_idx) // self.k
      acts_3_idx = (max_reward_idx_3 - (self.k * self.k) * acts_1_idx) % self.k
      print(f'Reward: {reward_3_list[max_reward_idx_3]} Best Depth: {best_depth} Best Action: {acts_1_idx}-{acts_2_idx}-{acts_3_idx}')
    elif best_depth == 3:
      acts_1_idx = max_reward_idx_4 // (self.k * self.k * self.k)
      acts_2_idx = (max_reward_idx_4 - (self.k * self.k * self.k) * acts_1_idx) // (self.k * self.k)
      acts_3_idx = (max_reward_idx_4 - (self.k * self.k * self.k) * acts_1_idx - (self.k * self.k) * acts_2_idx) // self.k
      acts_4_idx = (max_reward_idx_4 - (self.k * self.k * self.k) * acts_1_idx - (self.k * self.k) * acts_2_idx) % self.k
      print(f'Reward: {reward_4_list[max_reward_idx_4]} Best Depth: {best_depth} Best Action: {acts_1_idx}-{acts_2_idx}-{acts_3_idx}-{acts_4_idx}')
      
    return acts_1_idx
  
  def select_best_act_depth4_rollout1(self, next_imgs_means_1, next_imgs_means_2, next_imgs_means_3, next_imgs_means_4, next_imgs_5):
    """Select the best action based on L2 loss and model uncertainty."""

    goal_img = np.copy(self.goal_img)
    goal_img = self.preprocess(goal_img)

    h_diff_1_list = []
    h_diff_map_1_list = []
    h_diff_2_list = []
    h_diff_map_2_list = []
    h_diff_3_list = []
    h_diff_map_3_list = []
    h_diff_4_list = []
    h_diff_map_4_list = []
    h_diff_5_list = []
    h_diff_map_5_list = []

    # Compute the difference and variance for layer1 leaf.
    for i_0 in range(self.k):
      # Compute diff.
      h_diff_map = self.preprocess(next_imgs_means_1[i_0]) - goal_img
      h_diff_map = np.abs(h_diff_map[:, :, 3])
      h_diff = np.sum(h_diff_map) / (self.img_shape[0] * self.img_shape[1] * self.img_shape[2])

      h_diff_1_list.append(h_diff)
      h_diff_map_1_list.append(h_diff_map)

    # Compute the difference and variance for layer2 leaf.
    for i_0 in range(self.k):
      h_diff_2_d0_list = []
      h_diff_map_2_d0_list = []
      for i_1 in range(self.k):
        # Compute diff.
        h_diff_map = self.preprocess(next_imgs_means_2[i_0][i_1]) - goal_img
        h_diff_map = np.abs(h_diff_map[:, :, 3])
        h_diff = np.sum(h_diff_map) / (self.img_shape[0] * self.img_shape[1] * self.img_shape[2])

        h_diff_2_d0_list.append(h_diff)
        h_diff_map_2_d0_list.append(h_diff_map)
        
      h_diff_2_list.append(h_diff_2_d0_list)
      h_diff_map_2_list.append(h_diff_map_2_d0_list)

    # Compute the difference and variance for layer3 leaf.
    for i_0 in range(self.k):
      h_diff_3_d0_list = []
      h_diff_map_3_d0_list = []
      for i_1 in range(self.k):
        h_diff_3_d1_list = []
        h_diff_map_3_d1_list = []
        for i_2 in range(self.k):
          # Compute diff.
          h_diff_map = self.preprocess(next_imgs_means_3[i_0][i_1][i_2]) - goal_img
          h_diff_map = np.abs(h_diff_map[:, :, 3])
          h_diff = np.sum(h_diff_map) / (self.img_shape[0] * self.img_shape[1] * self.img_shape[2])

          h_diff_3_d1_list.append(h_diff)
          h_diff_map_3_d1_list.append(h_diff_map)

        h_diff_3_d0_list.append(h_diff_3_d1_list)
        h_diff_map_3_d0_list.append(h_diff_map_3_d1_list)

      h_diff_3_list.append(h_diff_3_d0_list)
      h_diff_map_3_list.append(h_diff_map_3_d0_list)

    # Compute the difference and variance for layer4 leaf.
    for i_0 in range(self.k):
      h_diff_4_d0_list = []
      h_diff_map_4_d0_list = []
      for i_1 in range(self.k):
        h_diff_4_d1_list = []
        h_diff_map_4_d1_list = []
        for i_2 in range(self.k):
          h_diff_4_d2_list = []
          h_diff_map_4_d2_list= []
          for i_3 in range(self.k):
            # Compute diff.
            h_diff_map = self.preprocess(next_imgs_means_4[i_0][i_1][i_2][i_3]) - goal_img
            h_diff_map = np.abs(h_diff_map[:, :, 3])
            h_diff = np.sum(h_diff_map) / (self.img_shape[0] * self.img_shape[1] * self.img_shape[2])

            h_diff_4_d2_list.append(h_diff)
            h_diff_map_4_d2_list.append(h_diff_map)

          h_diff_4_d1_list.append(h_diff_4_d2_list)
          h_diff_map_4_d1_list.append(h_diff_4_d1_list)

        h_diff_4_d0_list.append(h_diff_4_d1_list)
        h_diff_map_4_d0_list.append(h_diff_map_4_d1_list)

      h_diff_4_list.append(h_diff_4_d0_list)
      h_diff_map_4_list.append(h_diff_map_4_d0_list)

    # Compute the difference and variance for layer5 leaf.
    for i_0 in range(self.k):
      h_diff_5_d0_list = []
      h_diff_map_5_d0_list = []
      for i_1 in range(self.k):
        h_diff_5_d1_list = []
        h_diff_map_5_d1_list = []
        for i_2 in range(self.k):
          h_diff_5_d2_list = []
          h_diff_map_5_d2_list= []
          for i_3 in range(self.k):
            # Compute diff.
            h_diff_map = self.preprocess(next_imgs_5[i_0][i_1][i_2][i_3][-1]) - goal_img
            h_diff_map = np.abs(h_diff_map[:, :, 3])
            h_diff = np.sum(h_diff_map) / (self.img_shape[0] * self.img_shape[1] * self.img_shape[2])

            h_diff_5_d2_list.append(h_diff)
            h_diff_map_5_d2_list.append(h_diff_map)

          h_diff_5_d1_list.append(h_diff_5_d2_list)
          h_diff_map_5_d1_list.append(h_diff_5_d1_list)

        h_diff_5_d0_list.append(h_diff_5_d1_list)
        h_diff_map_5_d0_list.append(h_diff_map_5_d1_list)

      h_diff_5_list.append(h_diff_5_d0_list)
      h_diff_map_5_list.append(h_diff_map_5_d0_list)
    
    h_diff_1_list = np.array(h_diff_1_list) # (3, )
    h_diff_2_list = np.array(h_diff_2_list) # (3, 3)
    h_diff_3_list = np.array(h_diff_3_list) # (3, 3, 3)
    h_diff_4_list = np.array(h_diff_4_list) # (3, 3, 3, 3)
    h_diff_5_list = np.array(h_diff_5_list) # (3, 3, 3, 3)

    h_diff_1_flatten_list = h_diff_1_list.flatten()
    h_diff_2_flatten_list = h_diff_2_list.flatten()
    h_diff_3_flatten_list = h_diff_3_list.flatten()
    h_diff_4_flatten_list = h_diff_4_list.flatten()
    h_diff_5_flatten_list = h_diff_5_list.flatten()

    reward_1_list = 1 - h_diff_1_flatten_list
    reward_2_list = 0.99 * (1 - h_diff_2_flatten_list)
    reward_3_list = 0.99 * 0.99 * (1 - h_diff_3_flatten_list)
    reward_4_list = 0.99 * 0.99 * 0.99 * (1 - h_diff_4_flatten_list)
    reward_5_list = 0.99 * 0.99 * 0.99 * 0.99 * (1 - h_diff_5_flatten_list)
    
    max_reward_idx_1 = np.argsort(reward_1_list)[-1]
    max_reward_idx_2 = np.argsort(reward_2_list)[-1]
    max_reward_idx_3 = np.argsort(reward_3_list)[-1]
    max_reward_idx_4 = np.argsort(reward_4_list)[-1]
    max_reward_idx_5 = np.argsort(reward_5_list)[-1]

    max_reward_list = np.array([
      reward_1_list[max_reward_idx_1],
      reward_2_list[max_reward_idx_2],
      reward_3_list[max_reward_idx_3],
      reward_4_list[max_reward_idx_4],
      reward_5_list[max_reward_idx_5],
    ])

    best_depth = np.argsort(max_reward_list)[-1]

    if best_depth == 0:
      acts_1_idx = max_reward_idx_1
      print(f'Reward: {reward_1_list[max_reward_idx_1]} Best Depth: {best_depth} Best Action: {acts_1_idx}')
    elif best_depth == 1:
      acts_1_idx = max_reward_idx_2 // self.k
      acts_2_idx = max_reward_idx_2 % self.k
      print(f'Reward: {reward_2_list[max_reward_idx_2]} Best Depth: {best_depth} Best Action: {acts_1_idx}-{acts_2_idx}')
    elif best_depth == 2:
      acts_1_idx = max_reward_idx_3 // (self.k * self.k)
      acts_2_idx = (max_reward_idx_3 - (self.k * self.k) * acts_1_idx) // self.k
      acts_3_idx = (max_reward_idx_3 - (self.k * self.k) * acts_1_idx) % self.k
      print(f'Reward: {reward_3_list[max_reward_idx_3]} Best Depth: {best_depth} Best Action: {acts_1_idx}-{acts_2_idx}-{acts_3_idx}')
    elif best_depth == 3:
      acts_1_idx = max_reward_idx_4 // (self.k * self.k * self.k)
      acts_2_idx = (max_reward_idx_4 - (self.k * self.k * self.k) * acts_1_idx) // (self.k * self.k)
      acts_3_idx = (max_reward_idx_4 - (self.k * self.k * self.k) * acts_1_idx - (self.k * self.k) * acts_2_idx) // self.k
      acts_4_idx = (max_reward_idx_4 - (self.k * self.k * self.k) * acts_1_idx - (self.k * self.k) * acts_2_idx) % self.k
      print(f'Reward: {reward_4_list[max_reward_idx_4]} Best Depth: {best_depth} Best Action: {acts_1_idx}-{acts_2_idx}-{acts_3_idx}-{acts_4_idx}')
    elif best_depth == 4:
      acts_1_idx = max_reward_idx_5 // (self.k * self.k * self.k)
      acts_2_idx = (max_reward_idx_5 - (self.k * self.k * self.k) * acts_1_idx) // (self.k * self.k)
      acts_3_idx = (max_reward_idx_5 - (self.k * self.k * self.k) * acts_1_idx - (self.k * self.k) * acts_2_idx) // self.k
      acts_4_idx = (max_reward_idx_5 - (self.k * self.k * self.k) * acts_1_idx - (self.k * self.k) * acts_2_idx) % self.k
      print(f'Reward: {reward_5_list[max_reward_idx_5]} Best Depth: {best_depth} Best Action: {acts_1_idx}-{acts_2_idx}-{acts_3_idx}-{acts_4_idx}')

    return acts_1_idx

  ######################
  # Utility functions
  ######################

  def get_image(self, obs):
    """Stack color and height images image.
    
    Args:
      obs: observation from the envrionment.
    Returns:
      img: RGBHHH
    """

    # Get color and height maps from RGB-D images.
    cmap, hmap = utils.get_fused_heightmap(
        obs, self.cam_config, self.bounds, self.pix_size)
    img = np.concatenate((cmap,
                          hmap[Ellipsis, None],
                          hmap[Ellipsis, None],
                          hmap[Ellipsis, None]), axis=2)
    
    return img

  def preprocess(self, img_origin, h_only=False):
    """Pre-process input (subtract mean, divide by std)."""

    img = np.copy(img_origin)

    if h_only:
      assert img.shape[-1] == 1
      img[:, :, 0] = (img[:, :, 0] - self.depth_mean) / self.depth_std
    else:
      # assert img.shape[-1] == 6
      img[:, :, :3] = (img[:, :, :3] / 255 - self.color_mean) / self.color_std
      img[:, :, 3:] = (img[:, :, 3:] - self.depth_mean) / self.depth_std
    return img

  def make_rgbhhh(self, rgbh):
    """Make RGBHHH from RGBH."""

    h = rgbh[:, :, -1]
    rgbhhh = np.concatenate([rgbh, h[Ellipsis, None], h[Ellipsis, None]], axis=2)
    
    return rgbhhh