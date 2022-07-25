"""Dynamics module."""

import numpy as np
from ravens.models.resnet import ResNet36_4s_latent
import tensorflow as tf


class LatentDynamics(object):
  """Dynamics module."""
  
  def __init__(self, in_shape, out_channel):  
    # Initialize the model
    self.latent_shape = (40, 40, 6)
    # self.model = LatentModel(in_shape, latent_shape, out_channel)
    [d_in, action_in], d_out = ResNet36_4s_latent(in_shape, self.latent_shape, out_channel)
    self.model = tf.keras.models.Model(inputs=[d_in, action_in], outputs=[d_out])
    self.optim = tf.keras.optimizers.Adam(learning_rate=1e-4)
    self.metric = tf.keras.metrics.Mean(name='loss_dynamics')

    # Means and variances for color and depth images.
    self.color_mean = 0.18877631
    self.depth_mean = 0.00509261
    self.color_std = 0.07276466
    self.depth_std = 0.00903967
  
  def forward(self, init_img, p0, p1, p1_theta):
    """Forward pass.
    
    Args:
      in_img (numpy array): input image
    Returns:
      out_img (TF tensor): output image
    """

    # Debug
    if False:
      import matplotlib
      matplotlib.use('TkAgg')
      import matplotlib.pyplot as plt
      print(f"in_img: {in_img.shape}")
      print(f"pick_mask: {pick_mask.shape}")
      print(f"place_mask: {place_mask.shape}")
      print(f"sin_mask: {sin_mask.shape}")
      print(f"cos_mask: {cos_mask.shape}")

      # full
      f, ax = plt.subplots(1, 6)
      ax[0].imshow(in_img[:, :, 3] + pick_mask)
      ax[1].imshow(in_img[:, :, 3] + place_mask[:, :, 3])
      ax[2].imshow(in_img[:, :, 0])
      ax[3].imshow(in_img[:, :, 0] + place_mask[:, :, 0])
      ax[4].imshow(sin_mask)
      ax[5].imshow(cos_mask)
      plt.show() 
    
    # Expand a dimension for the init_img.
    in_img = np.expand_dims(init_img, axis=0)
    
    # Convert in_img to tensor.
    in_tens = tf.convert_to_tensor(in_img, dtype=tf.float32)

    # Action maps.
    pick_x_map  = np.ones(self.latent_shape[:2]) * p0[0]
    pick_y_map  = np.ones(self.latent_shape[:2]) * p0[1]
    place_x_map = np.ones(self.latent_shape[:2]) * p1[0]
    place_y_map = np.ones(self.latent_shape[:2]) * p1[1]
    sin_map     = np.ones(self.latent_shape[:2]) * np.sin(p1_theta)
    cos_map     = np.ones(self.latent_shape[:2]) * np.cos(p1_theta)

    # Concatenate the action maps on the latent map.
    action_input_numpy = np.concatenate([pick_x_map[Ellipsis, None],
                                         pick_y_map[Ellipsis, None],
                                         place_x_map[Ellipsis, None],
                                         place_y_map[Ellipsis, None],
                                         sin_map[Ellipsis, None],
                                         cos_map[Ellipsis, None]], axis=2)
    action_input_numpy = np.expand_dims(action_input_numpy, axis=0)
    action_input_tens = tf.convert_to_tensor(action_input_numpy, dtype=tf.float32)

    out_tens = self.model([in_tens, action_input_tens])

    return out_tens
  
  def train(self, init_img, target_img, p0, p1, p1_theta, backprop=True, repeat_H_lambda=1, h_only=False):
    """Train.
    
    Args:
      in_img (numpy array): input image (RGBHHH + pick + place)
      target_data (numpy array): target data (RGBH of the next step)
      backprop (bool): flag for backpropagation
      repeat_H_lambda (float): weights for training the height
      h_only (bool): whether to train only height
    
    Returns:
      loss: training loss
    """
    self.metric.reset_states()

    with tf.GradientTape() as tape:
      # Subtrack means and divide by std.
      init_img = self.preprocess_input(init_img, h_only)
      target_img = self.preprocess_target(target_img, h_only)        
      
      # Forward pass.
      out_tens = self.forward(init_img, p0, p1, p1_theta)

      # Get label.
      target_tens = tf.convert_to_tensor(target_img, dtype=tf.float32)

      # Get loss.
      diff = tf.abs(target_tens - out_tens)
      b, h, w, c = diff.shape

      if h_only:
        loss = tf.reduce_mean(diff)
      else:
        loss_R = tf.reduce_sum(diff[:, :, :, 0])
        loss_G = tf.reduce_sum(diff[:, :, :, 1])
        loss_B = tf.reduce_sum(diff[:, :, :, 2])
        loss_H = tf.reduce_sum(diff[:, :, :, 3])
        loss = (loss_R + loss_G + loss_B + repeat_H_lambda * loss_H) / (b * h * w * c)

    # Backpropagate
    if backprop:
      grad = tape.gradient(loss, self.model.trainable_variables)
      self.optim.apply_gradients(zip(grad, self.model.trainable_variables))
      self.metric(loss)
    
    return np.float32(loss)
  
  def test(self, init_img, target_img, p0, p1, p1_theta, repeat_H_lambda, h_only):
    """Test.
    
    Args:
      input_data (numpy array): input image (RGBHHH + pick + place)
    """
    
    # Subtrack means and divide by std.
    init_img = self.preprocess_input(init_img, h_only)
    target_img = self.preprocess_target(target_img, h_only)

    # Forward pass.
    out_tens = self.forward(init_img, p0, p1, p1_theta)

    # Get label.
    target_tens = tf.convert_to_tensor(target_img, dtype=tf.float32)

    # Get loss.
    diff = tf.abs(target_tens - out_tens)
    b, h, w, c = diff.shape

    if h_only:
      rgb_loss = 0.0
      height_loss = tf.reduce_mean(diff)
    else:
      assert c == 4
      loss_R = tf.reduce_sum(diff[:, :, :, 0])
      loss_G = tf.reduce_sum(diff[:, :, :, 1])
      loss_B = tf.reduce_sum(diff[:, :, :, 2])
      loss_H = tf.reduce_sum(diff[:, :, :, 3])
      rgb_loss = (loss_R + loss_G + loss_B) / (b * h * w * 3)
      height_loss = loss_H / (b * h * w)

    out_img = out_tens.numpy()[0]
    out_img = self.postprocess_output(out_img, h_only)

    return out_img, np.float32(rgb_loss), np.float32(height_loss)

  def load(self, path):
    """Load model weights."""
    self.model.load_weights(path)
  
  def save(self, filename):
    """Save the model."""
    self.model.save(filename)

  def preprocess_input(self, img, h_only=False):
    """Pre-process input (subtract mean, divide by std)."""

    if h_only:
      img[:, :, 0] = (img[:, :, 0] - self.depth_mean) / self.depth_std
    else:
      img[:, :, :3] = (img[:, :, :3] / 255 - self.color_mean) / self.color_std
      img[:, :, 3:] = (img[:, :, 3:] - self.depth_mean) / self.depth_std
    return img
  
  def preprocess_target(self, img, h_only=False):
    """Pre-process input (subtract mean, divide by std)."""

    if h_only:
      assert img.shape[-1] == 1
      img[:, :, 0] = (img[:, :, 0] - self.depth_mean) / self.depth_std 
    else:
      assert img.shape[-1] == 4
      img[:, :, :3] = (img[:, :, :3] / 255 - self.color_mean) / self.color_std
      img[:, :, -1] = (img[:, :, -1] - self.depth_mean) / self.depth_std
    return img

  def postprocess_output(self, img, h_only=False):
    """Post-process output (add mean, multiply by std)."""

    if h_only:
      img[:, :, 0] = img[:, :, 0] * self.depth_std + self.depth_mean
    else:
      img[:, :, :3] = 255 * (img[:, :, :3] * self.color_std + self.color_mean)
      img[:, :, -1] = img[:, :, -1] * self.depth_std + self.depth_mean
    return img