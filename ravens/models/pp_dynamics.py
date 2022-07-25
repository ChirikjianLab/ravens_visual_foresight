"""PP dynamics module."""

import numpy as np
from ravens.models.resnet import ResNet36_4s
from ravens.models.resnet import ResNet43_8s
from ravens.utils import utils
import tensorflow as tf
from tensorflow_addons import image as tfa_image


class PPDynamics(object):
  """PP dynamics module."""
  
  def __init__(self, in_shape, out_channel, mask_size, model_name):
    # Padding.
    self.mask_size = mask_size
    self.pad_size = int(self.mask_size / 2)
    self.padding = np.zeros((3, 2), dtype=int)
    self.padding[:2, :] = self.pad_size
    in_shape = np.array(in_shape)
    in_shape[0:2] += self.pad_size * 2
    in_shape = tuple(in_shape)
    print(f"[PPDynamics] in_shape after padding: {in_shape}")
    
    # Initialize the model.
    self.model_name = model_name
    print(f"[PPDynamics] Model is {self.model_name}")
    if self.model_name == 'resnet_lite':
      d_in, d_out = ResNet36_4s(in_shape, out_channel)
    elif self.model_name == 'resnet':
      d_in, d_out = ResNet43_8s(in_shape, out_channel)
    self.model = tf.keras.models.Model(inputs=[d_in], outputs=[d_out])
    self.optim = tf.keras.optimizers.Adam(learning_rate=1e-4)
    self.metric = tf.keras.metrics.Mean(name='loss_dynamics')

    # Means and variances for color and depth images.
    self.color_mean = 0.18877631
    self.depth_mean = 0.00509261
    self.color_std = 0.07276466
    self.depth_std = 0.00903967

  def train_pp(self, init_img, target_img, p0, p1, p1_theta, backprop=True, repeat_H_lambda=1, h_only=False):
    """Train the PP dynamics."""

    self.metric.reset_states()

    # Debug
    if False:
      import matplotlib
      matplotlib.use('TkAgg')
      import matplotlib.pyplot as plt
      if not h_only:
        f, ax = plt.subplots(1, 4)
        max_height = 0.14
        normalize = matplotlib.colors.Normalize(vmin=0.0, vmax=max_height)
        init_img[p0[0], p0[1], :] = 0.0
        target_img[p1[0], p1[1], :] = 0.0
        ax[0].imshow(init_img[:, :, :3] / 255.0)
        ax[1].imshow(init_img[:, :, 3], norm=normalize)
        ax[2].imshow(target_img[:, :, :3] / 255.0)
        ax[3].imshow(target_img[:, :, 3], norm=normalize)
        plt.show()
      else:
        f, ax = plt.subplots(2)
        ax[0].imshow(init_img[:, :, 0])
        ax[1].imshow(target_img[:, :, 0])
        plt.show()

    with tf.GradientTape() as tape:
      # Subtrack means and divide by std.
      init_img = self.preprocess_input(init_img, h_only)
      target_img = self.preprocess_target(target_img, h_only)        
      
      # Add padding.
      init_img  = np.pad(init_img, self.padding, mode='constant')
      target_img = np.pad(target_img, self.padding, mode='constant')

      # Forward pass.
      out_tens = self.forward_pp(init_img, p0, p1, p1_theta)

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

  def forward_pp(self, init_img, p0, p1, p1_theta):
    """Forward pass."""

    # Pick mask.
    init_shape = init_img.shape
    pick_mask = np.zeros((init_shape[0], init_shape[1]))
    pick_mask[p0[0]:(p0[0]+self.mask_size), p0[1]:(p0[1]+self.mask_size)] = 1.0

    # Place mask.
    init_tens = tf.convert_to_tensor(init_img, dtype=tf.float32)
    pivot = np.array([p0[1], p0[0]]) + self.pad_size
    rmat = utils.get_image_transform(p1_theta, (0, 0), pivot)
    rvec = rmat.reshape(-1)[:-1]
    init_tens_rot = tfa_image.transform(init_tens, rvec, interpolation='NEAREST')
    crop = init_tens_rot[p0[0]:(p0[0] + self.mask_size),
                         p0[1]:(p0[1] + self.mask_size), :]
    place_mask = np.zeros(init_shape)
    place_mask[p1[0]:(p1[0]+self.mask_size), p1[1]:(p1[1]+self.mask_size), :] = crop.numpy()

    # Concateante init_img, pick_mask, and place_mask.
    in_img = np.concatenate([init_img, pick_mask[Ellipsis, None], place_mask], axis=-1)

    # Debug paper
    if False:
      import matplotlib
      matplotlib.use('TkAgg')
      import matplotlib.pyplot as plt
      max_height = 0.14
      norm = matplotlib.colors.Normalize(vmin=0.0, vmax=max_height)
      f, ax = plt.subplots(3)
      init_img_copy = np.copy(init_img)
      original_init_img = self.postprocess_output(init_img_copy)
      original_init_img = original_init_img[int(self.mask_size/2): -int(self.mask_size/2), int(self.mask_size/2): -int(self.mask_size/2), :]
      original_init_img[p0[0], p0[1], :] = 0.0
      original_init_img[p1[0], p1[1], :] = 0.0
      no_pad_pick_mask = np.copy(pick_mask[int(self.mask_size/2): -int(self.mask_size/2), int(self.mask_size/2): -int(self.mask_size/2)])
      original_pick_mask = np.concatenate([no_pad_pick_mask[Ellipsis, None], 
          no_pad_pick_mask[Ellipsis, None], no_pad_pick_mask[Ellipsis, None]], axis=2)
      place_mask_copy = np.copy(place_mask)
      no_pad_place_mask = self.postprocess_output(place_mask_copy)
      original_place_mask = np.zeros(place_mask_copy.shape)
      original_place_mask[p1[0]:(p1[0]+self.mask_size), p1[1]:(p1[1]+self.mask_size), :] = no_pad_place_mask[p1[0]:(p1[0]+self.mask_size), p1[1]:(p1[1]+self.mask_size), :] 
      original_place_mask = original_place_mask[int(self.mask_size/2): -int(self.mask_size/2), int(self.mask_size/2): -int(self.mask_size/2)]
      ax[0].imshow(original_init_img[:, :, :3] / 255.0)
      ax[1].imshow(original_pick_mask * 255.0)
      # ax[2].imshow((original_place_mask[:, :, :3] + original_init_img[:, :, :3])/ 255.0)
      ax[2].imshow(original_place_mask[:, :, :3]/ 255.0)
      plt.show()

    # Debug
    if False:
      import matplotlib
      matplotlib.use('TkAgg')
      import matplotlib.pyplot as plt
      print(f"in_img: {in_img.shape}")
      print(f"pick_mask: {pick_mask.shape}")
      print(f"place_mask: {place_mask.shape}")
      
      # # h_only
      # f, ax = plt.subplots(1, 5)
      # ax[0].imshow(in_img[:, :, 0])
      # ax[1].imshow(in_img[:, :, 1])
      # ax[2].imshow(in_img[:, :, 2])
      # ax[3].imshow(in_img[:, :, 0] + in_img[:, :, 1])
      # ax[4].imshow(in_img[:, :, 0] + in_img[:, :, 2])
      
      # full
      f, ax = plt.subplots(4)
      ax[0].imshow(in_img[:, :, 3] + pick_mask)
      ax[1].imshow(in_img[:, :, 3] + place_mask[:, :, 3])
      ax[2].imshow(in_img[:, :, 0])
      ax[3].imshow(in_img[:, :, 0] + place_mask[:, :, 0])
      plt.show()    

    in_img = np.expand_dims(in_img, axis=0)
    
    # Convert in_img to tensor
    in_tens = tf.convert_to_tensor(in_img, dtype=tf.float32)
    
    # Forward pass.
    out_tens = self.model(in_tens)

    return out_tens

  def imagine(self, init_img, p0, p1, p1_theta, h_only):
    """Imagine the image after taking the action."""
    
    # Subtract means and divide by std.
    init_img = self.preprocess_input(init_img, h_only)

    # Add padding.
    init_img = np.pad(init_img, self.padding, mode='constant')

    # Forward pass.
    out_tens = self.forward_pp(init_img, p0, p1, p1_theta)

    # Postprocess output tensor.
    out_data = out_tens.numpy()[0][self.pad_size:(-self.pad_size), self.pad_size:(-self.pad_size)]
    out_img = self.postprocess_output(out_data, h_only)

    # Debug paper
    if False:
      import matplotlib
      matplotlib.use('TkAgg')
      import matplotlib.pyplot as plt
      plt.imshow(out_img[:, :, :3]/ 255.0)
      plt.show()

    return out_img

  def test_pp(self, init_img, target_img, p0, p1, p1_theta, repeat_H_lambda, h_only):
    """Test PP Dynamics."""

    # Subtract means and divide by std.
    init_img = self.preprocess_input(init_img, h_only)
    target_img = self.preprocess_input(target_img, h_only)

    # Add padding.
    init_img = np.pad(init_img, self.padding, mode='constant')
    target_img = np.pad(target_img, self.padding, mode='constant')

    # Forward pass.
    out_tens = self.forward_pp(init_img, p0, p1, p1_theta)

    # Get loss.
    target_tens = tf.convert_to_tensor(target_img, dtype=tf.float32)
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

    # Postprocess output tensor.
    out_data = out_tens.numpy()[0][self.pad_size:(-self.pad_size), self.pad_size:(-self.pad_size)]
    out_img = self.postprocess_output(out_data, h_only)

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
      assert img.shape[-1] == 1
      img[:, :, 0] = (img[:, :, 0] - self.depth_mean) / self.depth_std
    else:
      # assert img.shape[-1] == 6
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