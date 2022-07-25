"""Model to predict whether a primitive is permissible."""

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Sequential
from tensorflow.keras import optimizers, metrics


class BasicBlock(layers.Layer):
  """Basic Residual block."""

  def __init__(self, num_filters, stride=1):
    """Constructor."""

    super(BasicBlock, self).__init__()
    assert stride > 0
    self.stride = stride
    self.conv1 = layers.Conv2D(num_filters, (3, 3), strides=self.stride, padding='same')
    self.bn1   = layers.BatchNormalization()
    self.relu  = layers.Activation('relu')

    self.conv2 = layers.Conv2D(num_filters, (3, 3), strides=1, padding='same')
    self.bn2  = layers.BatchNormalization()

    self.conv_identity = layers.Conv2D(num_filters, (1, 1), strides=self.stride)

  def call(self, input):
    """Forward pass."""

    x = self.conv1(input)
    x = self.bn1(x)
    x = self.relu(x)

    x = self.conv2(x)
    x = self.bn2(x)

    if self.stride > 1:
      identity = self.conv_identity(input)
    else:
      identity = input
    x = layers.add([x, identity])
    x = tf.nn.relu(x)

    return x


class ResNet(keras.Model):
  """ResNet class."""

  def __init__(self, num_blocks_list, num_classes=2):
    """Constructor."""

    super(ResNet, self).__init__()
    
    # Preprocessing layers.
    self.preprocess = Sequential([
        layers.Conv2D(64, (3, 3), strides=(1, 1)),
        layers.BatchNormalization(),
        layers.Activation('relu'),
        layers.MaxPool2D(pool_size=(2, 2), strides=(1, 1), padding='same')])
      
    # Residual blocks.
    self.block0 = self.ResBlock(64 , num_blocks_list[0], stride=1)
    self.block1 = self.ResBlock(128, num_blocks_list[1], stride=2)
    self.block2 = self.ResBlock(256, num_blocks_list[2], stride=2)
    self.block3 = self.ResBlock(512, num_blocks_list[3], stride=2)

    # Pooling layer.
    self.avgpool = layers.GlobalAveragePooling2D()
    
    # FC layer.
    self.fc = layers.Dense(num_classes)
  
  def ResBlock(self, num_filters, num_blocks, stride):
    """Build Residual blocks with basic blocks."""

    blocks = Sequential()
    blocks.add(BasicBlock(num_filters, stride))
    for i in range(1, num_blocks):
      blocks.add(BasicBlock(num_filters, stride=1))
    return blocks

  def call(self, input):
    x = self.preprocess(input)
    x = self.block0(x)
    x = self.block1(x)
    x = self.block2(x)
    x = self.block3(x)
    x = self.avgpool(x)
    x = self.fc(x)

    return x

class Permissible:
  """Class to train the permissible function."""

  def __init__(self, in_shape):
    """Constructor."""

    # Initialize the model.
    self.model = ResNet(num_blocks_list=[2, 2, 2, 2], num_classes=1) # ResNet18 for binary classification
    self.model.build(input_shape=(None, in_shape[0], in_shape[1], in_shape[2]))
    self.model.summary()
    self.optim = tf.keras.optimizers.Adam(lr=1e-3)
    self.metric = tf.keras.metrics.Mean(name='loss_dynamics')
    self.bce = tf.keras.losses.BinaryCrossentropy(from_logits=True)

    # Means and variances for color and depth images.
    self.color_mean = 0.18877631
    self.depth_mean = 0.00509261
    self.color_std = 0.07276466
    self.depth_std = 0.00903967

  def train_permissible(self, in_img, label, backprop=True, h_only=False):
    """Train the permissible function.
    
    Args:
      in_img: input image.
      label: 0 or 1 indicating whether it is permissible.
    """

    self.metric.reset_states()

    with tf.GradientTape() as tape:
      # Subtrack means and divide by std.
      in_img = self.preprocess_input(in_img, h_only=h_only)

      # Convert numpy array to tensor.
      in_img = np.expand_dims(in_img, axis=0)
      in_tens = tf.convert_to_tensor(in_img, dtype=tf.float32)

      # Forward pass.
      y_pred = self.model(in_tens)

      # Get label.
      y_true = tf.constant([label])

      # Loss.
      loss = self.bce(y_true, y_pred)

    # # Debug
    # if np.float(loss) > 0.5:
    #   print(f'label: {label}')
    #   print(f'y_pred: {y_pred}')
    #   import matplotlib
    #   matplotlib.use('TkAgg')
    #   import matplotlib.pyplot as plt
    #   max_height = 0.14
    #   normalize = matplotlib.colors.Normalize(vmin=0.0, vmax=max_height)
    #   f, ax = plt.subplots(2)
    #   ax[0].imshow(in_img_copy[:, :, :3] / 255.0)
    #   ax[1].imshow(in_img_copy[:, :, 3], norm=normalize)
    #   plt.show()

    if backprop:
      grad = tape.gradient(loss, self.model.trainable_variables)
      self.optim.apply_gradients(zip(grad, self.model.trainable_variables))
      self.metric(loss)
    
    return np.float(loss)

  def test_permissible(self, in_img, label, h_only):
    """Test."""

    self.metric.reset_states()

    with tf.GradientTape() as tape:
      # Subtrack means and divide by std.
      in_img = self.preprocess_input(in_img, h_only=h_only)

      # Convert numpy array to tensor.
      in_img = np.expand_dims(in_img, axis=0)
      in_tens = tf.convert_to_tensor(in_img, dtype=tf.float32)

      # Forward pass.
      y_pred = self.model(in_tens)

      # Get label.
      y_true = tf.constant([label])

      # Loss.
      loss = self.bce(y_true, y_pred)
    
    return np.float(y_pred), np.float(loss)

  def load(self, path):
    """Load model weights."""
    self.model.load_weights(path)
  
  def save(self, filename):
    """Save the model."""
    self.model.save_weights(filename)

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

