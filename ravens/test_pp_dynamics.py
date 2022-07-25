"""Test the visual foresight (VF) model."""

import os

from absl import app
from absl import flags
from ravens.dataset_pp_dynamics import DatasetPPDynamics
from ravens.dynamics.pp_dynamics_trainer import PPDynamicsTrainer
import tensorflow as tf


flags.DEFINE_string('data_dir', './data_test', '')
flags.DEFINE_string('model_name', 'resnet_lite', '')
flags.DEFINE_string('task_set', 'unseen', '')
flags.DEFINE_integer('n_demos', 10, '')
flags.DEFINE_integer('n_runs', 1, '')
flags.DEFINE_integer('dynamics_total_steps', 60000, '')
flags.DEFINE_integer('gpu', 0, '')

FLAGS = flags.FLAGS

training_tasks = [
  'put-block-base',
  'stack-square',
  'stack-t',
  'stack-tower',
  'stack-pyramid',
  'stack-palace']

unseen_tasks = [
  'put-plane',
  'put-t',
  'stack-stair',
  'stack-twin-tower',
  'stack-big-stair',
  'stack-building',
  'stack-pallet',
  'stack-rectangle']

def main(unused_argv):
  # Configure which GPU to use.
  cfg = tf.config.experimental
  gpus = cfg.list_physical_devices('GPU')
  if not gpus:
    print('No GPUs detected. Running with CPU.')
  else:
    cfg.set_visible_devices(gpus[FLAGS.gpu], 'GPU')
  
  if FLAGS.task_set == 'training':
    tasks = training_tasks
  else:
    tasks = unseen_tasks
  
  # Load test dataset.
  dataset_list = [os.path.join(FLAGS.data_dir, f'{task}-mcts-pp-test') for task in tasks]
  dataset = DatasetPPDynamics(dataset_list)

  for train_run in range(FLAGS.n_runs):
    # Model path.
    dynamics_model_path = f'./dynamics_models/vf_{FLAGS.n_demos}/pp-dynamics-task6-demo{FLAGS.n_demos}-resnet_lite-seed{train_run}-step60000-ckpt-{FLAGS.dynamics_total_steps}.h5'
    
    print(f"Dynamics Model: {dynamics_model_path}")
    print(f'Evaluating on {FLAGS.task_set} tasks...')
    
    # Set seeds.
    tf.keras.utils.set_random_seed(train_run)
    tf.config.experimental.enable_op_determinism()

    # Initialize trainer and load the model.
    trainer = PPDynamicsTrainer(FLAGS.model_name)
    trainer.load_from_path(dynamics_model_path, total_steps=FLAGS.dynamics_total_steps)

    # Test.
    rgb_loss, height_loss, transition = trainer.validate_pp(dataset, episode_num=20)
    
    print('------------------')
    print(f'Total transition num: {transition}')
    print(f"Total avg RGB loss: {rgb_loss / transition}")
    print(f"Total avg height loss: {height_loss / transition}")

if __name__ == '__main__':
  app.run(main)