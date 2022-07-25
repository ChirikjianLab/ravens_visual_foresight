"""Test Transporters with Visual Foresight (TVF)."""

import os
import time
from absl import app
from absl import flags
import numpy as np

import tensorflow as tf
import pybullet as p

from ravens import tasks
from ravens import dataset
from ravens.mcts.tree_search_gctn import BestFirstSearch
from ravens.environments.environment_mcts import EnvironmentMCTS


flags.DEFINE_string('data_dir', './data_test', '')
flags.DEFINE_string('assets_root', './ravens/environments/assets/', '')
flags.DEFINE_string('dynamics_model_name', 'resnet_lite', '')
flags.DEFINE_string('method', 'tvf-large', '')
flags.DEFINE_string('task', 'stack-big-stair', '')
flags.DEFINE_bool('disp', False, '')
flags.DEFINE_integer('n_demos', 10, '')
flags.DEFINE_integer('n_runs', 1, '')
flags.DEFINE_integer('dynamics_total_steps', 60000, '')
flags.DEFINE_integer('gctn_total_steps', 40000, '')
flags.DEFINE_integer('gpu', 0, '')
flags.DEFINE_integer('n_tests', 20, '')

FLAGS = flags.FLAGS

def main(unused_argv):
  # Configure which GPU to use.
  cfg = tf.config.experimental
  gpus = cfg.list_physical_devices('GPU')
  if not gpus:
    print('No GPUs detected. Running with CPU.')
  else:
    cfg.set_visible_devices(gpus[FLAGS.gpu], 'GPU')

  np.set_printoptions(precision=3)

  if FLAGS.method == 'tvf-small':
    k = 2
  elif FLAGS.method == 'tvf-large':
    k = 3
  else:
    raise ValueError('method should be tvf-small or tvf-large!')

  print(f"Method: {FLAGS.method}, K-Means Clustering: {k}")
  print(f'Testing {FLAGS.task}...')

  for train_run in range(FLAGS.n_runs):
    # Set seeds.
    tf.keras.utils.set_random_seed(train_run)
    tf.config.experimental.enable_op_determinism()

    # Model path.
    dynamics_model_path = f'./dynamics_models/vf_{FLAGS.n_demos}/pp-dynamics-task6-demo{FLAGS.n_demos}-resnet_lite-seed{train_run}-step60000-ckpt-{FLAGS.dynamics_total_steps}.h5'
    attention_model_path = f'./gctn_models/GCTN-Multi-transporter-goal-{FLAGS.n_demos}-{train_run}-rots-36-fin_g/attention-ckpt-{FLAGS.gctn_total_steps}.h5'
    transport_model_path = f'./gctn_models/GCTN-Multi-transporter-goal-{FLAGS.n_demos}-{train_run}-rots-36-fin_g/transport-ckpt-{FLAGS.gctn_total_steps}.h5'
      
    # Initialize the tree.
    Planner = BestFirstSearch(
        1,
        k,
        FLAGS.dynamics_model_name)
    
    # Load the trained model for dynamics.
    Planner.load_dynamics_model(
        0,
        dynamics_model_path, 
        FLAGS.dynamics_total_steps)
    
    # Load the trained models for primitives.
    Planner.load_gctn_model(
        attention_model_path,
        transport_model_path,
        FLAGS.gctn_total_steps)

    # Initialize environment and task.
    env = EnvironmentMCTS(
        FLAGS.assets_root,
        disp=FLAGS.disp,
        shared_memory=False,
        hz=480)
    task = tasks.names[f'{FLAGS.task}-mcts'](pp=True)
    task.mode = 'test'

    # Load test dataset.
    ds = dataset.Dataset(os.path.join(FLAGS.data_dir, f'{FLAGS.task}-mcts-pp-test'))

    num_success = 0
    num_test = 0
    total_rewards = 0.0
    assert FLAGS.n_tests <= ds.n_episodes

    for i in range(FLAGS.n_tests):
      print(f'Test: {i + 1}/{ds.n_episodes}')
      episode, seed = ds.load(i)

      # Set the goal.
      goal = episode[-1]
      Planner.set_goal_obs(goal[0])
      
      # Reset env.
      np.random.seed(seed)
      env.seed(seed)
      env.set_task(task)
      obs, _, _, _, _, _ = env.reset()
      total_reward = 0
      reward = 0
      num_test += 1

      # task.max_steps includes two steps for random actions in data collection.
      for _ in range(task.max_steps-2):
        if FLAGS.method == 'tvf-small':
          # K2-M1-G0
          acts_1, _, next_imgs_means_1 = Planner.expand_obs_depth1(obs, visualize=False)
          best_act_idx = Planner.select_best_act_depth1(next_imgs_means_1)
        elif FLAGS.method == 'tvf-large':
          # K3-M3-G0
          acts, _, next_imgs_means = Planner.expand_obs_depth3(obs, visualize=False)
          acts_1 = acts[0]
          next_imgs_means_1 = next_imgs_means[0]
          next_imgs_means_2 = next_imgs_means[1]
          next_imgs_means_3 = next_imgs_means[2]
          best_act_idx = Planner.select_best_act_depth3(next_imgs_means_1, next_imgs_means_2, next_imgs_means_3)

        [_, obs], reward, done, _ = env.step(acts_1[best_act_idx])
        total_reward, _ = env.get_total_reward()

        print(f'Total Reward: {total_reward} Step Reward: {reward} Done: {done}')
        print('-------------')

        if done: 
          break
      
      if total_reward > 0.999:
        num_success += 1
      total_rewards += total_reward
    
    p.disconnect()
    success_rate = num_success / num_test
    avg_reward = total_rewards / num_test

    print('-----------------')
    print(f'Number of Demo: {FLAGS.n_demos}, Method: {FLAGS.method}, Training Run: {train_run}')
    print(f'Success rate: {success_rate}')
    print(f'Avg reward: {avg_reward}')

if __name__ == '__main__':
  app.run(main)