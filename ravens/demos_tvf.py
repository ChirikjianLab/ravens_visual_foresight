"""Data collection script for training TVF."""

import os
from absl import app
from absl import flags
import numpy as np
from ravens import tasks
from ravens.dataset_tvf_demo import DatasetTVFDemo
from ravens.environments.environment_mcts import EnvironmentMCTS

flags.DEFINE_string('assets_root', './ravens/environments/assets/', '')
flags.DEFINE_string('data_dir', './data_train', '')
flags.DEFINE_bool('disp', True, '')
flags.DEFINE_bool('shared_memory', False, '')
flags.DEFINE_string('task', 'put-block-base', '')
flags.DEFINE_string('mode', 'train', '')
flags.DEFINE_integer('n', 10, '')
flags.DEFINE_bool('pp', True, '') # Flag for discrete pick and place observation
flags.DEFINE_integer('steps_per_seg', 3, '')
flags.DEFINE_bool('random', True, '') # Flag for random data collection
flags.DEFINE_integer('max_num_random', 2, '') # Maximum number of random actions

FLAGS = flags.FLAGS

def main(unused_argv):
  # Initialize environment and task.
  env_cls = EnvironmentMCTS
  env = env_cls(
      FLAGS.assets_root,
      disp=FLAGS.disp,
      shared_memory=FLAGS.shared_memory,
      hz=480)
  task = tasks.names[FLAGS.task+'-mcts'](pp=FLAGS.pp)
  task.mode = 'train'

  # Initialize scripted oracle agent and dataset.
  agent = task.oracle(env, steps_per_seg=FLAGS.steps_per_seg)
  dataset = DatasetTVFDemo(os.path.join(FLAGS.data_dir, f'{FLAGS.task}-mcts-pp-{FLAGS.mode}'))

  # Train seeds are even and test seeds are odd.
  seed = dataset.max_seed
  if seed < 0:
    seed = -1 if (FLAGS.mode == 'test') else -2

  # Collect training data from oracle demonstrations.
  while dataset.n_episodes < FLAGS.n:
    print(f'Oracle demonstration: {dataset.n_episodes + 1}/{FLAGS.n}')
    episode, total_reward = [], 0
    seed += 2
    np.random.seed(seed)
    env.set_task(task)
    [obs, base_urdf, base_size, base_id, objs_id, info] = env.reset()
    max_steps = task.max_steps
    reward = 0

    # Test data does not need random actions.
    if FLAGS.mode == 'test':
      max_steps -= 2

    print(f'max_step: {max_steps}')

    if FLAGS.random:
      num_random = 0

    last_random_flag = None

    for step_i in range(max_steps):
      
      # Decide whether to take random actions.
      if FLAGS.random:
        if num_random == FLAGS.max_num_random:
          info['random'] = False
        else:
          if step_i == max_steps - 2 and num_random < FLAGS.max_num_random:
            info['random'] = True
          elif step_i == max_steps - 3 and num_random == 0:
            info['random'] = True
          else:
            # We don't want consecutive random actions.
            if last_random_flag:
              info['random'] = False
            else:
              random_flag = np.random.choice(range(2))
              if random_flag == 1:
                info['random'] = True
              else:
                info['random'] = False
        
        if info['random']:
          print('Random action')
          num_random += 1
          last_random_flag = True
        else:
          last_random_flag = False

        if step_i == max_steps - 1:
          assert not info['random']
          assert num_random == FLAGS.max_num_random
      else:
        info['random'] = False

      # Get the action.
      act, primitive = agent.act(obs, info)

      # Record the episode
      info['primitive'] = primitive
      info['base_size'] = base_size
      info['base_id'] = base_id
      info['objs_id'] = objs_id
      info['base_urdf'] = base_urdf
      episode.append((obs, act, reward, info))

      # Make a step.
      [_, obs], reward, done, info = env.step(act)

      total_reward += reward
      print(f'Total Reward: {total_reward} Done: {done} Primitive: {primitive} ')
      
      if done:
        info['primitive'] = -2
        info['base_size'] = base_size
        info['base_id'] = base_id
        info['objs_id'] = objs_id
        info['base_urdf'] = base_urdf
        break

    episode.append((obs, None, reward, info))

    # Only save completed demonstrations.
    if total_reward > 0.99:
      dataset.add(seed, episode)

if __name__ == '__main__':
  app.run(main)
