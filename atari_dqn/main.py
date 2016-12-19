import random
import tensorflow as tf

from dqn.agent import Agent
from dqn.myagent import MyAgent

from dqn.environment import GymEnvironment, SimpleGymEnvironment
from config import get_config

flags = tf.app.flags

# Environment
flags.DEFINE_string('env_name', 'Breakout-v0', 'The name of gym environment to use')
flags.DEFINE_integer('action_repeat', 4, 'The number of action to be repeated')

# Etc
flags.DEFINE_boolean('display', False, 'Whether to do display the game screen or not')
flags.DEFINE_boolean('is_train', True, 'Whether to do training or testing')

FLAGS = flags.FLAGS

# Set random seed
tf.set_random_seed(123)
random.seed(123)

def main(_):
  with tf.Session() as sess:
    config = get_config(FLAGS) or FLAGS

    if config.env_type == 'simple':
      env = SimpleGymEnvironment(config)
    else:
      env = GymEnvironment(config)

    config.cnn_format = 'NHWC'

    agent = MyAgent(config, env, sess)

    if FLAGS.is_train:
      agent.train()
    else:
      agent.play()

if __name__ == '__main__':
  tf.app.run()
