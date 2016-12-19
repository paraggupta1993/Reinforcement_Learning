import tensorflow as tf
from tensorflow.contrib.layers.python.layers import initializers
from tqdm import tqdm
from utils import get_time, save_pkl, load_pkl
from .base import BaseModel
from .history import History
from .replay_memory import ReplayMemory
import random
import numpy as np

def fully_connected(input_, output_size, name='fully_connected'):
  shape = input_.get_shape().as_list()
  stddev=0.02
  bias_start=0.0,

  activation_fn=tf.nn.relu
  with tf.variable_scope(name):
    w = tf.get_variable('Matrix', [shape[1], output_size], tf.float32,
        tf.random_normal_initializer(stddev=stddev))
    b = tf.get_variable('bias', [output_size],
        initializer=tf.constant_initializer(bias_start))

    out = tf.nn.bias_add(tf.matmul(input_, w), b)

    return activation_fn(out), w, b

def conv2d(x,
           output_dim,
           kernel_size,
           stride,
           name='conv2d'):

  activation_fn=tf.nn.relu
  initializer=tf.truncated_normal_initializer(0, 0.02)

  with tf.variable_scope(name):
    stride = [1, stride[0], stride[1], 1]
    kernel_shape = [kernel_size[0], kernel_size[1], x.get_shape()[-1], output_dim]

    w = tf.get_variable('w', kernel_shape, tf.float32, initializer=initializer)
    conv = tf.nn.conv2d(x, w, stride, 'VALID')

    b = tf.get_variable('biases', [output_dim], initializer=tf.constant_initializer(0.0))
    out = tf.nn.bias_add(conv, b)
    out = activation_fn(out)

    return out, w, b


elements = ['w1', 'w2', 'w3', 'w4', 'w5', 'b1', 'b2', 'b3', 'b4', 'b5']
class MyAgent(BaseModel):


    def __init__(self, config, environment, sess):
        super(MyAgent, self).__init__(config)
        self.sess = sess
        self.env = environment
        self.best_reward = 0.0
        self.episode_reward = 0.0
        self.loss_val = 0.0
        self.history = History(self.config)
        self.memory = ReplayMemory(self.config, self.model_dir)
        self.num_episode = 0
        self.build_dqn()

    def train(self):
        ## new game
        screen, reward, action, terminal = self.env.new_random_game()
        reward = self.clip_reward(reward)

        ## populate history with same screen
        for _ in range(self.config.history_length):
            self.history.add(screen)

        ## Take some random steps to fill memory
        for self.step in tqdm(range(1, int(self.config.learn_start)), ncols=70, initial=1):

            action = self.predict(self.history.get(), random_action=True)
            screen, reward, terminal = self.env.act(action, is_training=True)
            reward = self.clip_reward(reward)

            self.history.add(screen)
            self.memory.add(screen, reward, action, terminal)

            ## new game
            if terminal:
                screen, reward, action, terminal = self.env.new_random_game()
                reward = self.clip_reward(reward)

        self.episode_reward = 0
        self.best_reward = 0
        self.num_episode += 1

        # Start a new game
        screen, reward, action, terminal = self.env.new_random_game()
        reward = self.clip_reward(reward)

        ## Take steps and learn
        for self.step in tqdm(range(int(self.config.learn_start), int(self.config.max_step)), ncols=70, initial=int(self.config.learn_start)):

            ## Predict action
            action = self.predict(self.history.get())

            screen, reward, terminal = self.env.act(action, is_training=True)
            reward = self.clip_reward(reward)
            self.episode_reward += reward

            if self.episode_reward > self.best_reward:
                print
                print "Best Reward: ", self.episode_reward
                self.best_reward = self.episode_reward

            self.history.add(screen)
            self.memory.add(screen, reward, action, terminal)

            ## train NN with sample events
            if self.step % (self.train_frequency) == 0:
                merged_summaries = self.minibatching()

                ## Write Summaries into Summary writer
                # if self.step % self.test_step == 0:
                self.train_writer.add_summary(merged_summaries, self.step)

            ## Copy learned network to target network if necessary
            if self.step % 1000 == 0:

                for elem in elements:
                    self.copy_ops[elem].eval({self.copyplaceholders[elem]: self.net[elem].eval()})

                self.save_model(self.step + 1)

            if terminal:
                ## new game
                print self.num_episode, "Episode Reward: ", self.episode_reward, "loss:", self.loss_val
                self.num_episode += 1
                self.episode_reward = 0
                screen, reward, action, terminal = self.env.new_random_game()
                reward = self.clip_reward(reward)

    def minibatching(self):
        # FF state to find Q(s,a)
        # FF all s' to find max Q(s', a')
        # calculate target
        stateT, action, reward, stateT1, terminal = self.memory.sample()
        # qT1 = self.q.eval({self.state: stateT1})

        ## Doing evaluation using a different target network
        qT1 = self.q_t.eval({self.state: stateT1})
        terminal = np.array(terminal) + 0.
        maxqT1 = np.max(qT1, axis=1)
        target = (1. - terminal) * self.discount * maxqT1 + reward

        _, self.loss_val, merged_summaries = self.sess.run([self.train_op, self.loss, self.merged], {
            self.state: stateT,
            self.action_pl: action,
            self.y_pl: target,
            self.summary_br_pl: self.best_reward,
            self.summary_er_pl: self.episode_reward
            })

        return merged_summaries

    def clip_reward(self, reward):
        if reward > 1: return 1
        elif reward < -1: return -1
        return reward

    def predict(self, state, random_action=False, test_ep=None):
        if test_ep:
            ep = test_ep
        else:
            ep = (self.ep_end +
                    max(0., (self.ep_start - self.ep_end)
                                  * (self.ep_end_t - max(0., self.step - self.learn_start)) / self.ep_end_t))

        if random_action or random.random() < ep:
            action = random.randrange(self.env.action_size)
        else:
            action = self.q_action.eval({self.state: [state]})[0]

        return action


    def play(self, n_step=10000, episodes=100):
        ## Called when just testing the model
        best_reward = 0
        test_history = History(self.config)

        for idx in xrange(episodes):
            screen, reward, action, terminal = self.env.new_random_game()
            reward = self.clip_reward(reward)

            current_reward = 0

            for _ in range(self.history_length):
                test_history.add(screen)

            for t in tqdm(range(n_step), ncols=70):
                action = self.predict(test_history.get(), test_ep=0.1)
                screen, reward, terminal = self.env.act(action, is_training=False)
                reward = self.clip_reward(reward)


                test_history.add(screen)

                current_reward += reward
                if terminal: break

            if current_reward > best_reward:
                best_reward = current_reward
                best_idx = idx

            print ""
            print "Best reward: %d" %(best_reward)
            print ""

    def build_dqn(self):
        ## Here minibatches will flow through the tensor graph
        ## None == 32 for minibatches of 32 states

        # xavier = tf.contrib.layers.xavier_initializer()
        relu = tf.nn.relu
        self.net = {}
        target_net = {}
        net = self.net
        ## Pixels of recent 4 frames
        self.state = tf.placeholder(shape=[None, 84, 84,4], dtype=tf.float32, name='state')

        ## convolutions
        conv1, net['w1'], net['b1'] = conv2d(self.state, 32, [8, 8], [4,4], name='l1')
        conv2, net['w2'], net['b2'] = conv2d(conv1, 64, [4, 4], [2, 3], name='l2')
        conv3, net['w3'], net['b3'] = conv2d(conv2, 32, [3, 3], [1, 1], name='l3')

        flattened = tf.contrib.layers.flatten(conv3)

        ## fully connected hidden layer
        fc1, net['w4'], net['b4'] = fully_connected(flattened, 512, name='l4')

        ## output layer
        self.q, net['w5'], net['b5'] = fully_connected(fc1, self.env.action_size, name='q')

        self.q_action = tf.argmax(self.q, dimension=1)

        #### ---- Action for a state done --- ###

         # The TD target value
        self.y_pl = tf.placeholder(shape=[None], dtype=tf.float32, name="y")

        ## Place holder for which actions was taken in minibatch samples
        self.action_pl = tf.placeholder(shape=[None], dtype=tf.int32, name='actions')


        ## Get the predictions for the chosen action only
        #gather_indices = tf.range(self.batch_size) * tf.shape(self.q)[1] + self.action_pl
        #self.action_prediction = tf.gather(tf.reshape(self.q, [-1]), gather_indices)
        action_one_hot = tf.one_hot(self.action_pl, self.env.action_size, 1.0, 0.0, name='action_one_hot')
        self.action_prediction = tf.reduce_sum(self.q * action_one_hot, reduction_indices=1, name='q_acted')


        ## Calculate the loss
        self.losses = tf.squared_difference(self.y_pl, self.action_prediction)

        self.loss = tf.reduce_mean(self.losses)

        self.optimizer = tf.train.RMSPropOptimizer(0.00025, 0.99, 0.0, 1e-6)

        self.gvs = self.optimizer.compute_gradients(self.loss)

        ## Clip Error
        self.capped_gvs = [(tf.clip_by_value(grad, -1., 1.), var) for grad, var in self.gvs]

        self.train_op = self.optimizer.apply_gradients(self.capped_gvs, global_step=tf.contrib.framework.get_global_step())
        #self.train_op = self.optimizer.minimize(self.loss, global_step=tf.contrib.framework.get_global_step())

        # Merge all the summaries and write them out
        self.summary_br_pl =   tf.placeholder(shape=None, dtype=tf.float32, name='summary_br_pl')
        self.summary_er_pl =   tf.placeholder(shape=None, dtype=tf.float32, name='summary_er_pl')
        tf.summary.scalar('Loss', self.loss)
        tf.summary.scalar('BestReward', self.summary_br_pl)
        tf.summary.scalar('EpisodeReward', self.summary_er_pl)

        self.merged = tf.summary.merge_all()
        self.train_writer = tf.train.SummaryWriter('logs/' + '/train', self.sess.graph)
        # test_writer = tf.train.SummaryWriter('../log/' + '/test')
        # tf.train.SummaryWriter("../log")


        ## Build a target network
        self.state_t = tf.placeholder(shape=[None, 84, 84,4], dtype=tf.float32, name='state_t')
        conv1_t, target_net['w1'], target_net['b1'] = conv2d(self.state_t, 32, [8, 8], [4,4], name='l1_t')
        conv2_t, target_net['w2'], target_net['b2'] = conv2d(conv1_t, 64, [4, 4], [2, 3], name='l2_t')
        conv3_t, target_net['w3'], target_net['b3'] = conv2d(conv2_t, 32, [3, 3], [1, 1], name='l3_t')
        flattened_t = tf.contrib.layers.flatten(conv3_t)
        fc1_t, target_net['w4'], target_net['b4']= fully_connected(flattened_t, 512, name='l4_t')
        self.q_t, target_net['w5'], target_net['b5'] = fully_connected(fc1, self.env.action_size, name='q_t')
        self.q_action_t = tf.argmax(self.q_t, dimension=1)

        ## Build placeholders to copy from network to target network
        self.copyplaceholders = {}
        for elem in elements:
            self.copyplaceholders[elem] = tf.placeholder('float32', target_net[elem].get_shape().as_list(), name=elem)

        # defining ops to copy
        self.copy_ops = {}
        for elem in elements:
            self.copy_ops[elem] = target_net[elem].assign(self.copyplaceholders[elem])

        tf.initialize_all_variables().run()
        self.load_model()
        for elem in elements:
                self.copy_ops[elem].eval({self.copyplaceholders[elem]: self.net[elem].eval()})

