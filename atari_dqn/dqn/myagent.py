import tensorflow as tf
from tqdm import tqdm
from utils import get_time, save_pkl, load_pkl
from .base import BaseModel
from .history import History
from .replay_memory import ReplayMemory
import random
import numpy as np
class MyAgent(BaseModel):

    def __init__(self, config, environment, sess):
        super(MyAgent, self).__init__(config)
        self.sess = sess
        self.env = environment
        self.best_reward = 0.0
        self.loss_val = 0.0
        self.history = History(self.config)
        self.memory = ReplayMemory(self.config, self.model_dir)
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


        episode_reward = 0
        self.best_reward = 0
        action = 0
        ## Take steps and learn
        for self.step in tqdm(range(int(self.config.learn_start), int(self.config.max_step)), ncols=70, initial=int(self.config.learn_start)):

            ## Predict action
            action = self.predict(self.history.get())

            screen, reward, terminal = self.env.act(action, is_training=True)
            reward = self.clip_reward(reward)
            episode_reward += reward

            if episode_reward > self.best_reward:
                print "Best Reward: ", episode_reward
                self.best_reward = episode_reward

            self.history.add(screen)
            self.memory.add(screen, reward, action, terminal)

            if terminal:
                ## new game
                episode_reward = 0
                screen, reward, action, terminal = self.env.new_random_game()
                reward = self.clip_reward(reward)

            ## train NN with sample events
            if self.step % (self.train_frequency) == 0:
                self.minibatching()

            if self.step % self.test_step == 0:
                print "loss:", self.loss_val

    def minibatching(self):
        # FF state to find Q(s,a)
        # FF all s' to find max Q(s', a')
        # calculate target
        stateT, action, reward, stateT1, terminal = self.memory.sample()
        qT1 = self.q.eval({self.state: stateT1})
        terminal = np.array(terminal) + 0.
        maxqT1 = np.max(qT1, axis=1)
        target = (1. - terminal) * self.discount * maxqT1 + reward

        _, self.loss_val = self.sess.run([self.train_op, self.loss], {
            self.state: stateT,
            self.action_pl: action,
            self.y_pl: target
            })


    def clip_reward(reward):
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

        xavier = tf.contrib.layers.xavier_initializer()
        relu = tf.nn.relu

        ## Pixels of recent 4 frames
        self.state = tf.placeholder(shape=[None, 84, 84,4], dtype=tf.float32, name='state')

        ## convolutions


        conv1 = tf.contrib.layers.conv2d(self.state, 32, 8, 4, activation_fn=relu )
        conv2 = tf.contrib.layers.conv2d(conv1, 64, 4, 2, activation_fn=relu)
        conv3 = tf.contrib.layers.conv2d(conv2, 64, 3, 1, activation_fn=relu)

        flattened = tf.contrib.layers.flatten(conv3)

        ## fully connected hidden layer
        fc1 = tf.contrib.layers.fully_connected(flattened, 512)

        ## output layer
        self.q = tf.contrib.layers.fully_connected(fc1, self.env.action_size)

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
        self.train_op = self.optimizer.minimize(self.loss, global_step=tf.contrib.framework.get_global_step())

        tf.initialize_all_variables().run()
