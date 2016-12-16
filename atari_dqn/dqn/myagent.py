import tensorflow as tf
from tqdm import tqdm
from utils import get_time, save_pkl, load_pkl
from .base import BaseModel
from .history import History
from .replay_memory import ReplayMemory
import random
class MyAgent(BaseModel):

    def __init__(self, config, environment, sess):
        super(MyAgent, self).__init__(config)
        self.sess = sess
        self.env = environment
        self.history = History(self.config)
        self.memory = ReplayMemory(self.config, self.model_dir)
        self.build_dqn()
    def train(self):
        ## new game
        screen, reward, action, terminal = self.env.new_random_game()

        ## populate history with same screen
        for _ in range(self.config.history_length):
            self.history.add(screen)

        ## Take some random steps to fill memory
        for self.step in tqdm(range(1, int(self.config.learn_start)), ncols=70, initial=1):

            action = self.predict(self.history.get(), random_action=True)
            screen, reward, terminal = self.env.act(action, is_training=True)

            self.history.add(screen)
            self.memory.add(screen, reward, action, terminal)

        ## Take steps and learn
        for self.step in tqdm(range(int(self.config.learn_start), int(self.config.max_step)), ncols=70, initial=int(self.config.learn_start)):

            ## Predict action
            action = self.predict(self.history.get(), random_action=True)
            screen, reward, terminal = self.env.act(action, is_training=True)

            self.history.add(screen)
            self.memory.add(screen, reward, action, terminal)

            ## train NN with sample events

            self.minibatching()

    def minibatching(self):
        # FF state to find Q(s,a)
        # FF all s' to find max Q(s', a')
        # calculate target
        stateT, action, reward, stateT1, terminal = self.memory.sample()

        qT1 = self.q.eval({self.state: stateT1})
        terminal = np.array(terminal) + 0.
        maxqT1 = np.max(qT1, axis=1)
        target = (1. - terminal) * self.discount * maxqT1 + reward

        self.sess.run([self.train_op ], {
            self.state: stateT,
            self.action_pl: action,
            self.y_pl: target
            })

    def predict(self, state, random_action):
        ep = (self.ep_end +
                    max(0., (self.ep_start - self.ep_end)
                                  * (self.ep_end_t - max(0., self.step - self.learn_start)) / self.ep_end_t))

        if random_action or random.random() < ep:
            action = random.randrange(self.env.action_size)
        else:
            action = self.q_action.eval({self.state: [state]})[0]

        return action


    def play(self):
        ## Called when just testing the model
        pass

    def build_dqn(self):
        ## Here minibatches will flow through the tensor graph
        ## None == 32 for minibatches of 32 states

        ## Pixels of recent 4 frames
        self.state = tf.placeholder(shape=[None, 84, 84,4], dtype=tf.uint8, name='state')
        self.state = tf.to_float(self.state) / 255.0
        # The TD target value
        self.y_pl = tf.placeholder(shape=[None], dtype=tf.float32, name="y")

        ## convolutions
        conv1 = tf.contrib.layers.conv2d(self.state, 32, 8, 4, activation_fn=tf.nn.relu)
        conv2 = tf.contrib.layers.conv2d(conv1     , 64, 4, 2, activation_fn=tf.nn.relu)
        conv3 = tf.contrib.layers.conv2d(conv2     , 64, 3, 1, activation_fn=tf.nn.relu)

        flattened = tf.contrib.layers.flatten(conv3)

        ## fully connected hidden layer
        fc1 = tf.contrib.layers.fully_connected(flattened, 512)

        ## output layer
        self.q = tf.contrib.layers.fully_connected(fc1, self.env.action_size)

        self.q_action = tf.argmax(self.q, dimension=1)


        ## Place holder for which actions was taken in minibatch samples
        self.action_pl = tf.placeholder(shape=[None], dtype=tf.int32, name='actions')

        ## Get the predictions for the chosen action only
        gather_indices = tf.range(self.batch_size) * tf.shape(self.q)[1] + self.action_pl
        self.action_prediction = tf.gather(tf.reshape(self.q, [-1]), gather_indices)

        ## Calculate the loss
        self.losses = tf.squared_difference(self.y_pl, self.action_prediction)

        self.loss = tf.reduce_mean(self.losses)

        self.optimizer = tf.train.RMSPropOptimizer(0.00025, 0.99, 0.0, 1e-6)
        self.train_op = self.optimizer.minimize(self.loss, global_step=tf.contrib.framework.get_global_step())


