import tensortflow as tf
from utils import get_time, save_pkl, load_pkl
from .base import BaseModel
from .history import History
from .replay_memory import ReplayMemory

class Agent(BaseModel):

    def __init__(self, config, environment, sess):
        super(Agent, self).__init__(config)
        self.sess = sess
        self.env = environment
        self.memory = ReplayMemory(self.config, self.model_dir)
        self.build_dqn()

    def train(self):
        ## new game
        screen, reward, action, terminal = self.env.new_random_game()

        ## populate history with same screen
        for _ in range(self.history.length):
            self.history.add(screen)

        ## Take some random steps
        for self.step in tqdm(range(start_step, self.learn_step), ncols=70, initial=start_step):

            action = self.predict(self.history.get(), random_action=True)
            screen, reward, terminal = self.env.act(action, is_training=True)

            self.history.add(screen)
            self.memory.add(screen, reward, action, terminal)


        for self.step in tqdm(range(learn_step, self.max_step), ncols=70, initial=learn_step):

            ## Predict action
            action = self.predict(self.history.get(), random_action=True)
            screen, reward, terminal = self.env.act(action, is_training=True)

            self.history.add(screen)
            self.memory.add(screen, reward, action, terminal)

            ## train NN with sample events

            self.minibatching()

    def minibatching():
        stateT action, reward, stateT1, terminal = self.memory.sample()

        ##

    def predict(self, state, random_action):
        ep = (self.ep_end +
                    max(0., (self.ep_start - self.ep_end)
                                  * (self.ep_end_t - max(0., self.step - self.learn_start)) / self.ep_end_t))

        if random_action or random.random() < ep:
            action = random.randrange(self.env.action_size)
        else:
            pass
            ##action = ##

        return action


    def play(self):
        ## Called when just testing the model
        pass

    def build_dqn(self):
        ## Build placeholders for inputs
        ## Build the skeleton for tensors to flow through
        ## build convolutions
        ## build fully connected layers

        ## Pixels of recent 4 frames
        self.state = tf.placeholder('float32',
                [None, self.screen_height, self.screen_width, self.history_length], name='state']






