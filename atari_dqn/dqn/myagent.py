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
        ## Called when just training the model

        ## Predict action

        ## train NN with sample events

        ## Get state and Feed Replay Memory


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
                [None, self.history_length, self.screen_height, self.screen_width], name='state']




