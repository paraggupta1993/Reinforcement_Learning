# Reinforcement_Learning

Tensorflow implementation of [Playing Atari with Deep Reinforcement Learning](https://www.cs.toronto.edu/~vmnih/docs/dqn.pdf).

Experimenting with Reinforcement learning by Q-Tables-Learning, Deep Q-Learning and Double Deep-Q Learning using GYM OpenAI apis, atari games :)

Our implementation contains:

1. Toy Game - Reinforcement learning with Q-Table on frozen-lake text based game [./Q_tables]
2. Deep Q-network and Q-learning agent for playing atari games. [./atari_dqn/dqn/myagent.py]
    - Reward Clipping
    - Error Clipping

Contributors:
  - Leena Shekhar
  - Parag Gupta
  - Shantanu Patil

## Requirements

- Python 2.7 or Python 3.3+
- [gym](https://github.com/openai/gym)
- [tqdm](https://github.com/tqdm/tqdm)
- [OpenCV2](http://opencv.org/)
- [TensorFlow](https://www.tensorflow.org/)


## Usage

First, install prerequisites with:
```
    $ pip install tqdm gym[all] tensorflow
```
You might have to install tensorflow and opencv2 using platform-based package manager.

To train a model for Breakout:
   ```
    $ chmod +x train.sh
    $ ./train.sh 
```
To train a model for a different game (say Pacman):
```
    $ python main.py --env_name=MsPacman-v0 --is_train=True
    $ python main.py --env_name=MsPacman-v0 --is_train=True --display=True
```
To test and record the screen with gym:
```
    $ chmod +x test.sh 
    $ ./test.sh 
```
or
```
    $ python main.py --is_train=False
    $ python main.py --is_train=False --display=True
```

## Results

Result of training for 24 hours on CPU.

![best](assets/best.gif)




## References

- [simple_dqn](https://github.com/tambetm/simple_dqn.git) for replay memory
- [framework](https://github.com/devsisters/DQN-tensorflow)
- [Playing Atari with Deep Reinforcement Learning](https://www.cs.toronto.edu/~vmnih/docs/dqn.pdf)
