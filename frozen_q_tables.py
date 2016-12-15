import gym
import numpy as np
import random

env = gym.make('FrozenLake-v0')

Q = np.zeros([env.observation_space.n, env.action_space.n])
alpha = 0.1
gamma = 0.999
epsilon = 1.00
num_episodes = 10000.0
div = epsilon / num_episodes
episode = 0.0
not_stuck = 100
while episode < num_episodes:
    episode += 1.0
    state = env.reset()
    epsilon = epsilon - div
    not_stuck = 100
    # print epsilon
    while not_stuck:
        # env.render()
        not_stuck -= 1
        if random.random() < epsilon:
            action = env.action_space.sample()
            # print action
        else:
            action = np.argmax(Q[state, :])

        new_state, reward, done, x = env.step(action)
        Q[state, action] += alpha * (reward + gamma * np.max(Q[new_state, :]) - Q[state, action])

        #if reward: print reward, state
        if done: break
        state = new_state

print Q
import time
state = env.reset()
not_stuck = 500
while not_stuck:
    env.render()
    not_stuck -= 1
    action = np.argmax(Q[state, :])
    time.sleep(.1)
    new_state, reward, done, x = env.step(action)

    if reward: print reward
    if done: break
    state = new_state
print "Steps: ", 500 - not_stuck
