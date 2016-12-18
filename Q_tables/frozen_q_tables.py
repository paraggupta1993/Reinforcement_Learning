import gym
import numpy as np
import random

env = gym.make('FrozenLake-v0')
max_steps = 500
num_episodes = 3000
learning_rate = 0.1

for discount in range(1, 20):

    ## Training

    discount = .80 + float(discount) / 100.0
    Q = np.zeros([env.observation_space.n, env.action_space.n])
    epsilon = 1.00
    div = epsilon / float(num_episodes)
    for episode in range(1, 1 + num_episodes):

        epsilon = epsilon - div
        state = env.reset()

        for _ in range(max_steps):

            if random.random() < epsilon:
                action = env.action_space.sample()
            else:
                action = np.argmax(Q[state, :])

            new_state, reward, done, x = env.step(action)
            Q[state, action] += learning_rate * (reward + discount * np.max(Q[new_state, :]) - Q[state, action])

            if done: break

            state = new_state

    # print Q

    ## Testing
    test_episodes = 100
    solved = 0
    for episode in range(test_episodes):
        state = env.reset()

        for _ in range(max_steps):

            action = np.argmax(Q[state, :])
            state, reward, done, x = env.step(action)

            if reward:
                solved += reward

            if done: break

    print "discount:", discount, "solved:", solved,"/", test_episodes

