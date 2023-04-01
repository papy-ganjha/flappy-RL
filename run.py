import os, sys
import gymnasium as gym
import time
import joblib
import pickle
import text_flappy_bird_gym
from abc import ABC, abstractmethod
import numpy as np
class BaseAgent():
    """
    Backbone of an agent
    """
    def __init__(self, eps, step_size, discount, env, q):
        super(BaseAgent, self).__init__()
        self.eps = eps
        self.step_size = step_size
        self.discount = discount
        self.env = env
        self.q = q

    def choose_action(self, state):
        action = np.argmax(self.q[state])
        return action
        


if __name__ == '__main__':

    # initiate environment
    env = gym.make('TextFlappyBird-v0', height = 15, width = 20, pipe_gap = 4)
    obs, _ = env.reset()
    total_reward = 0

    # iterate
    with open("./q_values.pkl", "rb") as file:
        q = pickle.load(file)
        file.close()
    # print(q)
    # raise Exception
    agent = BaseAgent(eps=0.2, step_size = 0.7, discount=0.95, env=env, q=q)
    while True:

        # Select next action
        action = agent.choose_action(obs)
        # action = env.action_space.sample()  # for an agent, action = agent.policy(observation)

        # Appy action and return new observation of the environment
        obs, reward, done, _, info = env.step(action)
        total_reward += reward
        # print(obs, reward, info)

        # Render the game
        os.system("clear")
        sys.stdout.write(env.render())
        time.sleep(0.2) # FPS

        # If player is dead break
        if done:
            break
    print("Total reward is: ", total_reward)

    env.close()

