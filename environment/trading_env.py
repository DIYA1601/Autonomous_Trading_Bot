# Autonomous_Trading_Bot
Autonomous_Trading_Bot
import gym
from gym import spaces
import numpy as np
import pandas as pd

class TradingEnv(gym.Env):
    def __init__(self, data):
        super(TradingEnv, self).__init__()
        self.data = data
        self.n_steps = len(data)
        self.current_step = 0
        self.balance = 10000
        self.shares_held = 0
        self.action_space = spaces.Discrete(3)  # 0 = hold, 1 = buy, 2 = sell
        self.observation_space = spaces.Box(low=0, high=np.inf, shape=(5,), dtype=np.float32)

    def reset(self):
        self.current_step = 0
        self.balance = 10000
        self.shares_held = 0
        return self._get_obs()

    def _get_obs(self):
        obs = np.array([
            self.balance,
            self.shares_held,
            self.data['Open'][self.current_step],
            self.data['High'][self.current_step],
            self.data['Low'][self.current_step]
        ])
        return obs

    def step(self, action):
        price = self.data['Close'][self.current_step]
        reward = 0

        if action == 1:  # Buy
            self.shares_held += self.balance // price
            self.balance -= (self.balance // price) * price
        elif action == 2:  # Sell
            self.balance += self.shares_held * price
            self.shares_held = 0

        self.current_step += 1
        done = self.current_step >= self.n_steps - 1
        reward = self.balance + self.shares_held * price
        obs = self._get_obs()
        return obs, reward, done, {}
