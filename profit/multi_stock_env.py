#!/usr/local/bin/python3

"""This module creates an OpenAI Gym environment for trading multiple stocks

Created on May 28, 2020
@author: Matthew Sevrens
"""

# https://towardsdatascience.com/creating-a-custom-openai-gym-environment-for-stock-trading-be532be3910e
# https://github.com/bioothod/gym-stocks
# https://github.com/notadamking/Stock-Trading-Environment

#################### USAGE ##########################

# from profit.multi_stock_env import MultiStockEnv
# env = MultiStockEnv(data)

#####################################################

import sys
from dateutil.parser import parse

import numpy as np
import pandas as pd
from gym.utils import seeding
import gym
from gym import spaces

import matplotlib.pyplot as plt

from profit.get_market_data import get_dow

iteration = 0

class MultiStockEnv(gym.Env):
    """A multi-stock trading environment for OpenAI gym"""

    metadata = {'render.modes': ['human']}

    def __init__(self, data, day=0, train=True):

        self.dow_hist = pd.read_csv('data/aggregated_dow_1914-2020.csv')
        self.dow_hist.set_index('Date', inplace=True)
        self.daily_return = self.dow_hist['Close'].pct_change(1)[1:]
        self.daily_data = data
        self.day = day
        
        # Action Space: Buy or Sell Maximum 5 Shares
        self.action_space = spaces.Box(low=-5, high=5, shape=(28,), dtype=np.int8) 

        # Observation Space: [money] + [prices 1-28] * [owned shares 1-28]
        self.observation_space = spaces.Box(low=0, high=np.inf, shape=(57,))
        
        self.data = self.daily_data[self.day]
        self.date = parse(str(self.data.datadate.iloc[0])).strftime("%Y-%m-%d")
        self.terminal = False
        self.state = [10000] + self.data.adjcp.values.tolist() + [0 for i in range(28)]
        self.reward = 0
        self.asset_memory = [10000]
        self.dow_memory = [10000]

        self.reset()
        self._seed()

    def _sell_stock(self, index, action):
        if self.state[index+29] > 0:
            self.state[0] += self.state[index+1] * min(abs(action), self.state[index+29])
            self.state[index+29] -= min(abs(action), self.state[index+29])
        else:
            pass
    
    def _buy_stock(self, index, action):
        available_amount = self.state[0] // self.state[index+1]
        self.state[0] -= self.state[index+1] * min(available_amount, action)
        self.state[index+29] += min(available_amount, action)
        
    def step(self, actions):
        self.terminal = self.day >= len(self.daily_data) - 1

        if self.terminal:
            plt.plot(self.asset_memory,'r')
            plt.savefig('models/iteration_{}.png'.format(iteration))
            plt.close()
            print("total_reward:{}".format(self.state[0] + sum(np.array(self.state[1:29]) * np.array(self.state[29:])) - 10000))
            
            return self.state, self.reward, self.terminal, {}

        else:
            begin_total_asset = self.state[0] + sum(np.array(self.state[1:29]) * np.array(self.state[29:]))
            argsort_actions = np.argsort(actions)
            sell_index = argsort_actions[:np.where(actions < 0)[0].shape[0]]
            buy_index = argsort_actions[::-1][:np.where(actions > 0)[0].shape[0]]

            for index in sell_index:
                # print('take sell action'.format(actions[index]))
                self._sell_stock(index, actions[index])

            for index in buy_index:
                # print('take buy action: {}'.format(actions[index]))
                self._buy_stock(index, actions[index])

            self.day += 1
            self.data = self.daily_data[self.day]
            self.date = parse(str(self.data.datadate.iloc[0])).strftime("%Y-%m-%d")

            # print("stock_shares:{}".format(self.state[29:]))
            self.state =  [self.state[0]] + self.data.adjcp.values.tolist() + list(self.state[29:])
            end_total_asset = self.state[0] + sum(np.array(self.state[1:29]) * np.array(self.state[29:]))
            # print("end_total_asset:{}".format(end_total_asset))
            
            self.reward = end_total_asset - begin_total_asset            
            # print("step_reward:{}".format(self.reward))

            # Save Progress
            dow_growth = (self.dow_memory[-1] * self.daily_return.loc[self.date]) + self.dow_memory[-1]
            self.dow_memory.append(dow_growth)
            self.asset_memory.append(end_total_asset)

        return self.state, self.reward, self.terminal, {}

    def reset(self):
        self.asset_memory = [10000]
        self.dow_memory = [10000]
        self.day = 0
        self.data = self.daily_data[self.day]
        self.date = parse(str(self.data.datadate.iloc[0])).strftime("%Y-%m-%d")
        self.state = [10000] + self.data.adjcp.values.tolist() + [0 for i in range(28)]
        
        return self.state
    
    def render(self, mode='human'):

        cash = self.state[0]
        stock_values = sum(np.array(self.state[1:29]) * np.array(self.state[29:]))
        total_reward = cash + stock_values - 10000

        print("mode: " + mode)
        print("Cash Values: " + '%.2f'%(cash))
        print("Stock Values: " + '%.2f'%(stock_values))
        print("Total Assets: " + '%.2f'%(cash + stock_values))
        print("Dow Growth: " + '%.2f'%(self.dow_memory[self.day]))
        print("total_reward: {}".format('%.2f'%(total_reward)))
        print("")

        return total_reward

    def _seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]
