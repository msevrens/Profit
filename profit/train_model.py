#!/usr/local/bin/python3

"""This module trains a profit agent in an OpenAI Gym environment

Created on May 25, 2020
@author: Matthew Sevrens
"""

# https://towardsdatascience.com/creating-a-custom-openai-gym-environment-for-stock-trading-be532be3910e
# https://github.com/bioothod/gym-stocks
# https://github.com/notadamking/Stock-Trading-Environment

#################### USAGE ##########################

# python3 -m profit.train_model

#####################################################

import json
import datetime as dt

import pandas as pd
import numpy as np
import gym

from stable_baselines.common.policies import MlpPolicy
from stable_baselines.common.vec_env import DummyVecEnv
from stable_baselines import PPO2

from profit.trading_environment import StockTradingEnv

df = pd.read_csv('data/AAPL.csv')
df = df.sort_values('Date')

# Vectorize Environment
env = DummyVecEnv([lambda: StockTradingEnv(df)])
model = PPO2(MlpPolicy, env, verbose=1)
model.learn(total_timesteps=20000)
obs = env.reset()

for i in range(2000):
	action, _states = model.predict(obs)
	obs, rewards, done, info = env.step(action)
	env.render()