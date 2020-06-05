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

import sys
import json
import datetime as dt

import pandas as pd
import numpy as np
import gym

from stable_baselines.common.policies import MlpPolicy
from stable_baselines.common.vec_env import DummyVecEnv
from stable_baselines import PPO2

from profit.multi_stock_env import MultiStockEnv
from profit.get_market_data import load_dow_data, get_historical_prices


# Load Data
# train_data = load_dow_data(True)
# test_data = load_dow_data(False)

train_data = get_historical_prices(time_frame=("2000-01-01", "2014-12-31"))
validation_data = get_historical_prices(time_frame=("2015-01-01", "2015-12-31"))
test_data = get_historical_prices(time_frame=("2016-01-01", "2018-09-31"))

print("Number of days in train data: " + str(len(train_data)))
print("Number of days in test data: " + str(len(validation_data)))

# Vectorize Environment
trainEnv = DummyVecEnv([lambda: MultiStockEnv(train_data)])
model = PPO2(MlpPolicy, trainEnv, verbose=1)
model.learn(total_timesteps=25000)
obs = trainEnv.reset()

# Validate
validationEnv = DummyVecEnv([lambda: MultiStockEnv(validation_data, train=False)])

results = []

for ii in range(10):
    for i in range(len(validation_data) - 1):
        # print("Step: " + str(i))
        action, _states = model.predict(obs)
        obs, rewards, done, info = validationEnv.step(action)
        # testEnv.render(mode="test")
    print("Validation run #" + str(ii))
    total_reward = validationEnv.render(mode="test", suffix=ii)
    results.append(total_reward)
    obs = validationEnv.reset()

print(results)
print("\nAverage Reward: " + str(sum(results) / len(results)))

# Test
testEnv = DummyVecEnv([lambda: MultiStockEnv(test_data, train=False)])

results = []

for ii in range(10):
    for i in range(len(test_data) - 1):
        # print("Step: " + str(i))
        action, _states = model.predict(obs)
        obs, rewards, done, info = testEnv.step(action)
        # testEnv.render(mode="test")
    print("Test run #" + str(ii))
    total_reward = testEnv.render(mode="test", suffix=ii)
    results.append(total_reward)
    obs = testEnv.reset()

print(results)
print("\nAverage Reward: " + str(sum(results) / len(results)))
