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

df = pd.read_csv('data/AAPL.csv')
df = df.sort_values('Date')

def load_data(train):
    """Run module"""

    daily_data = []

    data_1 = pd.read_csv('data/dow_jones_30_daily_price.csv')

    equal_4711_list = list(data_1.tic.value_counts() == 4711)
    names = data_1.tic.value_counts().index

    select_stocks_list = list(names[equal_4711_list]) + ['NKE','KO']

    data_2 = data_1[data_1.tic.isin(select_stocks_list)][~data_1.datadate.isin(['20010912', '20010913'])]
    data_3 = data_2[['iid', 'datadate', 'tic', 'prccd', 'ajexdi']]
    data_3['adjcp'] = data_3['prccd'] / data_3['ajexdi']

    if train:
        time_frame = data_3[(data_3.datadate > 20090000) & (data_3.datadate < 20160000)]
    else:
        time_frame = data_3[data_3.datadate > 20160000]

    for date in np.unique(time_frame.datadate):
        daily_data.append(time_frame[time_frame.datadate == date])

    return daily_data

# Load Data
train_data = load_data(True)
test_data = load_data(False)

print("Number of days in train data: " + str(len(train_data)))
print("Number of days in test data: " + str(len(test_data)))

# Vectorize Environment
trainEnv = DummyVecEnv([lambda: MultiStockEnv(train_data)])
model = PPO2(MlpPolicy, trainEnv, verbose=1)
model.learn(total_timesteps=20000)
obs = trainEnv.reset()

testEnv = DummyVecEnv([lambda: MultiStockEnv(test_data, train=False)])

for i in range(len(test_data) - 1):
	print("Step: " + str(i))
	action, _states = model.predict(obs)
	obs, rewards, done, info = testEnv.step(action)
	testEnv.render(mode="test")