#!/usr/local/bin/python3

"""This module trains a profit agent in an OpenAI Gym environment

Created on May 25, 2020
@author: Matthew Sevrens
"""

# https://towardsdatascience.com/creating-a-custom-openai-gym-environment-for-stock-trading-be532be3910e
# https://github.com/bioothod/gym-stocks
# https://github.com/notadamking/Stock-Trading-Environment

#################### USAGE ##########################

# python3 -m profit.train_model [mode] [model]
# python3 -m profit.train_model test models/model_13K-V_20K-T.zip

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

def test_model(model_file):
    """Test model"""

    # Load Data and Environments
    validation_data = get_historical_prices(time_frame=("2016-01-01", "2018-09-30"))
    test_data = get_historical_prices(time_frame=("2018-09-30", "2020-04-02"))
    validationEnv = DummyVecEnv([lambda: MultiStockEnv(validation_data, train=False)])
    testEnv = DummyVecEnv([lambda: MultiStockEnv(test_data, train=False)])

    print("Number of days in validation data: " + str(len(validation_data)))
    print("Number of days in test data: " + str(len(test_data)) + "\n")

    # Load Model if Necessary
    if type(model_file) == str:
        model = PPO2(MlpPolicy, validationEnv, verbose=1)
        model.load(model_file, env=validationEnv)
    else:
        model = model_file

    results = []
    model.set_env(validationEnv)
    obs = validationEnv.reset()

    # Validate
    for i in range(10):
        for j in range(len(validation_data) - 1):
            # print("Step: " + str(j))
            action, _states = model.predict(obs)
            obs, rewards, done, info = validationEnv.step(action)
            # validationEnv.render(mode="test")
        print("Validation run #" + str(i))
        total_reward = validationEnv.render(mode="test", suffix="_validation_" + str(i))
        results.append(total_reward)
        obs = validationEnv.reset()

    print(results)
    print("\nAverage Validation Reward: " + str(sum(results) / len(results)))
    results = []
    model.set_env(testEnv)
    obs = testEnv.reset()

    # Test
    for i in range(10):
        for j in range(len(test_data) - 1):
            # print("Step: " + str(j))
            action, _states = model.predict(obs)
            obs, rewards, done, info = testEnv.step(action)
            # testEnv.render(mode="test")
        print("Test run #" + str(i))
        total_reward = testEnv.render(mode="test", suffix="_test_" + str(i))
        results.append(total_reward)
        obs = testEnv.reset()

    print(results)
    print("\nAverage Test Reward: " + str(sum(results) / len(results)))

def train_model():
    """Train model"""

    # Load Data and Environment
    train_data = get_historical_prices(time_frame=("2000-01-01", "2015-12-31"))
    trainEnv = DummyVecEnv([lambda: MultiStockEnv(train_data)])

    print("Number of days in train data: " + str(len(train_data)) + "\n")

    # Init Model
    model = PPO2(MlpPolicy, trainEnv, verbose=1)

    # Train Model
    model.learn(total_timesteps=25000)
    obs = trainEnv.reset()

    # Test Model
    test_model(model)

    # Save Final Model
    model.save("models/model")

def main():
    """Run module from command line"""

    mode = sys.argv[1] if len(sys.argv) > 1 else None
    model_file = sys.argv[2] if len(sys.argv) > 2 else None

    if mode == "test":
        test_model(model_file)
    else:
        train_model()

if __name__ == "__main__":
    main()