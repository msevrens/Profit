#!/usr/local/bin/python3

"""This module downloads yfinance price history"""

#################### USAGE ##########################

# python3 -m profit.dl_yfinance_history

#####################################################

import os
import shutil
import contextlib
from os.path import isfile, join

import yfinance as yf
import pandas as pd

def make_directory(path):
    if not os.path.exists(path):
        os.makedirs(path)

def move_symbols(symbols, dest):
    for s in symbols:
        filename = '{}.csv'.format(s)
        shutil.move(join('data/Kaggle 2020', filename), join(dest, filename))

def update_stocks():
    """Update individual stocks"""

    offset = 0
    limit = None
    period = 'max'

    data = pd.read_csv("http://www.nasdaqtrader.com/dynamic/SymDir/nasdaqtraded.txt", sep='|')
    data_clean = data[data['Test Issue'] == 'N']
    symbols = data_clean['NASDAQ Symbol'].tolist()

    print('total number of symbols traded = {}'.format(len(symbols)))

    limit = limit if limit else len(symbols)
    end = min(offset + limit, len(symbols))
    is_valid = [False] * len(symbols)

    # force silencing of verbose API
    with open(os.devnull, 'w') as devnull:
        with contextlib.redirect_stdout(devnull):
            for i in range(offset, end):
                s = symbols[i]
                data = yf.download(s, period=period)
                if len(data.index) == 0:
                    continue
            
                is_valid[i] = True
                data.to_csv('data/Kaggle 2020/{}.csv'.format(s))

    print('Total number of valid symbols downloaded = {}'.format(sum(is_valid)))

    valid_data = data_clean[is_valid]
    valid_data.to_csv('symbols_valid_meta.csv', index=False)

    etfs = valid_data[valid_data['ETF'] == 'Y']['NASDAQ Symbol'].tolist()
    stocks = valid_data[valid_data['ETF'] == 'N']['NASDAQ Symbol'].tolist()
        
    move_symbols(etfs, "data/Kaggle 2020/etfs")
    move_symbols(stocks, "data/Kaggle 2020/stocks")

def update_dow():
    """Download latest DOW data"""

    data = yf.download("^DJI", period="max")

    # Return Cached if Available
    if os.path.isfile("data/aggregated_dow_1914-2020.csv"):
        aggregated = pd.read_csv("data/aggregated_dow_1914-2020.csv")
    
    most_recent = aggregated['Date'].max()
    new_data = data[data.index > most_recent]
    aggregated.set_index('Date', inplace=True)
    aggregated.drop(columns=['Index'], inplace=True)
    aggregated.sort_index(inplace=True)

    # TODO: add new data to existing and save

if __name__ == "__main__":
    make_directory('data/Kaggle 2020')
    make_directory('data/Kaggle 2020/etfs')
    make_directory('data/Kaggle 2020/stocks')
    update_stocks()