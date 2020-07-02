#!/usr/local/bin/python3

"""This module fetches market data

Created on May 6, 2020
@author: Matthew Sevrens
"""

#################### USAGE ##########################

# python3 -m profit.get_market_data 

#####################################################

import os
import sys
import math
import json
import requests
import datetime as dt
from datetime import date, timedelta
from dateutil.parser import parse

import numpy as np
import pandas as pd
import pandas_datareader as web
import matplotlib.pyplot as plt
import yfinance as yf
import bs4 as bs
from pandas.plotting import register_matplotlib_converters

register_matplotlib_converters()

def success_signal():
	"""Calculate whether investments performing better than market as a whole"""

	return "probably not"

def plot_stock(ticker_symbol, rolling=1):
	"""Plot stock value since 1970"""

	start = dt.datetime(1958, 1, 1)
	end = dt.datetime(2020, 1, 1)

	# Load Data
	df = web.DataReader(ticker_symbol, 'yahoo', start, end)

	# Get Rolling Average
	df['Rolling'] = df['Adj Close'].rolling(window=rolling, min_periods=0).mean()
	df.dropna(inplace=True)

	# Plot
	ax1 = plt.subplot2grid((6, 1), (0, 0), rowspan=5, colspan=1)
	ax2 = plt.subplot2grid((6, 1), (5, 0), rowspan=1, colspan=1, sharex=ax1)

	ax1.plot(df.index, df['Adj Close'])
	ax1.plot(df.index, df['Rolling'])
	ax2.bar(df.index, df['Volume'])

	plt.show()

def get_dow():
	"""Merge multiple historical Dow Jones sources"""

	# Return Cached if Available
	if os.path.isfile("data/aggregated_dow_1914-2020.csv"):
		return pd.read_csv("data/aggregated_dow_1914-2020.csv")

	# Sources
	wsj = pd.read_csv('data/WSJ-Dow-HistoricalPrices-70-20.csv')
	mt = pd.read_csv('data/MT-dow-jones-industrial-average-daily.csv')
	yahoo = pd.read_csv('data/yahoo-DJI.csv')

	# Remove Adj Close
	yahoo.drop('Adj Close', axis=1, inplace=True)

	# yahoo = yf.Ticker("DJI")

	# Match Date Formats and Merge
	wsj['Date'] = wsj['Date'].apply(lambda x: parse(x).strftime("%Y-%m-%d"))
	merged = pd.merge(mt, wsj, on="Date", how="outer")
	three = pd.merge(merged, yahoo, on="Date", how="outer")
	three.rename({'Close': 'Yahoo Close', 'Closing Value': 'MT Close', ' Close': 'WSJ Close'}, axis=1, inplace=True)

	# Select Close Price from Sources
	match_map = (three['MT Close'] == three['WSJ Close']) & (three['MT Close'] == round(three['Yahoo Close'], 2))

	def merge_close(x):
		if not math.isnan(x['Yahoo Close']):
			return x['Yahoo Close']
		elif not math.isnan(x['WSJ Close']):
			return x['WSJ Close']
		else:
			return x['MT Close']

	three['Close'] = three.apply(merge_close, axis=1)

	# Save
	three.to_csv("data/aggregated_dow_1914-2020.csv")

	return three

def kaggle_stats():
	"""Get stats about Kaggle data"""

	data_count = {}

	# Return Cached if Available
	if os.path.isfile('data/kaggle_stats.json'):
		with open('data/kaggle_stats.json') as json_file:
			data_count = json.load(json_file)
	else:
		for s in os.listdir('data/Kaggle 2020/stocks/'):
			tic = os.path.splitext(s)[0]
			df = pd.read_csv('data/Kaggle 2020/stocks/' + s)
			data_count[tic] = len(df.index)
		with open('data/kaggle_stats.json', 'w') as fp:
			json.dump(data_count, fp)

	data_count = pd.DataFrame(data_count.items(), columns=['Ticker', 'Days Traded'])
	data_count.sort_values(by=['Days Traded'], ascending=False, inplace=True)

	pd.set_option("display.max_rows", 100)
	data_count = data_count.head(100)
	top_stocks = data_count['Ticker'].tolist()
	newest_ticker = data_count.iloc[99]['Ticker']
	newest_stock = pd.read_csv('data/Kaggle 2020/stocks/' + newest_ticker + ".csv")

	earliest_date = newest_stock['Date'].min()
	latest_date = newest_stock['Date'].max()

	print(top_stocks)
	print(earliest_date, latest_date)

	return top_stocks, (earliest_date, latest_date)

def load_baseline_data():
	"""Load benchmark data source"""

	baseline_data = pd.read_csv('data/dow_jones_30_daily_price.csv')
	equal_timeframe_list = list(baseline_data.tic.value_counts() >= 4711)
	names = baseline_data.tic.value_counts().index
	select_stocks_list = list(names[equal_timeframe_list])

	baseline_subset = baseline_data[baseline_data.tic.isin(select_stocks_list)]
	baseline_subset = baseline_subset[['iid', 'datadate', 'tic', 'prccd', 'ajexdi']]
	baseline_subset['adjcp'] = baseline_subset['prccd'] / baseline_subset['ajexdi']

	daily_data = []

	# Add Baseline Data
	for date in np.unique(baseline_subset.datadate):
		daily_data.append(baseline_subset[baseline_subset.datadate == date])

	tic_order = daily_data[0].tic.values

	return baseline_subset, tic_order, daily_data, select_stocks_list

def get_historical_prices(tickers=[], time_frame=("2009-01-01", "2016-01-01")):
	"""Construct and save historical prices of list of stock tickers"""

	print("\nLoading data from " + time_frame[0] + " to " + time_frame[1])

	# Load Benchmark Data
	if len(tickers) == 0:
		baseline_subset, tic_order, daily_data, select_stocks_list = load_baseline_data()
	else:
		daily_data, tic_order, select_stocks_list = {}, tickers, tickers 

	# Add Kaggle Data
	kaggle_data = {}

	for s in select_stocks_list:
		file_name = "data/Kaggle 2020/stocks/" + s + ".csv"
		file_available = os.path.isfile(file_name)
		if file_available:
			kaggle_data[s] = pd.read_csv(file_name)
			kaggle_data[s].set_index("Date", inplace=True)

	time_range = [kaggle_data[tic].index.tolist() for tic in kaggle_data.keys()]
	time_range = [item for sublist in time_range for item in sublist]
	time_range = pd.DataFrame(np.unique(time_range))

	# Subsample Time Range
	time_slice = time_range[(time_range[0] > time_frame[0]) & (time_range[0] < time_frame[1])]
	daily_data = []

	# Fill Dates
	for date in time_slice[0]:

		date = parse(date)
		rows = []

		for tic in tic_order:

			row = {
				"tic": tic, 
				"close": None, 
				"adjcp": None,
				"datadate": date.strftime("%Y%m%d")
			}

			# No Baseline Data
			if tic not in kaggle_data.keys():
				base_fill = baseline_subset[baseline_subset.tic.isin([tic])][baseline_subset.datadate.isin([date.strftime("%Y%m%d")])]
				row["adjcp"] = base_fill['adjcp'].values[0] if not base_fill.empty else 1
				rows.append(row)
				continue

			stock_data = kaggle_data[tic]

			# No Kaggle Data
			if date.strftime("%Y-%m-%d") not in stock_data.index:
				fill_value = daily_data[-1][daily_data[-1].tic == tic]['adjcp'].values[0]
				row["adjcp"] = fill_value # last available closing price
				rows.append(row)
				continue

			day_data = stock_data.loc[date.strftime("%Y-%m-%d")]
			row["close"] = day_data['Close']
			row["adjcp"] = day_data['Adj Close']
			rows.append(row)

		df = pd.DataFrame(rows)

		if not df["adjcp"].isnull().any():
			daily_data.append(df)
	
	return daily_data

def load_dow_data(train):
	"""Run module"""

	daily_data = []

	data_1 = pd.read_csv('data/dow_jones_30_daily_price.csv')

	equal_timeframe_list = list(data_1.tic.value_counts() >= 4711)
	names = data_1.tic.value_counts().index

	select_stocks_list = list(names[equal_timeframe_list])

	data_2 = data_1[data_1.tic.isin(select_stocks_list)][~data_1.datadate.isin(['20010912', '20010913'])]
	data_3 = data_2[['iid', 'datadate', 'tic', 'prccd', 'ajexdi']]
	data_3['adjcp'] = data_3['prccd'] / data_3['ajexdi']

	if train:
		time_frame = data_3[(data_3.datadate > 20090000) & (data_3.datadate < 20160000)]
	else:
		time_frame = data_3[(data_3.datadate > 20160000)]

	for date in np.unique(time_frame.datadate):
		daily_data.append(time_frame[time_frame.datadate == date])

	return daily_data

def get_sp_tickers():
	"""Scrape S&P 500 ticker symbols from wikipedia"""

	if os.path.isfile('data/S&P 500.txt'):
		with open('data/S&P 500.txt', 'r') as filehandle:
		 	tickers = [ticker.rstrip() for ticker in filehandle.readlines()]
		 	return tickers

	resp = requests.get('https://en.wikipedia.org/wiki/List_of_S%26P_500_companies')
	soup = bs.BeautifulSoup(resp.text, 'lxml')
	table = soup.find('table', {'class': 'wikitable sortable'})
	tickers = []

	for row in table.findAll('tr')[1:]:
		ticker = row.findAll('td')[0].text
		tickers.append(ticker.rstrip("\n"))

	with open('data/S&P 500.txt', 'w') as filehandle:
		filehandle.writelines("%s\n" % ticker for ticker in tickers)

	return tickers

if __name__ == "__main__":
	kaggle_stats()