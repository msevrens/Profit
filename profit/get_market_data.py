#!/usr/local/bin/python3

"""This module fetches market data

Created on May 6, 2020
@author: Matthew Sevrens
"""

#################### USAGE ##########################

# python3 -m profit.get_market_data 

#####################################################

"""

I want to know, if at each investment decision, instead of having made the decision I did, rather I invested in an index fund, the same amount, what would I have made

I think I could pull all my transactions made, when I purchased them, how much, what stock and then use those values to simulate each transaction having been 
made into an index fund. Because the number one thing I want to know is whether my efforts are actually contributing to doing better

"""

import os
import sys
import requests
import datetime as dt
from dateutil.parser import parse

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

	# Sources
	wsj = pd.read_csv('data/WSJ-Dow-HistoricalPrices-70-20.csv')
	mt = pd.read_csv('data/dow-jones-industrial-average-daily.csv')
	yahoo = pd.read_csv('data/yahoo-DJI.csv')

	# Remove Adj Close
	yahoo.drop('Adj Close', axis=1, inplace=True)

	# yahoo = yf.Ticker("DJI")

	# Match Date Formats and Merge
	wsj['Date'] = wsj['Date'].apply(lambda x: parse(x).strftime("%Y-%m-%d"))
	merged = pd.merge(mt, wsj, on="Date", how="outer")
	three = pd.merge(merged, yahoo, on="Date", how="outer")

	# Compare
	match_map = (three['Closing Value'] == three[' Close']) & (three['Closing Value'] == round(three['Close'], 2))

	# Merge 

	# Save

	return merged

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

get_dow()