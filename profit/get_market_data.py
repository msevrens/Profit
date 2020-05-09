#!/usr/local/bin/python3

"""This module fetches market data

Created on May 6, 2020
@author: Matthew Sevrens
"""

#################### USAGE ##########################

# python3 -m mind.get_market_data 

#####################################################

"""

I want to know, if at each investment decision, instead of having made the decision I did, rather I invested in an index fund, the same amount, what would I have made

I think I could pull all my transactions made, when I purchased them, how much, what stock and then use those values to simulate each transaction having been 
made into an index fund. Because the number one thing I want to know is whether my efforts are actually contributing to doing better

"""

import sys
import datetime as dt

import pandas
import pandas_datareader as web

start = dt.datetime(1970, 1, 1)
end = dt.datetime(2020, 1, 1)

df = web.DataReader('IBM', 'yahoo', start, end)
print(df.head(5))
print(df.tail(5))

def success_signal():
	"""Calculate whether investments performing better than market as a whole"""

	return "probably not"
