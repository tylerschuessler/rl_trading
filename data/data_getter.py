#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import datetime
import yfinance as yf


# In[2]:


def get_eod_min(ticker_name):
    '''
    gets eod minute data for a particular ticker_name
    writes it to file
    input(s): ticker_name string with the relevant company ticker label
    '''
    old_data = pd.read_csv('stored_data/{}.csv'.format(ticker_name))
    old_data = old_data[['Datetime', 'Open', 'High', 'Low', 'Close', 'Stock Splits', 'Dividends', 'Volume']]
    tick = yf.Ticker(ticker_name)
    new_data = tick.history(period = '1d', interval = '1m')
    new_data = new_data.tz_localize(None)
    new_data = new_data.reset_index(level=0)
    data = pd.concat([old_data, new_data], sort = False)
    data['Datetime'] = pd.to_datetime(data['Datetime'].values)
    data = data.drop_duplicates(subset=['Datetime'])
    data = data.reset_index(drop = True)
    data = data.sort_values(by='Datetime')
    data.to_csv('stored_data/{}.csv'.format(ticker_name))
    
    
    


# In[3]:


def first_write(ticker_name):
    '''
    for a new company/index gets the new data , i.e. past 7 days
    writes it to file
    input(s): ticker_name string with the relevant company ticker label
    '''
    tick = yf.Ticker(ticker_name)
    new_data = tick.history(period = '7d', interval = '1m')
    new_data = new_data.tz_localize(None)
    new_data.to_csv('stored_data/{}.csv'.format(ticker_name))


# In[11]:


def daily_update(basket, f = get_eod_min):
    '''
    iterates through basket of stocks to update their data on a daily basis
    input(s): basket - list of tickers to iterate through, f - function either end of day daily update 
        or first time getting data
    '''
    for stonk in basket:
        print('Fetching {}...'.format(stonk))
        f(stonk)
    print('done.')


# In[14]:


basket = ['AAPL', 'TSLA', 'GE', 'MSFT', 'VALE', 'NIO', 'TWTR', 'PTON', 'FB', 'F', 'BAC', 'NKLA', 'AAL', 'NKE', 'PINS',
         'T', 'ZM', 'CCL', 'FCX', 'PBR', 'C', 'SRNE', 'INTC', 'SNAP', 'ROKU', 'NVDA', 'WORK', 'UBER', 'AMAT', 'TSM', 'GOOG', 'AMZN']


# In[15]:


daily_update(basket)


# In[ ]:




