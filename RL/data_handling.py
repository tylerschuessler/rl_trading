#!/usr/bin/env python
# coding: utf-8

# In[1]:


from alpha_vantage.timeseries import TimeSeries
import matplotlib.pyplot as plt
from datetime import date
import numpy as np
import pandas as pd


# In[2]:


import datetime
import yfinance as yf
from sklearn.preprocessing import StandardScaler







def get_min_since(data):
    '''
    retrieves the min since market open for past data
    
    input(s): dataframe with past minute granularity data
    '''
    first_dex = 0
    min_since = []
    running_secs = 0
    for i in range(data.shape[0]):
        date = pd.Timestamp(data['Datetime'][i])
        temp = date - pd.Timestamp(year=date.year, month=date.month, day = date.day, hour=9, minute=30, second=0, microsecond=0)#,  tz='America/New_York')#data.index[first_dex] 
        min_since.append(temp.seconds)
    min_since = np.array(min_since)/60
    
    return min_since


# In[23]:


def process_timestamp(data, min_since):
    '''
    processes timestamp into usable machine learning data
    adds new features to data df
    might need adjustments in the future for better data rather than just ordinal encodings 
    
    input(s): dataframe with minute granularity data, array with min_since data
    output(s): df with new features
    '''
    df = data['Datetime'].apply(lambda x: [pd.Timestamp(x).dayofweek, pd.Timestamp(x).month, pd.Timestamp(x).day, pd.Timestamp(x).hour, pd.Timestamp(x).minute])
    temp = pd.DataFrame(df.to_list(), columns = ['day_of_week', 'month', 'day', 'hour', 'min'])
    dex = np.array(list(temp.index))+1
    
    temp['data_pt_dex'] = dex
    temp['min_since_open'] = min_since
    
    temp['open'] = data['Open'].values
    temp['high'] = data['High'].values
    temp['low'] = data['Low'].values
    temp['close'] = data['Close'].values
    temp['volume'] = data['Volume'].values
    
    return temp



def get_min_quote(company):
    '''
    gets the current minute data for a given stock
    
    input(s): string - the ticker label for a compant
    return(s): dataframe with all the data points collected
    '''
    ###make the ticker object
    company = yf.Ticker(company)
    not_done = True
    current_day = pd.DataFrame()
    
    ###open and close
    open_ = datetime.time(hour=7, minute=30, second=0, microsecond=0)
    close_ = datetime.time(hour=14, minute=1, second=0, microsecond=0)
    first_iter = 0
    diff_secs = []
    ###run until the day is over 
    while not_done:
        prev_time = current_time
        current_time = datetime.datetime.now()
        ct =  current_time.time()
        diff_secs.append((current_time.second+(current_time.microsecond/1000000)) - (prev_time.second + (prev_time.microsecond/1000000)))

        if ct >= open_ and ct <= close_:
            ###get stock data
            temp = company.history(period = '4m', interval = '1m')
            temp.reset_index(level=0, inplace=True)
            latest_price = temp.iloc[1]
            current_day = current_day.append(latest_price)
            
            ###wait 60 sec
            if first_iter == 0:
                sleep_time = 60
            else:
                sleep_time = (60 - np.mean(diff_secs))

            first_iter += 1
            time.sleep(sleep_time)

        else:
            not_done = False
    
    
    return current_day


# In[31]:


def read_data(ticker):
    '''
    reads data in from csv
    
    input(s): ticker string with the ticker label
    output(s): dataframe with the data
    '''
    return pd.read_csv('~/desktop/investin/data/stored_data/{}.csv'.format(ticker), index_col = 0)




class transforms():
    '''
    class for defining some methods for transforming some data
    '''
    def __init__(self, data):
        self.scaler = StandardScaler()
        
        self.pre_transformed = data.copy()
        
        self.transformed = data
        
    def standardize(self):
        '''
        ['open', 'high', 'low', 'close', 'volume'] standard normal transform
        '''
        temp = self.pre_transformed[['open', 'high', 'low', 'close', 'volume']]
        X = temp.values
        self.scaler.fit(X)
        X_trans = self.scaler.transform(X)
        self.transformed[['open', 'high', 'low', 'close', 'volume']] = X_trans
    
    def encode(self, data, col, max_val):
        '''
        helper function to convert cyclical time date to sin/cos       
        '''
        #data[col + '_cos'] = np.cos(2 * np.pi * data[col]/max_val)
        return np.sin(2 * np.pi * data[col].values/max_val)
        
    def sin_trans(self):
        '''
        perform sin transform, i.e. periodic transform for the time data
        '''
        
        for col in ['day_of_week', 'month', 'day', 'hour', 'min', 'min_since_open']:
            ###if you double the period vals are scaled between 0 and 1, but no two unique values map to same point
            max_val = 2*np.max(self.transformed[col].values)
            self.transformed[col] = self.encode(self.transformed, col, max_val)
            
    def data_pt_scaler(self):
        '''
        scale data_pt_dex by x/max_val
        '''
        temp = self.transformed['data_pt_dex'].values
        max_val = np.max(temp)
        self.transformed['data_pt_dex'] = temp/max_val
        
    def all_transforms(self):
        '''
        perform all transforms
        '''
        self.standardize()
        self.sin_trans()
        self.data_pt_scaler()
    
    
    def suplemental_factors(self, rolling_mean_sizes = [10, 30, 100], secant_mean_sizes = [3, 6, 12]):
        '''
        add some supplemental statistics 
        '''
        ###some rolling means
        secant = lambda x: (x[-1] - x[0])/x.shape[0]
        for size, size_2 in zip(rolling_mean_sizes, secant_mean_sizes):
            self.transformed['close_roll_{}'.format(size)] = self.transformed['close'].rolling(window=size, min_periods = 1).mean()
            self.transformed['close_secant_{}'.format(size_2)] = self.transformed['close'].rolling(window=size_2, min_periods = 1).apply(secant, raw = True)
            
        