
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import random

from data_handling import read_data, process_timestamp, get_min_since, transforms


class build_datas():
    '''
    class to build out a dictionary with data for each ticker
    '''
    def __init__(self, tickers = []):
        '''
        input(s): list of tickers to retrieve data for 
        '''
        if tickers:
            self.tickers = tickers
        else:
            self.tickers = ['MSFT', 'AAPL', 'AMZN', 'FB', 'GOOG', 'TSLA']
            
        self.ticker_dic, self.mean_dic, self.var_dic = self.get_data()
            
    def get_data(self):
        '''
        retrieves and transforms data for each ticker
        '''
        all_ = {}
        all_mean = {}
        all_var = {}
        min_ = 1000000
        min_ticker = None
        
        sets = []
        
        for ticker in self.tickers:
            data = read_data(ticker)
            sets.append(set(list(data['Datetime'].values)))
            all_[ticker] = data
        
        common = sets[0]
        for set_ in sets[1:]:
            common = common & set_
        common = list(common)
        
        for ticker in self.tickers:
            df = all_[ticker]
            df = df[df['Datetime'].isin(common)]
            df = df.reset_index()
            
            min_since = get_min_since(df)
            df = process_timestamp(df, min_since)
            
            t = transforms(df)
            t.all_transforms()
            t.suplemental_factors()
            
            all_[ticker] = t.transformed
            all_mean[ticker] = t.scaler.mean_
            all_var[ticker] = t.scaler.var_
            
        return all_, all_mean, all_var
