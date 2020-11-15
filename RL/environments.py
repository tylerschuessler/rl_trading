import copy
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
from torch.distributions import Categorical

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import random

###import my files needed####
import network_builds
import supp_funcs
import data_builder

class synthetic_environment():
    '''
    build some synthetic data
    '''
    def __init__(self):
        
        self.action_dim = 2
        self.state_dim = 4
        self.n_pts = 50000
        self.data_df = self.build()

        self.seq_len = 20
        self.sell_sigs = []
        self.buy_sigs = []
        self.monies_i = []
        self.monies_c = []

    def build(self):
        '''
        build out the data to look like sin
        output(s):
        '''
        x = 2*np.linspace(0, 100, self.n_pts)
        all_ = []
        def builder(x):
            y = np.sin(x) + np.sin(1.77*x) + np.sin(0.47*x) + np.cos(0.1*x) + 0.1*np.random.normal(size=self.n_pts)
            return y + 200
        for _ in range(self.state_dim):
            x = builder(x)
            mu = np.mean(x)
            std = np.std(x)
            all_.append((x - mu)/std)
        return pd.DataFrame(all_).T
        
    def initial_state(self, random = False):
        '''
        return an intial state, can be either random state or first state in the data
        
        input(s): random boolean if random intial state
        output(s): state vector
        '''
        if random:
            i = np.random.randint(0, int(self.data_length/2))
        else:
            i = 0
        self.initial_cash = 10000.0
        initial_investment = 0.0
        shares_outstanding = 0.0
        
        state = np.append(prelim_state, [[initial_investment, shares_outstanding, self.initial_cash] for _ in range(self.seq_len)], axis = 1)
        prelim_state = self.data_df.iloc[i:i+self.seq_len].values

        self.current_state = state
        return state
    
    def rewards(self, state):
        '''
        calculate the returns for given action 
        '''
        ###invested + penalty*cash - initial dollars
        penalty = 0.95
        return (state[:,-3].item() + penalty*(state[:,-1].item()) - self.initial_cash)
    
    def step(self, action, state, idx):
        '''
        take the next step based on the previous state and action choice
        
        input(s): current state, action to take
        output(s): next_state, reward
        '''
        #use low to update val of investment
        investment = state[:,0].item()*state[:,-2].item() 
        shares = state[:,-2].item()
        cash = state[:,-1].item()
        
        ###buy/sell action transition from invested to divested or vice versa
        if action == 0:
            #print('buy/sell')
            ###divest
            if cash == 0.0:

                shares = 0.0
                cash = investment
                investment = 0.0
                
                self.sell_sigs.append([idx, state[:,0].item()])
            ###invest    
            else:
                investment = cash
                shares = investment/state[:,1].item() #use high to calc buy
                cash = 0.0
                
                self.buy_sigs.append([idx, state[:,0].item()])
#         if action == 1:
#             cash = 0.999*cash
        
        self.monies_i.append(investment)
        self.monies_c.append(cash)
        

        prelim_state = self.data_df.iloc[idx].values.reshape(1, -1)
        state = np.append(prelim_state, [[investment, shares, cash]], axis = 1)
        
        return self.rewards(state), state

default_parameters = {
    'reward_type' : 'basic_trade_penalty',
    'discrete' : False,
    'action_dim' : 2,
    'initial_cash' : 10000.0,
    'running_trades_bool' : True,
    'delta_to_trade' : 0.25,
    'running_trades_freq' : 0.05, #1 every 20 min
    'c_log' : 0.0025,
    'c' : 0.35,
    'seq_len' : 1

}              
        
class environment():
    '''
    Class for defining the environment the agent interacts with
    '''
    def __init__(self, data_df, mean, var, params = default_parameters):
        '''
        input(s): data
        '''
        
        self.data_df = data_df
        self.mean = mean
        self.var = var
        
        
        cols = list(self.data_df.columns)
        self.l_i = cols.index('low')
        self.h_i = cols.index('high')
        self.c_i = cols.index('close')
        
        self.reward_type = params['reward_type']
        self.action_dim = params['action_dim']
        self.discrete = params['discrete']
        self.initial_cash = params['initial_cash']
        self.running_trades_bool = params['running_trades_bool']
        self.delta_to_trade = params['delta_to_trade']
        self.running_trades_freq = params['running_trades_freq']
        self.type_ = params['type_']
        self.seq_len = params['seq_len']

        self.c = params['c']
        self.c_log = params['c_log']

        self.state_dim = self.data_df.shape[1]
        self.data_length = self.data_df.shape[0]
        
        self.sell_sigs = []
        self.buy_sigs = []
        
        self.monies_i = [] 
        self.monies_c = []
        
        self.part_a = []
        self.part_b = []
        
        self.profits = []
        
        self.all_actions = []
        
        self.total_trades = []
        
        
    def initial_state(self, random = False, seed = 0):
        '''
        return an intial state, can be either random state or first state in the data
        
        input(s): random boolean if random intial state
        output(s): state vector
        '''
        if random:
            i = np.random.randint(0, int(0.75*self.data_length))
        else:
            i = seed


        initial_investment = 0.0
        shares_outstanding = 0.0
        
        self.prev_cash = self.initial_cash
        
        num_trades = 0.0
        prelim_state = self.data_df.iloc[i:i+self.seq_len].values

        running_trades = 3.0
        if self.running_trades_bool:
            state = np.append(prelim_state, [[running_trades, num_trades, initial_investment, shares_outstanding, self.initial_cash] for _ in range(self.seq_len)], axis = 1)
        else:
            state = np.append(prelim_state, [[num_trades, initial_investment, shares_outstanding, self.initial_cash] for _ in range(self.seq_len)], axis = 1)

        self.initial_shares = self.initial_cash/state[:,self.h_i][-1]

        self.state_dim = state.shape[1]

        i = i +self.seq_len - 1

        return state, i
    
    def rewards(self, state, prev_cash, prev_investment, prev_close):
        '''
        calculate the returns for given action 
        '''
        penalty = 1.0
        invested = state[:,-3].item()
        cash = state[:,-1].item()
        trades = state[:,-4].item()
        close = state[:,self.c_i].item()
        
        def log_clip(x):
            return np.log(np.clip(x, 1e-5, None))
        
        if self.reward_type == 'basic':
            r = invested + penalty*cash - self.initial_cash

        elif self.reward_type == 'basic_trade_penalty':
            part_a = invested + penalty*cash 
            part_b = trades*np.log((1 - self.c)/(1+self.c))
            
            r = part_a + part_b
            self.part_a.append(part_a)
            self.part_b.append(part_b)

        elif self.reward_type == 'excess_returns':
            #np.log(close/prev_close)
            sign = np.sign(close - prev_close)
            part_a = sign*((log_clip(invested)-log_clip(prev_investment)) - (log_clip(cash) - log_clip(prev_cash))) - (log_clip(close*self.initial_shares) - log_clip(prev_close*self.initial_shares)) 
            part_b = trades*np.log((1 - self.c_log)/(1+self.c_log))

            r = part_a + part_b
            self.part_a.append(part_a)
            self.part_b.append(part_b)
            
        elif self.reward_type == 'excess_returns_simple':
            part_a = log_clip(invested+cash)-log_clip(prev_investment+prev_cash) - (log_clip(close*self.initial_shares) -log_clip(prev_close*self.initial_shares)) 
            part_b = trades*np.log((1 - self.c_log)/(1+self.c_log))
            
            r = part_a + part_b
            
            self.part_a.append(part_a)
            self.part_b.append(part_b)
            
        elif self.reward_type == 'log_basic':
            part_a = log_clip(invested+cash) - log_clip(close*self.initial_shares)
            part_b = trades*np.log((1 - self.c_log)/(1+self.c_log))
            
            r = part_a + part_b
            
            self.part_a.append(part_a)
            self.part_b.append(part_b)

        elif self.reward_type == 'log_returns_modified':
            sign = np.sign(close - prev_close)
            part_a = sign*(log_clip(invested) - log_clip(cash)) - log_clip(close*self.initial_shares)
            part_b = trades*np.log((1 - self.c_log)/(1+self.c_log))

            r = part_a +part_b

            self.part_a.append(part_a)
            self.part_b.append(part_b)
            
        return r
    
    def step(self, action, state, idx):
        ''' 
        take the next step based on the previous state and action choice
        
        input(s): current state, action to take
        output(s): next_state, reward
        '''
       
        #open:0 high:1 low:2 close:3 volume:4
        low = (state[:,self.l_i].item() + self.mean[2])*np.sqrt(self.var[2])
        high = (state[:,self.h_i].item() + self.mean[1])*np.sqrt(self.var[1])
        close = (state[:,self.c_i].item() + self.mean[3])*np.sqrt(self.var[3])
        
        #use low to update val of investment
        prev_investment = low*state[:,-2].item() 
        shares = state[:,-2].item()
        prev_cash = state[:,-1].item()
        trades = state[:,-4].item()
        
        #if there should be running trades
        if self.running_trades_bool:
            running_trades = state[:,-5].item()
        else:
            running_trades = 1.0
        
        #discrete vs continious actions
        if self.discrete:
            self.all_actions.append(action)
            ###buy/sell action transition from invested to divested or vice versa
            cash, shares, investment, trades, running_trades = self.action_select(action, prev_cash, shares, prev_investment, trades, running_trades, high, close, idx)     
        else:
            self.all_actions.append([action[0], action[1]])
            cash, shares, investment, trades, running_trades = self.action_select_2(action, prev_cash, shares, prev_investment, trades, running_trades, high, close, low, idx) 

        self.monies_i.append(investment)
        self.monies_c.append(cash)
        self.total_trades.append(trades)
        
        prelim_state = self.data_df.iloc[idx].values.reshape(1, -1)
        #make sure running trades is required
        if self.running_trades_bool:
            state = np.append(prelim_state, [[running_trades, trades, investment, shares, cash]], axis = 1)
        else:
            state = np.append(prelim_state, [[trades, investment, shares, cash]], axis = 1)
        
        return self.rewards(state, prev_cash, prev_investment, close), state
    
    def action_select(self, action, cash, shares, investment, trades, running_trades, high, close, idx):
        '''
        helper to select the action
        
        input(s): action value, the cash value, the number of shares, investment amount
        '''
        ###2 action dimensions
        if self.action_dim == 2:
            if running_trades >= 1.0:
                if action == 0:
                    ###divest
                    if cash == 0.0:
                        shares = 0.0
                        cash += investment
                        investment = 0.0

                        self.sell_sigs.append([idx, close])
                    ###invest    
                    else:
                        investment += cash
                        shares = investment/high #use high to calc buy
                        cash = 0.0
                        
                        self.buy_sigs.append([idx, close])
                        
                    trades += 1
                    running_trades -= 1
             
        ###3 action dimensions
        elif self.action_dim == 3:
            if running_trades >= 1.0:
                ###divest
                if action == 0:
                    if investment > 0:
                        cash += investment
                        investment = 0.0
                        shares = 0.0

                        self.sell_sigs.append([idx, close])
                        trades += 1

                        running_trades -= 1
                
                elif action == 1:
                    if cash > 0:
                        investment += cash
                        shares = investment/high #use high to calc buy
                        cash = 0.0

                        self.buy_sigs.append([idx, close])
                        trades += 1

                        running_trades -= 1

        running_trades += self.running_trades_freq
        self.profits.append(cash+investment - self.prev_cash)
        self.prev_cash = cash+investment

        return cash, shares, investment, trades, running_trades
    
    def action_select_2(self, action, cash, shares, investment, trades, running_trades, high, close, low, idx):
        '''
        helper to select the action
        
        input(s): action value, the cash value, the number of shares, investment amount
        '''

        #portion invested
        total = cash + investment

        new_investment = action[0]*total
        new_cash = action[1]*total

        diff_invest = new_investment - investment
        
        if running_trades >= 1.0:
            if np.abs(diff_invest) >= self.delta_to_trade*investment:

                if diff_invest > 0:
                    shares += diff_invest/high
                    self.buy_sigs.append([idx, close])
                else:
                    shares += diff_invest/low
                    self.sell_sigs.append([idx, close])

                investment = new_investment
                cash = new_cash
                trades += 1
                running_trades -= 1

        running_trades += self.running_trades_freq
        self.profits.append(investment + cash - self.prev_cash)
        self.prev_cash = investment + cash

        return cash, shares, investment, trades, running_trades
        