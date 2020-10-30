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

import environments
import model_trainers


class plotter():
    def __init__(self, trainer, envs, envs_test):
        self.trainer = trainer
        self.envs = envs
        self.envs_test = envs_test
    
    def plot_rewards(self, test = False):
        if test:
            y = self.trainer.rewards_all_test
        else:
            y = self.trainer.rewards_all
            
        plt.plot(y)
        plt.title('rewards')
        plt.show()
    
    def plot_signals(self, test = False):
        if test:
            envs = self.envs_test
        else:
            envs = self.envs
            
        s = np.array(envs.sell_sigs)
        b = np.array(envs.buy_sigs)
        x = [i+1 for i in range(envs.data_df.shape[0])]
        y = (envs.data_df['close'] + envs.mean[3])*np.sqrt(envs.var[3])
        
        fig, ax = plt.subplots()
        ax.plot(x, y, label = 'close',color = 'black', alpha = 0.3)
        ax.scatter(s[:,0], s[:,1], label = 'sell signals', color = 'red', s = 35, alpha = 0.75, marker = '^')
        ax.scatter(b[:,0], b[:,1], label = 'buy signals', s =35, alpha = 0.35, marker = 'v')

        ax.legend()
        plt.show()
        
    def plot_monies(self, test = False):
        if test:
            envs = self.envs_test
        else:
            envs = self.envs
            
        c = np.array(envs.monies_c) 
        i = np.array(envs.monies_i)
        y = c+i
        
        plt.plot(y)
        plt.title('monies')
        plt.show()
        
    def plot_profits(self, test = False):
        if test:
            envs = self.envs_test
        else:
            envs = self.envs
        
        plt.plot(envs.profits)
        plt.title('per trade profits')
        plt.show()
    
    def plot_hist_trade_diff(self, test = False):
        if test:
            envs = self.envs_test
        else:
            envs = self.envs
        
        s = np.array(envs.sell_sigs)
        b = np.array(envs.buy_sigs)
        y = s[:,1].flatten() - b[:,1].flatten()[:s[:,1].shape[0]]
        
        plt.hist(y, bins = 30)
        plt.title('trade differential')
        plt.show()
        
    def plot_actions(self, test = False, discrete = False):
        if test:
            envs = self.envs_test
        else:
            envs = self.envs
        
        actions = np.array(envs.all_actions)
        
        plt.scatter([i+1 for i in range(actions.shape[0])], actions[:,1], label = 'cash', alpha = 0.5)
        plt.scatter([i+1 for i in range(actions.shape[0])], actions[:,0], label = 'investment', alpha = 0.5)
        plt.title('actions_taken')
        plt.show()
    
    def plot_trades(self, test = False):
        if test:
            envs = self.envs_test
        else:
            envs = self.envs
            
        plt.plot(envs.total_trades)
        plt.title('trades')
        plt.show()
        
    def plot_comparison(self, test = False):
        if test:
            envs = self.envs_test
        else:
            envs = self.envs
        
        fig, ax = plt.subplots()
        x = [i+1 for i in range(len(envs.part_a))]
        ax.scatter(x, envs.part_a, label = 'investment part')
        ax.scatter(x, envs.part_b, label = 'trade part')
        ax.legend()
        ax.set_title('portion of rewards')
        plt.show()