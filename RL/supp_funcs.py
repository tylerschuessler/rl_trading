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

###import network build classes###
import network_builds

class training_functionality():
    '''
    wrapper class for implementing training supplementary functions
    '''
    def __init__(self, params):
        self.mini_batch_size = params['mini_batch_size']
        self.ppo_epochs      = params['ppo_epochs']
        self.seq_len         = params['seq_len']


    def compute_gae(self, next_value, rewards, values, gamma=0.99, tau=0.95):
        '''
        General Advantage Estimation function from paper: used to estimate the advatage the actor gains 
            from a particular action
        input(s):
        '''
        values = values + [next_value]
        gae = 0
        returns = []
        for step in reversed(range(len(rewards))):
            delta = rewards[step] + gamma * values[step + 1]  - values[step]
            gae = delta + gamma * tau * gae
            returns.insert(0, gae + values[step])
        return returns


    def ppo_iter(self, states, actions, log_probs, returns, advantage):
        '''
        iterate through a set of the data using minibatches
        input(s):
        '''

        batch_size = states.size(0)
        for _ in range(batch_size // self.mini_batch_size):
            rand_ids = np.random.randint(0, batch_size, self.mini_batch_size)
            #rand_ids = np.random.randint(0, batch_size - mini_batch_size)
            yield states[rand_ids, :], actions[rand_ids, :], log_probs[rand_ids, :], returns[rand_ids, :], advantage[rand_ids, :]
            #yield states[rand_ids:rand_ids+mini_batch_size:, :], actions[rand_ids:rand_ids+mini_batch_size, :], log_probs[rand_ids:rand_ids+mini_batch_size, :], returns[rand_ids:rand_ids+mini_batch_size, :], advantage[rand_ids:rand_ids+mini_batch_size, :]

    def ppo_update(self, states, actions, log_probs, returns, advantages, actor_, critic_, optimizer_actor, optimizer_critic, clip_param=0.2):
        '''
        training loop: iterate through batches of states, actions, returns and advantages 
        '''
        for _ in range(self.ppo_epochs):
            for state, action, old_log_probs, return_, advantage in self.ppo_iter(states, actions, log_probs, returns, advantages):
                #with torch.autograd.set_detect_anomaly(True):
    
                ###obtain outputs from neural nets
                dist = actor_(state)
                value = critic_(state)

                ###calculate entropy and log_prob
                entropy = dist.entropy().mean()
                new_log_probs = dist.log_prob(action)

                # print(action.size())
                # print(new_log_probs.size())
                # print(advantage.size())

                ###calculate ppo ratio from paper
                ratio = (new_log_probs - old_log_probs).exp()
                surr1 = ratio * advantage
                surr2 = torch.clamp(ratio, 1.0 - clip_param, 1.0 + clip_param) * advantage

                ###actor and critic losses
                actor_loss  = - torch.min(surr1, surr2).mean() - 0.001 * entropy
                mse_loss = nn.MSELoss()
                critic_loss = mse_loss(value, return_) - 0.001 * entropy
        
                ###update
                optimizer_actor.zero_grad()
                optimizer_critic.zero_grad()
                

                actor_loss.backward(retain_graph=True)
                critic_loss.backward(retain_graph=True)
                
                optimizer_actor.step()
                optimizer_critic.step()



class training_functionality_LSTM(training_functionality):
    '''
    class for the LSTM training functionality 
    needed to overwrite a couple of functions
    '''
    def __init__(self, params):
        super(training_functionality_LSTM, self).__init__(params)

    def f(self, temp_tensor):
        ###depreciated
        '''
        helper function for building out the sequences 
        '''
        temp = [temp_tensor[j:j+self.seq_len, :].clone().unsqueeze(1) for j in range(self.mini_batch_size)]
        # for i in temp:
        #     print(i.size())
        return torch.cat(temp, 1)

    def ppo_iter(self, states, actions, log_probs, returns, advantage):
        '''
        iterate through a set of the data using minibatches
        input(s):
        '''
        #with torch.autograd.set_detect_anomaly(True):
        batch_size = states.size(0) - self.mini_batch_size
        for _ in range(batch_size // self.mini_batch_size):
            #rand_ids = np.random.randint(0, batch_size, mini_batch_size)
            #lump = self.mini_batch_size + self.seq_len 
            rand_ids = np.random.randint(self.mini_batch_size+self.seq_len, batch_size)
            #print(rand_ids)
            inds = np.array([[(rand_ids - 1) - i - j for i in reversed(range(self.seq_len))] for j in reversed(range(self.mini_batch_size))])
            inds= torch.tensor(inds.T)
            inds = inds.reshape(inds.size(0)*inds.size(1))
            temp = states.unsqueeze(1)
            #print(inds)
            x = torch.index_select(temp, 0, inds).view(self.seq_len, self.mini_batch_size, -1)
            #yield states[rand_ids, :], actions[rand_ids, :], log_probs[rand_ids, :], returns[rand_ids, :], advantage[rand_ids, :]
            yield x, actions[rand_ids- self.mini_batch_size:rand_ids, :], log_probs[rand_ids- self.mini_batch_size:rand_ids, :], returns[rand_ids- self.mini_batch_size:rand_ids, :], advantage[rand_ids- self.mini_batch_size:rand_ids, :]
    

