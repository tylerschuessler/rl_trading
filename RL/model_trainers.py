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

###import my files needed###
import network_builds
from supp_funcs import compute_gae, ppo_iter, ppo_update
import data_builder
import environments

###used for single actor###
class trainer_tester():
    '''
    class for implementing training/testing functionality
    '''
    def __init__(self, envs, discrete = True):
        ###make the environment
        self.envs = envs
        self.discrete = discrete
        
        ### intialize some hyper parameters
        self.state_dim = envs.state_dim + 5
        self.action_dim = envs.action_dim

        ###hyper params:
        self.lr               = 3e-4
        self.num_steps        = 300
        self.mini_batch_size  = 30
        self.ppo_epochs       = 6
        
        ###intialize actors and critics
        self.actor_ = network_builds.actor(self.state_dim, self.action_dim, discrete=self.discrete)#.to(device)
        self.optimizer_actor = torch.optim.Adam(self.actor_.parameters(), lr = self.lr)

        self.critic_ = network_builds.critic(self.state_dim)
        self.optimizer_critic = torch.optim.Adam(self.critic_.parameters(), lr = self.lr)
       

    def add_states(self, state, idx, entropy):
        '''
        add states to the batch of state, action, rewards ... etc pairs
        
        input(s): state-tensor with current state, current index in the data set, entropy-constant representing entropy
        '''
        state = torch.FloatTensor(state)#.to(device)
        
        dist = self.actor_(state)
        value = self.critic_(state)

        action = dist.sample()
        if self.discrete:
            temp = action.numpy().item()
        else:
            temp = F.softmax(action, dim = 1)
            temp = temp.numpy()[0]

        reward, next_state = self.envs.step(temp, state.numpy(), idx)

        log_prob = dist.log_prob(action)

        entropy += dist.entropy().mean()
        
        return log_prob.unsqueeze(1), value, torch.FloatTensor(np.array([reward])).unsqueeze(1), state, action.unsqueeze(1), next_state, entropy 

    
    def train(self, multiple_runs):
        torch.autograd.set_detect_anomaly(True)
        self.rewards_all = []
        for i in range(multiple_runs):
            if i == 0:
                state, j = self.envs.initial_state()
            else:
                state, j = self.envs.initial_state(random = True)
            max_idx = self.envs.data_df.shape[0] - j
            idx  = j + 1
            
            while idx < max_idx:
                log_probs = []
                values    = []
                states    = []
                actions   = []
                rewards   = []
                entropy = 0
                
                for _ in range(self.num_steps):

                    log_prob, value, reward, state, action, next_state, entropy = self.add_states(state, idx, entropy)
                    
                    log_probs.append(log_prob)
                    values.append(value)
                    rewards.append(reward)#.to(device))
                    states.append(state)
                    actions.append(action)

                    state = next_state
                    self.rewards_all.append(reward)
                    idx += 1
                    if idx >= max_idx:
                        break
                    
                    
                        
                next_state = torch.FloatTensor(next_state)#.to(device)
                next_value = self.critic_(next_state)
                returns = compute_gae(next_value, rewards, values)
                
                
                returns   = torch.cat(returns).detach()
                log_probs = torch.cat(log_probs).detach()
                values    = torch.cat(values).detach()
                states    = torch.cat(states)
                actions   = torch.cat(actions)
                advantage = returns - values
                
                
                ppo_update(self.ppo_epochs, self.mini_batch_size, states, actions, log_probs, returns, advantage, self.actor_, self.critic_, self.optimizer_actor, self.optimizer_critic)
                
    def test(self, envs):
        '''
        Implementation to test some of the data without updating any parameters of the networks
        '''
        self.envs = envs
        self.rewards_all_test = []
        state, j = self.envs.initial_state()
        max_idx = self.envs.data_df.shape[0] - j
        idx  = j + 1
        rewards_all = []
        entropy = 0
        while idx < max_idx:
            log_prob, value, reward, state, action, next_state, entropy = self.add_states(state, idx, entropy)
            state = next_state
            self.rewards_all_test.append(reward)
            idx += 1
            if idx >= max_idx:
                break


class trainer_tester_multiple():
    '''
    class for implementing training/testing functionality
    '''
    def __init__(self, db, discrete = True, key = 'MSFT'):
        ###make the environment
        self.ticker_dic = db.ticker_dic
        self.mean_dic = db.mean_dic
        self.var_dic = db.var_dic
        self.discrete = discrete

        ###hyper params:
        self.lr               = 3e-4
        self.num_steps        = 300
        self.mini_batch_size  = 30
        self.ppo_epochs       = 3
        
        ###intialize actors and critics
        self.envs = environments.environment(db.ticker_dic[key], db.mean_dic[key], db.var_dic[key], action_dim=2, discrete=False)
        
        ### intialize some hyper parameters
        self.state_dim = self.envs.state_dim + 5
        self.action_dim = self.envs.action_dim
        
        
        self.actor_ = network_builds.actor(self.state_dim, self.action_dim, discrete=self.discrete)#.to(device)
        self.optimizer_actor = torch.optim.Adam(self.actor_.parameters(), lr = self.lr)

        self.critic_ = network_builds.critic(self.state_dim)
        self.optimizer_critic = torch.optim.Adam(self.critic_.parameters(), lr = self.lr)
        
        ####set up lsit of actors, optimizers, envs
        self.actors_, self.actor_optimizers, self.envs_ = self.intialize_actors(db.ticker_dic, db.mean_dic, db.var_dic)
       
    def intialize_actors(self, ticker_dic, mean_dic, var_dic):
        '''
        intialize 1 or more actors 
        '''
        params_ = {'MSFT':3, 'AAPL' : 1, 'AMZN': 1, 'FB':1, 'GOOG':1, 'TSLA':1}
        
        actors_ = []
        actor_optimizers = []
        envs_ = []
        
        for key, val in params_.items():
            for _ in range(val):
                a = network_builds.actor(self.state_dim, self.action_dim, discrete=self.discrete)
                a.load_state_dict(self.actor_.state_dict())
                actors_.append(a)
                actor_optimizers.append(torch.optim.Adam(a.parameters(), lr = self.lr))
                envs_.append(environments.environment(ticker_dic[key], mean_dic[key], var_dic[key], action_dim=2, discrete=False))
        
        a = actors_
        b = actor_optimizers
        c = envs_
        
        d = list(zip(a, b, c))

        random.shuffle(d)
        actors_, actor_optimizers, envs_ = zip(*d)  
        
        self.n_workers = len(actors_)
        
        return actors_, actor_optimizers, envs_
            
    def add_states(self, actor_, envs, state, idx, entropy):
        '''
        add states to the batch of state, action, rewards ... etc pairs
        
        input(s): state-tensor with current state, current index in the data set, entropy-constant representing entropy
        '''
        state = torch.FloatTensor(state).to(device)
        
        dist = actor_(state)
        value = self.critic_(state)

        action = dist.sample()
        if self.discrete:
            temp = action.numpy().item()
        else:
            temp = F.softmax(action, dim = 1)
            temp = temp.numpy()[0]

        reward, next_state = envs.step(temp, state.numpy(), idx)

        log_prob = dist.log_prob(action)

        entropy += dist.entropy().mean()
        
        return log_prob.unsqueeze(1), value, torch.FloatTensor(np.array([reward])).unsqueeze(1), state, action.unsqueeze(1), next_state, entropy 

    
    def train(self):
        torch.autograd.set_detect_anomaly(True)
        self.rewards_all = []
        
        states_temp = [e.initial_state()[0] for e in self.envs_]
        max_idx = self.envs.data_df.shape[0] 
        idx  =  1
        
        counter = 1
        per = int(0.1*max_idx)
        
        state_main, _ = self.envs.initial_state()
        entropy_main = 0
        
        update_counter = 0
        
        while idx < max_idx:
            i = 0
            for actor_, optimizer, envs in zip(self.actors_, self.actor_optimizers, self.envs_):
                log_probs = []
                values    = []
                states    = []
                actions   = []
                rewards   = []
                entropy = 0
                
                state = states_temp[i]
                temp_idx = idx
                
                left_iterate = min(self.num_steps, max_idx - idx)
                
                
                for _ in range(left_iterate):
                    log_prob, value, reward, state, action, next_state, entropy = self.add_states(actor_, envs, state, temp_idx, entropy)

                    log_probs.append(log_prob)
                    values.append(value)
                    rewards.append(reward)#.to(device))
                    states.append(state)
                    actions.append(action)

                    state = next_state
                    
                    temp_idx += 1
                    

                next_state = torch.FloatTensor(next_state)#.to(device)
                next_value = self.critic_(next_state)
                returns = compute_gae(next_value, rewards, values)


                returns   = torch.cat(returns).detach()
                log_probs = torch.cat(log_probs).detach()
                values    = torch.cat(values).detach()
                states    = torch.cat(states)
                actions   = torch.cat(actions)
                advantage = returns - values


                ppo_update(self.ppo_epochs, self.mini_batch_size, states, actions, log_probs, returns, advantage, actor_, self.critic_, optimizer, self.optimizer_critic)
                
                states_temp[i] = state
                i += 1
                
                self.gradient_push(actor_)
            
            left_iterate = min(self.num_steps, max_idx - idx)
            temp_idx = idx
            
            with torch.no_grad():
                for _ in range(left_iterate): 
                    _, _, reward, state_main, _, next_state_main, entropy_main = self.add_states(self.actor_, self.envs, state_main, temp_idx, entropy_main)
                    state_main = next_state_main
                    self.rewards_all.append(reward)
                    
                    temp_idx += 1
                      
            idx += left_iterate
            update_counter += 1
                
            if update_counter%1 == 0:
                print('pushing network update')
                self.push_network_update()
            
            if update_counter % 3 == 0:
                print('{} % done.'.format((idx/max_idx)*100))
                 
            
    def gradient_push(self, actor_):
        '''
        push gradients to the main actor_
        '''
        self.optimizer_actor.zero_grad()
        for params, main_params in zip(actor_.parameters(), self.actor_.parameters()):
            main_params.grad = params.grad.clone()
        self.optimizer_actor.step()
        
    def push_network_update(self):
        '''
        after n iterations push the main network parameters to all exploring actors
        '''
        for actor_ in self.actors_:
            actor_.load_state_dict(self.actor_.state_dict())
    
    def baseline(self, envs):
        '''
        basline function for if trading were conducted with a random network
        '''
        state, _ = envs.initial_state()
        max_idx = self.envs.data_df.shape[0] 
        idx = 1
        self.rewards_all_test = []
        entropy = 0
        while idx < max_idx:
            actor_ = network_builds.actor(self.state_dim, self.action_dim, discrete=self.discrete)
            log_prob, value, reward, state, action, next_state, entropy = self.add_states(actor_, envs, state, idx, entropy)
            state = next_state
            self.rewards_all_test.append(reward)
            idx += 1
            
            if idx % 100 == 0:
                print('{}% done.'.format((idx/max_idx)*100))
                
            if idx >= max_idx:
                break
                    
    def test(self, envs):
        '''
        Implementation to test some of the data without updating any parameters of the networks
        '''
        self.envs = envs
        self.rewards_all_test = []
        state, j = envs.initial_state()
        max_idx = envs.data_df.shape[0] - j
        idx  = j + 1
        rewards_all = []
        entropy = 0
        while idx < max_idx:
            log_prob, value, reward, state, action, next_state, entropy = self.add_states(state, idx, entropy)
            state = next_state
            self.rewards_all_test.append(reward)
            idx += 1
            if idx >= max_idx:
                break
            
        
        