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
import supp_funcs 
import data_builder
import environments

default_parameters_single_actor = {
    'discrete' : False,
    'lr' : 3e-4,
    'num_steps' : 300,
    'mini_batch_size' : 30,
    'ppo_epochs' : 6,
    'key' : 'MSFT',
    'action_dim' : 2,
    'seq_len' : 1,
    'params_actors' : {'MSFT':3, 'AAPL' : 1, 'AMZN': 1, 'FB':1, 'GOOG':1, 'TSLA':1},
    'type_' : 'linear'
    
}

###used for single actor###
class trainer_tester():
    '''
    class for implementing training/testing functionality
    '''
    def __init__(self, envs, params = default_parameters_single_actor):
        ###make the environment
        self.envs = envs
        self.discrete = params['discrete']
        
        ### intialize some hyper parameters
        self.state_dim = envs.state_dim + 4 + self.envs.running_trades_bool
        self.action_dim = envs.action_dim

        ###hyper params:
        self.lr               = params['lr']
        self.num_steps        = params['num_steps']
        self.mini_batch_size  = params['mini_batch_size']
        self.ppo_epochs       = params['ppo_epochs']
        self.seq_len          = envs.seq_len
        self.type_            = params['type_']


        if self.type_ == 'linear':
            self.sf = supp_funcs.training_functionality(params)
            ###intialize actors and critics
            self.actor_ = network_builds.actor(self.state_dim, self.action_dim, params = params)
            self.critic_ = network_builds.critic(self.state_dim, params = params)

        elif self.type_ == 'LSTM':
            self.sf = supp_funcs.training_functionality_LSTM(params)
            ###intialize actors and critics
            self.actor_ = network_builds.actor_LSTM(self.state_dim, self.action_dim, params = params)
            self.critic_ = network_builds.critic_LSTM(self.state_dim, params = params)
            
        self.optimizer_actor = torch.optim.Adam(self.actor_.parameters(), lr = self.lr)
        self.optimizer_critic = torch.optim.Adam(self.critic_.parameters(), lr = self.lr)
       
    def add_states(self, actor_, envs, state, idx, entropy):
        '''
        add states to the batch of state, action, rewards ... etc pairs
        
        input(s): state-tensor with current state, current index in the data set, entropy-constant representing entropy
        '''
        state = torch.FloatTensor(state)#.to(device)
        
        if self.type_ == 'LSTM':
            ###create batch dimension
            state = state.unsqueeze(1)
        
        dist = actor_(state)
        value = self.critic_(state)

        action = dist.sample()

        if self.discrete:
            temp = action.numpy().item()
        else:
            temp = F.softmax(action, dim = 1)
            temp = temp.numpy()[0]

        if self.type_ == 'LSTM':
            state = torch.index_select(state, 0, torch.tensor([state.size(0) -1])).squeeze(1)
            
        reward, next_state = envs.step(temp, state.numpy(), idx)

        log_prob = dist.log_prob(action)

        entropy += dist.entropy().mean()
        
        return log_prob.unsqueeze(1), value, torch.FloatTensor(np.array([reward])).unsqueeze(1), state, action.unsqueeze(1), next_state, entropy 

    
    def train(self, multiple_runs):
        
        self.rewards_all = []
        update_counter = 0
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

                # self.actor_.eval()
                # self.critic_.eval()
                # with torch.no_grad():

                if self.type_ == 'LSTM':
                    saved_cell_actor = (self.actor_.hidden_cell[0].clone(),self.actor_.hidden_cell[1].clone())
                    saved_cell_critic = (self.critic_.hidden_cell[0].clone(),self.critic_.hidden_cell[1].clone())

                    self.actor_.hidden_cell = (self.actor_.hidden_cell[0][:, -1:, :].clone(), self.actor_.hidden_cell[1][:, -1:, :].clone())
                    self.critic_.hidden_cell = (self.critic_.hidden_cell[0][:, -1:, :].clone(), self.critic_.hidden_cell[1][:, -1:, :].clone())

                    # self.actor_.reset_hidden_cell(1)
                    # self.critic_.reset_hidden_cell(1)

                    for k in range(self.seq_len-1):
                        states.append(torch.FloatTensor(state[k:k+1,:]))

                for _ in range(self.num_steps):

                    if self.type_ == 'LSTM':
                        temp_state = state[1:, :]

                    log_prob, value, reward, state, action, next_state, entropy = self.add_states(self.actor_, self.envs, state, idx, entropy)
                    
                    log_probs.append(log_prob)
                    values.append(value)
                    rewards.append(reward)#.to(device))
                    states.append(state)
                    actions.append(action)

                    state = next_state


                    if self.type_ == 'LSTM':
                        state = np.concatenate((temp_state, state), axis = 0)

                    self.rewards_all.append(reward)

                    idx += 1
                    if idx >= max_idx:
                        break

                next_state = torch.FloatTensor(next_state)



                if self.type_ == 'LSTM':     
                    next_state = torch.FloatTensor(np.concatenate((state[1:, :], next_state), axis = 0)).unsqueeze(1)#.to(device)

                next_value = self.critic_(next_state)
                next_value = next_value.item()

                returns = self.sf.compute_gae(next_value, rewards, values)
                
                
                returns   = torch.cat(returns).detach()
                log_probs = torch.cat(log_probs).detach()
                values    = torch.cat(values).detach()
                states    = torch.cat(states)
                actions   = torch.cat(actions)
                advantage = returns - values

                print(states.saved_tensors)

                if self.type_ == 'LSTM':
                    # self.actor_.reset_hidden_cell(self.mini_batch_size)
                    # self.critic_.reset_hidden_cell(self.mini_batch_size)
                    self.actor_.hidden_cell = saved_cell_actor 
                    self.critic_.hidden_cell = saved_cell_critic

                
                # self.actor_.train()
                # self.critic_.train()

                update_counter += 1
                if update_counter % 3 == 0:
                    print('{} % done.'.format((idx/max_idx)*100))       
                    
                self.sf.ppo_update(states, actions, log_probs, returns, advantage, self.actor_, self.critic_, self.optimizer_actor, self.optimizer_critic)
                
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
            log_prob, value, reward, state, action, next_state, entropy = self.add_states(self.actor_, envs, state, idx, entropy)
            state = next_state
            self.rewards_all_test.append(reward)
            idx += 1
            if idx >= max_idx:
                break


class trainer_tester_multiple(trainer_tester):
    '''
    class for implementing training/testing functionality
    '''
    def __init__(self, db, params = default_parameters_single_actor):
        ###make the environment
        self.ticker_dic = db.ticker_dic
        self.mean_dic = db.mean_dic
        self.var_dic = db.var_dic
        key = params['key']
        envs = environments.environment(self.ticker_dic[key], self.mean_dic[key], self.var_dic[key], params = params)
        ###initialize the parent class
        super().__init__(envs, params = params)
        
        ####set up lsit of actors, optimizers, envs
        self.actors_, self.actor_optimizers, self.envs_ = self.intialize_actors(self.ticker_dic, self.mean_dic, self.var_dic, params['params_actors'], params)
       
    def intialize_actors(self, ticker_dic, mean_dic, var_dic, params_actors, params):
        '''
        intialize 1 or more actors 
        '''
        actors_ = []
        actor_optimizers = []
        envs_ = []
        
        for key, val in params_actors.items():
            for _ in range(val):
                if self.type_ == 'linear':
                    a = network_builds.actor(self.state_dim, self.action_dim, params = params)
                elif self.type_ == 'LSTM':
                    a = network_builds.actor_LSTM(self.state_dim, self.action_dim, params = params)

                a.load_state_dict(self.actor_.state_dict())
                actors_.append(a)
                actor_optimizers.append(torch.optim.Adam(a.parameters(), lr = self.lr))
                envs_.append(environments.environment(ticker_dic[key], mean_dic[key], var_dic[key], params = params))
        
        a = actors_
        b = actor_optimizers
        c = envs_
        
        d = list(zip(a, b, c))

        random.shuffle(d)
        actors_, actor_optimizers, envs_ = zip(*d)  
        
        self.n_workers = len(actors_)
        
        return actors_, actor_optimizers, envs_
            
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
                returns = self.sf.compute_gae(next_value, rewards, values)


                returns   = torch.cat(returns).detach()
                log_probs = torch.cat(log_probs).detach()
                values    = torch.cat(values).detach()
                states    = torch.cat(states)
                actions   = torch.cat(actions)
                advantage = returns - values

                

                self.sf.ppo_update(states, actions, log_probs, returns, advantage, actor_, self.critic_, optimizer, self.optimizer_critic)
                
                states_temp[i] = state
                i += 1
                
                self.gradient_push(actor_)
            
            left_iterate = min(self.num_steps, max_idx - idx)
            temp_idx = idx
            
            self.rewards_all = self.rewards_all + self.eval(self.actor_, self.envs, state_main, temp_idx, entropy_main, left_iterate)
                      
            idx += left_iterate
            update_counter += 1
                
            if update_counter%1 == 0:
                print('pushing network update')
                self.push_network_update()
            
            if update_counter % 3 == 0:
                print('{}% done.'.format((idx/max_idx)*100))
                 
    def eval(self, actor_, envs, state_main, temp_idx, entropy_main, left_iterate):
        '''
        function to evaluate the rewards of an actor 
        '''
        returns = []
        with torch.no_grad():
            for _ in range(left_iterate): 
                _, _, reward, state_main, _, next_state_main, entropy_main = self.add_states(actor_, envs, state_main, temp_idx, entropy_main)
                state_main = next_state_main
                returns.append(reward)
                
                temp_idx += 1

        return returns

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
                    


class trainer_tester_multiple_envolve(trainer_tester_multiple):
    '''
    class for implementing functionality to test some evolutionary techniques
    '''
    def __init__(self, db, params = default_parameters_single_actor):
        super(trainer_tester_multiple_envolve, self).__init__(db, params = params)
        
        self.prob_of_mutation = params['prob_of_mutation']
        self.percent_mutate = params['percent_mutate']
        self.breed_method = params['breed_method']
        self.sigma = params['sigma']

        self.envs_tester = environments.environment(self.ticker_dic[params['key']], self.mean_dic[params['key']], self.var_dic[params['key']], params = params)

    def push_network_update(self):
        '''
        after n iterations breed the networks and mutate 
        '''
        rewards_sum = []
        seed_ = np.random.randint(0, int(self.envs_tester.data_length) - 501)
        for actor_ in self.actors_:
            state_main, i = self.envs_tester.initial_state(seed = seed_)
            rewards_sum.append(np.mean(self.eval(actor_, self.envs_tester, state_main, i, 0, 500)))

        sorted_inds = np.argsort(rewards_sum)
        ###get the bottom/top half of the actors
        b_half = sorted_inds[:len(rewards_sum)]
        t_half = sorted_inds[len(rewards_sum):]

        for top, bot in zip(t_half, b_half):
            self.breed(self.actors_[top], self.actors_[bot])

        for actor_ in self.actors_:
            if np.random.binomial(size = 1, n=1, p = self.prob_of_mutation):
                self.mutate(actor_)

        # if shit out of whack:
        #     for actor_ in self.actors_:

        #         actor_.load_state_dict(self.actor_.state_dict())
    def mutate(self, model):
        '''
        function for mutating some of the paramters 
        '''
        self.percent_mutate = 0.05
        state_dict = model.state_dict().copy()

        for label, values in model.state_dict().items():
            norm_ = np.random.normal(size=values.size())
            binom = np.random.binomial(size=values.size(), n=1, p = self.percent_mutate)
            state_dict[label] = values + self.sigma*torch.FloatTensor(np.multiply(binom, norm_))
        
        model.load_state_dict(state_dict)

    def breed(self, model_t, model_b):
        '''
        breed the main network with the top model and replace the bottom model
        '''
        state_dict = model_t.state_dict().copy()
        main_state_dic = self.actor_.state_dict()
        for label, values in model_t.state_dict().items():
            if self.breed_method == 'avg':
                state_dict[label] = 0.5*(values + main_state_dic[label])
            elif self.breed_method == 'crossover':
                binom = np.random.binomial(size=values.size(), n=1, p = 0.5)
                state_dict[label] = np.multiply(values, binom) + np.multiply(main_state_dic[label], binom^1)

        model_b.load_state_dict(state_dict)

    def evaluate(self, state_main, temp_idx, entropy_main, left_iterate):
        rewards = []

        with torch.no_grad():
            for _ in range(left_iterate): 
                _, _, reward, state_main, _, next_state_main, entropy_main = self.add_states(self.actor_, self.envs, state_main, temp_idx, entropy_main)
                state_main = next_state_main
                rewards.append(reward)
                temp_idx += 1

        return np.sum(rewards)




class trainer_tester_LSTM(trainer_tester):
    '''
    class for overwriting some of the funcctionality for LSTMs
    '''
    def __init__(self, envs, params = default_parameters_single_actor):
        super(trainer_tester_LSTM, self).__init__(envs, params = params)

    def add_states(self, actor_, envs, state, idx, entropy):
        '''
        add states to the batch of state, action, rewards ... etc pairs
        
        input(s): state-tensor with current state, current index in the data set, entropy-constant representing entropy
        '''
        state = torch.FloatTensor(state)#.to(device)
        
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
            

                      
        
        