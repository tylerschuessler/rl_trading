import copy
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
from torch.distributions import Categorical
import numpy as np

###helper functions###
def init_weights(m):
    '''
    weight intialization function 
    '''
    if isinstance(m, nn.Linear):
        nn.init.normal_(m.weight, mean=0., std=0.1)
        nn.init.constant_(m.bias, 0.1)

def base_net(state_dim, dim_1, dim_2, type_, drop_out):
    '''
    function to define base_network
    '''
    if type_ == 'linear':
        layers = [
            nn.Linear(state_dim, dim_1),
            nn.Tanh(),
            nn.Linear(dim_1, dim_2),
            nn.Tanh()
        ]
        if drop_out:
            layers.intsert(2, nn.Dropout(p = 0.3))
            layers.append(nn.Dropout(p = 0.3))
        
    
    return layers

default_params = {
    
    'type_' : 'linear',
    'dropout' : False,
    'discrete' : False,
    'std' : 0.0,
    'hidden_dim' : 20,
    'dim_1' : 64,
    'dim_2' : 32,
    'mini_batch_size' : 90
}

###actor build###       
class actor(nn.Module):
    '''
    The actor class
    '''
    def __init__(self, state_dim, action_dim, params = default_params):
        '''
        input(s): dimensions of state and action, and the max action
        '''
        super(actor, self).__init__()
        
        ###for a discrete action space
        self.discrete = params['discrete']
        self.type_ = params['type_']
        self.dropout = params['dropout']
        self.std = params['std']
        self.hidden_dim = params['hidden_dim']
        self.dim_1 = params['dim_1']
        self.dim_2 = params['dim_2']
        self.mini_batch_size = params['mini_batch_size']
        
        self.log_std = nn.Parameter(torch.ones(1, action_dim) * self.std)
        
        
        self.set_up_layers(state_dim, action_dim)
        
        ###apply the intial weights if needed
        #self.apply(init_weights)
    
    def set_up_layers(self, state_dim, action_dim):
        '''
        make structure for the actor network
        '''
        if self.type_ == 'linear':

            layers = base_net(state_dim, self.dim_1, self.dim_2, self.type_, self.dropout)
            layers.append(nn.Linear(self.dim_2, action_dim))
        
            self.model = nn.Sequential(*layers)
        
        elif self.type_ == 'LSTM':
            self.lstm = nn.LSTM(state_dim, hidden_size = self.hidden_dim)
            self.lin_cap = nn.Linear(self.hidden_dim, action_dim)
            

    
    def forward(self, state):
        '''
        forward propagate the network
        input(s): state
        output(s): result of the actor network
        '''
        out = self.model(state)
        if self.discrete:
            probs = F.softmax(out, dim = 1)
            return Categorical(probs)
        else: 
            mu  = self.model(state)
            std   = self.log_std.exp().expand_as(mu)
            dist  = Normal(mu, std)
            return dist
            

###critic build###
class critic(nn.Module):
    '''
    critic class
    '''
    def __init__(self, state_dim, params = default_params):
        '''
        input(s): dimensions of state and action
        '''
        super(critic, self).__init__()
        self.type_ = params['type_']
        self.dropout = params['dropout']
        self.hidden_dim = params['hidden_dim']
        self.dim_1 = params['dim_1']
        self.dim_2 = params['dim_2']
        self.mini_batch_size = params['mini_batch_size']
        
        self.set_up_layers(state_dim)
        
    
    def set_up_layers(self, state_dim):
        '''
        make structure for the actor network
        '''
        if self.type_ == 'linear':
            layers = base_net(state_dim, self.dim_1, self.dim_2, self.type_, self.dropout)
            layers.append(nn.Linear(self.dim_2, 1))
            self.model = nn.Sequential(*layers)

        elif self.type_ == 'LSTM':
            self.lstm = nn.LSTM(state_dim, hidden_size = self.hidden_dim)
            self.lin_cap = nn.Linear(self.hidden_dim, 1)
        
    
    def forward(self, state):
        '''
        forward propagate the network
        input(s): state and action
        output(s): result of the critic network
        '''
        return self.model(state)




class actor_LSTM(actor):
    '''
    LSTM actor class
    need to overwrite the forward function
    '''
    def __init__(self, state_dim, action_dim, params = default_params):
        
        super(actor_LSTM, self).__init__(state_dim, action_dim, params = params)

        self.reset_hidden_cell(self.mini_batch_size)

    def reset_hidden_cell(self, mini_batch_size):
        '''
        resets the hidden cell state, or intializes it 
        '''
        self.hidden_cell = (torch.zeros(1,mini_batch_size,self.hidden_dim), torch.zeros(1,mini_batch_size,self.hidden_dim))

    def forward(self, x):
        
        lstm_out, self.hidden_cell = self.lstm(x, self.hidden_cell)
        mu = self.lin_cap(lstm_out[-1])

        if self.discrete:
            probs = F.softmax(mu, dim = 1)
            return Categorical(probs)
        else: 
            std   = self.log_std.exp().expand_as(mu)
            dist  = Normal(mu, std)
            return dist

class critic_LSTM(critic):
    '''
    LSTM critic class
    need to overwrite the forward function
    '''
    def __init__(self, state_dim, params = default_params):
        
        super(critic_LSTM, self).__init__(state_dim, params = params)
        
        self.reset_hidden_cell(self.mini_batch_size)
    
    def reset_hidden_cell(self, mini_batch_size):
        '''
        resets the hidden cell state, or intializes it 
        '''
        self.hidden_cell = (torch.zeros(1,mini_batch_size,self.hidden_dim), torch.zeros(1,mini_batch_size, self.hidden_dim))
    
    def forward(self, x):
        #.view(len(x) ,1, -1)
        lstm_out, self.hidden_cell = self.lstm(x, self.hidden_cell)
        #lstm_out[-1].clone()
        return self.lin_cap(lstm_out[-1])

                      
        