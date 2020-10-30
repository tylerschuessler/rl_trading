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
    else:
        print('not developed yet')
    
    return layers
    

###actor build###       
class actor(nn.Module):
    '''
    The actor class
    '''
    def __init__(self, state_dim, action_dim, type_ = 'linear', drop_out = False, discrete= True, std=0.0):
        '''
        input(s): dimensions of state and action, and the max action
        '''
        super(actor, self).__init__()
        
        ###for a discrete action space
        self.discrete = discrete
        
        self.log_std = nn.Parameter(torch.ones(1, action_dim) * std)
        
        self.set_params()
        
        self.set_up_layers(state_dim, action_dim, type_, drop_out)
        
        ###apply the intial weights if needed
        #self.apply(init_weights)
    
    def set_up_layers(self, state_dim, action_dim, type_, drop_out):
        '''
        make structure for the actor network
        '''

        layers = base_net(state_dim, self.dim_1, self.dim_2, type_, drop_out)
        layers.append(nn.Linear(self.dim_2, action_dim))
        
        self.model = nn.Sequential(*layers)
    
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
            
    def set_params(self, dim_1 = 64, dim_2 = 32):
        '''
        set the parameters for the number of neurons
        input(s): dims for neuron layers
        '''
        self.dim_1 = dim_1
        self.dim_2 = dim_2

###critic build###
class critic(nn.Module):
    '''
    critic class
    '''
    def __init__(self, state_dim, type_ = 'linear', drop_out = False):
        '''
        input(s): dimensions of state and action
        '''
        super(critic, self).__init__()
        
        self.set_params()
        self.set_up_layers(state_dim, type_, drop_out)
        
        self.apply(init_weights)
    
    def set_up_layers(self, state_dim, type_ , drop_out):
        '''
        make structure for the actor network
        '''
        layers = base_net(state_dim, self.dim_1, self.dim_2, type_, drop_out)
        layers.append(nn.Linear(self.dim_2, 1))
        self.model = nn.Sequential(*layers)
    
    def forward(self, state):
        '''
        forward propagate the network
        input(s): state and action
        output(s): result of the critic network
        '''
        return self.model(state)
        
    def set_params(self, dim_1 = 64, dim_2 = 32):
        '''
        set the parameters for the number of neurons
        input(s): dims for neuron layers
        '''
        self.dim_1 = dim_1
        self.dim_2 = dim_2