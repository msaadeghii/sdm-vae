#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Software dvae-speech
Copyright Inria
Year 2020
Contact : xiaoyu.bie@inria.fr
License agreement in LICENSE.txt

The code in this file is based part on the source code of:
- Simon Legaive (simon.leglaive@centralesupelec.fr)
- in “A recurrent variational autoencoder for speech enhancement” ICASSP, 2020
"""

from torch import nn
import torch
from collections import OrderedDict
import numpy as np


def odctdict(n,L):
    
    """
    %ODCTDICT Overcomplete DCT dictionary.
    %  D = ODCTDICT(N,L) returns the overcomplete DCT dictionary of size NxL
    %  for signals of length N.
    %
    %  See also ODCT2DICT, ODCT3DICT, ODCTNDICT.
    
    %  Ron Rubinstein
    %  Computer Science Department
    %  Technion, Haifa 32000 Israel
    %  ronrubin@cs
    %
    %  April 2009
    """

    D = np.zeros((n,L))
    D[:,0] = 1.0/np.sqrt(n) * np.ones((n))
    
    for k in range(1,L):
      v = np.cos(np.arange(n)*np.pi*(k)/L).T
      v = v-np.mean(v)
      D[:,k] = v/np.linalg.norm(v,2)

    return D
    
def build_VAE(cfg, device='cpu', vae_mode = 'VAE', exp_mode = 'test'):

    ### Load parameters
    # General
    x_dim = cfg.getint('Network', 'x_dim')
    z_dim = cfg.getint('Network','z_dim')
    activation = cfg.get('Network', 'activation')
    dropout_p = cfg.getfloat('Network', 'dropout_p')

    # Inference and generation
    dense_x_z = [] if cfg.get('Network', 'dense_x_z') == '' else [int(i) for i in cfg.get('Network', 'dense_x_z').split(',')]

    # Beta-vae
    beta = cfg.getfloat('Training', 'beta')

    # Build model
    if vae_mode == 'VAE':
        
        model = VAE(x_dim=x_dim, z_dim=z_dim,
                    dense_x_z=dense_x_z, activation=activation,
                    dropout_p=dropout_p, beta=beta, device=device).to(device)
    
    elif vae_mode == 'SDM-VAE':
        
        dictsize = cfg.getint('Network', 'dictsize')
        
        model = SDM_VAE(x_dim=x_dim, z_dim=z_dim,
                    dense_x_z=dense_x_z, activation=activation,
                    dropout_p=dropout_p, beta=beta, device=device, dictsize = dictsize).to(device)
        
    return model


#%% Standard VAE model
  
class VAE(nn.Module):

    '''
    VAE model class
    x: input data
    z: latent variables
    y: output data
    hidden_dim_enc: python list, the dimensions of hidden layers for encoder,
                        its reverse is the dimensions of hidden layers for decoder
    '''

    def __init__(self, x_dim=None, z_dim=16,
                 dense_x_z=[128], activation='tanh',
                 dropout_p = 0, beta=1, device='cpu'):

        super().__init__()
        ### General parameters for storn        
        self.x_dim = x_dim
        self.y_dim = self.x_dim
        self.z_dim = z_dim
        self.dropout_p = dropout_p
        if activation == 'relu':
            self.activation = nn.ReLU()
        elif activation == 'tanh':
            self.activation = nn.Tanh()
        else:
            raise SystemExit('Wrong activation type!')
        ### Inference
        self.dense_x_z = dense_x_z
        ### Generation
        self.dense_z_x = list(reversed(dense_x_z))
        ### Beta-loss
        self.beta = beta

        self.build()
        

    def build(self):
        
            
        ###################
        #### Inference ####
        ###################
        # 1. x_t to z_t
        
        dic_layers = OrderedDict()
        if len(self.dense_x_z) == 0:
            dim_x_z = self.dim_x
            dic_layers['Identity'] = nn.Identity()
        else:
            dim_x_z = self.dense_x_z[-1]
            for n in range(len(self.dense_x_z)):
                if n == 0:
                    dic_layers['linear'+str(n)] = nn.Linear(self.x_dim, self.dense_x_z[n])
                else:
                    dic_layers['linear'+str(n)] = nn.Linear(self.dense_x_z[n-1], self.dense_x_z[n])
                dic_layers['activation'+str(n)] = self.activation
                dic_layers['dropout'+str(n)] = nn.Dropout(p=self.dropout_p)
                
        self.mlp_x_z = nn.Sequential(dic_layers)
        self.inf_mean = nn.Linear(dim_x_z, self.z_dim)
        self.inf_logvar = nn.Linear(dim_x_z, self.z_dim)

        ######################
        #### Generation x ####
        ######################
        # 1. z_t to x_t
        
        dic_layers = OrderedDict()
        if len(self.dense_z_x) == 0:
            dim_z_x = self.dim_z
            dic_layers['Identity'] = nn.Identity()
        else:
            dim_z_x = self.dense_z_x[-1]
            for n in range(len(self.dense_z_x)):
                if n == 0:
                    dic_layers['linear'+str(n)] = nn.Linear(self.z_dim, self.dense_z_x[n])
                else:
                    dic_layers['linear'+str(n)] = nn.Linear(self.dense_z_x[n-1], self.dense_z_x[n])
                dic_layers['activation'+str(n)] = self.activation
                dic_layers['dropout'+str(n)] = nn.Dropout(p=self.dropout_p)
                
        self.mlp_z_x = nn.Sequential(dic_layers)
        self.gen_logvar = nn.Linear(dim_z_x, self.y_dim)


    def reparameterization(self, mean, logvar):

        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)

        return torch.addcmul(mean, eps, std) 


    def inference(self, x):

        x_z = self.mlp_x_z(x)
        z_mean = self.inf_mean(x_z)
        z_logvar = self.inf_logvar(x_z)
        z = self.reparameterization(z_mean, z_logvar)

        return z, z_mean, z_logvar

    
    def generation_x(self, z):

        z_x = self.mlp_z_x(z)
        log_y = self.gen_logvar(z_x)
        y = torch.exp(log_y)

        return y


    def forward(self, x, compute_loss=False, exp_mode = 'train'):
        
        # train input: (batch_size, x_dim, seq_len)
        # test input:  (x_dim, seq_len)
        # need input:  (seq_len, batch_size, x_dim)
       
        
        if len(x.shape) == 2:
            x = x.unsqueeze(0)
                
        seq_len = x.shape[0]
        batch_size = x.shape[1]
        
        # main part
        z, z_mean, z_logvar = self.inference(x)
        
        if exp_mode == 'train':
            y = self.generation_x(z)

        elif exp_mode == 'test':
            y = self.generation_x(z_mean)
        
        # calculate loss
        if compute_loss:
            loss_tot, loss_recon, loss_KLD = self.get_loss(x, y, z_mean, z_logvar, batch_size, seq_len, self.beta)
            self.loss = (loss_tot, loss_recon, loss_KLD)

        # output of NN:    (seq_len, batch_size, dim)
        # output of model: (batch_size, dim, seq_len) or (dim, seq_len)
        
        self.y = y.squeeze()

        return self.y


    def get_loss(self, x, y, z_mean, z_logvar, batch_size, seq_len, beta=1):

        loss_recon = torch.sum( x/y + torch.log(y) )
        loss_KLD = -0.5 * torch.sum(z_logvar -  z_logvar.exp() - z_mean.pow(2))

        loss_recon = loss_recon / (batch_size * seq_len)
        loss_KLD = loss_KLD / (batch_size * seq_len)
        loss_tot = loss_recon + beta * loss_KLD

        return loss_tot, loss_recon, loss_KLD


    def get_info(self):

        info = []
        info.append("----- Inference -----")
        for layer in self.mlp_x_z:
            info.append(str(layer))
        
        info.append("----- Bottleneck -----")
        info.append(str(self.inf_mean))
        info.append(str(self.inf_logvar))
        
        info.append("----- Decoder -----")
        for layer in self.mlp_z_x:
            info.append(str(layer))
        info.append(str(self.gen_logvar))

        return info

#%% VAE with a sparse dictionary model for the latent prior (SDM-VAE)

class SDM_VAE(nn.Module):

    '''
    VAE model class
    x: input data
    z: latent variables
    y: output data
    hidden_dim_enc: python list, the dimensions of hidden layers for encoder,
                        its reverse is the dimensions of hidden layers for decoder
    '''

    def __init__(self, x_dim=None, z_dim=16,
                 dense_x_z=[128], activation='tanh',
                 dropout_p = 0, beta=1, device='cpu', dictsize = 64):

        super().__init__()
        ### General parameters for storn        
        self.x_dim = x_dim
        self.y_dim = self.x_dim
        self.z_dim = z_dim
        self.dropout_p = dropout_p
        if activation == 'relu':
            self.activation = nn.ReLU()
        elif activation == 'tanh':
            self.activation = nn.Tanh()
        else:
            raise SystemExit('Wrong activation type!')
        ### Inference
        self.dense_x_z = dense_x_z
        ### Generation
        self.dense_z_x = list(reversed(dense_x_z))
        ### Beta-loss
        self.beta = beta

        ### dictionary
        self.dictsize = dictsize
        self.D = torch.from_numpy(odctdict(self.z_dim, dictsize).astype(np.float32)).to(device)# torch.eye(self.z_dim, dictsize).to(device) 
        self.D.requires_grad = False
        
        self.build()
        

    def build(self):
        
            
        ###################
        #### Inference ####
        ###################
        # 1. x_t to z_t
        
        dic_layers = OrderedDict()
        if len(self.dense_x_z) == 0:
            dim_x_z = self.dim_x
            dic_layers['Identity'] = nn.Identity()
        else:
            dim_x_z = self.dense_x_z[-1]
            for n in range(len(self.dense_x_z)):
                if n == 0:
                    dic_layers['linear'+str(n)] = nn.Linear(self.x_dim, self.dense_x_z[n])
                else:
                    dic_layers['linear'+str(n)] = nn.Linear(self.dense_x_z[n-1], self.dense_x_z[n])
                dic_layers['activation'+str(n)] = self.activation
                dic_layers['dropout'+str(n)] = nn.Dropout(p=self.dropout_p)
                
        self.mlp_x_z = nn.Sequential(dic_layers)

        self.inf_mean_alpha = nn.Linear(dim_x_z, self.dictsize)
        self.inf_logvar_alpha = nn.Linear(dim_x_z, self.dictsize)
        
        ######################
        #### Generation x ####
        ######################
        # 1. z_t to x_t
        
        dic_layers = OrderedDict()
        if len(self.dense_z_x) == 0:
            dim_z_x = self.dim_z
            dic_layers['Identity'] = nn.Identity()
        else:
            dim_z_x = self.dense_z_x[-1]
            for n in range(len(self.dense_z_x)):
                if n == 0:
                    dic_layers['linear'+str(n)] = nn.Linear(self.z_dim, self.dense_z_x[n])
                else:
                    dic_layers['linear'+str(n)] = nn.Linear(self.dense_z_x[n-1], self.dense_z_x[n])
                dic_layers['activation'+str(n)] = self.activation
                dic_layers['dropout'+str(n)] = nn.Dropout(p=self.dropout_p)
                
        self.mlp_z_x = nn.Sequential(dic_layers)
        self.gen_logvar = nn.Linear(dim_z_x, self.y_dim)

    def reparameterization(self, mean, logvar):

        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)

        return torch.addcmul(mean, eps, std) 


    def inference_alpha(self, x):

        x_z = self.mlp_x_z(x)
        alpha_mean = self.inf_mean_alpha(x_z)
        alpha_logvar = self.inf_logvar_alpha(x_z)
        alpha = self.reparameterization(alpha_mean, alpha_logvar)

        return alpha, alpha_mean, alpha_logvar
    
    
    def generation_x(self, z):

        z_x = self.mlp_z_x(z)
        log_y = self.gen_logvar(z_x)
        y = torch.exp(log_y)

        return y


    def forward(self, x, compute_loss=False):
        
        # train input: (batch_size, x_dim, seq_len)
        # test input:  (x_dim, seq_len)
        # need input:  (seq_len, batch_size, x_dim)


        # shape: (batch_size, x_dim)
        seq_len = 1
        batch_size = x.shape[0]
        
        # main part
        alpha, alpha_mean, alpha_logvar = self.inference_alpha(x) # shape: (batch_size, dictsize)

        y = self.generation_x(alpha @ self.D.t())

        with torch.no_grad():
            
            # gamma update
            gamma = alpha_logvar.exp() + alpha_mean.pow(2) 
            
        # calculate loss
        if compute_loss:
            loss_tot, loss_recon, loss_KLD = self.get_loss(x, y, alpha, alpha_mean, alpha_logvar, gamma, batch_size, seq_len, beta = self.beta)
            self.loss = (loss_tot, loss_recon, loss_KLD)
            

        # output of NN:    (seq_len, batch_size, dim)
        # output of model: (batch_size, dim, seq_len) or (dim, seq_len)
        
        self.y = y.squeeze()
        
        return self.y


    def get_loss(self, x, y, alpha, alpha_mean, alpha_logvar, gamma, batch_size, seq_len, beta=1):
        
        loss_recon = torch.sum( x/y + torch.log(y) )
        loss_KLD = -0.5 * torch.sum(alpha_logvar - torch.log(gamma) - (alpha_logvar.exp() + (alpha_mean).pow(2))/gamma)
        
        loss_recon = loss_recon / (batch_size * seq_len)
        loss_KLD = loss_KLD / (batch_size * seq_len)
        loss_tot = loss_recon + beta * loss_KLD

        return loss_tot, loss_recon, loss_KLD


    def get_info(self):

        info = []
        info.append("----- Inference -----")
        for layer in self.mlp_x_z:
            info.append(str(layer))
        
        info.append("----- Bottleneck -----")
        info.append(str(self.inf_mean_alpha))
        info.append(str(self.inf_logvar_alpha))
        
        info.append("----- Decoder -----")
        for layer in self.mlp_z_x:
            info.append(str(layer))
        info.append(str(self.gen_logvar))

        return info

    
if __name__ == '__main__':
    x_dim = 513
    device = 'cpu'
    vae = VAE(x_dim = x_dim).to(device)
    model_info = vae.get_info()
    for i in model_info:
        print(i)
    
