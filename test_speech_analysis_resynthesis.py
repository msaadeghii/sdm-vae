#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep  3 15:35:08 2021

@author: smostafa
"""


from . import LearningAlgorithm
import os
from random import shuffle
import numpy as np

root_dir = './'

speech_dir = './'

    
file_list = [os.path.join(root, name)
              for root, dirs, files in os.walk(speech_dir)
              for name in files
              if name.endswith('.wav')]

#shuffle(file_list)

file_list = file_list[:1000]

# Create score list
list_score_isdr = []
list_score_pesq= []
list_score_stoi = []
  
list_spm = []


target_snr = -10 # for oracle speech enhancement


save_dir = './'
save_flag = False
denoise = False

for ind_mix, speech_file in enumerate(file_list):
    
    # vae_mode == 'VAE' 
        
    state_dict_file = './saved_model/model.pt'    
        
    path, fname = os.path.split(state_dict_file)
    cfg_file = os.path.join(path, 'config.ini')

    audio_recon = os.path.join(save_dir, 'vae.wav')

    learning_algo = LearningAlgorithm(config_file=cfg_file)
    score_isdr, score_pesq, score_stoi, spm = learning_algo.generate(audio_orig = speech_file, audio_recon = audio_recon, save_flag = save_flag,  state_dict_file = state_dict_file, denoise = denoise, target_snr = target_snr, seed = ind_mix, model_type = 'VAE')

    list_score_isdr.append(score_isdr)
    list_score_pesq.append(score_pesq)
    list_score_stoi.append(score_stoi)
    list_spm.append(spm)
    
    
    if ind_mix % 50 ==0:
        print('File {} / {} processed ...'.format(ind_mix, len(file_list)))
    

print('SDR = {} '.format(np.mean(np.asarray(list_score_isdr))))
print('PESQ = {} '.format(np.mean(np.asarray(list_score_pesq))))
print('STOI = {} '.format(np.mean(np.asarray(list_score_stoi))))
print('SPM = {} '.format(np.mean(np.asarray(list_spm))))
