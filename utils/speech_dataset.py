#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Software dvae-speech
Copyright Inria
Year 2020
Contact : xiaoyu.bie@inria.fr
License agreement in LICENSE.txt

Class SpeechSequencesFull():
- generate Pytorch dataloader
- data sequence is clipped from the beginning of each audio signal
- every speech sequence can be divided into multiple data sequences, as long as audio_len >= seq_len
- usually, this method will give larger training sequences

Class SpeechSequencesRandom():
- generate Pytorch dataloader
- data sequence is clipped from a random place in each audio signal
- every speech sequence can only be divided into one single data sequence
- this method will introduce some randomness into training dataset

Both of these two Class use librosa.effects.trim()) to trim leading and trailing silence from an audio signal
"""

import numpy as np
import soundfile as sf
import librosa
import random
import torch
from torch.utils import data


class SpeechDatasetFrames(data.Dataset):
    """
    Customize a dataset of speech sequences for Pytorch
    at least the three following functions should be defined.
    
    This is a quick speech sequence data loader which allow multiple workers
    """
    def __init__(self, file_list, STFT_dict, shuffle, sequence_len = 'NaN', name='WSJ0'):

        super().__init__()

        # STFT parameters
        self.fs = STFT_dict['fs']
        self.nfft = STFT_dict['nfft']
        self.hop = STFT_dict['hop']
        self.wlen = STFT_dict['wlen']
        self.win = STFT_dict['win']
        self.trim = STFT_dict['trim']
        
        # data parameters
        self.file_list = file_list
        self.sequence_len = sequence_len
        self.name = name
        self.shuffle_file_list = shuffle
        self.current_frame = 0
        self.tot_num_frame = 0
        self.cpt_file = 0
        
        self.compute_len()


    def compute_len(self):

        self.num_samples = 0
        
        for cpt_file, wavfile in enumerate(self.file_list):

            x, fs_x = sf.read(wavfile)
            if self.fs != fs_x:
                raise ValueError('Unexpected sampling rate')        
                
            # remove beginning and ending silence
            x, index = librosa.effects.trim(x, top_db=30)

            x = np.pad(x, int(self.nfft // 2), mode='reflect') 
            # (cf. librosa.core.stft)
            
            n_frames = 1 + int((len(x) - self.wlen) / self.hop)
            
            self.num_samples += n_frames



    def __len__(self):
        """
        arguments should not be modified
        Return the total number of samples
        """
        return self.num_samples


    def __getitem__(self, index):
        """
        input arguments should not be modified
        torch data loader will use this function to read ONE sample of data 
        from a list that can be indexed by parameter 'index'
        """
        
        if self.current_frame == self.tot_num_frame:
        
            if self.cpt_file == len(self.file_list):
                self.cpt_file = 0
                if self.shuffle_file_list:
                    random.shuffle(self.file_list)
            
            wavfile = self.file_list[self.cpt_file]
            self.cpt_file += 1

            x, fs_x = sf.read(wavfile)
            if self.fs != fs_x:
                raise ValueError('Unexpected sampling rate')        
            x_orig = x/np.max(np.abs(x))
            
            # remove beginning and ending silence
            x_trimmed, index = librosa.effects.trim(x_orig, top_db=30)
                        
            x_pad = np.pad(x_trimmed, int(self.nfft // 2), mode='reflect') 
            # (cf. librosa.core.stft)
            
            X = librosa.stft(x_pad, n_fft=self.nfft, hop_length=self.hop, 
                             win_length=self.wlen,
                             window=self.win) # STFT

            
            self.data = np.abs(X)**2
            
            self.current_frame = 0
            self.tot_num_frame = self.data_a.shape[1]
            
        frame = self.data[:,self.current_frame]  
        
        self.current_frame += 1
        
        # turn numpy array to torch tensor with torch.from_numpy#
        """
        e.g.
        matrix = torch.from_numpy(matrix.astype(np.float32))
        target = torch.from_numpy(np.load(t_pth).astype(np.int32))
        """
        frame = torch.from_numpy(frame.astype(np.float32))
        
        return frame

