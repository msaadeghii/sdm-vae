#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Software dvae-speech
Copyright Inria
Year 2020
Contact : xiaoyu.bie@inria.fr
License agreement in LICENSE.txt

The main python file for model training, data test and performance evaluation, see README.md for further information
"""


import os
import shutil
import socket
import datetime
import pickle
import numpy as np
import torch
import librosa
import soundfile as sf
import matplotlib.pyplot as plt
from .utils import myconf, get_logger, EvalMetrics, SpeechDatasetFrames
from .model import build_VAE
import pyloudnorm as pyln

meter = pyln.Meter(16000)

def Hoyer_sparsity(z):

    D = len(z)

    hoyer_sp = (np.sqrt(D) - np.linalg.norm(z, 1)/ np.linalg.norm(z, 2))/(np.sqrt(D) - 1)

    return hoyer_sp

def sparsity_measure(z):
    spm = 0
    N = z.shape[0]
    for i in range(N):
        spm += Hoyer_sparsity(z[i,:])
        
    return spm / N
        
def compute_rmse(x_est, x_ref):

    # align
    len_x = len(x_est)
    x_ref = x_ref[:len_x]
    # scaling
    alpha = np.sum(x_est*x_ref) / np.sum(x_est**2)
    # x_est_ = np.expand_dims(x_est, axis=1)
    # alpha = np.linalg.lstsq(x_est_, x_ref, rcond=None)[0][0]
    x_est_scaled = alpha * x_est
    return np.sqrt(np.square(x_est_scaled - x_ref).mean())


class LearningAlgorithm():

    """
    Basical class for model building, including:
    - read common paramters for different models
    - define data loader
    - define loss function as a class member
    """

    def __init__(self, config_file='config_default.ini'):

        # Load config parser
        self.config_file = config_file
        if not os.path.isfile(self.config_file):
            raise ValueError('Invalid config file path')    
        
        self.cfg = myconf()
        
        self.cfg.read(self.config_file)
        
        self.model_name = self.cfg.get('Network', 'name')
        self.vae_mode = self.cfg.get('Network', 'vae_mode')
        self.dataset_name = self.cfg.get('DataFrame', 'dataset_name')

        # Get host name and date
        self.hostname = socket.gethostname()
        self.date = datetime.datetime.now().strftime("%Y-%m-%d-%Hh%M")
        
        # Load STFT parameters
        wlen_sec = self.cfg.getfloat('STFT', 'wlen_sec')
        hop_percent = self.cfg.getfloat('STFT', 'hop_percent')
        fs = self.cfg.getint('STFT', 'fs')
        zp_percent = self.cfg.getint('STFT', 'zp_percent')
        wlen = wlen_sec * fs
        wlen = np.int(np.power(2, np.ceil(np.log2(wlen)))) # pwoer of 2
        hop = np.int(hop_percent * wlen)
        nfft = wlen + zp_percent * wlen
        win = np.sin(np.arange(0.5, wlen+0.5) / wlen * np.pi)

        STFT_dict = {}
        STFT_dict['fs'] = fs
        STFT_dict['wlen'] = wlen
        STFT_dict['hop'] = hop
        STFT_dict['nfft'] = nfft
        STFT_dict['win'] = win
        STFT_dict['trim'] = self.cfg.getboolean('STFT', 'trim')
        self.STFT_dict = STFT_dict

        # Load model parameters
        self.use_cuda = self.cfg.getboolean('Training', 'use_cuda')
        self.device = 'cuda' if torch.cuda.is_available() and self.use_cuda else 'cpu'

        # Build model
        self.build_model()


    def build_model(self):


        self.model = build_VAE(cfg=self.cfg, device=self.device, vae_mode = self.vae_mode, exp_mode = 'train')
        

    def init_optimizer(self):

        # Load 
        self.optimization  = self.cfg.get('Training', 'optimization')
        lr = self.cfg.getfloat('Training', 'lr')
        
        # Init optimizer (Adam by default)
            
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)


    def build_dataloader(self, train_data_dir, val_data_dir, sequence_len, batch_size, STFT_dict, use_random_seq=False):

        # List all the data with certain suffix
        data_suffix = self.cfg.get('DataFrame', 'suffix')
        train_file_list = librosa.util.find_files(train_data_dir, ext=data_suffix)
        val_file_list = librosa.util.find_files(val_data_dir, ext=data_suffix)
        # Generate dataloader for pytorch
        num_workers = self.cfg.getint('DataFrame', 'num_workers')
        shuffle_file_list = self.cfg.get('DataFrame', 'shuffle_file_list')
        shuffle_samples_in_batch = self.cfg.get('DataFrame', 'shuffle_samples_in_batch')
        data_dir = self.cfg.get('User', 'data_dir')
        
        train_file_list = [os.path.join(root, name) for root, dirs, files in os.walk(os.path.join(data_dir, 'train_data_NTCD')) for name in files if name.endswith('.wav')]
        
        val_file_list = [os.path.join(root, name) for root, dirs, files in os.walk(os.path.join(data_dir, 'val_data_NTCD')) for name in files if name.endswith('.wav')]


        train_dataset = SpeechDatasetFrames(file_list=train_file_list, sequence_len=sequence_len,
                                              STFT_dict=self.STFT_dict, shuffle=shuffle_file_list, name=self.dataset_name)
        val_dataset = SpeechDatasetFrames(file_list=val_file_list, sequence_len=sequence_len,
                                                STFT_dict=self.STFT_dict, shuffle=shuffle_file_list, name=self.dataset_name)


        train_num = train_dataset.__len__()
        val_num = val_dataset.__len__()

        # Create dataloader
        train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, 
                                                       shuffle=shuffle_samples_in_batch,
                                                       num_workers = num_workers)
        val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size,
                                                     shuffle=shuffle_samples_in_batch,
                                                     num_workers = num_workers)

        return train_dataloader, val_dataloader, train_num, val_num


    def get_basic_info(self):

        basic_info = []
        basic_info.append('HOSTNAME: ' + self.hostname)
        basic_info.append('Time: ' + self.date)
        basic_info.append('Device for training: ' + self.device)
        if self.device == 'cuda':
            basic_info.append('Cuda verion: {}'.format(torch.version.cuda))
        basic_info.append('Model name: {}'.format(self.model_name))
        basic_info.append('VAE mode: {}'.format(self.vae_mode))
        
        return basic_info


    def train(self):

        # Set module.training = True
        self.model.train()
        torch.autograd.set_detect_anomaly(True)

        # Create directory for results
        saved_root = self.cfg.get('User', 'saved_root')
        z_dim = self.cfg.getint('Network','z_dim')
        tag = self.cfg.get('Network', 'tag')
        filename = "{}_{}_{}_z_dim={}".format(self.dataset_name, self.date, tag, z_dim)
        save_dir = os.path.join(saved_root, filename)
        if not(os.path.isdir(save_dir)):
            os.makedirs(save_dir)

        # Save the model configuration
        save_cfg = os.path.join(save_dir, 'config.ini')
        shutil.copy(self.config_file, save_cfg)

        # Create logger
        log_file = os.path.join(save_dir, 'log.txt')
        logger_type = self.cfg.getint('User', 'logger_type')
        logger = get_logger(log_file, logger_type)

        # Print basical infomation
        for log in self.get_basic_info():
            logger.info(log)
        logger.info('In this experiment, result will be saved in: ' + save_dir)

        # Print model infomation (optional)
        if self.cfg.getboolean('User', 'print_model'):
            for log in self.model.get_info():
                logger.info(log)

        # Init optimizer
        self.init_optimizer()

        
        batch_size = self.cfg.getint('Training', 'batch_size')
        sequence_len = self.cfg.getint('DataFrame','sequence_len')
        use_random_seq = self.cfg.getboolean('DataFrame','use_random_seq')
        
        # Create data loader
        train_data_dir = self.cfg.get('User', 'train_data_dir')
        val_data_dir = self.cfg.get('User', 'val_data_dir')
        loader = self.build_dataloader(train_data_dir=train_data_dir, val_data_dir=val_data_dir,
                                       sequence_len=sequence_len, batch_size=batch_size,
                                       STFT_dict=self.STFT_dict, use_random_seq=use_random_seq)
        train_dataloader, val_dataloader, train_num, val_num = loader
        log_message = 'Training samples: {}'.format(train_num)
        logger.info(log_message)
        print(log_message)
        log_message = 'Validation samples: {}'.format(val_num)
        logger.info(log_message)
        print(log_message)

        self.train_normal(logger, save_dir, train_dataloader, val_dataloader, train_num, val_num)



    def train_normal(self, logger, save_dir, train_dataloader, val_dataloader, train_num, val_num):

        # Load training parameters
        epochs = self.cfg.getint('Training', 'epochs')
        early_stop_patience = self.cfg.getint('Training', 'early_stop_patience')
        save_frequency = self.cfg.getint('Training', 'save_frequency')

        # Create python list for loss
        train_loss = np.zeros((epochs,))
        val_loss = np.zeros((epochs,))
        train_recon = np.zeros((epochs,))
        train_KLD = np.zeros((epochs,))
        val_recon = np.zeros((epochs,))
        val_KLD = np.zeros((epochs,))
        best_val_loss = np.inf
        cpt_patience = 0
        cur_best_epoch = epochs
        best_state_dict = self.model.state_dict()

        # Define optimizer (might use different training schedule)
        optimizer = self.optimizer
        
        # Train with mini-batch SGD
        for epoch in range(epochs):

            start_time = datetime.datetime.now()
            
            # Batch training
            for batch_idx, batch in enumerate(train_dataloader):
                
                batch = batch.to(self.device)
                
                self.model(batch, compute_loss=True)

                loss_tot, loss_recon, loss_KLD = self.model.loss
                
                optimizer.zero_grad()
                loss_tot.backward()
                optimizer.step()
                
                train_loss[epoch] += loss_tot.item()
                train_recon[epoch] += loss_recon.item()
                train_KLD[epoch] += loss_KLD.item()
                
            # Validation
            for batch_idx, batch in enumerate(val_dataloader):

                batch = batch.to(self.device)
                
                self.model(batch, compute_loss=True)


                loss_tot, loss_recon, loss_KLD = self.model.loss
                
                val_loss[epoch] += loss_tot.item()
                val_recon[epoch] += loss_recon.item()
                val_KLD[epoch] += loss_KLD.item()

            # Loss normalization
            train_loss[epoch] = train_loss[epoch]/ train_num
            val_loss[epoch] = val_loss[epoch] / val_num
            train_recon[epoch] = train_recon[epoch] / train_num 
            train_KLD[epoch] = train_KLD[epoch]/ train_num
            val_recon[epoch] = val_recon[epoch] / val_num 
            val_KLD[epoch] = val_KLD[epoch] / val_num
            
            # Early stop patiance
            if val_loss[epoch] < best_val_loss:
                best_val_loss = val_loss[epoch]
                cpt_patience = 0
                best_state_dict = self.model.state_dict()
                cur_best_epoch = epoch
            else:
                cpt_patience += 1

            # Training time
            end_time = datetime.datetime.now()
            interval = (end_time - start_time).seconds / 60
            log_message = 'Epoch: {} train loss: {:.4f} val loss {:.4f} training time {:.2f}m'.format(epoch, train_loss[epoch], val_loss[epoch], interval)
            logger.info(log_message)
            print(log_message)

            # Stop traning if early-stop triggers
            if cpt_patience == early_stop_patience:
                logger.info('Early stop patience achieved')
                print('Early stopping occured ...')
                break

            # Save model parameters regularly
            if epoch % save_frequency == 0:
                save_file = os.path.join(save_dir, self.model_name + '-' + self.vae_mode + '_epoch' + str(cur_best_epoch) + '.pt')
                torch.save(self.model.state_dict(), save_file)
        
        # Save the final weights of network with the best validation loss
        train_loss = train_loss[:epoch+1]
        val_loss = val_loss[:epoch+1]
        train_recon = train_recon[:epoch+1]
        train_KLD = train_KLD[:epoch+1]
        val_recon = val_recon[:epoch+1]
        val_KLD = val_KLD[:epoch+1]
        save_file = os.path.join(save_dir, self.model_name + '-' + self.vae_mode + '_final_epoch' + str(cur_best_epoch) + '.pt')
        torch.save(best_state_dict, save_file)
        
        # Save the training loss and validation loss
        loss_file = os.path.join(save_dir, 'loss_model.pckl')
        with open(loss_file, 'wb') as f:
            pickle.dump([train_loss, val_loss, train_recon, train_KLD, val_recon, val_KLD], f)


        # Save the loss figure
        tag = self.vae_mode
        plt.clf()
        fig = plt.figure(figsize=(8,6))
        plt.rcParams['font.size'] = 12
        plt.plot(train_loss, label='training loss')
        plt.plot(val_loss, label='validation loss')
        plt.legend(fontsize=16, title=self.model_name, title_fontsize=20)
        plt.xlabel('epochs', fontdict={'size':16})
        plt.ylabel('loss', fontdict={'size':16})
        fig_file = os.path.join(save_dir, 'loss_{}.png'.format(tag))
        plt.savefig(fig_file)

        plt.clf()
        fig = plt.figure(figsize=(8,6))
        plt.rcParams['font.size'] = 12
        plt.plot(train_recon, label='Reconstruction')
        plt.plot(train_KLD, label='KL Divergence')
        plt.legend(fontsize=16, title='{}: Training'.format(self.model_name), title_fontsize=20)
        plt.xlabel('epochs', fontdict={'size':16})
        plt.ylabel('loss', fontdict={'size':16})
        fig_file = os.path.join(save_dir, 'loss_train_{}.png'.format(tag))
        plt.savefig(fig_file) 

        plt.clf()
        fig = plt.figure(figsize=(8,6))
        plt.rcParams['font.size'] = 12
        plt.plot(val_recon, label='Reconstruction')
        plt.plot(val_KLD, label='KL Divergence')
        plt.legend(fontsize=16, title='{}: Validation'.format(self.model_name), title_fontsize=20)
        plt.xlabel('epochs', fontdict={'size':16})
        plt.ylabel('loss', fontdict={'size':16})
        fig_file = os.path.join(save_dir, 'loss_val_{}.png'.format(tag))
        plt.savefig(fig_file)


    def generate(self, audio_orig, audio_recon=None, state_dict_file=None, save_flag = False, denoise = False, target_snr = 10, seed = 0, model_type = 'A-VAE'):
        """
        Input: a reference audio (and a predefined path for generated audio
        Output: generated audio
        """

        # Define generated 
        if audio_recon == None:
            #print('Generated audio file will be saved in the same directory as reference audio')
            audio_dir, audio_file = os.path.split(audio_orig)
            file_name, file_ext = os.path.splitext(audio_file)
            audio_recon = os.path.join(audio_dir, file_name+'_recon'+file_ext)
        else:
            root_dir, filename = os.path.split(audio_recon)
            #if not os.path.isdir(root_dir):
            #    os.makedirs(root_dir)
        
        # Load model state
        if state_dict_file != None:
            self.model.load_state_dict(torch.load(state_dict_file, map_location=self.device))

        # Read STFT parameters
        fs = self.STFT_dict['fs']
        nfft = self.STFT_dict['nfft']
        hop = self.STFT_dict['hop']
        wlen = self.STFT_dict['wlen']
        win = np.sin(np.arange(0.5, wlen+0.5) / wlen * np.pi)
        
        # Read original audio file
        x, fs_x = sf.read(audio_orig)
        if fs != fs_x:
            raise ValueError('Unexpected sampling rate')

        # Scaling
        scale = np.max(np.abs(x))
        x = x / scale

        # STFT
        X = librosa.stft(x, n_fft=nfft, hop_length=hop, win_length=wlen, window=win)

        # Prepare data input        
        data_orig = np.abs(X) ** 2 # (x_dim, seq_len)
        data_orig = torch.from_numpy(data_orig.astype(np.float32)).to(self.device) 
                
        # Set module.training = False
        self.model.eval()

        # Reconstruction
        with torch.no_grad():
            data_recon = self.model(data_orig.t(), compute_loss=False).to('cpu').detach().numpy().T
            
            if model_type == 'VAE':
                _,z_mean, _ = self.model.inference(data_orig.t()) # corresponds to the encoder mean
            elif model_type == 'SDM-VAE':
                _,z_mean, _ = self.model.inference_alpha(data_orig.t()) # corresponds to the encoder mean


            z_mean = z_mean.to('cpu').detach().numpy()
            #print(z_mean[30,:]) 
        if not denoise:
            # Re-synthesis
            X_recon = np.sqrt(data_recon) * np.exp(1j * np.angle(X))
            x_recon = librosa.istft(X_recon, hop_length=hop, win_length=wlen, window=win)
        
            # Wrtie audio file
            scale_norm = 1 / (np.maximum(np.max(np.abs(x_recon)), np.max(np.abs(x)))) * 0.9
        else:
            np.random.seed(seed)
            b_n = np.random.normal(loc=0, scale=1, size = len(x))

            s_loudness = meter.integrated_loudness(x)
            n_loudness = meter.integrated_loudness(b_n)

            input_snr = s_loudness - n_loudness
            scale_factor = 10**((input_snr - target_snr)/20)

            b_n = b_n * scale_factor

            B_n = np.abs(librosa.stft(b_n, n_fft=nfft, hop_length=hop, win_length=wlen, window=win))** 2
            
            y_n = x + b_n

            Y_n = librosa.stft(y_n, n_fft=nfft, hop_length=hop, win_length=wlen, window=win)

            X_recon = (data_recon/(data_recon + B_n)) * Y_n

            x_recon = librosa.istft(X_recon, hop_length=hop, win_length=wlen, window=win)

            scale_norm = 1 / (np.maximum(np.max(np.abs(x_recon)), np.max(np.abs(x)))) * 0.9

        if save_flag == True:
            sf.write(audio_recon, scale_norm*x_recon, fs_x)
            path, name = os.path.split(audio_recon)
            sf.write(os.path.join(path, name[:-4]+'-noisy.wav'), y_n, fs_x)
        
        score_isdr, score_pesq, score_stoi = self.eval_metrics(audio_ref = x, audio_est = scale_norm*x_recon)
        spm_a = sparsity_measure(z_mean)
        
        return score_isdr, score_pesq, score_stoi, spm_a        
    
    def eval_metrics(self, audio_ref, audio_est, metric='all'):
        """
        Input: a reference audio and a generated audio
        Output: score(s) from different evaluation metrics
        """

        eval_metrics = EvalMetrics(metric=metric)
        
        score_isdr, score_pesq, score_stoi = eval_metrics.eval(audio_est, audio_ref)
        
        return score_isdr, score_pesq, score_stoi