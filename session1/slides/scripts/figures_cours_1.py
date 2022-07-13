#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 13 17:25:59 2020

@author: sleglaive
"""

import numpy as np
import soundfile as sf 
import matplotlib.pyplot as plt
import librosa
import os

plt.close('all')


#%%

if False:

    data_dir = '/data/enseignement/2019-2020/electif_image_son/slides/cours_1/audio'
    
    file_list = librosa.util.find_files(data_dir)
    
    for file in file_list:
        
        x, fs = sf.read(file)
        
        # mono and scale
        if len(x.shape)>1:
            x = x[:,0]
        x = x/np.max(x)*0.9
        sf.write(file, x, fs)
        
        #
        file_name = os.path.split(file)[1][:-4]
        
        # plot
        time = np.arange(0, x.shape[0])/fs
        plt.figure(figsize=(6,2))
        plt.plot(time, x, 'k')
        plt.xlabel('time (s)')
        plt.ylabel('amplitude')
        plt.title(file_name)
        plt.tight_layout()
        
        plt.savefig(file[:-4] + '.png')
        
    
#%%
    
if False:
    
    T_vec = np.array([8, 16, 32])
#    T_vec = np.array([16, 32, 64])
    
    plt.subplots(3,2, figsize=(10,6))
    F = 128
    f = np.arange(F)
    T_max = np.max(T_vec)
    
    
    for ind, T in enumerate(T_vec):
    
        t = np.arange(T_max)
        nu0 = 0.2
        
#        x = np.sin(2*np.pi*nu0*t)
        x = np.zeros(T_max)
        x[:T] = 1
        
        x_hat = np.fft.fft(x, F)
        X = np.fft.fft(x, T)
    
        plt.subplot(3,2,ind*2+1)
        plt.stem(t, x)
        plt.xlim(0,T_max)
        plt.ylim(0,1.1)
        
        if ind==0:
            plt.title('Finite-length signal')
        
        if ind==2:
            plt.xlabel('time')
            
        plt.yticks(np.array([0,1]), np.array([0,1]))
            
        plt.subplot(3,2,ind*2+2)
        plt.plot(f/F, np.abs(x_hat), '-')
        plt.plot(np.arange(T)/T, np.abs(X), 'o', fillstyle='none')
        
        if ind==0:
            plt.title('Magnitude spectra')
#            plt.legend({r'DFT $X(f) = \hat{x}(f/T)$', r'DTFT $\hat{x}(\nu)$'}, loc='upper right')
            plt.legend([r'DTFT $\hat{x}(\nu)$', r'DFT $X(f) = \hat{x}(f/T)$'], loc='upper right')
        
        if ind==2:
            plt.xlabel('frequency')

    plt.tight_layout()
    
#%%
    
if True:
    
#    T_vec = np.array([8, 16, 32])
    T_vec = np.array([16, 32, 64])
    
    plt.subplots(3,2, figsize=(10,6))
    F = 128
    f = np.arange(F)
    T_max = np.max(T_vec)
    
    
    for ind, T in enumerate(T_vec):
    
        t = np.arange(T_max)
        nu0 = 0.1
        
        
        ## Porte
#        x = np.zeros(T_max)
#        x[:T] = 1
        
        ## Sinus
        x = np.sin(2*np.pi*nu0*t)
        x[T:] = 0
        
        x_hat = np.fft.fft(x, F)
        X = np.fft.fft(x, T)
    
        plt.subplot(3,2,ind*2+1)
        plt.plot(t, x, '-')
        plt.plot(t, x, 'o', fillstyle='none')
#        plt.stem(t, x)
        plt.xlim(0,T_max)
        plt.ylim(-1.1,1.1)
        
        if ind==0:
            plt.title('Finite-length signal')
        
        if ind==2:
            plt.xlabel('time')
            
        plt.yticks(np.array([0,1]), np.array([0,1]))
            
        plt.subplot(3,2,ind*2+2)
        plt.plot(f/F, np.abs(x_hat), '-')
        plt.plot(np.arange(T)/T, np.abs(X), 'o', fillstyle='none')
        
        if ind==0:
            plt.title('Magnitude spectra')
#            plt.legend({r'DFT $X(f) = \hat{x}(f/T)$', r'DTFT $\hat{x}(\nu)$'}, loc='upper right')
            plt.legend([r'DTFT $\hat{x}(\nu)$', r'DFT $X(f) = \hat{x}(f/T)$'], loc='upper right')
        
        if ind==2:
            plt.xlabel('frequency')

    plt.tight_layout()
    
#%%
    
if False:
    
    T_vec = np.array([16, 32, 64])
    
    plt.subplots(3,2, figsize=(10,6))
    F = 128
    f = np.arange(F)
    T_max = np.max(T_vec)
    
    
    for ind, T in enumerate(T_vec):
    
        t = np.arange(T_max)
        nu0 = 0.1
        nu1 = 0.13
        
        ## Sinus
        x = .5*np.sin(2*np.pi*nu0*t) + .5*np.sin(2*np.pi*nu1*t)
        x[T:] = 0
        
        x_hat = np.fft.fft(x, F)
        X = np.fft.fft(x, T)
    
        plt.subplot(3,2,ind*2+1)
        plt.plot(t, x, '-')
        plt.plot(t, x, 'o', fillstyle='none')
#        plt.stem(t, x)
        plt.xlim(0,T_max)
        plt.ylim(-1.1,1.1)
        
        if ind==0:
            plt.title('Finite-length signal')
        
        if ind==2:
            plt.xlabel('time')
            
        plt.yticks(np.array([0,1]), np.array([0,1]))
            
        plt.subplot(3,2,ind*2+2)
        plt.plot(f/F, np.abs(x_hat), '-')
        plt.plot(np.arange(T)/T, np.abs(X), 'o', fillstyle='none')
        
        if ind==0:
            plt.title('Magnitude spectra')
#            plt.legend({r'DFT $X(f) = \hat{x}(f/T)$', r'DTFT $\hat{x}(\nu)$'}, loc='upper right')
            plt.legend([r'DTFT $\hat{x}(\nu)$', r'DFT $X(f) = \hat{x}(f/T)$'], loc='upper right')
        
        if ind==2:
            plt.xlabel('frequency')

    plt.tight_layout()

#%%
    
if False:
    
    x, fs = sf.read('/data/enseignement/2019-2020/electif_image_son/slides/cours_1/audio/acdll.wav')
    x = x/np.max(x)
    
    T = x.shape[0]
    f = np.arange(T)*fs/T
    t = np.arange(T)/fs
    
    plt.subplot(2,1,1)
    plt.plot(t, x, 'k')
    plt.title('Signal')
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude')
    
    X = np.fft.fft(x)
    plt.subplot(2,1,2)
    plt.plot(f[:int(T/2)], 20*np.log10(np.abs(X[:int(T/2)])), 'k')
    plt.title('Power spectrum')
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Magnitude in dB')
    
    plt.tight_layout()
    
#%%
    
if False:
    t = np.arange(256)
    nu0 = 0.01
    nu1 = 0.04
    nu2 = 0.08
    x0 = np.sin(2*np.pi*nu0*t)
    x1 = np.sin(2*np.pi*nu1*t)
    x2 = np.sin(2*np.pi*nu2*t)
    
    x = x0 + .8*x1 + .5*x2
    
    plt.close()
    plt.figure()
    plt.subplot(4,1,1)
    plt.plot(x0)
    plt.subplot(4,1,2)
    plt.plot(x1)
    plt.subplot(4,1,3)
    plt.plot(x2)
    plt.subplot(4,1,4)
    plt.plot(x)