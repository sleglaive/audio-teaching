#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb  7 14:33:45 2020

@author: simon
"""

import numpy as np
import soundfile as sf 
import matplotlib.pyplot as plt
import librosa
import os
import librosa.display


plt.close('all')

#%%
audio_files = ['trump_cut', 'trump_toilet', 'trump_usina', 'trump_inchindown']

fs = 44100

wlen_sec=32e-3
hop_percent=.5

wlen = int(wlen_sec*fs) # window length of 64 ms
wlen = np.int(np.power(2, np.ceil(np.log2(wlen)))) # next power of 2
nfft = wlen
hop = np.int(hop_percent*wlen) # hop size
win = np.sin(np.arange(.5,wlen-.5+1)/wlen*np.pi); # sine analysis window

for ind, audio_file in enumerate(audio_files):
    
    plt.subplot(2,2,ind+1)
    
    x, fs = sf.read('./data/' + audio_file + '.wav')    
    
    X = librosa.stft(x, n_fft=nfft, hop_length=hop, win_length=wlen, window=win) # STFT
    
    librosa.display.specshow(librosa.power_to_db(np.abs(X)**2), sr=fs, hop_length=hop, x_axis='time', y_axis='hz')
    
    plt.ylim([0,4000])
#    plt.xlim([0,10])
    plt.set_cmap('magma')
    
#    
    plt.colorbar()
    plt.clim(vmin=-30)
#    
    
    plt.ylabel('frequency (Hz)')
    plt.xlabel('time (s)')

plt.tight_layout()

#%%

T = 21
t = np.linspace(-10,10,T)

u = np.zeros(T)
u[5:15] = 1
    
v = np.zeros(T)
v[10:15] = np.flip(np.linspace(0,1,5))

v_t_moins_tau = np.zeros(T)
v_t_moins_tau[0:5] = np.linspace(0,1,5)

u_conv_v = np.zeros(T)
u_conv_v[5:5+14] = np.convolve(u[5:15],v[10:15])

plt.subplot(5,1,1)
plt.stem(t,u,basefmt=" ")
plt.xticks(t[0:T+1:2].astype(int), [])
plt.title(r'$u(\tau)$', fontsize=15)

plt.subplot(5,1,2)
plt.stem(t,v, basefmt=" ")
plt.xticks(t[0:T+1:2].astype(int), [])
plt.title(r'$v(\tau)$', fontsize=15)

plt.subplot(5,1,3)
plt.stem(t,np.flip(v), basefmt=" ")
plt.xticks(t[0:T+1:2].astype(int), [])
plt.title(r'$v(-\tau)$', fontsize=15)

plt.subplot(5,1,4)
plt.stem(t,v_t_moins_tau, basefmt=" ")
plt.xticks(t[0:T+1:2].astype(int), [])
plt.title(r'$v(t-\tau)$', fontsize=15)


plt.subplot(5,1,5)
plt.stem(t,u_conv_v, basefmt=" ")
plt.xticks(t[0:T+1:2].astype(int), t[0:T+1:2].astype(int))
plt.title(r'$[u \star v](t)$', fontsize=15)


plt.tight_layout()


#%%
import scipy.io

mat = scipy.io.loadmat('./data/RIRs_MIRD/160/mix_filters_pos1.mat')
h_160 = mat['a']

mat = scipy.io.loadmat('./data/RIRs_MIRD/360/mix_filters_pos1.mat')
h_360 = mat['a']

mat = scipy.io.loadmat('./data/RIRs_MIRD/610/mix_filters_pos1.mat')
h_610 = mat['a']

fs = 16000

h_all = [h_160, h_360, h_610]

T60 = ['160 ms', '360 ms', '610 ms']

T = int(0.610*fs) + 1

plt.close()
for i, h in enumerate(h_all):
    
    plt.subplot(1,3,i+1)
    
    t = np.arange(T)/fs
    
    h_pad = np.zeros(T)
    h_pad[:h.shape[0]] = h[:,0]
    
    plt.plot(t, h_pad)
    plt.title('RIR with T60 = ' + T60[i])
    
    
    plt.xlabel('time (s)')
    if i > 0:
        plt.yticks(np.arange(-0.2,0.15,0.05), [])
        
    if  i == 0:
        plt.ylabel('amplitude')    

    plt.ylim([-0.2, 0.15])
    plt.grid('on')

#%%

plt.figure()
t = np.arange(T)/fs
plt.plot(t, 20*np.log10(np.abs(h_all[-1][:,0])))
plt.ylabel('magnitude (dB)')
plt.title('RIR with a reverberation time of 610 ms')
plt.xlabel('time (s)')
