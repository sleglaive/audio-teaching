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


audio_files = ['trump', 'trump_toilet', 'trump_usina', 'trump_inchindown']

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