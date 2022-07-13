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

plt.close('all')


x, fs = sf.read('./data/trump.wav')

x = x[:int(15.5*fs)]

x = x/(1.1*np.max(np.abs(x)))
sf.write('./data/trump_cut.wav',x,fs )


rooms = ['bathroom', 'inchindown', 'usina', 'bathroom']

for room in rooms:

    a, fs_a = sf.read('./data/' + room + '.wav')
    if len(a.shape) > 1: 
        a = a[:,-1]
        
    if fs_a != fs:
        a = librosa.resample(a, fs_a, fs)
    
    if a.shape[0] > int(20*fs):
        a = a[:int(20*fs)]
    
    y = np.convolve(x, a)
    y = y/(1.1*np.max(np.abs(y)))
    sf.write('./data/trump_' + room + '.wav', y, fs)
    
