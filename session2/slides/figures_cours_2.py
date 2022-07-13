#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 22 14:45:41 2020

@author: sleglaive
"""

import numpy as np
import soundfile as sf 
import matplotlib.pyplot as plt
import librosa
import librosa.display
import scipy as sp

import os


#%%

x, fs = sf.read('/data/enseignement/2019-2020/electif_image_son/slides/cours_2/audio/SA2.WAV')

t0 = 1500
tend = 17120

x = x[t0:tend]

x = x/np.max(np.abs(x))

T = x.shape[0]
t = np.arange(T)/fs

plt.plot(t, x, 'k')
plt.title('\"don\'t ask me\"', fontsize=16)
plt.xlabel('Time (s)', fontsize=16)
plt.ylabel('Amplitude', fontsize=16)
plt.tight_layout()


#%% aeiou Simon

def LPC(x, P):
    
    T = x.shape[0]
    
    r = np.zeros(P+1)
    
    for i in np.arange(P+1):
        
        x1 = x[i:T]
        x2 = x[:T-i]
        r[i] = 1/T*np.sum(x1*x2)
        
    R = sp.linalg.toeplitz(r[:P])
    a = -np.linalg.inv(R)@r[1:]
    sigma2 = np.sum(r * np.concatenate([np.array([1.]), a]))

    return a, sigma2

x_cplt, fs = sf.read('/data/enseignement/2019-2020/electif_image_son/slides/cours_2/audio/simon.wav')

t0vec = [np.int(0.25*fs), np .int(1.1*fs), np.int(2.2*fs), np.int(3.1*fs), np.int(4.1*fs)]

vowel = ['\\a\\', '\\e\\', '\\i\\', '\\o\\', '\\u\\']


fig, axs = plt.subplots(5,1)

for n, t0 in enumerate(t0vec):
    
    tend = t0 + np.int(0.2*fs)
    
    x = x_cplt[t0:tend]
    x = x/np.max(np.abs(x))
    
    x = np.convolve(x, np.array([1, -0.976]))
    T = x.shape[0]
    
    a, sigma2 = LPC(x, 16)

    X = np.fft.fft(x)[:T//2+1]    
    
    perio_x = np.abs(X)**2/T
    perio_x_db = 10*np.log10(perio_x)

    den = np.abs( np.fft.fft( np.concatenate([np.array([1.]), a]), T) )[:T//2+1]
    spec_lpc_x_db = 10*np.log10(sigma2) - 20*np.log10(den)
        
    f = np.arange(T//2+1)*fs/T
    axs[n].plot(f, perio_x_db, 'k')
    axs[n].plot(f, spec_lpc_x_db)

    
    axs[n].grid('on')
    
#    if n < 4:
#        axs[n].set_xticks([])

    
    axs[n].set_xlim(0,4000)
    
    axs[n].set_title(vowel[n], fontsize=10)
    axs[n].set_ylabel('magnitude (dB)', fontsize=10)
    
    

plt.xlabel('frequency (Hz)', fontsize=10)
    

plt.tight_layout()


#%% aeiou

wlen_sec=32e-3
hop_percent=.5

wlen = int(wlen_sec*fs) # window length of 64 ms
wlen = np.int(np.power(2, np.ceil(np.log2(wlen)))) # next power of 2
nfft = wlen
hop = np.int(hop_percent*wlen) # hop size
win = np.sin(np.arange(.5,wlen-.5+1)/wlen*np.pi); # sine analysis window

x, fs = sf.read('/data/enseignement/2019-2020/electif_image_son/slides/cours_2/audio/simon.wav')    

X = librosa.stft(x, n_fft=nfft, hop_length=hop, win_length=wlen, window=win) # STFT

plt.figure(figsize=(10,7))
librosa.display.specshow(librosa.power_to_db(np.abs(X)**2), sr=fs, hop_length=hop, x_axis='time', y_axis='hz')
plt.set_cmap('gray_r')
axes = plt.gca()
axes.set_ylim([0,4000])

plt.colorbar()
plt.clim(vmin=-30)


plt.ylabel('frequency (Hz)')
plt.xlabel('time (s)')
plt.tight_layout()

#%% assa_azza

wlen_sec=32e-3
hop_percent=.5

wlen = int(wlen_sec*fs) # window length of 64 ms
wlen = np.int(np.power(2, np.ceil(np.log2(wlen)))) # next power of 2
nfft = wlen
hop = np.int(hop_percent*wlen) # hop size
win = np.sin(np.arange(.5,wlen-.5+1)/wlen*np.pi); # sine analysis window

x, fs = sf.read('/data/enseignement/2019-2020/electif_image_son/slides/cours_2/audio/assa_azza.wav')    

X = librosa.stft(x, n_fft=nfft, hop_length=hop, win_length=wlen, window=win) # STFT

plt.figure(figsize=(10,7))
librosa.display.specshow(librosa.power_to_db(np.abs(X)**2), sr=fs, hop_length=hop, x_axis='time', y_axis='hz')
plt.set_cmap('gray_r')
axes = plt.gca()
#axes.set_ylim([0,4000])

plt.colorbar()
plt.clim(vmin=-30)


plt.ylabel('frequency (Hz)')
plt.xlabel('time (s)')
plt.tight_layout()

#%% CS

wlen_sec=16e-3
hop_percent=.5

wlen = int(wlen_sec*fs) # window length of 64 ms
wlen = np.int(np.power(2, np.ceil(np.log2(wlen)))) # next power of 2
nfft = wlen
hop = np.int(hop_percent*wlen) # hop size
win = np.sin(np.arange(.5,wlen-.5+1)/wlen*np.pi); # sine analysis window

x, fs = sf.read('/data/enseignement/2019-2020/electif_image_son/slides/cours_2/audio/CS.wav')    

X = librosa.stft(x, n_fft=nfft, hop_length=hop, win_length=wlen, window=win) # STFT

plt.figure(figsize=(10,7))
librosa.display.specshow(librosa.power_to_db(np.abs(X)**2), sr=fs, hop_length=hop, x_axis='time', y_axis='hz')
plt.set_cmap('gray_r')
axes = plt.gca()
#axes.set_ylim([0,4000])

plt.colorbar()
#plt.clim(vmin=-30)


plt.ylabel('frequency (Hz)')
plt.xlabel('time (s)')
plt.tight_layout()

