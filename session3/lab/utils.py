#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Provides utilities for running nussl algorithms that do not belong to
any specific algorithm or that are shared between algorithms.
"""

from __future__ import division
import warnings
import numpy as np
import matplotlib.pyplot as plt
import soundfile as sf 
import librosa

def anechoic_FD_auralization(s, a=1, delta=0, wlen=512, hop=256, win='hann'):

    T = s.shape[0] # signal length
    S = librosa.stft(s, n_fft=wlen, hop_length=hop, win_length=wlen, window=win) # STFT of the source signal
    F, N = S.shape
    
    X = np.zeros((F, N, 2), dtype='complex') # STFT of the microphone signals
    
    ########## TO COMPLETE ###########
      
    freq_vec = np.arange(F) # f = [0,...,F-1]  
    H = a*np.exp(-1j*2*np.pi*delta*freq_vec/wlen)
    X[:,:,0] = S
    X[:,:,1] = S*H[:,np.newaxis]
    
    ##################################
    
    x = np.zeros((T,2)) 
    
    # iSTFT to get the time-domain microphone signals
    x[:,0] = librosa.istft(X[:,:,0], hop_length=hop, win_length=wlen, window=win, length=T)
    x[:,1] = librosa.istft(X[:,:,1], hop_length=hop, win_length=wlen, window=win, length=T)

    return x

def create_mixture(anechoic_FD_auralization, source_wavefiles, a, delta, wlen, hop, win):
        
    stereo_sources = []
    
    for j, wavefile in enumerate(source_wavefiles):
        
        s, fs = sf.read(wavefile) # mono source signal
        s = s/np.max(np.abs(s)) # normalize
        
        sim = anechoic_FD_auralization(s, a[j], delta[j], wlen, hop, win) # stereo source image
        stereo_sources.append(sim)

    T = np.max([sim.shape[0] for sim in stereo_sources]) # number of samples longest source signal
    
    x = np.zeros((T, 2)) # mixture signal
    for sim in stereo_sources:
        x[:sim.shape[0]] += sim
        
    return x

def plot_recording_config(q_m1, q_m2, q_s):
    # A function to plot the recording configuration given the source and microphone coordinates
    
    if len(q_s.shape)>1:
        n_src = q_s.shape[1]
    else:
        n_src = 1         
        q_s = q_s[:,np.newaxis]
    
    plt.plot(q_m1[0], q_m1[1], 'o')
    plt.plot(q_m2[0], q_m2[1], 'o')
    
    legend = ['mic1', 'mic2']
    
    for j in np.arange(n_src):

        plt.plot(q_s[0,j], q_s[1,j], 'x')
        legend.append('source' + str(j+1))
        
    plt.xlim([-1, 1])
    plt.ylim([-1, 1])
    plt.xlabel('x coordinate (m)', fontsize=15)
    plt.ylabel('y coordinate (m)', fontsize=15)
    plt.title('recording configuration', fontsize=15)
    plt.legend(legend, loc='lower right')


def find_peak_indices(input_array, n_peaks, min_dist=None, do_min=False, threshold=0.5):
    """
    This function will find the indices of the peaks of an input n-dimensional numpy array.
    This can be configured to find max or min peak indices, distance between the peaks, and
    a lower bound, at which the algorithm will stop searching for peaks (or upper bound if
    searching for max). Used exactly the same as :func:`find_peak_values`.

    This function currently only accepts 1-D and 2-D numpy arrays.

    Notes:
        * This function only returns the indices of peaks. If you want to find peak values,
        use :func:`find_peak_values`.

        * min_dist can be an int or a tuple of length 2.
            If input_array is 1-D, min_dist must be an integer.
            If input_array is 2-D, min_dist can be an integer, in which case the minimum
            distance in both dimensions will be equal. min_dist can also be a tuple if
            you want each dimension to have a different minimum distance between peaks.
            In that case, the 0th value in the tuple represents the first dimension, and
            the 1st value represents the second dimension in the numpy array.


    See Also:
        :: :func:`find_peak_values` ::

    Args:
        input_array: a 1- or 2- dimensional numpy array that will be inspected.
        n_peaks: (int) maximum number of peaks to find
        min_dist: (int) minimum distance between peaks. Default value: len(input_array) / 4
        do_min: (bool) if True, finds indices at minimum value instead of maximum
        threshold: (float) the value (scaled between 0.0 and 1.0)

    Returns:
        peak_indices: (list) list of the indices of the peak values

    """
    input_array = np.array(input_array, dtype=float)

    if input_array.ndim > 2:
        raise ValueError('Cannot find peak indices on data greater than 2 dimensions!')

    is_1d = input_array.ndim == 1
    zero_dist = zero_dist0 = zero_dist1 = None
    min_dist = len(input_array) // 4 if min_dist is None else min_dist

    if is_1d:
        zero_dist = min_dist
    else:
        if type(min_dist) is int:
            zero_dist0 = zero_dist1 = min_dist
        elif len(min_dist) == 1:
            zero_dist0 = zero_dist1 = min_dist[0]
        else:
            zero_dist0, zero_dist1 = min_dist

    # scale input_array between [0.0, 1.0]
    if np.min(input_array) < 0.0:
        input_array += np.min(input_array)
    elif np.min(input_array) > 0.0:
        input_array -= np.min(input_array)

    input_array /= np.max(input_array)

    # flip sign if doing min
    input_array = -input_array if do_min else input_array

    # throw out everything below threshold
    input_array = np.multiply(input_array, (input_array >= threshold))

    # check to make sure we didn't throw everything out
    if np.size(np.nonzero(input_array)) == 0:
        raise ValueError('Threshold set incorrectly. No peaks above threshold.')
    if np.size(np.nonzero(input_array)) < n_peaks:
        warnings.warn('Threshold set such that there will be less peaks than n_peaks.')

    peak_indices = []
    for i in range(n_peaks):
        # np.unravel_index for 2D indices e.g., index 5 in a 3x3 array should be (1, 2)
        # Also, wrap in list for duck typing
        cur_peak_idx = list(np.unravel_index(np.argmax(input_array), input_array.shape))

        # zero out peak and its surroundings
        if is_1d:
            cur_peak_idx = cur_peak_idx[0]
            peak_indices.append(cur_peak_idx)
            lower, upper = _set_array_zero_indices(cur_peak_idx, zero_dist, len(input_array))
            input_array[lower:upper] = 0
        else:
            peak_indices.append(cur_peak_idx)
            lower0, upper0 = _set_array_zero_indices(cur_peak_idx[0], zero_dist0,
                                                     input_array.shape[0])
            lower1, upper1 = _set_array_zero_indices(cur_peak_idx[1], zero_dist1,
                                                     input_array.shape[1])
            input_array[lower0:upper0, lower1:upper1] = 0

        if np.sum(input_array) == 0.0:
            break

    return peak_indices


def _set_array_zero_indices(index, zero_distance, max_len):
    lower = index - zero_distance - 1
    upper = index + zero_distance + 1
    lower = 0 if lower < 0 else lower
    upper = max_len if upper >= max_len else upper
    return lower, upper


def find_peak_values(input_array, n_peaks, min_dist=None, do_min=False, threshold=0.5):
    """
    Finds the values of the peaks in a 1-D or 2-D numpy array. Used exactly the same as
    :func:`find_peak_indices`. This function will find the values of the peaks of an input
    n-dimensional numpy array.

    This can be configured to find max or min peak values, distance between the peaks, and
    a lower bound, at which the algorithm will stop searching for peaks (or upper bound if
    searching for max).

    This function currently only accepts 1-D and 2-D numpy arrays.

    Notes:
        * This function only returns the indices of peaks. If you want to find peak values,
        use :func:`find_peak_indices`.

        * min_dist can be an int or a tuple of length 2.
            If input_array is 1-D, min_dist must be an integer.
            If input_array is 2-D, min_dist can be an integer, in which case the minimum
            distance in both dimensions will be equal. min_dist can also be a tuple if
            you want each dimension to have a different minimum distance between peaks.
            In that case, the 0th value in the tuple represents the first dimension, and
            the 1st value represents the second dimension in the numpy array.


    See Also:
        :: :func:`find_peak_indices` ::

    Args:
        input_array: a 1- or 2- dimensional numpy array that will be inspected.
        n_peaks: (int) maximum number of peaks to find
        min_dist: (int) minimum distance between peaks. Default value: len(input_array) / 4
        do_min: (bool) if True, finds indices at minimum value instead of maximum
        threshold: (float) the value (scaled between 0.0 and 1.0)

    Returns:
        peak_values: (list) list of the values of the peak values

    """
    if input_array.ndim > 2:
        raise ValueError('Cannot find peak indices on data greater than 2 dimensions!')

    if input_array.ndim == 1:
        return [input_array[i] for i in find_peak_indices(input_array, n_peaks, min_dist,
                                                          do_min, threshold)]
    else:
        return [input_array[i, j] for i, j in find_peak_indices(input_array, n_peaks, min_dist,
                                                                do_min, threshold)]
