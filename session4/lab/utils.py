import os, sys
import numpy as np
import matplotlib.pyplot as plt

def power_to_db(V, amin=1e-10, top_db=80.0):
    """
    Taken from https://librosa.github.io/librosa/generated/librosa.core.power_to_db.html
    
    Essentially computes a power spectrogram in dB.
    """
    
    ref = np.max(V)
    V_dB = 10.0 * np.log10(np.maximum(amin, V))
    V_dB -= 10.0 * np.log10(np.maximum(amin, ref))
    V_dB = np.maximum(V_dB, V_dB.max() - top_db)
    return V_dB

def amp_to_db(V, amin=1e-10, top_db=80.0):
    """
    Taken from https://librosa.github.io/librosa/generated/librosa.core.power_to_db.html
    
    Essentially computes a power spectrogram in dB.
    """
    return power_to_db(V**2, amin, top_db)
    
def plot_NMF(V, W, H, V_hat, error, figsize=(10,2), aspect='auto', wr=[1, 0.5, 1, 1], power_spec=False): 
    """
    Plot NMF.
    
    Args:
        'V': the original data matrix
        'W': the dictionary matrix
        'H': the activation matrix
        'V_hat': the reconstructed data matrix
        'error': value of the IS divergence that will be put in the title of the plot.
        'figsize': size of the figure
        'aspect': controls the aspect ratio of the axes (see matplotlib.axes.Axes.imshow)
        'wr': width ratios of the columns.
        'power_spec': must be set to True if V corresponds to a power spectrogram.
    
    """
    if power_spec:
        V = power_to_db(V)
        V_hat = power_to_db(V_hat)
        W = power_to_db(W)
        H = H**(1/4) # reduces the dynamic for better visualization
    
    fig, ax = plt.subplots(1, 4, gridspec_kw={'width_ratios': wr}, figsize=figsize)    
    
    K = W.shape[1]
    cmap = 'gray_r'
    # im = ax[0].imshow(V, aspect=aspect, origin='lower', cmap=cmap, clim=[0, np.max(V)])
    im = ax[0].imshow(V, aspect=aspect, origin='lower', cmap=cmap)
    ax[0].set_title(r'$V$')
    plt.sca(ax[0])
    plt.colorbar(im)   
    # im = ax[1].imshow(W, aspect=aspect, origin='lower', cmap=cmap, clim=[0, np.max(W)])
    im = ax[1].imshow(W, aspect=aspect, origin='lower', cmap=cmap)
    ax[1].set_title(r'$W$')
    plt.sca(ax[1])
    plt.colorbar(im)
    plt.xticks(np.arange(K), np.arange(1, K+1))
    # im = ax[2].imshow(np.flip(H, axis=0), aspect=aspect, origin='lower', cmap=cmap, clim=[0, np.max(H)])
    im = ax[2].imshow(np.flip(H, axis=0), aspect=aspect, origin='lower', cmap=cmap)
    ax[2].set_title(r'$H$')
    plt.sca(ax[2])    
    plt.colorbar(im)
    plt.yticks(np.arange(K), np.arange(K, 0, -1))
    # im = ax[3].imshow(V_hat, aspect=aspect, origin='lower', cmap=cmap, clim=[0, np.max(V_hat)])
    im = ax[3].imshow(V_hat, aspect=aspect, origin='lower', cmap=cmap)
    ax[3].set_title(r'$WH$ (error = %0.2f)'%error)
    plt.sca(ax[3])    
    plt.colorbar(im)
    plt.tight_layout() 
    plt.show() 
    
