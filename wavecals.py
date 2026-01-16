import os
import fnmatch
from datetime import datetime
import pickle

import numpy as np
import matplotlib.pyplot as mpl
from matplotlib.dates import MinuteLocator, DateFormatter
from scipy.signal import fftconvolve
from scipy.ndimage.interpolation import shift
from scipy.ndimage.filters import gaussian_filter
from gaussfitter import gaussfit
from . import jFits

from .climb import climb
from .register import SubPixelRegister

def get_spots_grid(f1, smoothed=False, plot=True, stride=30, 
                   start_offsets=(240,100), dims=(67,64)):
    print(start_offsets)
    gf1=gaussian_filter(f1,2)
    if plot:
        disp=jFits.jInteractive_Display(f1,vmin=-1,vmax=600)
    out_to_in_0=np.empty(dims)
    out_to_in_1=np.empty(dims)
    start=[start_offsets[1]-stride,start_offsets[0]-stride]
    for ii in range(dims[1]):
        start=(start[0],start[1]+int(stride))#increment column
        for ll in range(dims[0]):
            start=(int(stride)+start[0],start[1])#increment row
            print(start)
            gpeak_pix=climb(gf1,start)
            peak_pix=climb(f1,gpeak_pix)
            peak_pix=gpeak_pix
            out_to_in_0[ll,ii]=peak_pix[0]
            out_to_in_1[ll,ii]=peak_pix[1]
            if plot:
                disp.a.plot(peak_pix[1],peak_pix[0],'ro')
                mpl.show()
            start=(peak_pix[0],peak_pix[1])
        start=(out_to_in_0[0,ii]-int(stride),out_to_in_1[0,ii])#go back to first row
    return (out_to_in_0,out_to_in_1)

def spoof_spots(target, to_shift, direction='-+',
                best_spots_slice=(slice(920,1020),slice(900,1000))):
    """modified from Zack Briesemeister"""
    sl = best_spots_slice

    unsharp = (target[sl] - gaussian_filter(target[sl], .5))
    subreg = SubPixelRegister(unsharp)
    sh = subreg(to_shift[sl], 1000, direction=direction)
    return shift(to_shift, sh)

def shift_spots_to_anchor(target, wavecal_dict, target_filter='nb39',
                          direction='-+',
                          best_spots_slice=(slice(920,1020),slice(900,1000))):
    """modified from Zack Briesemeister
    the "direction" keyword should encode the known sign of the entries to
    the "shift" parameter fed to scipy.ndimage.interpolation.shift"""
    sl = best_spots_slice

    unsharp = (target[sl] - gaussian_filter(target[sl], .5))
    subreg = SubPixelRegister(unsharp)
    sh = subreg(wavecal_dict[target_filter][sl], 1000, direction=direction)
    sh_dict = {}
    for nbfilt in wavecal_dict.keys:
        sh_dict[nbfilt] = shift(wavecal_dict[nbfilt], sh)
    return sh_dict

def w_func(px, A, B, C):
    """functional form of wavelength solution, inverted for ease of use"""
    return A * (px - B)**.5 + C


def onedmoments(profile):
    """calculates the first and second moment of a 1D profile

    Returns:
    std    : standard devation
    mu     : centroid
    A      : Amplitude
    H      : Height
    """
    data = np.copy(profile)
    xs = np.arange(len(data))
    H = data.min()
    A = data.max() - data.min()
    data -= H
    data /= data.sum()
    mu = xs.dot(data)
    mom2 = np.power(xs, 2).dot(data)
    var = mom2 - mu**2
    return np.sqrt(var), mu, A, H

def help_nbs(nb_dict):
    bg = np.median([nb_dict['nb29'], nb_dict['nb33'], nb_dict['nb35'], nb_dict['nb39']], axis=0)
    out = {}
    for k in nb_dict:
        out[k] = nb_dict[k]
    nb29_ds = nb_dict['nb29'] - bg
    #nb29_ds = nb_dict['nb29'] - nb_dict['nb35']
    #ave = np.mean(nb29_ds[4:150, 4:-4])
    #sig = np.std(nb29_ds[4:150, 4:-4])
    #inds = nb29_ds < (ave - 3.5*sig)
    #filler = np.random.randn(inds.sum())
    #nb29_ds[inds] = filler
    out['nb29'] = nb29_ds

    nb33_ds = nb_dict['nb33'] - bg
    #nb33_ds = nb_dict['nb33'] - nb_dict['nb29']
    #ave = np.mean(nb33_ds[4:150, 4:-4])
    #sig = np.std(nb33_ds[4:150, 4:-4])
    #inds = nb33_ds < (ave - 3.5*sig)
    #filler = np.random.randn(inds.sum())
    #nb33_ds[inds] = filler
    out['nb33'] = nb33_ds

    nb35_ds = nb_dict['nb35'] - bg 
    #nb35_ds = nb_dict['nb35'] - nb_dict['nb29']
    #ave = np.mean(nb35_ds[4:150, 4:-4])
    #sig = np.std(nb35_ds[4:150, 4:-4])
    #inds = nb35_ds < (ave - 3.5*sig)
    #filler = np.random.randn(inds.sum())
    #nb35_ds[inds] = filler
    out['nb35'] = nb35_ds

    nb39_ds = nb_dict['nb39'] - bg
    #nb39_ds = nb_dict['nb39'] - nb_dict['nb33']
    #ave = np.mean(nb39_ds[4:150, 4:-4])
    #sig = np.std(nb39_ds[4:150, 4:-4])
    #inds = nb39_ds < (ave - 3.5*sig)
    #filler = np.random.randn(inds.sum())
    #nb39_ds[inds] = filler
    out['nb39'] = nb39_ds

    return out



