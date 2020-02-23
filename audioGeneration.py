#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 18 09:47:29 2019

@author: luozhidan
"""

# =============================================================================
# CPE 7993 - Independent study
# Project: Generation of non-repetitive sound effects
# =============================================================================

import matplotlib.pyplot as plt
import numpy as np
import peakutils
import pywt
from scipy.io import wavfile
from sklearn.decomposition import PCA

def plot(sig, title = None, data = None):   
    
    if data is None: pass
    else: plt.scatter(data[:,0],data[:,1],c = 'red',marker = 'x') 
      
    plt.plot(sig)
    plt.xlabel("Data length")
    plt.ylabel("Normalized amplitude")
    plt.title(title)
    my_x_ticks = np.linspace(0, len(sig), 5)
    plt.xticks(my_x_ticks)
    plt.grid(True)
    plt.show()

def specgramplot(sig, f):
    plt.specgram(sig, Fs = f)
    plt.xlabel("time(second)")
    plt.ylabel("Frequency(Hz)")
    plt.title("Spectrogram")
    plt.show()
    
def load_audio(filename, show = False):
    rate, sig = wavfile.read(filename)
    # normalization
    nsig = sig/np.max(np.abs(sig))
    
    if show == True:
        # plot audio
        plot(nsig, 'Normalized Audio Signal')
    return [nsig, rate]

def find_peaks(sig, num_peak):
    indexes = peakutils.peak.indexes(sig, thres=0.8, min_dist=len(sig)/num_peak)
    magnitudes = sig[indexes]
    peakdata = np.array([indexes, magnitudes]).T
    plot(sig, 'Peak Detection', data = peakdata)
    return indexes

def divide(sig, peak_idx, buffer = 1000, show = False):
    # sound_events = np.array([])
    # segmentation points
    seg_pts = np.append(peak_idx - buffer, len(sig))
    # lengths between peaks
    peak_diff = np.diff(seg_pts)
    # find maximum length
    max_diff = np.max(np.append(peak_diff, peak_idx[0]))
    events = np.zeros((max_diff, len(peak_idx)))
   
    for i in range(len(peak_idx)):
        tmp = sig[seg_pts[i]:seg_pts[i+1]]
        # add 0 to make equal lengths
        n_diff = max_diff - len(tmp)
        data_i = np.append(tmp, np.zeros(n_diff))
        # record data_i into ith column of events
        events[:,i] = data_i
    
    if show == True:
        num_show = 2
        idx = np.random.randint(0,len(peak_idx),num_show)
        for i in range(len(idx)):
            plot(events[:,idx[i]], '$%dth$ sound events' % (idx[i]+1))
        print('\n%d random sound events are shown\n' % num_show)       
    return events

def get_feature(events, wname = 'db5'):
    wavelet = pywt.Wavelet(wname)
    level = pywt.dwt_max_level(len(events[:,0]), wavelet)
    print("DWT by wavelet = '%s' and level = %d\n" % (wname, level))
    
    for k in range(np.size(events,1)):
        coeff = pywt.wavedec(events[:,k], wavelet, level = level)
        feature_i = []       
        for i in range(level+1):
            feature_i = np.append(feature_i,coeff[i])    
            
        if k == 0:
            feature = feature_i
            length = []
            for j in range(len(coeff)):
                length.append(len(coeff[j]))               
        else:
            feature = np.column_stack((feature, feature_i))
       
    return [feature, length]
        
def pca(feature, explained):
    ipca = PCA(n_components = explained)
    p_components = ipca.fit_transform(feature)
    n_components = ipca.n_components_
    ratio = sum(ipca.explained_variance_ratio_)
    return [p_components, n_components, ratio]
        
def morph(feature_old, bw):
    randnum = np.random.uniform(-bw,bw,size=(np.shape(feature_old)))
    feature_tmp = feature_old + np.abs(feature_old)*randnum
    feature_new = np.mean(feature_tmp, axis=1)
    return feature_new
    
def morph_feature(information, explained = 0.90, bw = 0.01, show = False):
    feature = information[0]
    length = information[1]    
    nlength = np.insert(length,0,0)
    idx = np.cumsum(nlength)
    inf_new = np.array([])    
    
    for i in range(len(idx)-1):
        sample = feature[idx[i]:idx[i+1], : ]
        p_components, n_components, ratio = pca(sample, explained)    
        feature_new = morph(p_components, bw)
        inf_new = np.append(inf_new, feature_new)
        
        if show == True:
            if i == 0:
                print('Level-%d approximation coefficient is reduced from %d to %d'\
                      % (len(length)-1, np.size(sample,1), n_components))
                print('The amount of variance explained: %.2f%%\n' % (100*ratio))
            else:
                print('Level-%d detail coefficient is reduced from %d to %d'\
                      % (len(length)-i, np.size(sample,1), n_components))
                print('The amount of variance explained: %.2f%%\n' % (100*ratio))
                
    return [inf_new, length]

def arr2list(arr, length):
    nlength = np.insert(length,0,0)
    idx = np.cumsum(nlength)
    mylist = []    
    for i in range(len(idx)-1):
        list_tmp = arr[idx[i]:idx[i+1]]
        mylist.append(list_tmp)
    return mylist

def reconstruct(new_inf, f, wname = 'db5', show = False):
    arr = new_inf[0]
    length = new_inf[1]
    coeffs = arr2list(arr, length)
    wavelet = pywt.Wavelet(wname)
    new_event = pywt.waverec(coeffs, wavelet)
    norm = new_event/np.max(np.abs(new_event))
    
    if show == True:
        plot(norm, 'Reconstructed Sound Event')
        specgramplot(norm, f = f)
    return norm

def synthesize(feature, f, bw = 0.1, show = False):
    n = len(bw)
    audio = []
    for i in range(n):        
        # morph feature
        new_inf = morph_feature(feature, explained = 0.90, bw = bw[i], show = show)
        # get new sound events
        new_event = reconstruct(new_inf, f = f, show = show)
        # synthesize
        audio = np.append(audio, new_event)
    plot(audio, 'Synthesized audio')
    specgramplot(audio, f = f)
    return audio
 
    
if __name__ == "__main__":
    sig, rate = load_audio('sound samples/TIC-fing-50.wav')
    # num_peak general should set to greater than expected values
    peak_idx = find_peaks(sig, num_peak = 80)
    # get sound events
    events = divide(sig, peak_idx, show = True)
    # get feature based on dwt
    feature = get_feature(events, wname = 'db5')   
    # Morphing part
    bandwidth = [0.1, 0.5, 1, 5, 10]
    audio = synthesize(feature, f = rate, bw = bandwidth, show = False)

 








