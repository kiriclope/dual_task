import os, sys, importlib 

import numpy as np
import matplotlib 
matplotlib.use('GTK3cairo') 

import matplotlib.pyplot as plt 
from scipy.ndimage import uniform_filter1d
from scipy.signal import savgol_filter

from sklearn.preprocessing import StandardScaler

from joblib import Parallel, delayed 
from oasis.functions import deconvolve
from sklearn.feature_selection import VarianceThreshold 

from . import constants as gv 
from . import get_data as data 
from . import preprocessing as pp 
from . import plot_utils as pl 
from . import plot_settings as pl_set

from .options import * 
from .get_days import *

def set_title_labels():
    # if gv.DCV_THRESHOLD:
    #     figtitle = 'deconvolve'
    #     ylabel = 'Deconvolve Activity'
    if gv.Z_SCORE:
        figtitle = 'z_score'
        ylabel = 'Z-score'
    elif gv.Z_SCORE_BL:
        figtitle = 'z_score_baseline' 
        ylabel = '$Z-score_{Baseline}$'
    elif gv.Z_SCORE_TRIALS: 
        figtitle = 'z_score_baseline_trials' 
        ylabel = '$Z-score_{Trials}$'
    else:
        figtitle = 'raw_fluo'
        ylabel = 'Fluorescence (a.u.)'
        
    return figtitle, ylabel 

def create_figdir(**kwargs): 
    globals().update(kwargs) 
    pl.figDir() 
    
    gv.figdir = gv.figdir + '/n_days_%d/fluo_traces/%s' % (n_days, gv.mouse) 
    
    if not os.path.isdir(gv.figdir): 
        os.makedirs(gv.figdir)
        print(gv.figdir) 
        
def FluoTrace(day='all', mean=0, i_neuron=None, seed=None, **kwargs): 
    # set options and get globals 
    options = set_options(**kwargs) 
    set_globals(**options) 

    trial_type = options['trial_type'] 
    
    if trial_type == 'correct': 
        trial_type = 1 
    elif trial_type == 'incorrect': 
       trial_type = 2 
    else: 
       trial_type = 0
    
    data.get_days() 
    
    X_days, y_days = get_X_y_days(day=day, stimulus='sample')
    create_figdir(**options) # must come after getAllDays() 
    
    y_task = y_days[i_task] 
    trial_list_S1 = np.nonzero( np.in1d(y_task[0][0], y_task[trial_type][0]))[0] 
    trial_list_S2 = np.nonzero( np.in1d(y_task[0][1], y_task[trial_type][1]))[0]
    
    X_S1 = X_days[i_task, 0][trial_list_S1] 
    X_S2 = X_days[i_task, 1][trial_list_S2] 
    
    X_S1_S2 = np.vstack( (X_S1, X_S2) ) 
    X_task = pp.preprocess_X(X_S1_S2)    
    
    print('mouse', gv.mouse, 'day', gv.day, 'task', gv.task_str[options['i_task']], 'X_task', X_task.shape) 
    
    # set trial, neuron and seed 
    if seed is None:
        seed = np.random.randint(0, 1e6) 
    np.random.seed(seed)
    
    if i_neuron is None:
        i_neuron = np.random.randint(0, X_task.shape[1])

    figtitle, ylabel = set_title_labels() 
    if mean==0:
        figtitle += '_neuron_%d' % i_neuron
        title = 'neuron %d' % i_neuron 
    else:
        figtitle = 'population_' + figtitle 
        title = 'population average'
        X_task = np.nanmean(X_task, axis=1) # avg over neurons
        
    plt.figure(figtitle, figsize=pl.set_size(200)) 
    plt.title(title) 
    plt.xlabel('Time (s)') 
    plt.xticks([0,2,4,6,8,10,12,14])    
    plt.ylabel(ylabel) 
    pl.add_vlines() 
        
    for i_trial in range(5):
        # i_trial = np.random.randint(0, X_task.shape[0]) 
        if mean==0: 
            plt.plot(gv.time, X_task[i_trial, i_neuron], 'k', alpha=0.1, color=gv.pal[i_task]) 
        else: 
            plt.plot(gv.time, X_task[i_trial], 'k', alpha=0.1, color=gv.pal[i_task]) 
            
    # plot median over trials and percentiles 
    X_task_avg = np.nanmean(X_task, axis=0) 
    
    alpha = 0.95 
    p = ((1.0-alpha)/2.0)*100 
    lower = np.percentile(X_task, p, axis=0) 
    
    p = ( alpha + (1.0-alpha)/2.0)*100 
    upper = np.percentile(X_task, p, axis=0)
    
    # X_task_avg = np.nanmean(X_task, axis=0) 
    # ci = 1.96 * np.std(X_task, axis=0)/np.nanmean(X_task, axis=0) 
    # lower = X_task_avg - ci
    # upper = X_task_avg + ci 
    
    if mean==0: 
        plt.plot(gv.time,  X_task_avg[i_neuron], color=gv.pal[i_task], label='raw') 
        plt.fill_between(gv.time, lower[i_neuron], upper[i_neuron], color=gv.pal[i_task], alpha=.1) 
    else: 
        plt.plot(gv.time,  X_task_avg, color=gv.pal[i_task], label='raw') 
        plt.fill_between(gv.time, lower, upper, color=gv.pal[i_task], alpha=.1) 
    
    pl.save_fig(figtitle) 

def FluoDist(day=None, mean=0, i_neuron=None, seed=None, **kwargs): 
    # set options and get globals 
    options = set_options(**kwargs) 
    set_globals(**options) 
    
    data.get_days() 
    data.get_stimuli_times() 
    data.get_delays_times() 
    
    X = getAllDays(day=day)
    create_figdir(**options) # must come after getAllDays() 
    
    X = np.swapaxes(X, 0, 1) 
    X = np.hstack(X)
    
    X = X[i_task] 
    
    print('mouse', gv.mouse, 'day', gv.day, 'task', gv.task_str[options['i_task']], 'X', X.shape) 

    if gv.SAVGOL:
        X = savgol_filter(X, int(np.ceil(gv.frame_rate / 2.) * 2 + 1), polyorder = gv.SAVGOL_ORDER, deriv=0, axis=-1)
    
    BL = X[..., gv.bins_BL] 
    F0 = np.nanmean(BL, axis=-1) 
    
    print(gv.bins_BL) 
    
    trial = np.random.randint(0, F0.shape[0])
    
    print(trial, X.shape, X[trial, 0:5, 0])
    print(trial, F0.shape, F0[trial, 0:5])
    
    plt.figure('single_trial_F0')
    plt.hist(F0[trial], int(F0.shape[1]/10)) 
    plt.axvline(np.nanmean(F0[trial]), c='k', ls='-') 
    plt.ylabel('# neurons') 
    plt.xlabel('F0') 

    avgF0 = np.nanmean(F0, axis=0)
    
    plt.figure('trial_averaged_F0')
    plt.hist(avgF0, int(avgF0.shape[0]/10)) 
    plt.axvline(np.nanmean(avgF0), c='k', ls='-') 
    plt.ylabel('# neurons') 
    plt.xlabel('$<F0>_{trials}$') 

    gv.DCV_THRESHOLD=0 
    rates = pp.deconvolveFluo(X) 
    ratesBL = np.nanmean(rates[..., gv.bins_BL], axis=-1) 

    print(trial, rates.shape, rates[trial, 0:5, 0])
    print(trial, ratesBL.shape, ratesBL[trial, 0:5])
    
    plt.figure('single_trial_dcv')
    plt.hist(ratesBL[trial], int(ratesBL.shape[1]/10)) 
    plt.axvline(np.nanmean(ratesBL[trial]), c='k', ls='-') 
    plt.ylabel('# neurons') 
    plt.xlabel('$r_0$') 

    avgRatesBL= np.nanmean(ratesBL, axis=0)
    
    plt.figure('trial_averaged_dcv')
    plt.hist(avgRatesBL, int(avgRatesBL.shape[0]/10)) 
    plt.axvline(np.nanmean(avgRatesBL), c='k', ls='-') 
    plt.ylabel('# neurons') 
    plt.xlabel('$<r_0>_{trials}$') 
    
