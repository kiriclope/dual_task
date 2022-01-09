from importlib import reload
import inspect, sys
import numpy as np 
import matplotlib.pyplot as plt

sys.path.insert(0, '../')

import utils.constants as gv 
reload(gv)
from utils.options import *

import utils.get_data as data
reload(data)
from utils.get_days import * 

import utils.preprocessing as pp
reload(pp)
import utils.plot_utils as pl 
reload(pl)

from senc.utils import * 
from senc.plot_utils import * 
from senc.statistics import * 

def stimulus_axis(**kwargs):
    options = set_options(**kwargs) 
    set_globals(**options) 
    
    create_figdir(**options) 
    
    X_S1, X_S2 = get_X_S1_X_S2_days_task(day=options['day'], stimulus=options['stimulus'], task=options['task'], trials=options['trials']) 
    
    # standardize the data 
    X_S1_S2 = np.vstack((X_S1, X_S2)) 
    mean = np.mean(X_S1_S2, axis=0) 
    std = np.std(X_S1_S2, axis=0) 
    
    X_S1 = (X_S1 - mean) / std 
    X_S2 = (X_S2 - mean) / std 
    
    print('X_S1', X_S1.shape, 'X_S2', X_S2.shape) 
    X_S1_epochs = pp.avg_epochs(X_S1) 
    X_S2_epochs = pp.avg_epochs(X_S2) 
    
    # X_S1_S2_epochs = np.vstack((X_S1_epochs, X_S2_epochs)) 
    # mean = np.mean(X_S1_S2_epochs, axis=0) 
    # std = np.std(X_S1_S2_epochs, axis=0) 
    
    # X_S1_epochs = (X_S1_epochs - mean) / std 
    # X_S2_epochs = (X_S2_epochs - mean) / std 
    
    dX = get_coding_direction(X_S1_epochs, X_S2_epochs, options['pval']) 
    print('coding direction', dX.shape) 
    return dX

def stimulus_axis_day(**kwargs):
    options = set_options(**kwargs) 
    set_globals(**options) 
    
    create_figdir(**options) 
    print(options['trials']) 
    
    X_S1, X_S2 = get_X_S1_X_S2_day_task(day=options['day'], stimulus=options['stimulus'], task=options['task'], trials=options['trials']) 
    print(options['trials']) 
    
    # # standardize the data 
    X_S1_S2 = np.vstack((X_S1, X_S2)) 
    mean = np.mean(X_S1_S2, axis=0) 
    std = np.std(X_S1_S2, axis=0) 
    
    X_S1 = (X_S1 - mean) / std 
    X_S2 = (X_S2 - mean) / std 
    
    print('X_S1', X_S1.shape, 'X_S2', X_S2.shape)
        
    X_S1_epochs = pp.avg_epochs(X_S1) 
    X_S2_epochs = pp.avg_epochs(X_S2) 
    
    # X_S1_S2_epochs = np.vstack((X_S1_epochs, X_S2_epochs)) 
    # mean = np.mean(X_S1_S2_epochs, axis=0) 
    # std = np.std(X_S1_S2_epochs, axis=0) 
    
    # X_S1_epochs = (X_S1_epochs - mean) / std 
    # X_S2_epochs = (X_S2_epochs - mean) / std 

    dX = get_coding_direction(X_S1_epochs, X_S2_epochs, kwargs['pval']) 
    print('coding direction', dX.shape) 
    return dX, X_S1_epochs, X_S2_epochs 

def sel_time(**kwargs):
    # gv.tasks = ['DPA', 'DualGo', 'DualNoGo'] 
    
    options = set_options(**kwargs) 
    set_globals(**options)
    
    create_figdir(**options)
    
    data.get_days() # do not delete that !!
    if options['obj']=='norm':
        gv.epochs = ['ED','MD','LD','BL']
    else:
        gv.epochs = ['ED','MD','LD'] 
                
    figtitle = 'Sample_Sel_%s_time_day_%s_%s_trials' % ( options['obj'], str(options['day']), options['trial_type'] ) 
        
    X_S1, X_S2 = get_X_S1_X_S2_days_task(day=options['day'], stimulus=options['stimulus'], task=options['task'], trials=options['trials'])

    if options['stimulus']=='sample':
        options['bins'] = gv.bins_ED
    else:
        options['bins'] = gv.bins_cue 
        
    print('bins', options['bins']) 

    sel = get_sel(X_S1, X_S2, options['obj'], options['pval'], bins=options['bins'], standardize=options['standardize']) 
    # print('sel', sel.shape) 
    
    if options['obj']=='norm':
        bins_BL = np.arange(0, 12) 
        norm_sel_BL = np.nanmean( sel[bins_BL] )
        if options['stimulus']=='sample':
            norm_sel_ED = np.nanmean( sel[gv.bins_ED] )
        else:
            print(gv.bins_cue)
            norm_sel_ED = np.nanmean( sel[gv.bins_cue] )
        
        norm_sel = (sel-norm_sel_BL) / np.amax(sel) 
        # norm_sel = (sel-norm_sel_BL) / norm_sel_BL 
        plt.plot(gv.time, norm_sel, color=gv.pal[i_task]) 
    else:
        plt.plot(gv.time, sel, color=gv.pal[i_task])
        
        #     # if options['obj']=='norm':
        #     #     # ci_BL = np.nanmean(ci[gv.bins_BL, :], axis=0) 
        #     #     # ci = ci / ci_BL + norm_sel[:, np.newaxis] 
        #     #     ci = ci / np.absolute(norm_sel_BL) + norm_sel[:, np.newaxis] 
        #     #     # ci = (ci-norm_sel_BL) / norm_sel_BL 
        #     #     # plt.fill_between(gv.time, ci[:,0], ci[:,1], alpha=0.1)
        #     #     print('norm_sel', norm_sel[gv.bins_ED], 'ci', ci[gv.bins_ED]) 
        #     #     plt.fill_between(gv.time, norm_sel - ci[:,0], norm_sel + ci[:,1], alpha=0.1) 
        #     # else: 
        #     # ci = ci + sel[:, np.newaxis] 
        #     # plt.fill_between(gv.time, ci[:,0], ci[:,1], alpha=0.1)  
        #     plt.fill_between(gv.time, sel-ci[:,0], sel+ci[:,1], alpha=0.1) 
        
    # get shuffle 
    if options['obj']!= 'auc':
        print('shuffle statistics') 
        sel_shuffle = shuffle_stat(X_S1, X_S2,
                                   lambda x,y: get_sel(x, y, options['obj'],
                                                       options['pval'],
                                                       bins=options['bins'],
                                                       standardize=options['standardize']), 
                                   n_samples=options['n_samples'] ) 
        
    if options['obj']=='norm': 
        norm_sel_BL_shuff = np.nanmean( sel_shuffle[:, bins_BL], axis=-1 ) 
        # sel_shuffle = ( sel_shuffle - norm_sel_BL ) / norm_sel_BL
        sel_shuffle = ( sel_shuffle - norm_sel_BL_shuff[:, np.newaxis] ) / np.amax(sel)
        # sel_shuffle = ( sel_shuffle - norm_sel_BL_shuff[:, np.newaxis] ) / norm_sel_BL_shuff[:, np.newaxis] 
            
    print('sel_shuffle', sel_shuffle.shape) 
    mean = np.nanmean(sel_shuffle, axis=0) 
    std = np.nanpercentile(sel_shuffle, [2.5, 97.5], axis=0)          
    plt.plot(gv.time, mean, '--' , color=gv.pal[i_task])         
    plt.fill_between(gv.time, std[0], std[1], color=gv.pal[i_task], alpha=.1) 
    
    if options['stimulus']=='distractor' or options['task']=='dual':
        if options['i_mice']==4:
            pl.add_vlines()
    else:
        pl.add_vlines()
    
    plt.xlabel('Time (s)') 
    plt.xticks([0,2,4,6,8,10,12,14])
        
    if options['obj']=='norm':
        if options['stimulus']=='distractor': 
            plt.ylabel('Distractor Sel.') 
        else: 
            plt.ylabel('Sample Sel.') 
        
        # plt.ylim([-.5, 1.5]) 
        # plt.yticks([-0.5, 0, .5, 1, 1.5]) 
    if options['obj']=='cos':
        plt.ylabel('cos($\\alpha$)') 
        plt.ylim([-.25, 1.25]) 
        plt.yticks([0, .25, .5, .75, 1]) 
    if options['obj']=='frac':
        plt.ylabel('Frac. Selective') 
        plt.ylim([-.1, 0.4]) 
    if options['obj'] == 'auc':
        plt.ylabel('AUC') 
        # plt.ylim([0, .2]) 
        # plt.yticks([0, .1, .2, .3, .4, .5]) 
    
    # pl.save_fig(figtitle) 
    # plt.close('all') 

def performance_day(**kwargs): 
    options = set_options(**kwargs) 
    set_globals(**options)
    
    create_figdir(**options) 
    
    X_S1_correct, X_S2_correct = get_X_S1_X_S2_day_task(day=options['day'], stimulus='sample', task=options['task'], trials='correct_unpair') 
    performance = (X_S1_correct.shape[0] + X_S2_correct.shape[0]) / (32) 
    
    return performance 

if __name__ == '__main__':

    kwargs = dict() 
    kwargs['pval']= 0.05 
    
    if(len(sys.argv)>1): 
        kwargs['i_mice'] = int(sys.argv[1]) 
        kwargs['task'] = sys.argv[2] 
        kwargs['day'] = sys.argv[3]
        kwargs['trials'] = sys.argv[4] 
        
    cosine = np.zeros((3, 6)) * np.nan 
    performance = np.zeros((3,6)) * np.nan 
    
    for i_mice in range(3): 
        for i_day in range(6): 
            kwargs['i_mice'] = i_mice+1 
            kwargs['day'] = i_day+1 
            kwargs['obj'] = 'norm' 
            
            kwargs['stimulus'] = 'sample' 
            dX_sample, X_S1, X_S2 = stimulus_axis_day(**kwargs) 
            
            # sel_time(**kwargs) 
            
            kwargs['stimulus'] = 'distractor' 
            # sel_time(**kwargs) 
            dX_distractor, X_D1, X_D2 = stimulus_axis_day(**kwargs) 
            # print(gv.epochs)
            cosine[i_mice, i_day] = np.abs(cos_between(dX_sample[:,0], dX_sample[:,-1] ) ) 
            # cosine[i_mice, i_day] = np.abs(cos_between(dX_sample[:,0], np.mean(X_S1[...,-1], axis=0) )) /2 
            # cosine[i_mice, i_day] += np.abs(cos_between(dX_sample[:,0], np.mean(X_S2[...,-1], axis=0) )) /2 
            
            # cosine[i_mice, i_day] = np.abs(cos_between(np.mean(X_S1[...,0], axis=0) , np.mean(X_S1[...,-1], axis=0) )) /2 
            # cosine[i_mice, i_day] += np.abs(cos_between(np.mean(X_S2[...,0], axis=0) , np.mean(X_S2[...,-1], axis=0) ))/2 
            
            performance[i_mice, i_day] = performance_day(**kwargs) 
            
    mean_cos = np.nanmean(cosine, axis=0) 
    std_cos = np.nanstd(cosine, axis=0) 
    
    mean_perf = np.nanmean(performance, axis=0) 
    std_perf = np.nanstd(performance, axis=0) 
    
    days = np.arange(1,7,1) 
    
    # plt.scatter(mean_cos, mean_perf) 
    # plt.scatter(cosine.T, performance.T) 
    # plt.ylabel('DPA performance') 
    # plt.xlabel('$\Delta_{sample}.\Delta_{distractor}$') 
    
    # # plt.plot(days, mean_perf, '-o') 
    # # plt.fill_between(days, mean_perf-std_perf, mean_perf+std_perf, alpha=.1) 
    # # plt.ylabel('DPA performance') 
    # # plt.xlabel('days') 
    
    # # # plt.figure(trials) 
    # plt.plot(days, cosine.T, '-o') 
    plt.plot(days, mean_cos, '-o') 
    # plt.fill_between(days, mean_cos-std_cos, mean_cos+std_cos, alpha=.1) 
    plt.ylabel('$cos(\Delta_{sample},\Delta_{distractor})$') 
    plt.xlabel('days') 
    
    
