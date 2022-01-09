from importlib import reload
import inspect
import numpy as np 
import matplotlib.pyplot as plt

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

from .utils import *
from .plot_utils import *
from .statistics import * 

def sample_dist(**kwargs):
    gv.tasks = ['DPA', 'DualGo', 'DualNoGo'] 
    
    options = set_options(**kwargs) 
    set_globals(**options)
    
    create_figdir(**options)
    
    data.get_days() # do not delete that !!

    if options['obj']=='norm':
        gv.epochs = ['ED','MD','LD','BL']
    else:
        gv.epochs = ['ED','MD','LD'] 
    
    trial_type = get_trial_type(options['trial_type'])
    
    # X_trials, y_trials = get_X_y_days(day=kwargs['day'], stimulus=options['stimulus'])  
    X_trials, y_trials = get_X_y_days(day=kwargs['day']) 
    X_trials = pp.preprocess_X(X_trials) 
 
    print('X_trials', X_trials.shape, 'y_trials', y_trials.shape)
    
    X_sample = np.zeros( (2, 2*X_trials.shape[2], X_trials.shape[3], X_trials.shape[-1]) ) * np.nan 
    X_dist = np.zeros( (2, 2*X_trials.shape[2], X_trials.shape[3], X_trials.shape[-1]) ) * np.nan 

    figtitle = 'cos_sample_dist_time_day_%s_' % ( str(options['day'])) 
    # figtitle = 'cos_sample_dist_time_day_%s_%s_trials' % ( str(options['day']), options['trial_type'] ) 
    fig = plt.figure(figtitle, figsize=set_size(200)) 
    
    # get go vs no go for dual task conditions 
    for i_task in range( 1, len(gv.tasks) ): 
            
        y_task = y_trials[i_task] 
        trial_list_S1 = np.nonzero( np.in1d(y_task[0][0], y_task[trial_type][0]))[0] 
        trial_list_S2 = np.nonzero( np.in1d(y_task[0][1], y_task[trial_type][1]))[0] 
                
        X_S1 = X_trials[i_task, 0][trial_list_S1] 
        X_S2 = X_trials[i_task, 1][trial_list_S2] 
            
        X_S1_S2 = np.vstack((X_S1, X_S2)) 
        print('X_S1_S2', X_S1_S2.shape) 
        
        X_dist[i_task-1, 0:X_S1_S2.shape[0]] = X_S1_S2 
            
    # get S1 and S2 for dual task conditions 
    y_task = y_trials[1] 
    trial_list_S1 = np.nonzero( np.in1d(y_task[0][0], y_task[trial_type][0]))[0] 
    trial_list_S2 = np.nonzero( np.in1d(y_task[0][1], y_task[trial_type][1]))[0] 
    
    X_S1_Go = X_trials[1, 0][trial_list_S1] 
    X_S2_Go = X_trials[1, 1][trial_list_S2] 
        
    y_task = y_trials[2] 
    trial_list_S1 = np.nonzero( np.in1d(y_task[0][0], y_task[trial_type][0]))[0] 
    trial_list_S2 = np.nonzero( np.in1d(y_task[0][1], y_task[trial_type][1]))[0] 
        
    X_S1_NoGo = X_trials[2, 0][trial_list_S1] 
    X_S2_NoGo = X_trials[2, 1][trial_list_S2] 
    
    X_S1_stack = np.vstack((X_S1_Go, X_S1_NoGo)) 
    X_S2_stack = np.vstack((X_S2_Go, X_S2_NoGo))
    
    print('X_S1_stack', X_S1_stack.shape, 'X_S2_stack', X_S2_stack.shape) 
        
    X_sample[0, 0:X_S1_stack.shape[0]] = X_S1_stack 
    X_sample[1, 0:X_S2_stack.shape[0]] = X_S2_stack 
    
    dX_sample = get_coding_direction(X_sample[0], X_sample[1], options['pval']) 
    dX_dist = get_coding_direction(X_dist[0], X_dist[1], options['pval']) 
    
    print('dX_sample', dX_sample.shape, 'dX_dist', dX_dist.shape) 

    cosine = np.zeros( dX_sample.shape[-1])
    dX_bins = np.nanmean( dX_sample[:, gv.bins_STIM], axis=-1) 
    
    for i_epoch in range(dX_sample.shape[-1]):        
        cosine[i_epoch] = cos_between(dX_bins, dX_dist[:, i_epoch])
        
    if options['i_mice']==2:
        gv.pal = ['violet']
    elif options['i_mice']==4:
        gv.pal = ['orange']
    elif options['i_mice']==1:
        gv.pal = ['magenta']
    elif options['i_mice']==2:
        gv.pal = ['gray']
    
    # plt.plot(gv.time, cosine, color=gv.pal[0])
    if options['trial_type']=='correct':
        plt.plot(gv.time, np.arccos(cosine)*180/np.pi, color=gv.pal[0])
        # plt.plot(gv.time, np.absolute( np.arccos(cosine)*180/np.pi-90), color=gv.pal[0]) 
    else:
        plt.plot(gv.time, np.absolute( np.arccos(cosine)*180/np.pi-90), color=gv.pal[0], ls='--') 
        
    pl.add_vlines()            
    plt.xlabel('Time (s)') 
    plt.xticks([0,2,4,6,8,10,12,14]) 
    plt.ylabel('cos($\\alpha_{sample, dist}$)')
    # plt.ylim([-.25,.5]) 
    plt.ylabel('\alpha_{sample, dist}$ (deg)')    
    # plt.ylabel('$\delta \\alpha_{sample, dist}$ (deg)')    
    # plt.ylim([0, 25]) 
    # plt.yticks([50, 60, 70, 80, 90, 100, 110, 120]) 
    
    pl.save_fig(figtitle) 
    # plt.close('all') 
    
def sel_time(**kwargs):
    gv.tasks = ['DPA', 'DualGo', 'DualNoGo'] 
    
    options = set_options(**kwargs) 
    set_globals(**options)
    
    create_figdir(**options)
    
    data.get_days() # do not delete that !!

    if options['obj']=='norm':
        gv.epochs = ['ED','MD','LD','BL']
    else:
        gv.epochs = ['ED','MD','LD'] 
    
    trial_type = get_trial_type(options['trial_type'])
    
    # X_trials, y_trials = get_X_y_days(day=kwargs['day'], stimulus=options['stimulus'])  
    X_trials, y_trials = get_X_y_days(day=kwargs['day']) 
    X_trials = pp.preprocess_X(X_trials) 
    # X_trials = pp.avg_epochs(X_trials) 
 
    print('X_trials', X_trials.shape, 'y_trials', y_trials.shape) 
    
    X_stack = np.zeros( (2, 2*X_trials.shape[2], X_trials.shape[3], X_trials.shape[-1]) ) * np.nan 
    
    if options['stimulus']=='distractor':
        figtitle = 'Dist_Sel_%s_time_day_%s_%s_trials' % ( options['obj'], str(options['day']), options['trial_type'] ) 
        fig = plt.figure(figtitle, figsize=set_size(200))  
        for i_task in range( 1, len(gv.tasks) ): 
            
            y_task = y_trials[i_task] 
            trial_list_S1 = np.nonzero( np.in1d(y_task[0][0], y_task[trial_type][0]))[0] 
            trial_list_S2 = np.nonzero( np.in1d(y_task[0][1], y_task[trial_type][1]))[0] 
                
            X_S1 = X_trials[i_task, 0][trial_list_S1] 
            X_S2 = X_trials[i_task, 1][trial_list_S2] 
            
            X_S1_S2 = np.vstack((X_S1, X_S2)) 
            print('X_S1_S2', X_S1_S2.shape) 
        
            X_stack[i_task-1, 0:X_S1_S2.shape[0]] = X_S1_S2
            
    elif options['task']=='dual':
        figtitle = 'Sample_Sel_%s_time_day_%s_%s_trials' % ( options['obj'], str(options['day']), options['trial_type'] ) 
        fig = plt.figure(figtitle, figsize=set_size(200)) 
        
        y_task = y_trials[1] 
        trial_list_S1 = np.nonzero( np.in1d(y_task[0][0], y_task[trial_type][0]))[0] 
        trial_list_S2 = np.nonzero( np.in1d(y_task[0][1], y_task[trial_type][1]))[0] 
        
        X_S1_Go = X_trials[1, 0][trial_list_S1] 
        X_S2_Go = X_trials[1, 1][trial_list_S2] 
        
        y_task = y_trials[2] 
        trial_list_S1 = np.nonzero( np.in1d(y_task[0][0], y_task[trial_type][0]))[0] 
        trial_list_S2 = np.nonzero( np.in1d(y_task[0][1], y_task[trial_type][1]))[0] 
        
        X_S1_NoGo = X_trials[2, 0][trial_list_S1] 
        X_S2_NoGo = X_trials[2, 1][trial_list_S2] 
        
        X_S1_stack = np.vstack((X_S1_Go, X_S1_NoGo)) 
        X_S2_stack = np.vstack((X_S2_Go, X_S2_NoGo)) 
        print('X_S1_stack', X_S1_stack.shape, 'X_S2_stack', X_S2_stack.shape) 
        
        X_stack[0, 0:X_S1_stack.shape[0]] = X_S1_stack 
        X_stack[1, 0:X_S2_stack.shape[0]] = X_S2_stack 
    else:
        figtitle = '%s_%s_time_day_%s_%s_trials' % ( gv.mouse, options['obj'], str(options['day']), options['trial_type'] )      
        fig = plt.figure(figtitle, figsize=set_size(160)) 
        
    if options['stimulus']=='distractor' or options['task']=='dual': 
        tasks_list = ['DPA']
        if options['i_mice']==2:
            gv.pal = ['violet']
        else:
            gv.pal = ['orange'] 
    else:
        tasks_list = ['DPA', 'DualGo', 'DualNoGo'] 
    
    for i_task, gv.task in enumerate(tasks_list): 
        if options['stimulus']=='distractor' or options['task']=='dual':            
            X_S1 = X_stack[0]
            X_S2 = X_stack[1] 
        else:
            y_task = y_trials[i_task] 
            trial_list_S1 = np.nonzero( np.in1d(y_task[0][0], y_task[trial_type][0]))[0] 
            trial_list_S2 = np.nonzero( np.in1d(y_task[0][1], y_task[trial_type][1]))[0] 
            
            X_S1 = X_trials[i_task, 0][trial_list_S1] 
            X_S2 = X_trials[i_task, 1][trial_list_S2] 
        
        # print('X_S1', X_S1.shape, 'X_S2', X_S2.shape) 
        print('task', gv.tasks[i_task]) 
        print('sample statistics') 
        
        options['bins'] = gv.bins_ED 
        print('bins', options['bins']) 
        
        sel = get_sel(X_S1, X_S2, options['obj'], options['pval'], bins=options['bins'], standardize=options['standardize']) 
        # print('sel', sel.shape) 
        
        if options['obj']=='norm':
            bins_BL = np.arange(0, 12)            
            norm_sel_BL = np.nanmean( sel[bins_BL] )             
            norm_sel = (sel-norm_sel_BL) /  norm_sel_BL             
            plt.plot(gv.time, norm_sel, color=gv.pal[i_task]) 
        else:
            plt.plot(gv.time, sel, color=gv.pal[i_task])
        
        # if options['obj']!= 'auc': 
        #     print('bootstraped ci') 
        #     # ci = boot.ci((X_S1, X_S2), statfunction=lambda x, y: get_sel(x, y, options['obj'], options['pval'], bins=options['bins'] ),
        #     #              n_samples=n_samples, multi='independent', method='bca').T
            
        #     # ci = boot.ci((X_S1, X_S2), statfunction=lambda x, y: get_sel(x, y, options['obj'], options['pval'], bins=options['bins'] ),
        #     #              n_samples=n_samples, multi='independent', method='pi').T 
            
        #     # ci = my_bootstraped_ci(X_S1, X_S2, statfunction=lambda x, y: get_sel(x, y, options['obj'], options['pval'], bins=options['bins'] ) 
        #     #                        , n_samples=options['n_samples'] ) 
            
        #     print('ci', ci.shape) 
            
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
            sel_shuffle = ( sel_shuffle - norm_sel_BL_shuff[:, np.newaxis] ) / norm_sel_BL 
            # sel_shuffle = ( sel_shuffle - norm_sel_BL_shuff[:, np.newaxis] ) / norm_sel_BL_shuff[:, np.newaxis] 
            
        # print('sel_shuffle', sel_shuffle.shape) 
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
        
        plt.ylim([-.5, 1.5]) 
        plt.yticks([-0.5, 0, .5, 1, 1.5]) 
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
    
    pl.save_fig(figtitle) 
    plt.close('all') 
    
def sel_epochs(**kwargs): 
    options = set_options(**kwargs)  
    set_globals(**options) 
    
    create_figdir(**options) 
    n_samples = options['n_samples']
    
    data.get_days() # do not delete that !!
    
    i_trial_type = get_trial_type(options['trial_type'])    
    X_trials, y_trials = get_X_y_days(day=kwargs['day'],  stimulus=options['stimulus'])    
    X_trials = pp.preprocess_X(X_trials)
        
    if options['obj']=='norm':
        gv.epochs = ['ED','MD','LD','BL']
    else:
        gv.epochs = ['ED','MD','LD'] 

    X_trials = pp.avg_epochs(X_trials) 
    
    if not gv.inter_trials: 
        X_trials = np.swapaxes(X_trials, 0, -1) 
        
    sel = np.zeros( (X_trials.shape[0],  X_trials.shape[-1]) )
    sel_shuffle = np.zeros( (X_trials.shape[0], X_trials.shape[-1], n_samples) ) 
    sel_perm = np.zeros( (X_trials.shape[0], X_trials.shape[-1], n_samples) ) 
    
    ci = np.zeros(( X_trials.shape[0], X_trials.shape[-1], 2) ) 
    pval_shuffle = np.zeros( ( X_trials.shape[0], X_trials.shape[-1]) ) 
    pval_task = np.zeros( ( X_trials.shape[0], X_trials.shape[-1]) ) 
    
    # sample statistics 
    for i_task, gv.task in enumerate(gv.tasks): 
        y_task = y_trials[i_task] 
        
        trial_list_S1 = np.nonzero( np.in1d(y_task[0][0], y_task[i_trial_type][0]))[0] 
        trial_list_S2 = np.nonzero( np.in1d(y_task[0][1], y_task[i_trial_type][1]))[0] 
            
        X_S1 = X_trials[i_task, 0][trial_list_S1] 
        X_S2 = X_trials[i_task, 1][trial_list_S2]
                
        # sample statistics 
        sel[i_task] = get_sel( X_S1, X_S2, options['obj'], options['pval'], standardize=options['standardize'])
                    
        # bootstrapped confidence interval             
        ci[i_task] = my_bootstraped_ci(X_S1, X_S2, statfunction=lambda x,y: get_sel(x, y,
                                                                                    options['obj'],
                                                                                    options['pval'],
                                                                                    standardize=options['standardize'])
                                       , n_samples=options['n_samples'])
        
        # ci = ci + sel[i_task, :, np.newaxis] 
        # ci = np.absolute(ci - sel[i_task, :, np.newaxis])
        
        # shuffle statistics
        sel_shuffle[i_task] = shuffle_stat(X_S1, X_S2, lambda x,y: get_sel(x, y,
                                                                           options['obj'],
                                                                           options['pval'],
                                                                           standardize=options['standardize'] ),
                                           n_samples=options['n_samples'] ).T 
            
        for i_epoch in range(X_S1.shape[-1]): 
            pval_shuffle[i_task, i_epoch] = sum( abs( sel_shuffle[i_task, i_epoch] ) 
                                                 >= abs(sel[i_task, i_epoch]) ) / n_samples 
                
            if gv.inter_trials: 
                if i_epoch==0: 
                    print(gv.tasks[i_task], 'epoch MD vs LD', options['trial_type'], 'trial', 
                          options['obj'], sel[i_task, i_epoch], 'ci', ci[i_task, i_epoch],
                          'pval shuffle',  pval_shuffle[i_task, i_epoch] ) 
                else: 
                    print(gv.tasks[i_task], 'epoch ED vs', gv.epochs[i_epoch], options['trial_type'], 'trial', 
                          options['obj'], sel[i_task, i_epoch], 'ci', ci[i_task, i_epoch],
                          'pval shuffle',  pval_shuffle[i_task, i_epoch] ) 
            else: 
                print(gv.tasks[i_epoch], 'epoch', gv.epochs[i_task], options['trial_type'], 'trial', 
                      options['obj'], sel[i_task, i_epoch], 'ci', ci[i_task, i_epoch],
                      'pval shuffle',  pval_shuffle[i_task, i_epoch] ) 
                    
    # permutation test             
    for i_task in range(1, len(gv.tasks)): # Dual Go and Dual No Go
        sel_perm_DPA = np.zeros( (n_samples, len(gv.epochs) ) ) 
        sel_perm_Other = np.zeros( (n_samples, len(gv.epochs) ) ) 
            
        print('permutate S1 trials across tasks') 
        X_S1_DPA, X_S1_other = perm_task(X_trials[:,0], y_trials[..., 0, :], i_task, i_trial_type, n_samples=n_samples) 
        print('permutate S2 trials across tasks') 
        X_S2_DPA, X_S2_other = perm_task(X_trials[:,1], y_trials[..., 1, :], i_task, i_trial_type, n_samples=n_samples) 
        
        print('X_S1_DPA', X_S1_DPA.shape, 'X_S2_DPA', X_S2_DPA.shape) 
        print('X_S1_other', X_S1_other.shape, 'X_S2_other', X_S2_other.shape)
            
        for i_iter in range(n_samples): 
            # DPA perm 
            sel_perm_DPA[i_iter] = get_sel( X_S1_DPA[..., i_iter], 
                                            X_S2_DPA[..., i_iter], 
                                            options['obj'], options['pval'],
                                            standardize=options['standardize']) 
            # Dual perm 
            sel_perm_Other[i_iter] = get_sel( X_S1_other[..., i_iter], 
                                              X_S2_other[..., i_iter], 
                                              options['obj'], options['pval'],
                                                  standardize=options['standardize'] )  
            
        # print('sel_perm', sel_perm_DPA.shape, sel_perm_Other.shape) 
            
        for i_epoch in range(3): 
                
            # compare Delta_perm = DPA_perm - Other_perm vs Delta = DPA - Other 
            pval_task[i_task-1, i_epoch] = sum( abs(sel_perm_DPA[..., i_epoch] - sel_perm_Other[..., i_epoch]) 
                                                >= abs( sel[0, i_epoch] 
                                                        - sel[i_task, i_epoch]) ) / n_samples 
            if gv.inter_trials:
                if i_epoch==0: 
                    print('shuffle DPA and', gv.tasks[i_task], 'epoch MD vs LD', options['trial_type'], 'trial',
                          'pval', pval_task[i_task-1, i_epoch] ) 
                else: 
                    print('shuffle DPA and', gv.tasks[i_task], 'epoch ED vs', gv.epochs[i_epoch], options['trial_type'], 'trial',
                          'pval', pval_task[i_task-1, i_epoch] ) 
            else:
                print('epoch', gv.tasks[i_epoch], 'ED vs', gv.epochs[i_task], options['trial_type'], 'trial',
                      'pval', pval_task[i_task-1, i_epoch] ) 
                    
    figtitle = '%s_%s_epochs_%s_day_%s' % (gv.mouse, options['obj'], options['trial_type'], str(options['day']))  
    
    if gv.inter_trials==0:
        figtitle=  figtitle + '_tasks'
    
    fig = plt.figure(figtitle, figsize=set_size(160) ) 
    cols = [-.1,0,.1]
    
    if options['obj']=='frac': 
        high = [0.575, 0.5] 
        low = [-.095,-.095,-.095] 
        corr = [-0.025, -0.025, -0.025] 
    if options['obj']=='norm': 
        high = [2.1, 1.9] 
        # high = [9-.25, 8-.25] 
        low = [-.95,-.95,-.95] 
        # corr = [-0.25, -0.25, -0.25] 
        corr = [-0.025, -0.025, -0.025] 
    if options['obj']=='cos': 
        high = [1.1, 0.95] 
        low = [-.15,-.15,-.15] 
        corr = [-0.025, -0.025, -0.025] 
    if options['obj']=='proj': 
        high = [5, 4.5] 
        low = [-.25,-.25,-.25] 
        corr = [-0.1, -0.1, -0.1] 
                
    for i_task, gv.task in enumerate(gv.tasks): 
        ci[i_task, :, 0] = - ci[i_task, :, 0] 
        
        if options['obj']=='norm':

            BL = sel[i_task, -1]
            
            print('sel', (sel[i_task]-BL) / BL, 'ci', (ci[i_task]) / np.absolute(BL) ) 
            
            plt.plot([cols[i_task], .4 + cols[i_task]] , (sel[i_task, 1:3]-BL) / BL, 'o', color=gv.pal[i_task], ms=2) 
            
            plt.errorbar([cols[i_task], .4 + cols[i_task]], (sel[i_task, 1:3]-BL) / BL , yerr = (ci[i_task, 1:3]) / np.absolute(BL) , 
                         ecolor=gv.pal[i_task], color=gv.pal[i_task], ls='none') 
        else:
            
            plt.plot([cols[i_task], .4 + cols[i_task]] , sel[i_task, 1:3], 'o', color=gv.pal[i_task], ms=2) 
            plt.errorbar([cols[i_task], .4 + cols[i_task]], sel[i_task, 1:3], yerr=ci[i_task, 1:3], 
                         ecolor=gv.pal[i_task], color=gv.pal[i_task], ls='none') 
        
        for i_epoch in range(2): 
                
            if pval_shuffle[i_task, i_epoch+1]<=0.001: 
                plt.text( i_epoch * .4 + cols[i_task], low[i_task], "***",
                          ha='center', va='bottom', color='k', fontsize=5) 
            elif pval_shuffle[i_task, i_epoch+1]<=.01: 
                plt.text( i_epoch*.4 + cols[i_task], low[i_task], "**",
                          ha='center', va='bottom', color='k', fontsize=5) 
            elif pval_shuffle[i_task, i_epoch+1]<=.05: 
                plt.text( i_epoch*.4 + cols[i_task], low[i_task], "*",
                          ha='center', va='bottom', color='k', fontsize=5) 
            elif pval_shuffle[i_task, i_epoch+1]>.05: 
                plt.text( i_epoch*.4 + cols[i_task], low[i_task], "ns",
                          ha='center', va='bottom', color='k', fontsize=5) 
        
    for i_task in range(len(gv.tasks)-1): 
        for i_epoch in range(1,3): 
            plt.plot( [(i_epoch-1)*.4 + cols[0], (i_epoch-1)*.4  + cols[i_task+1]] , [high[i_task], high[i_task]] , lw=1, c='k') 
                
            if pval_task[i_task, i_epoch]<=.001: 
                plt.text(( 2*(i_epoch-1)*.4 +cols[0] + cols[i_task+1])*.5, high[i_task] + corr[i_task], "***", 
                         ha='center', va='bottom', color='k', fontsize=7) 
            elif pval_task[i_task, i_epoch]<=.01: 
                plt.text(( 2*(i_epoch-1)*.4 +cols[0] + cols[i_task+1])*.5, high[i_task] + corr[i_task], "**", 
                         ha='center', va='bottom', color='k', fontsize=7) 
            elif pval_task[i_task, i_epoch]<=.05: 
                plt.text(( 2*(i_epoch-1)*.4 +cols[0] + cols[i_task+1])*.5, high[i_task] + corr[i_task], "*",
                         ha='center', va='bottom', color='k', fontsize=7) 
            elif pval_task[i_task, i_epoch]>.05: 
                plt.text(( 2*(i_epoch-1)*.4 +cols[0] + cols[i_task+1])*.5, high[i_task], "ns",
                         ha='center', va='bottom', color='k', fontsize=5) 

    plt.xlabel('Epochs')
                        
    plt.xticks([0,.4], ['Middle', 'Late']) 
    plt.xlim([-0.25, .65]) 
        
    if options['obj']=='norm':  
        plt.ylabel('Sample Sel.') 
        plt.ylim([-1, 2.5]) 
        # plt.yticks([0,2,4,6,8,10]) 
        # plt.yticks([0,.25,.5,.75,1])
        plt.yticks([-1,-.5,0,.5,1,1.5,2, 2.5]) 
            
    if options['obj']=='cos': 
        plt.ylabel('cos($\\alpha$)') 
        plt.ylim([-.25, 1.25]) 
        plt.yticks([0,.25,.5,.75,1]) 
            
    if options['obj']=='frac': 
        plt.ylabel('Frac. Selective') 
        plt.ylim([-.1, 0.65]) 

    if options['obj']=='proj': 
        plt.ylabel('S1/S2 memory axis') 
        plt.ylim([-.5, 6]) 
            
    # if isinstance(kwargs['day'], str): 
    #     plt.suptitle(options['mouse_name'][options['i_mice']] + ', ' + options['day']+' days') 
    # else: 
    #     plt.suptitle(options['mouse_name'][options['i_mice']] + ', day %d' % options['day'])

    # plt.tight_layout()
    pl.save_fig(figtitle) 
    plt.close('all') 
    
    
def sel_days(**kwargs):
    options = set_options(**kwargs)  
    set_globals(**options) 
    
    create_figdir(**options) 
    n_samples = options['n_samples']
           
    data.get_days() # do not delete that !!

    if options['obj']=='norm':
        gv.epochs = ['ED','MD','LD','BL'] 
        gv.epoch_str = ['Delay', 'Early', 'Middle', 'Late'] 
    else:
        gv.epochs = ['ED','MD','LD'] 
        gv.epoch_str = ['Early', 'Middle', 'Late'] 
    
    sel = np.zeros((len(gv.days), len(gv.tasks), len(gv.epochs)) ) 
    sel_shuffle = np.zeros((len(gv.days), len(gv.tasks), len(gv.epochs), n_samples) ) 
    sel_perm = np.zeros((len(gv.days), len(gv.tasks), len(gv.epochs), n_samples) ) 
    
    ci = np.zeros((len(gv.days), len(gv.tasks), len(gv.epochs), 2) ) 
    pval_shuffle = np.zeros((len(gv.days), len(gv.tasks), len(gv.epochs)) ) 
    pval_task = np.zeros((len(gv.days), len(gv.tasks), len(gv.epochs)) )
    
    trial_type = options['trial_type'] 
    
    if trial_type == 'correct': 
        i_trial_type = 1 
    elif trial_type == 'incorrect': 
       i_trial_type = 2 
    else: 
       i_trial_type = 0 
    
    # print('trial_type', trial_type, i_trial_type)
    for i_day in range(len(gv.days)): 
        
        X_trials, y_trials = get_X_y_day(day=gv.days[i_day],  stimulus=options['stimulus']) 
        X_trials = pp.preprocess_X(X_trials) 
        X_trials = pp.avg_epochs(X_trials) 
        
        # print('X_trials', X_trials.shape, 'y_trials', y_trials.shape) 
        
        if not gv.inter_trials: 
            X_trials = np.swapaxes(X_trials, 0, -1) 
        
        X_S1_list=[]
        X_S2_list=[]
                    
        for i_task, gv.task in enumerate(gv.tasks): 
            y_task = y_trials[i_task] 
            
            trial_list_S1 = np.nonzero( np.in1d(y_task[0][0], y_task[i_trial_type][0]))[0] 
            trial_list_S2 = np.nonzero( np.in1d(y_task[0][1], y_task[i_trial_type][1]))[0] 
            
            X_S1 = X_trials[i_task, 0][trial_list_S1] 
            X_S2 = X_trials[i_task, 1][trial_list_S2] 
            
            # print('trial type', trial_type, 'X_S1', X_S1.shape, 'X_S2', X_S2.shape) 
                        
            # sample statistics 
            sel[i_day, i_task] = get_sel( X_S1, X_S2, options['obj'], options['pval'], standardize=options['standardize']) 
            # print('sel', sel[i_task].shape) 
                        
            ci[i_day, i_task] = my_bootstraped_ci(X_S1, X_S2, statfunction = lambda x, y: get_sel(x,y,
                                                                                                  options['obj'],
                                                                                                  options['pval'],
                                                                                                  standardize=options['standardize']) 
                                                  , n_samples=options['n_samples'] )  
            
            # ci[i_day, i_task, :, 0] = sel[i_day, i_task] - ci[i_day, i_task, :, 0] 
            # ci[i_day, i_task, :, 1] = -sel[i_day, i_task] + ci[i_day, i_task, :, 1] 
            
            for i_epoch in range(X_S1.shape[-1]): 
                
                print(gv.tasks[i_task], 'epoch', gv.epochs[i_epoch], 'trial', 
                      options['obj'], sel[i_day, i_task, i_epoch], 'ci', ci[i_day, i_task, i_epoch],
                      'pval shuffle',  pval_shuffle[i_day, i_task, i_epoch] ) 
    
    figtitle = '%s_%s_across_days_%s' % (gv.mouse, options['obj'], str(options['day']))
    
    if not gv.inter_trials: 
        figtitle = figtitle + '_tasks' 
    
    # fig = plt.figure(figtitle, figsize=(1.25*1.618*  min(len(gv.epochs),4) , 1.618*1.25)) 
    fig = plt.figure(figtitle, figsize=( set_size(160) ) ) 
    
    # for i_epoch in range( min(len(gv.epochs),4) ): 
    for i_epoch in range(1,2) : 
        if options['obj']=='cos': 
            if i_epoch==0:
                ax = fig.add_subplot( int( '133') ) 
            else:
                # ax = fig.add_subplot( int( '12'+ str(i_epoch)) ) 
                ax = fig.add_subplot() 
        else:
            ax = fig.add_subplot() 
            # ax = fig.add_subplot( int( '1' + str( min(len(gv.epochs),4) ) + str(i_epoch+1)) ) 
        
        for i_task, gv.task in enumerate(gv.tasks): 

            if options['obj']=='norm':
                BL = sel[:, i_task, -1] 
                
                plt.plot(gv.days , (sel[:, i_task, i_epoch]-BL)/BL , '-o', color=gv.pal[i_task], ms=2) 
                plt.errorbar(gv.days, (sel[:, i_task, i_epoch]-BL)/BL, yerr=((ci[:, i_task, i_epoch]/BL[:, np.newaxis]) ).T, 
                             ecolor=gv.pal[i_task], color=gv.pal[i_task], ls='none', alpha=.5, capsize=2) 

                # plt.errorbar(gv.days, (sel[:, i_task, i_epoch]-BL)/(norm-BL), yerr=((ci[:, i_task, i_epoch]-ci[:, i_task, -1])/ (norm[:, np.newaxis]-BL[:, np.newaxis]) ).T, 
                #              ecolor=gv.pal[i_task], color=gv.pal[i_task], ls='none', alpha=.5, capsize=2) 
                
            else:
                plt.plot(gv.days , sel[:, i_task, i_epoch], '-o', color=gv.pal[i_task], ms=2) 
                plt.errorbar(gv.days, sel[:, i_task, i_epoch], yerr=ci[:, i_task, i_epoch].T,
                             ecolor=gv.pal[i_task], color=gv.pal[i_task], ls='none', alpha=.5, capsize=2) 
        
        plt.xlabel('Days') 
        
        # if gv.inter_trials:
        #     if options['obj']=='cos': 
        #         if i_epoch==0 :
        #             plt.title('Middle vs Late')
        #         else: 
        #             plt.title('Early vs ' + gv.epoch_str[i_epoch])
        #     else:
        #         plt.title(gv.epoch_str[i_epoch])                 
        # else: 
        #     plt.title('DPA vs ' + gv.task_str[i_epoch]) 
        
        plt.xticks(gv.days) 
        
        if options['obj']=='norm':
            plt.ylabel('Sample Sel.')
            plt.ylim([-.5, 2]) 
            # plt.yticks([0, 1, 2, 3 ]) 
            plt.yticks([0, 0.5, 1, 1.5, 2]) 
        if options['obj']=='cos': 
            plt.ylabel('cos($\\alpha$)') 
            plt.ylim([-.1, 1.1]) 
            plt.yticks([0, .25, .5, .75, 1]) 
        if options['obj']=='frac':
            plt.ylabel('Frac. Selective') 
            plt.ylim([-.1, 0.3]) 
        
        # if gv.inter_trials:
        #     plt.ylim([-.1, 1]) 
        # else:
        #     plt.ylim([-.1, .8]) 
            
    pl.save_fig(figtitle) 
    plt.close('all') 
    

def performance_days(**kwargs): 
    options = set_options(**kwargs) 
    set_globals(**options)
    
    create_figdir(**options) 
    
    data.get_days() 
    
    day_list = gv.days

    figtitle = 'perf_days' 
    fig = plt.figure(figtitle, figsize=set_size(200)) 

    for i_task in range(3):
        perf_S1_S2 = np.zeros(len(day_list)) 
        for i_day, day in enumerate(day_list):
            X_trials, y_trials = get_X_y_day(day=day) 
            
            perf_S1_S2[i_day] = np.count_nonzero(~np.isnan(y_trials[i_task][1])) / y_trials[i_task][0].shape[0] / 32 
            
            print('day', day_list[i_day], 'performance: S1_S2', perf_S1_S2[i_day]) 

        plt.plot(day_list, perf_S1_S2, '-o', color=gv.pal[i_task]) 
        
    plt.xlabel('Day')
    plt.ylabel('Performance') 
    plt.ylim([0.25, 1.25]) 
    
    pl.save_fig(figtitle) 
    plt.close('all') 


def proj_memory_axis(**kwargs):
    
    options = set_options(**kwargs) 
    set_globals(**options) 
    
    create_figdir(**options) 
    
    X_trials, y_trials = get_X_y_days(day=kwargs['day'], stimulus=options['stimulus']) 
    X_trials = pp.preprocess_X(X_trials) 
    y_task = y_trials[options['i_task']] 
    
    X_S1 = X_trials[options['i_task'], 0]
    X_S2 = X_trials[options['i_task'], 1] 
    
    if standardize is not None: 
        X_S1_S2 = np.vstack((X_S1, X_S2)) 
        m = np.nanmean( X_S1_S2, axis=0 ) 
        s = np.nanstd( X_S1_S2, axis=0 ) 
        
        # print('m', m.shape, np.mean(m), 's', s.shape, np.std(s)) 
        Xz = ( X_S1_S2 - m ) / (s + 1e-16) 
        # Xz = pp.preprocess_X(X_S1_S2) 
    
        # pick S1 and S2 trials from shuffle 
        X_S1 = Xz[0:X_S1.shape[0]] 
        X_S2 = Xz[X_S1.shape[0]:] 
    
    X_S1_S2 = np.vstack( (X_S1, X_S2) )
    print('individual trials', X_S1_S2.shape) 
    
    X_epochs = pp.avg_epochs(X_trials) 
    # X_epochs = np.hstack(X_epochs) 
    print('X_epochs', X_epochs.shape) 
    
    correct_trial_list_S1 = np.nonzero( np.in1d(y_task[0][0], y_task[1][0]))[0] 
    correct_trial_list_S2 = np.nonzero( np.in1d(y_task[0][1], y_task[1][1]))[0] 

    print(correct_trial_list_S1)
    
    X_S1_epoch = X_epochs[options['i_task'], 0][correct_trial_list_S1, :, options['i_epoch']]
    X_S2_epoch = X_epochs[options['i_task'], 1][correct_trial_list_S2, :, options['i_epoch']]
    
    print('X_S1_epoch', X_S1_epoch.shape, 'X_S2_epoch', X_S2_epoch.shape) 
    
    # X_S1_epoch = X_epochs[0, ..., options['i_epoch'] ] 
    # X_S2_epoch = X_epochs[1, ..., options['i_epoch'] ] 
    print('X_S1_epoch', X_S1_epoch.shape, 'X_S2_epoch', X_S2_epoch.shape)
    
    if standardize is not None : 
        X_S1_S2_epoch =  np.vstack((X_S1_epoch, X_S2_epoch))        
        m = np.nanmean( X_S1_S2_epoch, axis=0 ) 
        s = np.nanstd( X_S1_S2_epoch, axis=0 )
        
        # print('m', m.shape, np.mean(m), 's', s.shape, np.std(s)) 
        Xz = ( X_S1_S2_epoch - m ) / (s + 1e-16) 
    
        # pick S1 and S2 trials from shuffle 
        X_S1_epoch = Xz[0:X_S1_epoch.shape[0]] 
        X_S2_epoch = Xz[X_S1_epoch.shape[0]:] 
    
    dX = get_coding_direction(X_S1_epoch, X_S2_epoch, options['pval']) 
    print('coding direction', dX.shape) 
    
    figtitle = 'trialVsMemory_%s_%s' % (gv.epochs[options['i_epoch']], str(kwargs['day']) ) 
    fig = plt.figure(figtitle, figsize=(2.1*1.25*2, 1.85*1.25))
    
    # if isinstance(kwargs['day'], str) :
    #     plt.suptitle(options['mouse_name'][options['i_mice']] + ', ' + options['tasks'][options['i_task']] + ', ' + options['day']+' days') 
    # else: 
    #     plt.suptitle(options['mouse_name'][options['i_mice']] + ', ' + options['tasks'][options['i_task']] + ', day %d' % options['day']) 
        
    print('mouse', gv.mice[options['i_mice']], 'task', gv.tasks[options['i_task']]) 
    for i_correct in range(2): 
        ax = fig.add_subplot('12'+str(i_correct+1)) 
        
        X_proj = np.zeros( (2, X_S1.shape[0], X_S1.shape[-1]) ) * np.nan 
        
        title = ''
        if i_correct==0: 
            title = 'correct trials'
            print('correct trials')
        else: 
            title = 'incorrect trials' 
            print('incorrect trials') 
        
        for i_sample in range(2): 
            X_sample = X_trials[options['i_task'], i_sample] # * (-1)**(i_sample) 
            # print(X_sample.shape) 
            
            trial_list = np.nonzero( np.in1d(y_task[0][i_sample], y_task[i_correct+1][i_sample]))[0] 
            print(trial_list) 
            
            print('sample', gv.samples[i_sample], 
                  'per', len(trial_list)/ y_task[0][i_sample].shape[0] * 100 , '%') 

            if i_sample==0: 
                title = title + ': ' + 'S1 ' + str(len(trial_list)) + '/' + str(y_task[0][i_sample].shape[0]) 
            else: 
                title = title + ', ' + 'S2 ' + str(len(trial_list)) + '/' + str(y_task[0][i_sample].shape[0]) 
                
            if len(trial_list)!=0: 
                
                for i_trial in trial_list: 
                    for i_time in range(X_sample.shape[-1]): 
                        
                        X_proj[i_sample, i_trial, i_time] = np.dot( X_sample[i_trial, :, i_time], dX) / np.linalg.norm(dX) / np.linalg.norm(X_sample[i_trial, :, i_time]) 
                    
                        # plt.plot(gv.time, X_proj[i_sample, i_trial], color=gv.pal[i_sample], alpha=0.1) 
                    
                mean = np.nanmean(X_proj[i_sample], axis=0) 
                plt.plot(gv.time, mean, color=gv.pal[i_sample] ) 
                
                # ci = np.nanpercentile(X_proj[i_sample], [2.5, 97.5], axis=0) 
                # plt.fill_between(gv.time, ci[0], ci[1], color=gv.pal[i_sample], alpha=.1) 
                
                std = np.nanstd(X_proj[i_sample], axis=0) 
                plt.fill_between(gv.time, mean-std, mean+std, color=gv.pal[i_sample], alpha=.1) 
                
                pl.add_vlines() 
                plt.axhline(0, color='k', ls='--') 
                plt.xlabel('Time (s)')
                
                if kwargs['code']=='memory':
                    if options['stimulus'] =='sample': 
                        plt.ylabel('S1/S2 memory axis')
                    else:
                        plt.ylabel('T1/T2 memory axis')
                else:
                    if options['stimulus'] =='sample': 
                        plt.ylabel('S1/S2 sensory axis') 
                    else:
                        plt.ylabel('T1/T2 sensory axis') 
                        
                # if kwargs['pval']==0.001: 
                #     plt.ylim([-0.15, 0.15]) 
                # else: 
                #     plt.ylim([-0.2, 0.2]) 
                plt.ylim([-0.25, 0.25]) 
                    
        plt.title(title)
                    
    pl.save_fig(figtitle) 
    plt.close('all') 
