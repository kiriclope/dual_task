import os
import numpy as np

import matplotlib
# matplotlib.use('Agg') 
import matplotlib.pyplot as plt 
from mpl_toolkits.axes_grid1 import make_axes_locatable

import utils.constants as gv 
import utils.plot_utils as pl 

from utils.plot_settings import SetPlotParams 
SetPlotParams() 

def set_size(width, fraction=1):
    """ Set aesthetic figure dimensions to avoid scaling in latex.

    Parameters
    ----------
    width: float
            Width in pts
    fraction: float
            Fraction of the width which you wish the figure to occupy

    Returns
    -------
    fig_dim: tuple
            Dimensions of figure in inches
    """
    # Width of figure
    fig_width_pt = width * fraction

    # Convert from pt to inches
    inches_per_pt = 1 / 72.27

    # Golden ratio to set aesthetic figure height
    golden_ratio = (5 ** 0.5 - 1) / 2

    # Figure width in inches
    fig_width_in = fig_width_pt * inches_per_pt
    # Figure height in inches
    fig_height_in = fig_width_in * golden_ratio

    return fig_width_in, fig_height_in 

def create_figdir(**kwargs): 
    globals().update(kwargs) 
    pl.figDir() 
    
    gv.figdir = gv.figdir + '/n_days_%d' % n_days
    
    # gv.figdir = gv.figdir + '/selectivity/pval_%.3f/%s/%s' % (kwargs['pval'], kwargs['tasks'][kwargs['i_task']], kwargs['stimulus']) 
    gv.figdir = gv.figdir + '/selectivity/pval_%.3f/%s' % (kwargs['pval'], kwargs['stimulus']) 
    
    gv.figdir = gv.figdir + '/%s' % code 
            
    gv.figdir = gv.figdir + '/%s' % gv.mouse 
    
    if not os.path.isdir(gv.figdir): 
        os.makedirs(gv.figdir)
        print(gv.figdir) 
        
    if not os.path.isdir(gv.filedir): 
        os.makedirs(gv.filedir) 
        print(gv.filedir)

def plotFracSelBars(mean, errorbar=None, pval_shuffle=None, **kwargs): 
    
    labels = np.arange(len(gv.epochs)) 
    width=0.25 

    if isinstance(kwargs['day'], int): 
        figtitle = '%s_fracSelBars_day_%d' % (gv.mouse, kwargs['day']) 
        ax = plt.figure(figtitle).add_subplot() 
        ax.set_title(kwargs['mouse_name'][kwargs['i_mice']] + ', ' + kwargs['tasks'][kwargs['i_task']] + ', day %d' % kwargs['day']) 
    else: 
        figtitle = '%s_fracSelBars_%s_days' % (gv.mouse, kwargs['day']) 
        ax = plt.figure(figtitle).add_subplot() 
        ax.set_title(kwargs['mouse_name'][kwargs['i_mice']] + ', ' + kwargs['tasks'][kwargs['i_task']] + ', ' + kwargs['day'] +' days') 
        
    for i_task, task in enumerate(gv.tasks): 
        values = mean[i_task]
        if errorbar is not None: 
            error = errorbar[i_task].T 
        else:
            error = None
            
        ax.bar(labels + i_task*width, values , yerr=error,  color = gv.pal[i_task], width = width) 
        
    if 'ED' in gv.epochs:
        epochs = ['Early', 'Middle', 'Late']
    else:
        epochs = ['Sample', 'Distractor', 'Test']
        
    tasks = ['DPA', 'Dual Go', 'Dual NoGo'] 
    
    if not gv.inter_trials: 
        plt.xticks([i + width for i in range(len(gv.epochs))], epochs) 
        if 'ED' in gv.epochs:
            plt.xlabel('Delay') 
        else:
            plt.xlabel('Epoch') 
    else: 
        plt.xticks([i + width for i in range(len(gv.tasks))], tasks) 
        plt.xlabel('Task') 
        
    plt.ylabel('Fraction Selective') 

    if pval_shuffle is not None:
        add_shuffle(pval_shuffle) 
        
    # if samples is not None: 
    #     p_values = get_p_values(samples) 
    #     add_pvalue(p_values[:,:], [0.5, 0.4]) 
    
    plt.ylim([-.1, .6]) 
    
    pl.save_fig(figtitle) 
    plt.close('all') 
    
def plotSelBars(obj, mean, errorbar=None, pval_perm=None, pval_shuffle=None, **kwargs): 
    
    labels = np.arange(len(gv.epochs)) 
    width=0.25 
    
    if isinstance(kwargs['day'], int): 
        figtitle = '%s_%sSelBars_day_%d_%s_trials' % (obj, gv.mouse, kwargs['day'], kwargs['trial_type']) 
        ax = plt.figure(figtitle).add_subplot() 
        ax.set_title(kwargs['mouse_name'][kwargs['i_mice']] + ', day %d' % kwargs['day'] + ', ' + kwargs['trial_type'] + ' trials') 
    else: 
        figtitle = '%s_%sSelBars_%s_days_%s_trials' % (obj, gv.mouse, kwargs['day'], kwargs['trial_type']) 
        ax = plt.figure(figtitle).add_subplot() 
        ax.set_title(kwargs['mouse_name'][kwargs['i_mice']] + ', ' + kwargs['day'] +' days' + ', ' + kwargs['trial_type'] + ' trials') 
        
    for i_task, task in enumerate(gv.tasks): 
        values = mean[i_task]
        if errorbar is not None: 
            error = errorbar[i_task].T 
        else:
            error = None
            
        ax.bar(labels + i_task*width, values , yerr=error,  color = gv.pal[i_task], width = width) 
        
    if 'ED' in gv.epochs:
        epochs = ['Early', 'Middle', 'Late']
    else:
        epochs = ['Sample', 'Distractor', 'Test']
        
    tasks = ['DPA', 'Dual Go', 'Dual NoGo'] 
    
    if not gv.inter_trials: 
        plt.xticks([i + width for i in range(len(gv.epochs))], epochs) 
        if 'ED' in gv.epochs:
            plt.xlabel('Delay') 
        else:
            plt.xlabel('Epoch') 
    else: 
        plt.xticks([i + width for i in range(len(gv.tasks))], tasks) 
        plt.xlabel('Task') 

    if obj=='sel':
        plt.ylabel('Selectivity')
        
        if pval_perm is not None:
            # add_pvalue(pval_perm[1:], [5.5, 5.0]) 
            # plt.ylim([-.1, 6]) 

            add_pvalue(pval_perm[1:], [8.5, 8.0]) 
            plt.ylim([-.1, 10]) 
            
    if obj=='frac':
        plt.ylabel('Frac. Selective')
        
        if pval_perm is not None:
            add_pvalue(pval_perm[1:], [0.55, 0.5])
            
        plt.ylim([-.1, .6]) 

    if obj=='angle':
        plt.ylabel('Angle')
        
        if pval_perm is not None:
            add_pvalue(pval_perm[1:], [0.85, 0.8]) 
            
        plt.ylim([-.1, 1]) 
        
    if pval_shuffle is not None:
        add_shuffle(pval_shuffle) 

    pl.save_fig(figtitle) 
    plt.close('all') 
    
def add_shuffle(p_values, high=None): 

    if high is None:
        high = [-0.05,- 0.05,- 0.05] #2 lines: NDvsD1 or NDvsD2  
    
    cols = 0.4*np.arange(p_values.shape[0]+1) #3 cols: ND, D1 and D2 
        
    for i_task in range(p_values.shape[0]): 
        for i_epoch in range(p_values.shape[1]): 
            
            if p_values[i_task,i_epoch]<=.001:
                plt.text((2*i_epoch + cols[0] + cols[i_task+1])*.5, high[i_task]-.02, "***", ha='center', va='bottom', color='k', fontsize=6) 
                
            elif p_values[i_task,i_epoch]<=.01: 
                plt.text((2*i_epoch + cols[0] + cols[i_task+1])*.5, high[i_task]-.02, "**", ha='center', va='bottom', color='k', fontsize=6)
                
            elif p_values[i_task-1,i_epoch]<=.05: 
                plt.text((2*i_epoch + cols[0] + cols[i_task+1])*.5, high[i_task]-.02, "*", ha='center', va='bottom', color='k', fontsize=6)
                
            elif p_values[i_task-1,i_epoch]>.05: 
                plt.text((2*i_epoch + cols[0] + cols[i_task+1])*.5, high[i_task]-.005, "ns", ha='center', va='bottom', color='k', fontsize=4) 
                
def add_pvalue(p_values, high=None, width=0.25):

    if high is None:
        high = [10.1, 10.05] #2 lines: NDvsD1 or NDvsD2 
    
    cols = width*np.arange(p_values.shape[0]+1) #3 cols: ND, D1 and D2 
    
    for i_task in range(p_values.shape[0]): 
        for i_epoch in range(p_values.shape[1]): 
            plt.plot( [i_epoch + cols[0], i_epoch + cols[i_task+1]] , [high[i_task], high[i_task]] , lw=1, c='k') 
            
            if p_values[i_task,i_epoch]<=.001:
                plt.text((2*i_epoch+cols[0]+cols[i_task+1])*.5, high[i_task]-.02, "***", ha='center', va='bottom', color='k', fontsize=8) 
                
            elif p_values[i_task,i_epoch]<=.01: 
                plt.text((2*i_epoch + cols[0]+cols[i_task+1])*.5, high[i_task]-.02, "**", ha='center', va='bottom', color='k', fontsize=8)
                
            elif p_values[i_task-1,i_epoch]<=.05: 
                plt.text((2*i_epoch + cols[0]+cols[i_task+1])*.5, high[i_task]-.02, "*", ha='center', va='bottom', color='k', fontsize=8)
                
            elif p_values[i_task-1,i_epoch]>.05: 
                plt.text((2*i_epoch + cols[0]+cols[i_task+1])*.5, high[i_task]-.005, "ns", ha='center', va='bottom', color='k', fontsize=6) 

def add_pvalue_CI(p_values, high=None, width=0.25):

    if high is None:
        high = [10.1, 10.05] #2 lines: NDvsD1 or NDvsD2 
    
    cols = width*np.arange(p_values.shape[0]+1) #3 cols: ND, D1 and D2 

    cols = [-.1, 0, 1]
    
    for i_task in range(p_values.shape[0]): 
        for i_epoch in range(p_values.shape[1]): 
            plt.plot( [i_epoch + cols[0], i_epoch + cols[i_task+1]] , [high[i_task], high[i_task]] , lw=1, c='k') 
            
            if p_values[i_task,i_epoch]<=.001:
                plt.text((2*i_epoch+cols[0]+cols[i_task+1])*.5, high[i_task]-.02, "***", ha='center', va='bottom', color='k', fontsize=8) 
                
            elif p_values[i_task,i_epoch]<=.01: 
                plt.text((2*i_epoch + cols[0]+cols[i_task+1])*.5, high[i_task]-.02, "**", ha='center', va='bottom', color='k', fontsize=8)
                
            elif p_values[i_task-1,i_epoch]<=.05: 
                plt.text((2*i_epoch + cols[0]+cols[i_task+1])*.5, high[i_task]-.02, "*", ha='center', va='bottom', color='k', fontsize=8)
                
            elif p_values[i_task-1,i_epoch]>.05: 
                plt.text((2*i_epoch + cols[0]+cols[i_task+1])*.5, high[i_task]-.005, "ns", ha='center', va='bottom', color='k', fontsize=6) 
