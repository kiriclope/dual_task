from importlib import reload
import inspect, sys
import gc 
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

import senc.utils
reload(senc.utils)
from senc.utils import * 
from senc.plot_utils import * 
from senc.statistics import * 

def get_delta_time(**options): 
    
    X_S1_tasks=[]
    X_S2_tasks=[]

    print(options['tasks'])
    
    for options['i_task'] in range(len(options['tasks'])): 
        options['task'] = options['tasks'][options['i_task']]
        X_S1, X_S2 = get_X_S1_X_S2_days_task(day=options['day'], stimulus=options['stimulus'], task=options['task'], trials=options['trials']) 
        X_S1_tasks.append(X_S1) 
        X_S2_tasks.append(X_S2) 
    
    X_S1_all = np.vstack(X_S1_tasks) 
    X_S2_all = np.vstack(X_S2_tasks) 
    
    X_S1_all, X_S2_all = pp.preprocess_X_S1_X_S2(X_S1_all, X_S2_all,
                                                 scaler=options['scaler_BL'],
                                                 center=options['center_BL'], scale=options['scale_BL'],
                                                 avg_mean=options['avg_mean'], avg_noise=options['avg_noise'],
                                                 unit_var=options['unit_var']
    ) 
    
    # X_S1_all = pp.avg_epochs(X_S1_all, gv.epochs) 
    # X_S2_all = pp.avg_epochs(X_S2_all, gv.epochs) 
    
    print('X_S1_all', X_S1_all.shape, 'X_S2_all', X_S2_all.shape)
    
    X_S1_all, X_S2_all, center, scale = scale_data(X_S1, X_S2, scaler=options['scaler'], return_center_scale=1)
    
    Delta_all = get_coding_direction(X_S1_all, X_S2_all, **options) 
    print('Delta_all', Delta_all.shape)  
    
    if options['return_center_scale']: 
        return Delta_all, center, scale
    else:
        return Delta_all
    
def sel_time(**options): 
    
    data.get_days() # do not delete that !!
    
    X_S1, X_S2 = get_X_S1_X_S2_days_task(day=options['day'], stimulus=options['stimulus'], task=options['task'], trials=options['trials'])
    # that must be before bins    
    X_S1, X_S2 = pp.preprocess_X_S1_X_S2(X_S1, X_S2,
                                         scaler=options['scaler_BL'],
                                         center=options['center_BL'], scale=options['scale_BL'],
                                         avg_mean=options['avg_mean'], avg_noise=options['avg_noise'], unit_var=options['unit_var']) 
    
    if options['stimulus']=='sample':
        if options['bins']=='ED': 
            options['bins'] = gv.bins_ED 
        if options['bins']=='MD': 
            options['bins'] = gv.bins_MD 
        if options['bins']=='Dist': 
            options['bins'] = gv.bins_DIST 
        if options['bins']=='Sample': 
            options['bins'] = gv.bins_STIM 
    else: 
        options['bins'] = gv.bins_DIST
    
    print('bins', options['bins']) 
    # options['bins'] = None 
    
    print('task', options['task'], 'X_S1', X_S1.shape, 'X_S2', X_S2.shape) 
    
    if options['obj']=='cos' or options['obj']=='proj' : 
        sel, Delta = get_sel(X_S1, X_S2, return_Delta=1, **options) 
        options['Delta0'] = Delta # this fixes delta for the stats. 
    else:
        sel = get_sel(X_S1, X_S2, **options) 
    
    print('sel', sel.shape) 
    
    sel_ci = None 
    if options['ci']==1: 
        print('bootstraped ci') 
        sel_ci = my_bootstraped_ci(X_S1, X_S2, statfunction=lambda x, y: get_sel(x, y, **options) 
                                   , n_samples=options['n_samples'] ) 
        print('ci', sel_ci.shape)
    
    sel_shuffle = None 
    if options['shuffle']==1:
        print('shuffle statistics') 
        sel_shuffle = shuffle_stat(X_S1, X_S2, lambda x,y: get_sel(x, y,**options), n_samples=options['n_shuffles']) 
        mean_shuffle = np.nanmean(sel_shuffle, axis=0) 
        perc_shuffle = np.nanpercentile(sel_shuffle, [2.5, 97.5], axis=0) 
        print('sel_shuffle', sel_shuffle.shape, 'mean', mean_shuffle.shape, 'perc', perc_shuffle.shape) 

    if options['obj']=='norm':
        # bins = np.arange(0, 12) 
        # BL = np.nanmean( sel[bins] ) 
        # sel -= BL 
        max_ = np.amax(sel)
        
        sel /= max_ 
        # sel /= BL
        
        if options['ci']: 
            # sel_ci -= BL 
            sel_ci /= max_ 
            # sel_ci /= BL 
        if options['shuffle']: 
            # sel_shuffle -= BL 
            sel_shuffle /= max_ 
            # sel_shuffle /= BL
    
    return sel, sel_ci, sel_shuffle
 
def plot_sel_time(sel, sel_ci=None, sel_shuffle=None, **options): 
    create_figdir(**options)
    
    if(len(options['tasks'])==2): 
        figtitle = 'sel_%s_time_Dual_%s_day_%s_trials' % ( options['obj'], str(options['day']), options['trials'] ) 
    else: 
        figtitle = 'sel_%s_time_day_%s_%s_trials' % ( options['obj'], str(options['day']), options['trials'] ) 
    
    fig = plt.figure(figtitle) 
    
    plt.plot(gv.time, sel, color=gv.pal[options['i_task']]) 
    if options['ci'] == 1: 
        plt.fill_between(gv.time, sel-sel_ci[:,0], sel+sel_ci[:,1], alpha=0.1) 
    if options['shuffle']==1: 
        mean_shuffle = np.nanmean(sel_shuffle, axis=0) 
        perc_shuffle = np.nanpercentile(sel_shuffle, [2.5, 97.5], axis=0) 
        
        plt.plot(gv.time, mean_shuffle, '--' , color=gv.pal[options['i_task']]) 
        plt.fill_between(gv.time, perc_shuffle[0], perc_shuffle[1], color=gv.pal[options['i_task']], alpha=.1) 
    
    if options['add_vlines']==1:    
        pl.add_vlines()
        if options['obj']=='proj':
            plt.text(2.5, 2.6, 'Sample', horizontalalignment='center', fontsize=10) 
            plt.text(5, 2.6, 'Dist.', horizontalalignment='center', fontsize=10) 
            plt.text(7, 2.6, 'Cue', horizontalalignment='center', fontsize=10) 
            plt.text(9.5, 2.6, 'Test', horizontalalignment='center', fontsize=10) 
        else:
            plt.text(2.5, 1.1, 'Sample', horizontalalignment='center', fontsize=10) 
            plt.text(5, 1.1, 'Dist.', horizontalalignment='center', fontsize=10) 
            plt.text(7, 1.1, 'Cue', horizontalalignment='center', fontsize=10) 
            plt.text(9.5, 1.1, 'Test', horizontalalignment='center', fontsize=10) 
            
    plt.xlabel('Time (s)') 
    plt.xticks([0, 2, 4, 6, 8, 10, 12, 14]) 
    plt.xlim([0, 14]) 
    
    if options['obj']=='norm':
        if options['stimulus']=='distractor': 
            plt.ylabel('Distractor Sel.') 
        else: 
            plt.ylabel('Sample Sel.')         
        plt.ylim([-.25, 1.1]) 
        plt.yticks([-0.25, 0, .25, .5, .75, 1.0]) 
        
    if options['obj']=='cos':
        # plt.ylabel('Cosine') 
        plt.ylabel('Overlap\n' r'Early Sample vs. Sample') 
        plt.ylim([-.25, 1.05]) 
        plt.yticks([-0.25, 0, .25, .5, .75, 1.0]) 
    
    if options['obj']=='frac':
        plt.ylabel('Frac. Selective') 
        plt.ylim([-.05, 0.3]) 
        plt.yticks([0, .1, .2, .3])
        
    if options['obj']=='proj': 
        # plt.ylabel('Overlap') 
        # plt.ylim([-1, 1]) 
        # plt.yticks([-1,-0.5, 0, .5, 1.0]) 

        plt.ylabel('Sample Memory Axis') 
        plt.ylim([-0.5, 1]) 
        plt.yticks([-0.5, 0, .5, 1.0, 1.5, 2.0, 2.5]) 
        
    if options['obj']=='score': 
        plt.ylabel('Score') 
        plt.ylim([0.45, 1.25]) 
        plt.yticks([.5, 0.75, 1.0])         
    
    if(options['IF_SAVE']==1):
        pl.save_fig(figtitle) 
    plt.show()
    
if __name__ == '__main__':
    
    kwargs = dict() 
    
    kwargs['T_WINDOW'] = 0.5 
    kwargs['bins'] = 'ED' 
    kwargs['sample'] = 'S1' 
    
    kwargs['ci'] = 1 
    kwargs['n_samples'] = 1000 
    kwargs['shuffle'] = 1 
    kwargs['n_shuffles'] = 1000 
    
    kwargs['pval']= .05 # .05, .01, .001 
    
    kwargs['scaler'] = 'standard' #'standard' # if not standardized gives strange results for norm 
    kwargs['scaler_BL'] = 'robust' 
    kwargs['avg_mean'] = 0 
    kwargs['avg_noise'] = 1 
    
    kwargs['clf']='logitnetAlphaCV' # 'LogisticRegressionCV'    
    kwargs['clf']='dot' # 'LogisticRegressionCV'
    
    kwargs['tasks'] = np.array(['DPA', 'DualGo', 'DualNoGo', 'Dual']) 
    # kwargs['tasks'] = ['DPA', 'Dual'] 
    
    kwargs['fold_type'] = 'stratified' 
    
    if(len(sys.argv)>1): 
        kwargs['i_mice'] = int(sys.argv[1]) 
        kwargs['task'] = sys.argv[2] 
        kwargs['day'] = sys.argv[3] 
        kwargs['trials'] = sys.argv[4] 
        kwargs['obj'] = sys.argv[5] 
        kwargs['stimulus'] = sys.argv[6]         
        kwargs['sample'] = sys.argv[7]
    
    options = set_options(**kwargs) 
    set_globals(**options)

    print(options['task'])
    
    if options['task']=='all':
        options['add_vlines']=0 
        options['tasks'] = np.array(['DPA', 'DualGo', 'DualNoGo']) 
        
        # options['Delta0'] = get_delta_time(**options) 
        
        for options['i_task'] in range(len(options['tasks'])): 
            options['task'] = options['tasks'][options['i_task']] 
            
            if options['i_task']==len(options['tasks'])-1: 
                options['add_vlines']=1 
                options['IF_SAVE']=1 
            
            sel, sel_ci, sel_shuffle = sel_time(**options) 
            plot_sel_time(sel, sel_ci, sel_shuffle, **options) 
    
    else: 
        options['tasks'] = np.array(['DualGo', 'DualNoGo']) 
        options['tasks'] = np.array(['all']) 
        options['stimulus'] = 'sample' 
        options['return_center_scale'] = 0 
        options['Delta0'] = get_delta_time(**options) 
        
        options['tasks'] = np.array(['DPA', 'DualGo', 'DualNoGo', 'Dual']) 
        options['bins'] = 'ED' 
        options['stimulus']= 'sample' 
        options['i_task'] = np.argwhere(options['tasks']==options['task'])[0][0] 
        print(options['tasks'], options['task'], options['i_task']) 
        
        options['add_vlines'] = 1 
        sel, sel_ci, sel_shuffle = sel_time(**options) 
        plot_sel_time(sel, sel_ci, sel_shuffle, **options) 
    
    
