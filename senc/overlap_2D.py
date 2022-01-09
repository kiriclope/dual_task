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

import senc.utils
reload(senc.utils)
from senc.utils import * 
from senc.plot_utils import * 
from senc.statistics import * 
from senc.sel_time import *

if __name__ == '__main__':
    
    kwargs = dict() 

    kwargs['ci'] = 0 
    kwargs['n_samples'] = 1000 
    kwargs['shuffle'] = 0 
    kwargs['n_shuffles'] = 1000 
    
    kwargs['T_WINDOW'] = 0.5 
    kwargs['pval']= .05 # .05, .01, .001 
    kwargs['sample'] = 'S1' 
    
    kwargs['scaler'] = 'standard' 
    kwargs['scaler_BL'] = 'standard' 
    kwargs['avg_mean'] = 0 
    kwargs['avg_noise'] = 1 
    
    kwargs['tasks'] = np.array(['DPA', 'DualGo', 'DualNoGo']) 
    # kwargs['tasks'] = ['DPA', 'Dual'] 
    kwargs['clf']='logitnetAlphaCV' 
    
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
    
    # # get coding direction of the sample 
    # options['add_vlines'] = 0 
    # options['stimulus'] = 'sample' 
    # options['tasks'] = ['all'] 
    # options['task'] = 'all' 
    # options['Delta0'] = None 
    # options['Delta0'] = get_delta_time(**options) 
    # print('Delta_Sample', options['Delta0'].shape) 
    
    # get projection of sample trials onto sample axis 
    options['tasks']=np.array(['DPA', 'DualGo', 'DualNoGo', 'Dual']) 
    options['task'] = sys.argv[2] 
    options['bins'] = 'ED' 
    options['trials']= sys.argv[4] 
    options['stimulus'] = sys.argv[6] 
    sel_sample, sel_sample_ci, sel_sample_shuffle = sel_time(**options) 
    
    # get coding direction of the distractor 
    options['stimulus'] = 'distractor' 
    options['tasks'] = ['Dual'] 
    options['task'] = 'Dual'
    options['Delta0'] = None 
    options['Delta0'] = get_delta_time(**options) 
    print('Delta_Dist', options['Delta0'].shape) 
    
    # get projection of sample trials onto distractor axis 
    options['tasks']=np.array(['DPA', 'DualGo', 'DualNoGo']) 
    options['stimulus'] = sys.argv[6]
    options['bins'] = 'MD' 
    options['task'] = sys.argv[2] 
    sel_dist, sel_dist_ci, sel_dist_shuffle = sel_time(**options) 
    
    fig = plt.figure('overlap_2D') 
    
    options['tasks']=np.array(['DPA', 'DualGo', 'DualNoGo', 'Dual']) 
    print(options['tasks']==options['task']) 
    options['i_task'] = np.argwhere(options['tasks']==options['task'])[0][0] 
    print(options['tasks'], options['task'], options['i_task']) 
    
    options['add_vlines'] = 1
    # plot_sel_time(sel_sample, None, None, **options) 
    # plot_sel_time(sel_dist, None, None, **options) 
    create_figdir(**options)
    
    plt.plot(sel_sample[0], sel_dist[0], 'kd', ms=5)
    plt.plot(sel_sample[-1], sel_dist[-1], 'ks', ms=5) 
    
    gv.T_WINDOW = 0 
    data.get_bins()
    
    # radius = np.sqrt( (sel_sample[gv.bins_BL[-1]])**2 + (sel_dist[gv.bins_BL[-1]])**2 ) 
    # circ = plt.Circle((0, 0), radius, color='b', alpha=0.1) 
    # ax = plt.gca()
    # ax.add_patch(circ) 
    
    bins_ = sum([gv.bins_STIM[:], gv.bins_DIST[:], gv.bins_CUE[:], gv.bins_TEST[:]], []) 
    
    plt.plot(sel_sample, sel_dist, '--', color=gv.pal[options['i_task']]) 
    x_text = ( sel_sample[gv.bins_STIM[-1]] + sel_sample[gv.bins_STIM[0]] ) / 2.0 
    y_text = ( sel_dist[gv.bins_STIM[-1]] + sel_dist[gv.bins_STIM[0]] ) / 2.0 - 0.05 
    plt.text(x_text, y_text, 'Sample', horizontalalignment='center', fontsize=10) 

    x_text = ( sel_sample[gv.bins_DIST[-1]] + sel_sample[gv.bins_DIST[0]] ) / 2.0 
    y_text = ( sel_dist[gv.bins_DIST[-1]] + sel_dist[gv.bins_DIST[0]] ) / 2.0 + .05
    plt.text(x_text, y_text, 'Dist.', horizontalalignment='center', fontsize=10) 
    
    x_text = ( sel_sample[gv.bins_CUE[-1]] + sel_sample[gv.bins_CUE[0]] ) / 2.0 
    y_text = ( sel_dist[gv.bins_CUE[-1]] + sel_dist[gv.bins_CUE[0]] ) / 2.0 + .05
    plt.text(x_text, y_text, 'Cue', horizontalalignment='center', fontsize=10) 
    
    x_text = ( sel_sample[gv.bins_TEST[-1]] + sel_sample[gv.bins_TEST[0]] ) / 2.0 
    y_text = ( sel_dist[gv.bins_TEST[-1]] + sel_dist[gv.bins_TEST[0]] ) / 2.0 -.05 
    plt.text(x_text, y_text, 'Test', horizontalalignment='center', fontsize=10)
    
    plt.plot(sel_sample[gv.bins_STIM[0]], sel_dist[gv.bins_STIM[0]], 'o', ms=5, color=gv.pal[options['i_task']]) 
    plt.plot(sel_sample[gv.bins_STIM[-1]], sel_dist[gv.bins_STIM[-1]], 'o', ms=5, color=gv.pal[options['i_task']]) 
        
    plt.plot(sel_sample[gv.bins_DIST[0]], sel_dist[gv.bins_DIST[0]], 'o', ms=5, color=gv.pal[options['i_task']]) 
    plt.plot(sel_sample[gv.bins_DIST[-1]], sel_dist[gv.bins_DIST[-1]], 'o', ms=5, color=gv.pal[options['i_task']]) 
    
    plt.plot(sel_sample[gv.bins_CUE[0]], sel_dist[gv.bins_CUE[0]], 'o', ms=5, color=gv.pal[options['i_task']]) 
    plt.plot(sel_sample[gv.bins_CUE[-1]], sel_dist[gv.bins_CUE[-1]], 'o', ms=5, color=gv.pal[options['i_task']]) 
    
    plt.plot(sel_sample[gv.bins_TEST[0]], sel_dist[gv.bins_TEST[0]], 'o', ms=5, color=gv.pal[options['i_task']]) 
    plt.plot(sel_sample[gv.bins_TEST[-1]], sel_dist[gv.bins_TEST[-1]], 'o', ms=5, color=gv.pal[options['i_task']]) 
    
    sel_sample[bins_] = np.nan 
    sel_dist[bins_] = np.nan 
    plt.plot(sel_sample, sel_dist, '-', color=gv.pal[options['i_task']]) 
    
    plt.xlabel('Sample Memory Axis') 
    plt.ylabel('Distractor Memory Axis') 
    
    pl.save_fig('overlap_2D') 
    plt.show()
