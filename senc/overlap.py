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

def overlap_time(**kwargs):
    
    options = set_options(**kwargs) 
    set_globals(**options)
    
    create_figdir(**options)
    
    data.get_days() # do not delete that !!
        
    X_S1, X_S2 = get_X_S1_X_S2_days_task(day=options['day'], stimulus='sample', task=options['task'], trials=options['trials'])    
    X_S1, X_S2 = pp.preprocess_X_S1_X_S2(X_S1, X_S2,
                                         scaler=options['scaler_BL'],
                                         center=options['center'], scale=options['scale'],
                                         avg_mean=options['avg_mean'], avg_noise=options['avg_noise'], unit_var=options['unit_var']) 
    
    X_S1, X_S2 = scale_data(X_S1, X_S2, scaler=options['scaler']) 
    dX_sample = get_coding_direction(X_S1, X_S2, **options) 
    
    X_D1, X_D2 = get_X_S1_X_S2_days_task(day=options['day'], stimulus='distractor', task=options['task'], trials=options['trials']) 
    X_D1, X_D2 = pp.preprocess_X_S1_X_S2(X_D1, X_D2,
                                         scaler=options['scaler_BL'],
                                         center=options['center'], scale=options['scale'],
                                         avg_mean=options['avg_mean'], avg_noise=options['avg_noise'], unit_var=options['unit_var']) 
    
    # X_D1, X_D2 = scale_data(X_D1, X_D2, scaler=options['scaler']) 
    # dX_distractor = get_coding_direction(X_D1, X_D2, pval) 
    
    if(options['bins']=='ED'):
        options['bins'] = gv.bins_ED
    else:
        options['bins'] = None 
    
    if(options['bins'] ==None):
        figtitle = 'overlap_equal_time_%s_day_%s_%s_trials' % ( options['task'], str(options['day']), options['trials'] )
    else:
        figtitle = 'overlap_ED_%s_day_%s_%s_trials' % ( options['task'], str(options['day']), options['trials'] )
    
    fig = plt.figure(figtitle) 
    
    overlap = -get_overlap(X_D1, X_D2, dX_sample, **options) 
    
    plt.plot(gv.time, overlap,'k') 
    plt.xlabel('Time (s)')
    
    if options['bins'] is not None : 
        plt.ylabel('Overlap\n' r'Early Sample vs. Dist') 
    else: 
        plt.ylabel('Co-overlap\n' r'Sample vs. Dist') 
    
    plt.ylim([-.3,.3])
    plt.yticks([-.3,-.2,-.1,0,.1,.2,.3]) 
    plt.xlim([0,.14]) 
    plt.xticks([0,2,4,6,8,10,12,14]) 
    
    if options['ci']:
        print('bootstrap ci')
        ci = my_bootstraped_ci(X_D1, X_D2, statfunction=lambda x, y: get_overlap(x, y, dX_sample, **options) 
                               , n_samples=options['n_samples'] ) 
    
        print('ci', ci.shape) 
        plt.fill_between(gv.time, overlap-ci[:,0], overlap+ci[:,1], alpha=0.1) 
    
    if options['shuffle']:
        print('shuffle statistics') 
        sel_shuffle = shuffle_stat(X_D1, X_D2,
                                   lambda x,y: -get_overlap(x, y, dX_sample,**options), 
                                   n_samples=options['n_shuffles'] ) 
    
        print('sel_shuffle', sel_shuffle.shape) 
        mean = np.nanmean(sel_shuffle, axis=0) 
        std = np.nanpercentile(sel_shuffle, [2.5, 97.5], axis=0) 
        plt.plot(gv.time, mean, 'k--') 
        plt.fill_between(gv.time, std[0], std[1], alpha=.1, color='k') 
    
    plt.ylim([-.25, 0.5]) 
    plt.yticks([-0.25, 0, .25, .5]) 
    plt.xticks([0,2,4,6,8,10,12,14]) 
    
    pl.add_vlines() 
    plt.text(2.5, 0.55, 'Sample', horizontalalignment='center', fontsize=10) 
    plt.text(5, 0.55, 'Dist.', horizontalalignment='center', fontsize=10) 
    plt.text(7, 0.55, 'Cue', horizontalalignment='center', fontsize=10) 
    plt.text(9.5, 0.55, 'Test', horizontalalignment='center', fontsize=10) 
    
    pl.save_fig(figtitle) 
    
if __name__ == '__main__':
    
    kwargs = dict() 
    kwargs['T_WINDOW']= 0.5 
    kwargs['pval']= 0.05 
    
    kwargs['n_samples']= 1000 
    kwargs['n_shuffles']= 1000 
    
    kwargs['scaler']= 'standard' 
    kwargs['scaler_BL']= 'robust' 
    
    kwargs['avg_center']= 0 
    kwargs['avg_noise']= 1 
    
    if(len(sys.argv)>1): 
        kwargs['i_mice'] = int(sys.argv[1]) 
        kwargs['task'] = sys.argv[2] 
        kwargs['day'] = sys.argv[3]
        kwargs['trials'] = sys.argv[4] 
        kwargs['bins'] = sys.argv[5] 
    
    overlap_time(**kwargs) 
    
