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

from utils.plot_settings import SetPlotParams
SetPlotParams()

def get_modulation_index(X_S1, X_S2):
    ''' X_Si [n_trials, n_neurons, n_epochs] '''
    mean_S1 = np.abs( np.nanmean(X_S1, axis=0) ) # across trials 
    mean_S2 = np.abs( np.nanmean(X_S2, axis=0) ) 
    modulation_index = (mean_S1 - mean_S2) /  (mean_S1 + mean_S2) 
    
    return modulation_index 

def modulation_dist(**kwargs):
    
    options = set_options(**kwargs) 
    set_globals(**options)
    
    create_figdir(**options)
    
    data.get_days() # do not delete that !!
    
    X_S1, X_S2 = get_X_S1_X_S2_days_task(day=options['day'], stimulus='sample', task=options['task'], trials=options['trials'])
    X_S1, X_S2 = pp.preprocess_X_S1_X_S2(X_S1, X_S2,
                                         scaler=options['scaler_BL'],
                                         center=options['center'], scale=options['scale'],
                                         avg_mean=options['avg_mean'], avg_noise=options['avg_noise']) 
    
    X_S1 = pp.avg_epochs(X_S1.copy(), epochs=['ED', 'LD'])
    X_S2 = pp.avg_epochs(X_S2.copy(), epochs=['ED', 'LD'])

    print('X_S1', X_S1.shape,'X_S2', X_S2.shape) 
    modulation_index = get_modulation_index(X_S1, X_S2).T
    print('modulation index', modulation_index.shape) 

    shuffle_mi = shuffle_stat(X_S1, X_S2, lambda x,y: get_modulation_index(x, y), n_samples=options['n_shuffles'] ).T  
    print('shuffle_mi', shuffle_mi.shape) 
    
    # p value with respect to shuffle 
    pval_shuffle = 2.0*np.amin( np.stack( [np.mean( shuffle_mi >= modulation_index[..., np.newaxis], axis=-1 ), 
                                           np.mean( shuffle_mi <= modulation_index[..., np.newaxis], axis=-1 ) ] ) 
                                , axis=0 ) 
    print('pval_shuffle', pval_shuffle.shape) 
    
    # modulation_index[pval_shuffle>=0.05] = np.nan 
    # modulation_index[0][pval_shuffle[0]>=0.05] = np.nan 
    # modulation_index[1][pval_shuffle[1]>=0.05] = np.nan 
    
    plt.hist(modulation_index[0], bins=50, alpha=0.5, label='Early Delay') 
    plt.hist(modulation_index[1], bins=50, alpha=0.5, label='Late Delay') 
    # plt.legend(bbox_to_anchor=(1.04,1), loc="upper left") 
    plt.xlabel('Modulation Index') 
    plt.ylabel('Count') 
    
    # plt.scatter(modulation_index[0], modulation_index[1]) 
    # plt.xlabel('Early Delay') 
    # plt.xlim([-1,1]) 
    # plt.ylabel('Late Delay') 
    # plt.ylim([-1,1]) 
    
if __name__ == '__main__':

    kwargs = dict() 
    kwargs['pval']= 0.05  
    kwargs['n_samples']= 1000
    
    if(len(sys.argv)>1): 
        kwargs['i_mice'] = int(sys.argv[1]) 
        kwargs['task'] = sys.argv[2] 
        kwargs['day'] = sys.argv[3]
        kwargs['trials'] = sys.argv[4] 
    
    modulation_dist(**kwargs) 
    
