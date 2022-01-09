import numpy as np 
import scipy.stats as stats
from scipy.spatial.distance import mahalanobis

import utils.constants as gv 

from statistics import * 
from .decoder import *

import utils.preprocessing as pp

def my_mahalanobis(X_S1, X_S2):

    X = np.vstack([X_S1 , X_S2]) 
    V = np.cov(X.T) 
    VI = np.linalg.inv(V)
    
    return np.diag(np.sqrt(np.dot(np.dot((X_S1-X_S2),VI),(X_S1-X_S2).T))) 
    
def scale_data(X_S1, X_S2, scaler='standard', center=None, scale=None, return_center_scale=0): 
    
    X_S1_S2 = np.vstack((X_S1, X_S2)) 

    if center is None or scale is None : 
        if scaler == 'standard': 
            center, scale = standard_scaler(X_S1_S2) 
        elif scaler == 'robust': 
            center, scale = robust_scaler(X_S1_S2) 
        elif scaler == 'center': 
            center = np.nanmean(X_S1_S2, axis=0) 
            scale = 1 
        else : 
            center = 0 
            scale = 1 
    
    X_scale = ( X_S1_S2 - center ) / scale 
    
    X_S1_scale = X_scale[0:X_S1.shape[0]] 
    X_S2_scale = X_scale[X_S1.shape[0]:] 
    
    if return_center_scale:
        return X_S1_scale, X_S2_scale, center, scale 
    else:
        return X_S1_scale, X_S2_scale
    
def standard_scaler(X):
    center = np.nanmean(X, axis=0) 
    scale = np.nanstd(X, axis=0) 
    scale = _handle_zeros_in_scale(scale, copy=False)  
    
    return center, scale

def robust_scaler(X): 
 
    center = np.nanmedian(X, axis=0)         
    quantiles = np.nanpercentile(X, q=[25,75], axis=0) 
    scale = quantiles[1] - quantiles[0] 
    scale = _handle_zeros_in_scale(scale, copy=False)  
    
    adjust = (stats.norm.ppf(75 / 100.0) - stats.norm.ppf(25 / 100.0)) 
    scale = scale / adjust 

    return center, scale

def _handle_zeros_in_scale(scale, copy=True):
    """Makes sure that whenever scale is zero, we handle it correctly.
    This happens in most scalers when we have constant features.
    """

    # if we are fitting on 1D arrays, scale might be a scalar
    if np.isscalar(scale):
        if scale == .0: 
            scale = 1.
        return scale
    elif isinstance(scale, np.ndarray):
        if copy:
            # New array to avoid side-effects
            scale = scale.copy()
        scale[scale == 0.] = 1.0 
        return scale 
    
def get_trial_type(trial_str):
    if trial_str == 'correct': 
        trial_type = 1 
    elif trial_str == 'incorrect': 
       trial_type = 2 
    else: 
       trial_type = 0

    return trial_type

def get_coding_direction(X_S1, X_S2, **kwargs): 
    ''' returns coding direction defined as 
                       Delta = <X_S1>_trials - <X_S2>_trials 
    
    inputs: - X_Si is n_trials x n_neurons x n_time/n_epochs 
            - epoch in ['ED', 'MD', 'LD'] 
    
    outputs: - Delta (n_neurons x n_time/n_epochs) 
    ''' 
    
    if kwargs['feature_sel']=='ttest_ind': 
        # # # independant ttest 
        # _, p_val = stats.ttest_ind(X_S1, X_S2, equal_var = False, nan_policy='propagate', axis=0) 
        # _, p_val = stats.mannwhitneyu(X_S1, X_S2, axis=0) 
        # # _, p_val = stats.chisquare(X_S1, X_S2, ddof=0, axis=0) 

        # X_S1_sel = X_S1.copy() 
        # X_S2_sel = X_S2.copy()         

        # X_S1_sel[:, p_val>=kwargs['pval']] = 0
        # X_S2_sel[:, p_val>=kwargs['pval']] = 0 
        
        if X_S1.shape[-1]<=4:
            bins = -1
        else:
            bins = np.arange(0, 12)

        X_S1_sel = X_S1.copy() 
        X_S2_sel = X_S2.copy()         
        
        X_S1_BL = np.mean(X_S1[..., bins], axis=-1) 
        _, p_val = stats.ttest_ind(X_S1, X_S1_BL[..., np.newaxis], equal_var = False, nan_policy='propagate', axis=0) 
        # _, p_val = stats.mannwhitneyu(X_S1,  X_S1_BL[..., np.newaxis], axis=0) 
        X_S1_sel[:, p_val>=kwargs['pval']] = 0 
        
        X_S2_BL = np.mean(X_S2[...,bins], axis=-1) 
        _, p_val = stats.ttest_ind(X_S2, X_S2_BL[..., np.newaxis], equal_var = False, nan_policy='propagate', axis=0) 
        # _, p_val = stats.mannwhitneyu(X_S2,  X_S2_BL[..., np.newaxis], axis=0) 
        X_S2_sel[:, p_val>=kwargs['pval']] = 0 
        
        # # diff of population vectors averaged over trials 
        # Delta = np.nanmean(X_S1, axis=0) - np.nanmean(X_S2, axis=0) 
        Delta = np.median(X_S1_sel, axis=0) - np.median(X_S2_sel, axis=0) 
        
    # using lasso 
    elif kwargs['feature_sel']=='lasso': 
        Delta = - get_coefs(X_S1.copy(), X_S2.copy(), **kwargs).T 
        # print('Delta', Delta.shape) 
    return Delta 

def unit_vector(vector): 
    """ Returns the unit vector of the vector.  """ 
    norm = np.linalg.norm(vector, axis=0) 
    
    if norm>0: 
        u = vector / norm  
    else: 
        u = np.zeros(vector.shape) 
    
    return u 

def cos_between(v1, v2): 
    """ Returns the angle in radians between vectors 'v1' and 'v2' """
    
    v1_u = unit_vector(v1) 
    v2_u = unit_vector(v2) 
    return np.clip( np.dot(v1_u, v2_u), -1.0, 1.0) 

def get_cos_sel(X_S1, X_S2, return_Delta=0, **kwargs): 
    
    pval=kwargs['pval'] 
    bins=kwargs['bins'] 
    Delta0=kwargs['Delta0'] 
    j_epoch=0 
    
    Delta = get_coding_direction(X_S1, X_S2, **kwargs) 
    # print('Delta', Delta.shape, 'bins', bins) 
    
    if bins is not None:
        if Delta0 is not None:
            try :
                Delta_bins = np.nanmean( Delta0[:, bins], axis=-1)
            except:
                Delta_bins = Delta0[:, 0] 
        else:
            Delta_bins = np.nanmean( Delta[:, bins], axis=-1) 
        
        # print('Delta_bins', Delta_bins.shape) 
        
        cosine = np.zeros( Delta.shape[-1] ) 
        for i_epoch in range(Delta.shape[-1]): 
            cosine[i_epoch] = cos_between(Delta_bins, Delta[:, i_epoch]) 
    
    elif Delta0 is not None: 
        cosine = np.zeros( Delta.shape[-1] ) 
        for i_epoch in range(Delta.shape[-1]): 
            cosine[i_epoch] = cos_between(Delta0[:,0], Delta[:, i_epoch]) 
    else: 
        cosine = np.zeros( Delta.shape[-1] ) 
        for i_epoch in range(Delta.shape[-1]): 
            if i_epoch==0 : 
                dum = np.arange(3) 
                dum = dum[dum!=j_epoch] 
                cosine[i_epoch] = cos_between(Delta[:, dum[0]], Delta[:, dum[1]]) 
            else: 
                cosine[i_epoch] = cos_between(Delta[:, j_epoch], Delta[:, i_epoch]) 
    
    if return_Delta : 
        return cosine, Delta 
    else: 
        return cosine 
    
def get_overlap(X_D1, X_D2, Delta_sample, **kwargs): 
    
    # X_S1, X_S2 = scale_data(X_S1, X_S2, scaler=standardize) 
    # Delta_sample = get_coding_direction(X_S1, X_S2, pval) 
    
    X_D1, X_D2 = scale_data(X_D1, X_D2, scaler=kwargs['scaler'])
    Delta_distractor = get_coding_direction(X_D1, X_D2, **kwargs) 
    
    overlap = np.zeros( Delta_sample.shape[-1] )
    
    if kwargs['bins'] is not None: 
        Delta_bins = np.nanmean( Delta_sample[:, kwargs['bins']], axis=-1) 
        
        for i_epoch in range(Delta_sample.shape[-1]): 
            overlap[i_epoch] = cos_between(Delta_bins, Delta_distractor[:, i_epoch]) 
    else: 
        for i_epoch in range(Delta_sample.shape[-1]): 
            overlap[i_epoch] = cos_between(Delta_sample[:,i_epoch], Delta_distractor[:, i_epoch]) 
    
    return overlap 

def get_frac_sel(X_S1, X_S2, pval=0.05): 
    ''' returns fraction of sel neurons (permutation test) 
    inputs: - X_Si is n_trials x n_neurons 
    outputs: - frac_sel (float) 
    ''' 
    
    # # independant ttest 
    _, p_val = stats.ttest_ind(X_S1, X_S2, equal_var = False, nan_policy='propagate', axis=0) 
    
    if p_val.ndim>1: 
        frac_sel = np.zeros( X_S1.shape[-1] ) 
        for i_epoch in range( X_S1.shape[-1]): 
            idx_sel = np.where(p_val[:, i_epoch]<pval)[0] 
            frac_sel[i_epoch] = len(idx_sel) / len(p_val) 
    else: 
        idx_sel = np.where(p_val<pval)[0] 
        frac_sel = len(idx_sel) / len(p_val) 
    
    return frac_sel 

def average_distance(X_S1, X_S2, pval=.05):
    
    _, p_val = stats.ttest_ind(X_S1, X_S2, equal_var = False, nan_policy='propagate', axis=0) 

    X_S1_sel = X_S1.copy() 
    X_S1_sel[:, p_val>=pval] = 0
                            
    X_S2_sel = X_S2.copy() 
    X_S2_sel[:, p_val>=pval] = 0 
    
    # if X_S1.shape[-1]<=4:
    #     bins = -1
    # else:
    #     bins = np.arange(0, 12)
    
    # X_S1_BL = np.mean(X_S1[..., bins], axis=-1) 
    # _, p_val = stats.ttest_ind(X_S1, X_S1_BL[..., np.newaxis], equal_var = False, nan_policy='propagate', axis=0) 
    # # _, p_val = stats.mannwhitneyu(X_S1,  X_S1_BL[..., np.newaxis], axis=0) 
    # X_S1[:, p_val>=pval] = 0 
        
    # X_S2_BL = np.mean(X_S2[...,bins], axis=-1) 
    # _, p_val = stats.ttest_ind(X_S2, X_S2_BL[..., np.newaxis], equal_var = False, nan_policy='propagate', axis=0) 
    # # _, p_val = stats.mannwhitneyu(X_S2,  X_S2_BL[..., np.newaxis], axis=0) 
    # X_S2[:, p_val>=pval] = 0 
    
    distance = 0
    for i_trial in range(X_S1.shape[0]):
        for j_trial in range(X_S2.shape[0]): 
            # distance += np.sqrt( X_S1[i_trial]**2 - X_S2[j_trial]**2 ) 
            distance += np.linalg.norm(X_S1_sel[i_trial]-X_S2_sel[j_trial], axis=0) 
    
    return distance / X_S1.shape[0] / X_S2.shape[0] 

def get_norm_sel(X_S1, X_S2, **kwargs): 
    ''' returns norm of coding direction 
    inputs: - X_Si is n_trials x n_neurons x time
    outputs: - norm_ (float) 
    '''
    
    norm_ = average_distance(X_S1, X_S2, kwargs['pval']) 
    # Delta = get_coding_direction(X_S1, X_S2, **kwargs) 
    # # # print('Delta', Delta.shape) 
    # norm_ = np.linalg.norm(Delta, axis=0) 
    # # # print('norm_', norm_.shape) 
    
    if norm_.shape[0]<=4 :         
        norm_ = (norm_ - norm_[-1]) 
    else: 
        bins = np.arange(0, 12) 
        norm_bins = np.nanmean( norm_[bins] ) 
        norm_ -= norm_bins
        # norm_ /= norm_bins 
        # max_ = np.amax(norm_) 
        # norm_ /= max_ 
    
    return norm_ 

def get_proj(X_S1, X_S2, return_Delta=0, **kwargs): 
    
    bins=kwargs['bins'] 
    Delta0=kwargs['Delta0'] 
    
    # if kwargs['Delta0'] is not None:
    _, p_val = stats.ttest_ind(X_S1, X_S2, equal_var = False, nan_policy='propagate', axis=0) 

    # X_S1_sel = X_S1
    # X_S2_sel = X_S2
    
    X_S1_sel = X_S1.copy() 
    X_S2_sel = X_S2.copy() 
    
    X_S1_sel[:, p_val>=kwargs['pval']] = 0 
    X_S2_sel[:, p_val>=kwargs['pval']] = 0 
    
    non_zero = sum(p_val<kwargs['pval']) 
    
    X_S1_avg = np.median(X_S1_sel, axis=0) 
    X_S2_avg = np.median(X_S2_sel, axis=0) 
    
    Delta = X_S1_avg - X_S2_avg 
    # Delta = Delta / np.linalg.norm(Delta, axis=0) 
    # else:
    #     Delta = Delta0
        
    if bins is not None: 
        if Delta0 is not None: 
            # Delta0_norm = Delta0 / np.linalg.norm(Delta0, axis=0) 
            try :
                Delta_bins = np.nanmean( Delta0[:, bins], axis=-1)
            except:
                Delta_bins = Delta0[:, 0] 
        else: 
            Delta_bins = np.nanmean( Delta[:, bins], axis=-1) 
        
        Delta_bins /= np.linalg.norm(Delta_bins) 
        
        X_proj = np.zeros( Delta.shape[-1] ) 
        
        # for i_epoch in range(Delta.shape[-1]): 
        #     X_proj[i_epoch] = ( np.dot(Delta_bins, X_S1_avg[:, i_epoch]) - np.dot(Delta_bins, X_S2_avg[:, i_epoch]) ) / 2.0 
        
        if kwargs['sample'] == 'S1_S2':
            X_S1_avg = np.mean( X_S1, axis=0 ) 
            X_S2_avg = np.mean( X_S2, axis=0 ) 
            
            # X_S1_avg /= np.linalg.norm(X_S1_avg, axis=0) 
            # X_S2_avg /= np.linalg.norm(X_S2_avg, axis=0) 
            
            X_proj= ( np.dot(Delta_bins,  X_S1_avg ) +  np.dot(Delta_bins,  X_S2_avg ) ) / 2.0 
            
            # for i_epoch in range(Delta.shape[-1]): 
            #     X_proj[i_epoch] = ( np.mean( np.dot(Delta_bins, X_S1[..., i_epoch].T), axis=0 ) 
            #                         - np.mean( np.dot(Delta_bins, X_S2[..., i_epoch].T), axis=0 ) ) / 2.0 
        elif kwargs['sample'] == 'S1':
            # print('S1')
            # X_S1_avg = np.mean( X_S1, axis=0 ) 
            # # X_S1_avg /= np.linalg.norm(X_S1_avg, axis=0) 
            # X_proj= np.dot(Delta_bins,  X_S1_avg ) 
            for i_epoch in range(Delta.shape[-1]): 
                X_proj[i_epoch] = np.mean( np.dot(Delta_bins, X_S1[..., i_epoch].T), axis=0 )  
            
        elif kwargs['sample'] == 'S2': 
            # print('S2')
            # X_S2_avg = np.mean( X_S2, axis=0 ) 
            # # X_S2_avg /= np.linalg.norm(X_S2_avg, axis=0) 
            # X_proj = np.dot(Delta_bins,  X_S2_avg ) 
            for i_epoch in range(Delta.shape[-1]): 
                X_proj[i_epoch] = np.mean( np.dot(Delta_bins, X_S2[..., i_epoch].T), axis=0 ) 
    
    elif Delta0 is not None:
        # X_proj = np.zeros( Delta.shape[-1] ) 
        # for i_epoch in range(Delta.shape[-1]): 
        #     X_proj[i_epoch] = (np.dot(Delta0[:,i_epoch], X_S1_avg[:, i_epoch])
        #                        - np.dot(Delta0[:,i_epoch], X_S2_avg[:, i_epoch]) ) / 2.0 
        
        X_proj = np.zeros( Delta.shape[-1] ) 
        for i_epoch in range(Delta.shape[-1]): 
            X_proj[i_epoch] = ( np.mean( np.dot(Delta0[:,i_epoch], X_S1[..., i_epoch].T), axis=0 ) 
                                - np.mean( np.dot(Delta0[:,i_epoch], X_S2[..., i_epoch].T), axis=0 ) ) / 2.0  
    else: 
        # X_proj = np.zeros( Delta.shape[-1] ) 
        # for i_epoch in range(Delta.shape[-1]): 
        #     X_proj[i_epoch] = (np.dot(Delta[:,i_epoch], X_S1_avg[:, i_epoch])
        #                        - np.dot(Delta[:,i_epoch], X_S2_avg[:, i_epoch]) ) / 2.0 
        
        X_proj = np.zeros( Delta.shape[-1] ) 
        for i_epoch in range(Delta.shape[-1]): 
            X_proj[i_epoch] = ( np.mean( np.dot(Delta[:,i_epoch], X_S1[..., i_epoch].T), axis=0 ) 
                                - np.mean( np.dot(Delta[:,i_epoch], X_S2[..., i_epoch].T), axis=0 ) ) / 2.0 
    
    if return_Delta : 
        return X_proj , Delta 
    else:
        return X_proj 
    
def get_sel(X_S1, X_S2, return_Delta=0, **kwargs):
    
    if kwargs['obj']!='score' and kwargs['obj']!='coefs' : 
        X_S1, X_S2 = scale_data(X_S1, X_S2, scaler=kwargs['scaler'], center=kwargs['center'], scale=kwargs['scale']) 
    
    if kwargs['obj']=='cos':
        if return_Delta :
            out, Delta = get_cos_sel(X_S1, X_S2, return_Delta=1, **kwargs) 
        else:
            out = get_cos_sel(X_S1, X_S2, return_Delta=0, **kwargs)
    
    elif kwargs['obj']=='norm': 
        out = get_norm_sel(X_S1, X_S2, **kwargs) 
    elif kwargs['obj']=='frac': 
        out = get_frac_sel(X_S1, X_S2, pval=kwargs['pval']) 
    elif kwargs['obj']=='proj':
        # out = get_score(X_S1, X_S2, **kwargs) 
        if return_Delta:
            out, Delta = get_proj(X_S1, X_S2, return_Delta=1, **kwargs) 
        else: 
            out = get_proj(X_S1, X_S2, return_Delta=0, **kwargs)
    
    elif kwargs['obj']=='score':
        out = get_score(X_S1, X_S2, **kwargs)
        
    elif kwargs['obj']=='coefs': 
        out = get_coefs(X_S1, X_S2, **kwargs) 
    
    if return_Delta : 
        return out, Delta 
    else: 
        return out 
    
