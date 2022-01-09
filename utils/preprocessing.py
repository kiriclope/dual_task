import numpy as np
from scipy.signal import savgol_filter, butter, filtfilt 
import scipy.stats as stats 
from sklearn.preprocessing import StandardScaler 
from sklearn.feature_selection import SelectKBest, chi2, VarianceThreshold, f_regression, mutual_info_classif, f_classif

from . import constants as gv 

def preprocess_X_S1_X_S2(X_S1, X_S2, scaler='standard', center=None, scale=None, avg_mean=0, avg_noise=0, unit_var=0, return_center_scale=0):
    
    X = np.vstack( (X_S1, X_S2) )

    # X = savgol_filter(X, int(np.ceil(gv.frame_rate/2.0) * 2 + 1), polyorder = gv.SAVGOL_ORDER, deriv=0, axis=-1, mode='mirror') 
    
    if scaler=='standard':
        X_scale, center, scale = standard_scaler_BL(X, center, scale, avg_mean, avg_noise) 
    elif scaler=='robust': 
        X_scale = robust_scaler_BL(X, center, scale, avg_mean, avg_noise, unit_var) 
    elif scaler=='center': 
        X_scale = center_BL(X, center, avg_mean) 
    else:
        X_scale = X 
    
    # X_scale = savgol_filter(X_scale, int(np.ceil(gv.frame_rate/2.0) * 2 + 1), polyorder = gv.SAVGOL_ORDER, deriv=0, axis=-1, mode='mirror') 

    if return_center_scale :
        return X_scale[:X_S1.shape[0]], X_scale[X_S1.shape[0]:], center, scale 
    else:
        return X_scale[:X_S1.shape[0]], X_scale[X_S1.shape[0]:]

def butter_lowpass_filter(data, cutoff, fs, order):
    normal_cutoff = cutoff / nyq 
    # Get the filter coefficients 
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    y = filtfilt(b, a, data)
    return y 

def center(X): 
    scaler = StandardScaler(with_mean=True, with_std=False) 
    Xc = scaler.fit_transform(X.T).T 
    return Xc 

def z_score(X): 
    scaler = StandardScaler()
    
    if X.ndim>3: # task x sample x trial x neurons x time 
        Xz = X
        for i_task in range(3): 
            for i_sample in range(2): 
                for i in range(X.shape[2]): 
                    Xt = X[i_task, i_sample, i] 
                    if gv.Z_SCORE: 
                        Xz[i_task, i_sample, i] = scaler.fit_transform(Xt.T).T 
                    elif gv.Z_SCORE_BL: 
                        scaler.fit(Xt[...,gv.bins_BL].T) 
                        Xz[i_task, i_sample, i] = scaler.transform(Xt.T).T 
                        
    elif X.ndim>2: # trial x neurons x time 
        Xz = X 
        for i in range(X.shape[0]): 
            Xt = X[i] 
            if gv.Z_SCORE: 
                Xz[i] = scaler.fit_transform(Xt.T).T 
            elif gv.Z_SCORE_BL:
                scaler.fit(Xt[...,gv.bins_BL].T)  
                Xz[i] = scaler.transform(Xt.T).T 
                
                # mean = np.mean(Xt[:,gv.bins_BL],axis=1) # mean over baseline bins 
                
                # std = np.std(Xt[:,gv.bins_BL],axis=1) # std over baseline bins 
                # Xz[i] = ( Xt-mean[:,np.newaxis] ) /std[:,np.newaxis]
                
                # Xz[i] = ( Xt-mean[:,np.newaxis] ) 
                         
                # for i_neuron in range(X.shape[1]): 
                #     Xn = X[:,i_neuron] 
                #     std = np.std(Xn[:,gv.bins_BL]) # std over baseline in all trials 
                #     Xz[i, i_neuron] = Xz[i, i_neuron]/std
                
    else: # trial x neurons 
        
        if gv.Z_SCORE: 
            Xz = scaler.fit_transform(X.T).T 
        elif gv.Z_SCORE_BL : 
            scaler.fit(X[...,gv.bins_BL].T) 
            Xz = scaler.transform(X.T).T
            
    return Xz 

def z_score_trials(X):

    print('X', X.shape) 
    # average over trials
    if X.ndim>3: # task x sample x trial x neurons x time
        z_trials = np.zeros(X.shape)
        
        for i_task in range(3):
            X_BL = X[i_task, ..., gv.bins_BL]
            X_BL = np.moveaxis(X_BL, 0, -1)
            print('X_BL', X_BL.shape)
            
            m = np.nanmean( X_BL, axis=-1) # average for each trial over BL 
            
            X_task = np.vstack(X[i_task]) 
            X_task = X_task[..., gv.bins_BL] 
            X_task = np.hstack(X_task) 
            print('X_task', X_task.shape) 
            
            s = np.nanstd( X_task, axis=-1 ) 
            s[s == 0.0] = 1.0 
            
            print('m', m.shape, np.mean(m), 's', s.shape, np.std(s) ) 
            z_trials[i_task] = ( X[i_task] - m[..., np.newaxis] ) \
                               / s[np.newaxis, np.newaxis, :, np.newaxis] 
        
    else: # trials x neurons x time 
        X_BL = X[..., gv.bins_BL]
        
        print('X_BL', X_BL.shape) 
        
        X_trials = np.hstack(X_BL) 
        print('X_trials', X_trials.shape)
        
        m = np.nanmean( X_BL, axis=-1) 
        s = np.nanstd( X_BL, axis=-1) 
        s[s == 0.0] = 1.0 
        
        print('m', m.shape, np.mean(m), 's', s.shape, np.std(s)) 
        z_trials = ( X - m[..., np.newaxis] ) / s[np.newaxis, :, np.newaxis] 
        # z_trials = ( X - m[np.newaxis, ..., np.newaxis] ) / s[np.newaxis, :, np.newaxis] 
    
    return z_trials 

def normalize(X):
    # Xmin = np.amin(X, axis=-1) 
    # Xmax = np.amax(X, axis=-1) 
    
    Xmin = np.amin(X[..., gv.bins_BL], axis=-1) 
    Xmax = np.amax(X[..., gv.bins_BL], axis=-1) 
    
    Xmin = Xmin[..., np.newaxis] 
    Xmax = Xmax[..., np.newaxis] 
    
    return (X-Xmin)/(Xmax-Xmin+gv.eps) 

def normalize_trials(X):
    avg_trials = np.mean(X, axis=0)
    avg_trials = avg_trials[np.newaxis, ...]
    
    Xmin = np.amin(avg_trials[..., gv.bins_BL], axis=-1) 
    Xmax = np.amax(avg_trials[..., gv.bins_BL], axis=-1) 
    
    Xmin = Xmin[..., np.newaxis] 
    Xmax = Xmax[..., np.newaxis] 
    
    return (X-Xmin)/(Xmax-Xmin+gv.eps) 

def dFF0_remove_silent(X): 
    ''' N_trials, N_neurons, N_times '''

    print('X', X.shape) 
    if X.ndim>3:
        X_stack = np.vstack( np.vstack(X) )
    else:
        X_stack = X 
    
    print('X_stack', X_stack.shape) 
    F0 = np.mean( np.mean(X_stack[..., gv.bins_BL], axis=-1), axis=0) 
    print('X', X.shape, 'X_stack', X_stack.shape, 'F0', F0.shape, F0[:5]) 
    F0 = F0[np.newaxis,:, np.newaxis] 
    print('X', X.shape, 'X_stack', X_stack.shape, 'F0', F0.shape) 
    
    # if gv.AVG_F0_TRIALS: 
    #     F0 = np.mean( np.mean(X[...,gv.bins_BL],axis=-1), axis=0) 
    #     F0 = F0[np.newaxis,:, np.newaxis] 
    # else:
    #     F0 = np.mean(X[...,gv.bins_BL],axis=-1) 
    #     # F0 = np.percentile(X, 1, axis=-1) 
    #     F0 = F0[..., np.newaxis] 
    
    # print('X', X.shape,  X[0,0, 0:3])
    # print('F0', F0.shape, F0[0,0,0:3]) 
    
    # if gv.F0_THRESHOLD is not None: 
    #     # removing silent neurons 
        
    #     # idx = np.where(F0<=gv.F0_THRESHOLD) 
    #     # F0 = np.delete(F0, idx, axis=-2) 
    #     # X = np.delete(X, idx, axis=-2)

    # idx = np.where(F0<=0.00001)
    # F0[idx] = np.nan 
    # X[idx] = np.nan 
        
    # print('X', X.shape, 'F0', F0.shape)
    
    # F0 = _handle_zeros_in_scale(F0, copy=False)
    
    # return (X-F0) / F0 
    return (X-F0) / _handle_zeros_in_scale(F0, copy=False) 
    
def dFF0(X): 
    if not gv.AVG_F0_TRIALS: 
        # F0 = np.mean(X[...,gv.bins_BL],axis=-1)        
        F0 = np.percentile(X, 15, axis=-1) 
        F0 = F0[..., np.newaxis] 
    else: 
        F0 = np.mean( np.mean(X[...,gv.bins_BL],axis=-1), axis=0) 
        F0 = F0[np.newaxis,:, np.newaxis]

    F0 = _handle_zeros_in_scale(F0, copy=False)
    
    return (X-F0) / (F0 + gv.eps) 

def dF(X): 
    if not gv.AVG_F0_TRIALS: 
        F0 = np.mean(X[...,gv.bins_BL],axis=-1)        
        # F0 = np.percentile(X, 15, axis=-1) 
        F0 = F0[..., np.newaxis] 
    else: 
        F0 = np.mean( np.mean(X[...,gv.bins_BL],axis=-1), axis=0) 
        F0 = F0[np.newaxis,:, np.newaxis]
        
    return (X-F0)

def bin_data(data, bin_step, bin_size):
    # bin_step number of pts btw bins, bin_size number of size in each bin
    bin_array = [np.mean(np.take(data,np.arange(int(i*bin_step),int(i*bin_step+bin_size)), axis=2), axis=2) for i in np.arange(data.shape[2]//bin_step-1)]
    bin_array = np.array(bin_array)
    bin_array = np.rollaxis(bin_array,0,3)
    return bin_array

def avg_epochs(X, epochs=None): 

    X_avg = np.mean(X, axis=-1) 
    X_epochs = np.empty( tuple([len(gv.epochs)])+ X_avg.shape ) 
    # print('X', X_epochs.shape, 'X_avg', X_avg.shape) 
    # print('start', gv.bin_start, 'epochs', gv.epochs) 

    if epochs==None:
        epochs = gv.epochs
        
    # print('average over epochs', epochs) 
    
    for i_epoch, epoch in enumerate(epochs):
        
        if epoch=='BL':
            X_BL = np.nanmean(X[...,gv.bins_BL],axis=-1) 
            X_epochs[i_epoch] = X_BL
        elif epoch == 'STIM':
            X_STIM = np.nanmean(X[...,gv.bins_STIM],axis=-1) 
            X_epochs[i_epoch] = X_STIM
        elif epoch == 'ED':            
            X_ED = np.nanmean(X[...,gv.bins_ED],axis=-1)
            # print('X_ED', X_ED.shape, 'bins', gv.bins_ED) 
            X_epochs[i_epoch] = X_ED
        elif epoch == 'DIST':
            X_DIST = np.nanmean(X[...,gv.bins_DIST],axis=-1) 
            X_epochs[i_epoch] = X_DIST
        elif epoch == 'MD':
            X_MD = np.nanmean(X[...,gv.bins_MD],axis=-1) 
            X_epochs[i_epoch] = X_MD
        elif epoch == 'CUE':
            X_CUE = np.nanmean(X[...,gv.bins_CUE],axis=-1) 
            X_epochs[i_epoch] = X_CUE
        elif epoch == 'LD':
            X_LD = np.nanmean(X[...,gv.bins_LD],axis=-1) 
            X_epochs[i_epoch] = X_LD
        elif epoch=='RWD':
            X_RWD = np.nanmean(X[...,gv.bins_RWD],axis=-1) 
            X_epochs[i_epoch] = X_RWD 
        elif epoch == 'TEST':
            X_TEST = np.nanmean(X[...,gv.bins_TEST],axis=-1) 
            X_epochs[i_epoch] = X_TEST 
        elif epoch == 'DELAY':
            X_DELAY = np.nanmean(X[...,gv.bins_DELAY],axis=-1) 
            X_epochs[i_epoch] = X_DELAY 
        
    X_epochs = np.moveaxis(X_epochs,0,-1)  
    
    return X_epochs 

def prescreening(X, y, alpha=0.05, scoring=f_classif): 
    ''' X is trials x neurons 
    alpha is the level of significance 
    scoring is the statistics, use f_classif or mutual_info_classif 
    '''
    
    model = SelectKBest(score_func=scoring, k=X.shape[1])    
    model.fit(X,y) 
    pval = model.pvalues_.flatten() 
    non_sel = np.argwhere(pval>alpha) 
    X_sel = np.delete(X, non_selected, axis=1) 
    return X_sel 

def _handle_zeros_in_scale(scale, copy=True):
    """Makes sure that whenever scale is zero, we handle it correctly.
    This happens in most scalers when we have constant features.
    """

    # if we are fitting on 1D arrays, scale might be a scalar
    if np.isscalar(scale):
        if scale == 0.0:
            scale = 1.
        return scale
    elif isinstance(scale, np.ndarray):
        if copy:
            # New array to avoid side-effects
            scale = scale.copy()
        scale[scale == 0.0] = 1.0
        return scale

def standard_scaler_BL(X, center=None, scale=None, avg_mean=0, avg_noise=0):

    X_BL = X[..., gv.bins_BL]
    if center is None:
        if avg_mean:
            print('avg mean over trials')
            center = np.mean(np.hstack(X_BL), axis=-1)
        else:
            center = np.nanmean(X_BL, axis=-1) 
                
    if scale is None:
        if avg_noise:
            print('avg noise over trials')
            scale = np.std(np.hstack(X_BL), axis=-1) 
            # scale = np.nanstd(np.reshape(X_BL, (X_BL.shape[0], X_BL.shape[1], X_BL.shape[3], X_BL.shape[2]*X_BL.shape[4])) , axis=-1)
        else:
            scale = np.nanstd(X_BL, axis=-1)

        scale = _handle_zeros_in_scale(scale, copy=False) 
        
    if avg_mean and avg_noise:
        X_scale = ( X - center[np.newaxis, ..., np.newaxis] ) / scale[np.newaxis, ..., np.newaxis] 
    elif avg_mean:
        X_scale = ( X - center[np.newaxis, ..., np.newaxis] ) / scale[..., np.newaxis] 
    elif avg_noise:    
        X_scale = ( X - center[..., np.newaxis] ) / scale[np.newaxis, ..., np.newaxis] 
    else:        
        X_scale = ( X - center[..., np.newaxis] ) / scale[..., np.newaxis] 
    
    return X_scale, center, scale 
    
def robust_scaler_BL(X, center=None, scale=None, avg_mean=0, avg_noise=0, unit_var=0):
    
    X_BL = X[..., gv.bins_BL] 

    if center is None:
        if avg_mean:
            print('avg mean over trials')
            center = np.nanmedian(np.hstack(X_BL), axis=-1) 
        else:
            center = np.nanmedian(X_BL, axis=-1)
    
    if scale is None:
        if avg_noise:
            print('avg noise over trials')
            quantiles = np.nanpercentile(np.hstack(X_BL), q=[25, 75], axis=-1) 
        else: 
            quantiles = np.nanpercentile(X_BL, q=[25,75], axis=-1)
        
        scale = quantiles[1] - quantiles[0] 
        scale = _handle_zeros_in_scale(scale, copy=False)  
        
        if unit_var:
            adjust = (stats.norm.ppf(75 / 100.0) - stats.norm.ppf(25 / 100.0)) 
            scale = scale / adjust 
    
    if avg_mean and avg_noise:
        X_scale = ( X - center[np.newaxis, ..., np.newaxis] ) / scale[np.newaxis, ..., np.newaxis] 
    elif avg_mean:
        X_scale = ( X - center[np.newaxis, ..., np.newaxis] ) / scale[..., np.newaxis] 
    elif avg_noise: 
        X_scale = ( X - center[..., np.newaxis] ) / scale[np.newaxis, ..., np.newaxis] 
    else:
        X_scale = ( X - center[..., np.newaxis] ) / scale[..., np.newaxis] 
    
    return X_scale 

def center_BL(X, center=None, avg_mean=0):
    
    if center is None:
        X_BL = X[..., gv.bins_BL]
        
        if avg_mean:
            print('avg mean over trials')
            center = np.mean(np.hstack(X_BL), axis=-1)
        else:
            center = np.nanmean(X_BL, axis=-1)
            
    if avg_mean:
        X_scale = X-center[np.newaxis, ..., np.newaxis] 
    else: 
        X_scale = X-center[..., np.newaxis] 
    
    return X_scale

def preprocess_X(X, scaler='standard', center=None, scale=None, avg_mean=0, avg_noise=0):
    
    # if gv.SAVGOL: 
    #     X = savgol_filter(X, int(np.ceil(gv.frame_rate/2.0) * 2 + 1), polyorder = gv.SAVGOL_ORDER, deriv=0, axis=-1, mode='mirror')

    print('scaler BL', scaler) 
    if scaler=='standard':
        X_scale, center, scale = standard_scaler_BL(X, center, scale, avg_mean, avg_noise) 
    elif scaler=='robust': 
        X_scale = robust_scaler_BL(X, center, scale) 
    elif scaler=='center': 
        X_scale = center_BL(X, center) 
    else:
        X_scale = X 
    # X = dFF0_remove_silent(X) 
    
    # if gv.SAVGOL: 
    #     X = savgol_filter(X, int(np.ceil(gv.frame_rate/2.0) * 2 + 1), polyorder = gv.SAVGOL_ORDER, deriv=0, axis=-1, mode='mirror') 
    
    # if gv.F0_THRESHOLD is not None: 
    #     X = dFF0_remove_silent(X) 
    #     # print(X.shape) 
    #     # gv.n_neurons = X.shape[1] 
        
    # if gv.DECONVOLVE: 
    #     if X.ndim>3: # task x sample x trial x neurons x time 
    #         for i_task in range(3): 
    #             for i_sample in range(2): 
    #                 X[i_task, i_sample] = deconvolveFluo(X[i_task, i_sample]) 
    #     else:
    #         X = deconvolveFluo(X) 
    # else:            
        
    #     if gv.Z_SCORE | gv.Z_SCORE_BL :
    #         # print('z_score')
    #         X = z_score(X)
            
    #     if gv.Z_SCORE_TRIALS:
    #         X = z_score_trials(X) 
            
    #     if gv.NORMALIZE:
    #         # print('normalize') 
    #         X = normalize(X)
            
    #     if gv.NORMALIZE_TRIALS: 
    #         # print('normalize') 
    #         X = normalize_trials(X) 
            
    #     # if gv.DETREND:
    #     #     X = detrend_X(X, order=gv.DETREND_ORDER)
    
    return X_scale
