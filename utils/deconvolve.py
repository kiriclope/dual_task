import numpy as np
from joblib import Parallel, delayed, parallel_backend 
from oasis.functions import deconvolve 

from . import constants as gv 
from . import progressbar as pg  

def deconvolveFluo(X):

    # F0 = np.empty( (X.shape[0], X.shape[1]) ) 
    # F0[:] = np.mean( np.mean(X[...,gv.bins_BL],axis=-1), axis=0 ) 
    F0 = np.mean(X[..., gv.bins_BL], axis=-1)
    
    # F0 = np.percentile(X, 15, axis=-1) 
    
    # def F0_loop(X, n_trial, n_neuron, bins): 
    #     X_ij = X[n_trial, n_neuron]        
    #     c, s, b, g, lam = deconvolve(X_ij, penalty=1) 
    #     return b
    
    # # loop over trials and neurons 
    # with pg.tqdm_joblib(pg.tqdm(desc='F0', total=X.shape[0]*X.shape[1])) as progress_bar: 
    #     F0 = Parallel(n_jobs=gv.num_cores)(delayed(F0_loop)(X, n_trial, n_neuron, gv.bins_BL) 
    #                                         for n_trial in range(X.shape[0]) 
    #                                         for n_neuron in range(X.shape[1]) )
        
    # F0 = np.array(F0).reshape( (X.shape[0], X.shape[1]) ) 
    
    # def X_loop(X, F0, n_trial, n_neuron):
    #     X_ij = X[n_trial, n_neuron]
    #     F0_ij = F0[n_trial, n_neuron]
    #     c, s, b, g, lam = deconvolve(X_ij, penalty=1, b=F0_ij) 
    #     return c 
    
    def S_loop(X, F0, n_trial, n_neuron):
        X_ij = X[n_trial, n_neuron]
        F0_ij = F0[n_trial, n_neuron]
        c, s, b, g, lam = deconvolve(X_ij, penalty=1) 
        # c, s, b, g, lam = deconvolve(X_ij, penalty=1, b=F0_ij) 
        return s 
    
    # # loop over trials and neurons 
    # with pg.tqdm_joblib(pg.tqdm(desc='denoise', total=X.shape[0]*X.shape[1])) as progress_bar: 
    #     X_dcv = Parallel(n_jobs=gv.num_cores)(delayed(X_loop)(X, F0, n_trial, n_neuron) 
    #                                         for n_trial in range(X.shape[0]) 
    #                                         for n_neuron in range(X.shape[1]) ) 
    # X_dcv = np.array(X_dcv).reshape(X.shape) 
    
    with pg.tqdm_joblib(pg.tqdm(desc='deconvolve', total=X.shape[0]*X.shape[1])) as progress_bar: 
        S_dcv = Parallel(n_jobs=gv.num_cores)(delayed(S_loop)(X, F0, n_trial, n_neuron) 
                                              for n_trial in range(X.shape[0]) 
                                              for n_neuron in range(X.shape[1]) ) 
        
    S_dcv = np.array(S_dcv).reshape(X.shape)    
    # S_flt = savgol_filter(S_dcv, int(np.ceil(gv.frame_rate / 2.) * 2 + 1), polyorder = 5, deriv=0)
    
    def threshold_spikes(S_dcv, threshold): 
        # S_dcv[S_dcv<=threshold] = 0 
        # S_dcv[S_dcv>threshold] = 1 
        # S_dcv = uniform_filter1d( S_dcv, int(gv.frame_rate/2) ) 
        return S_dcv*1000
    
    S_th = threshold_spikes(S_dcv, gv.DCV_THRESHOLD)  
    S_avg = np.mean(S_th[...,gv.bins_BL],axis=-1) 
    S_avg = S_avg[..., np.newaxis]
    
    print('X_avg', np.mean(S_avg)) 
    # removing silent neurons 
    # idx = np.argwhere(S_avg<=0) 
    # S_th = np.delete(S_th, idx, axis=1)
    
    # print('X_dcv', S_th.shape[1]) 
    # gv.n_neurons = S_th.shape[1] 
    
    if gv.Z_SCORE | gv.Z_SCORE_BL: 
        
        if gv.Z_SCORE_BL: 
            gv.bins_z_score = gv.bins_BL 
        else: 
            gv.bins_z_score = gv.bins 
            
        def scaler_loop(S, n_trial, bins): 
            S_i = S[n_trial] 
            scaler = StandardScaler() 
            scaler.fit(S_i[:,bins].T) 
            return scaler.transform(S_i.T).T 
        
        with pg.tqdm_joblib(pg.tqdm(desc='standardize', total=X.shape[0])) as progress_bar: 
            S_scaled = Parallel(n_jobs=gv.num_cores)(delayed(scaler_loop)(S_th, n_trial, gv.bins_z_score) 
                                                     for n_trial in range(X.shape[0]) ) 
        
        # def scaler_loop(S, i_neuron, bins): 
        #     S_neuron = S[:, i_neuron] 
            
        #     avg_trial = np.mean(S_neuron[:, bins], axis=-1) 
        #     avg_trial = avg_trial[:, np.newaxis] 
            
        #     std_trial = np.std(S_neuron[:, bins], axis=-1) 
        #     std_trial = std_trial[:, np.newaxis] 
            
        #     return (S_neuron - avg_trial) / (std_trial) 
            
        #     # avg_all = np.mean(S_neuron[:, bins], axis=-1) 
        #     # std_all = np.std(S_neuron[:, bins]) 
        #     # return (S_neuron - avg_trial) / (std_all) 
        
        # with pg.tqdm_joblib(pg.tqdm(desc='standardize', total=X.shape[0])) as progress_bar: 
        #     S_scaled = Parallel(n_jobs=gv.num_cores)(delayed(scaler_loop)(S_th, i_neuron, gv.bins_z_score) 
        #                                              for i_neuron in range(gv.n_neurons) ) 
            
        S_scaled = np.asarray(S_scaled) 
        # S_scaled = np.swapaxes(S_scaled, 0, 1) 
        
        return S_scaled 
    
    return S_th 
