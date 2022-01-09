import numpy as np
from meegkit.detrend import detrend 
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

from . import constants as gv 
from . import progressbar as pg  

def detrend_loop(X, trial, neuron, order):
    X_det, _, _ = detrend(X[trial, neuron], order)
    return X_det

def detrend_X(X, order=3):
    with pg.tqdm_joblib(pg.tqdm(desc='detrend X', total=int(X.shape[0]*X.shape[1]) ) ) as progress_bar: 
        dum = Parallel(n_jobs=gv.num_cores)(delayed(detrend_loop)(X, trial, neuron, order) 
                                            for trial in range(X.shape[0]) 
                                            for neuron in range(X.shape[1]) )
               
        X = np.asarray(dum).reshape(X.shape[0], X.shape[1], X.shape[2])
    return X

def detrend_data(X_trial, poly_fit=1, degree=7): 
    """ Detrending of the data, if poly_fit=1 uses polynomial fit else linear fit. """
    # X_trial : # neurons, # times 
    
    model = LinearRegression()
    fit_values_trial = []

    indexes = range(0, X_trial.shape[1]) # neuron index 
    values = np.mean(X_trial,axis=0) # mean fluo value 
    
    indexes = np.reshape(indexes, (len(indexes), 1))
    
    if poly_fit:
        poly = PolynomialFeatures(degree=degree) 
        indexes = poly.fit_transform(indexes) 
            
    model.fit(indexes, values)
    fit_values = model.predict(indexes) 
    fit_values_trial = np.array(fit_values)
    
    # for i in range(0, X_trial.shape[0]): # neurons 
    #     indexes = range(0, X_trial.shape[1]) # neuron index 
    #     values = X_trial[i] # fluo value 
                
    #     indexes = np.reshape(indexes, (len(indexes), 1))

    #     if poly_fit:
    #         poly = PolynomialFeatures(degree=degree) 
    #         indexes = poly.fit_transform(indexes) 

    #     model.fit(indexes, values)
    #     fit_values = model.predict(indexes) 
        
    #     fit_values_trial.append(fit_values) 
        
    # fit_values_trial = np.array(fit_values_trial)
    return fit_values_trial
