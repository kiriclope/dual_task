import os, sys, importlib 
from importlib import reload

import warnings
warnings.filterwarnings("ignore")

import numpy as np

import scipy.stats as stats 
from sklearn.preprocessing import StandardScaler 

from joblib import Parallel, delayed 
import multiprocessing

import utils.constants as gv 
reload(gv) 
from utils.options import *

from . import bootstrap 
reload(bootstrap) 
from .bootstrap import bootstrap, crossVal 

import utils.glms 
reload(utils.glms) 
from utils.glms import * 

''' 
We want to estimate the statistical significance of a model,
to do so, we bootstrap with replacement, fit the model 
on each bootstrap sample, and compute the quantity of interest (mean, score, corr, ...) for each boot.
'''

def getAlphaLambda(X, y, **options):
    
    clf_ = logitnetAlphaIterCV(lbd=options['lbd'], n_alpha=options['n_alpha'], n_lambda=options['n_lambda'], n_splits=options['inner_splits'],
                               standardize=True, fit_intercept=options['fit_intercept'], prescreen=options['prescreen'],
                               fold_type=options['fold_type'],
                               scoring=options['inner_scoring'], thresh=options['tol'] , maxit=options['max_iter'],
                               n_jobs=-1, verbose=0) 
    
    # clf = logitnetIterCV(lbd=options['lbd'], alpha=options['alpha'], n_lambda=options['n_lambda'], n_splits=options['n_splits'],
    #                           standardize=options['standardize'], fit_intercept=options['fit_intercept'], prescreen=options['prescreen'],
    #                           fold_type=options['fold_type'],
    #                           scoring=options['inner_scoring'], thresh=options['tol'] , maxit=options['max_iter'],
    #                           n_jobs=-1, verbose=options['verbose']) 
    
    # scaler = StandardScaler() 
    # X = scaler.fit_transform(X) 
    clf_.fit(X,y) 
    
    return clf_.alpha_, clf_.lbd_min_, clf_.non_zero_min 

def get_coefs(X_S1, X_S2, return_hyper=0, **kwargs):
    
    clf = get_clf(**kwargs) 
    model = bootstrap(clf, method=kwargs['bootstrap_method'], n_boots=kwargs['n_iter'],
                      scoring=kwargs['scoring'], scaler=kwargs['scaler'], n_jobs=None, verbose=kwargs['verbose']) 
    
    X_S1_S2 = np.vstack( ( X_S1, X_S2 ) ) 
    y = np.hstack((np.zeros(X_S1.shape[0]), np.ones(X_S2.shape[0]) )) 
    
    coefs = [] 
    for i_epochs in range(X_S1.shape[-1]): 
        
        X = X_S1_S2[:,:,i_epochs] 
        
        # get estimation of best (alpha, lambda) on the concatenated trials for the train epoch 
        # if i_epochs==0 : 
        #     print('fix_alpha_lbd') 
        #     # alpha, lbd, non_zero = getAlphaLambda(X_concat[..., i_epochs], y_concat, **kwargs)  
        #     alpha, lbd, non_zero = getAlphaLambda(X, y, **kwargs) 
        #     print('alpha', alpha, 'lambda', lbd, 'non_zero', non_zero) 
        #     clf.alpha = alpha 
        #     clf.lbd = lbd 
        
        dum = model.get_coefs(X, y) 
        coefs.append(dum) # N_samples x N_neurons 

    coefs = np.array(coefs)

    # print('coefs', coefs.shape)
    
    if return_hyper:
        return coefs, alpha, lbd 
    else:
        return coefs  

def get_hyperparam(X_S1, X_S2, **kwargs): 
    
    X_S1_S2 = np.vstack( ( X_S1, X_S2 ) ) 
    y = np.hstack((np.zeros(X_S1.shape[0]), np.ones(X_S2.shape[0]) )) 
    
    # get estimation of best (alpha, lambda) on the concatenated trials for the train epoch 
    if i_epochs==0 : 
        print('fix_alpha_lbd') 
        alpha, lbd, non_zero = getAlphaLambda(X, y, **kwargs) 
        print('alpha', alpha, 'lambda', lbd, 'non_zero', non_zero) 
        gv.clf.alpha = alpha 
        gv.clf.lbd = lbd 
    
def get_score(X_S1, X_S2, return_hyper=0, **kwargs):
    
    clf = get_clf(**kwargs) 
    # if kwargs['verbose'] :
    #     print('alpha', clf.alpha, 'lbd', clf.lbd) 
    # print('l1_ratio', clf.l1_ratio, 'C', clf.C) 
    
    # print('clf', clf) 
    model = crossVal(clf, method=kwargs['fold_type'], n_iter=kwargs['n_iter'], scoring=kwargs['scoring']
                     , scaler=kwargs['scaler'], n_jobs=None, verbose=kwargs['verbose']) 
    
    # model=clf 
    X_S1_S2 = np.vstack( ( X_S1, X_S2 ) ) 
    y = np.hstack((np.zeros(X_S1.shape[0]), np.ones(X_S2.shape[0]) )) 
    
    score = [] 
    
    # X_train = np.mean( X_S1_S2[..., [24,25,26]], axis=-1)
    
    for i_epochs in range(X_S1.shape[-1]): 
        X_train = X_S1_S2[..., i_epochs].copy()
        X_test = X_S1_S2[..., i_epochs].copy()
        
        # get estimation of best (alpha, lambda) on the concatenated trials for the train epoch 
        if i_epochs==0 and return_hyper: 
            # print('fix_alpha_lbd') 
            # alpha, lbd, non_zero = getAlphaLambda(X_concat[..., i_epochs], y_concat, **kwargs) 
            alpha, lbd, non_zero = getAlphaLambda(X, y, **kwargs) 
            if kwargs['verbose'] : 
                print('alpha', alpha, 'lambda', lbd, 'non_zero', non_zero) 
            
            clf.alpha = alpha 
            clf.lbd = lbd[0] 
            
        #     clf.l1_ratio = alpha 
        #     clf.C = 1.0 / lbd[0] 
        
        # if kwargs['verbose'] :
        #     print('alpha', clf.alpha, 'lbd', clf.lbd) 
        # print('l1_ratio', clf.l1_ratio, 'C', clf.C) 
            
        # model_fit = model.fit(X,y) 
        # dum = model_fit.score(X,y) 
        
        dum = model.get_scores(X_train, X_test, y) 
        score.append(dum) # N_samples x N_neurons 
    
    if return_hyper:
        return np.array(score), alpha, lbd 
    else:
        return np.array(score)        
    
    
