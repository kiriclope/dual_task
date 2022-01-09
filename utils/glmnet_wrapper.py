import warnings 
warnings.filterwarnings('ignore') 

from copy import deepcopy

import pandas as pd

import scipy
import scipy.stats as stats
import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin 
from sklearn.metrics import roc_auc_score, accuracy_score, log_loss, mean_squared_error 
from sklearn.model_selection import KFold, StratifiedKFold 
from sklearn.feature_selection import SelectKBest, chi2, VarianceThreshold, f_regression, mutual_info_classif, f_classif

from glmnet_python.glmnetSet import glmnetSet 

from glmnet_python.glmnet import glmnet 
from glmnet_python.glmnetCoef import glmnetCoef 
from glmnet_python.glmnetPredict import glmnetPredict 
from glmnet_python.glmnetPlot import glmnetPlot

from glmnet_python.cvglmnet import cvglmnet 
from glmnet_python.cvglmnetCoef import cvglmnetCoef 
from glmnet_python.cvglmnetPredict import cvglmnetPredict 

from glmnet_python.cvglmnetPlot import cvglmnetPlot

from joblib import Parallel, delayed

from . import progressbar as pg

class logitnet(BaseEstimator, ClassifierMixin): 
    
    def __init__(self, alpha=1, n_lambda=20, lbd=1, scoring='accuracy',
                 prescreen=False, standardize=True, fit_intercept=True, thresh=1e-4 , maxit=1e6):
        
        self.prescreen = prescreen

        if isinstance(lbd, str):
            self.lbd = lbd
        else:
            self.lbd = scipy.float64([lbd]) 
        
        self.alpha = alpha 
        self.n_lambda = n_lambda 
        self.scoring = scoring 
        
        self.fit_intercept = fit_intercept 
        self.thresh = thresh 
        self.maxit = maxit 
        
        opts = dict() 
        opts['alpha'] = scipy.float64(alpha) 
        opts['nlambda'] = scipy.int32(n_lambda) 
        
        opts['standardize'] = standardize 
        opts['intr'] = fit_intercept 
        
        opts['thresh'] = scipy.float64(thresh) 
        opts['maxit'] = scipy.int32(maxit) 
        
        self.options = glmnetSet(opts) 

    def pre_screen_fold(X, y, p_alpha=0.05, scoring=f_classif): 
        ''' X is trials x neurons 
        alpha is the level of significance 
        scoring is the statistics, use f_classif or mutual_info_classif 
        '''    
        model = SelectKBest(score_func=scoring, k=X.shape[1])    
        model.fit(X,y) 
        pval = model.pvalues_.flatten() 
        idx_out = np.argwhere(pval>p_alpha) 
        X_screen = np.delete(X, idx_out, axis=1) 
        
        return X_screen
        
    def fit(self, X, y):
        
        # if self.prescreen:
        #     X = scipy.array( self.pre_screen_fold(X, y) ) 
        
        model_ = glmnet(x = X.copy(), y = y.copy(), family = 'binomial', **self.options) 
        self.model_ = model_ # for some reason I have to pass it like that a = funcnet() then self.a = a
                
        coefs = glmnetCoef(self.model_, s= scipy.float64([self.lbd]), exact=False) 
        self.intercept_ = coefs[0] 
        self.coef_ = coefs[1:] 
        
        return self 
    
    def get_coefs(self, obj=None, lbd=None, exact=False):
        coefs = glmnetCoef(self.model_, s=scipy.float64([self.lbd]), exact=exact) 
        self.intercept_ = coefs[0] 
        self.coef_ = coefs[1:] 
        
        return coefs 
    
    def predict(self, X): 
        return glmnetPredict(self.model_, newx=X, ptype='class', s= scipy.float64([self.lbd]) ) 
    
    def predict_proba(self, X): 
        return glmnetPredict(self.model_, newx=X, ptype='response', s= scipy.float64([self.lbd]) ) 
    
    def score(self, X, y):
        if self.scoring=='class' or self.scoring=='accuracy': 
            y_pred = self.predict(X) 
            return accuracy_score(y, y_pred) 
        if self.scoring=='auc' or self.scoring=='roc_auc': 
            y_pred = self.predict_proba(X) 
            return roc_auc_score(y, y_pred) 
        if self.scoring=='deviance' or self.scoring=='log_loss': 
            y_pred = self.predict_proba(X) 
            return log_loss(y, y_pred) 
        if self.scoring=='mse': 
            y_pred = self.predict(X) 
            return mean_squared_error(y, y_pred) 
    
class logitnetCV(BaseEstimator, ClassifierMixin): 
    
    def __init__(self, alpha=1, n_lambda=100, n_splits=10, standardize=False, fit_intercept=False, lbd ='lambda_1se',
                 fold_type='kfold', shuffle=True, random_state=None, prescreen=True, f_screen='f_classif',
                 scoring='accuracy', thresh=1e-4 , maxit=1e6, n_jobs=1):

        opts = dict() 
        opts['alpha'] = scipy.float64(alpha) 
        opts['nlambda'] = scipy.int32(n_lambda) 
        opts['lambdau'] = scipy.array( -np.sort(-np.logspace(np.log(0.5), np.log(0.01), opts['nlambda'], base=np.exp(1)) ) ) 
        
        opts['standardize'] = standardize 
        opts['intr'] = fit_intercept 
        
        opts['thresh'] = scipy.float64(thresh) 
        opts['maxit'] = scipy.int32(maxit) 
        
        opts['prescreen'] = prescreen
        opts['f_screen'] = f_screen 
        
        self.lbd = lbd 
        self.options = glmnetSet(opts) 
        self.scoring = scoring # 'deviance', 'class', 'auc', 'mse' or 'mae'
        
        if 'accuracy' in scoring: 
            self.scoring = 'class' 
        if 'roc_auc' in scoring: 
            self.scoring = 'auc' 
        if 'log_loss' in scoring: 
            self.scoring = 'deviance' 

        self.fold_type = fold_type 
        self.shuffle = shuffle 
        self.random_state = random_state 
        self.n_splits = scipy.int32(n_splits) 
        self.n_jobs = n_jobs 
        
    def fit(self, X, y): 
        
        if self.fold_type=='stratified': 
            folds = StratifiedKFold(n_splits=self.n_splits, shuffle=self.shuffle, random_state=self.random_state) 
        else: 
            folds = KFold(n_splits=self.n_splits, shuffle=self.shuffle, random_state=self.random_state) 
            
        foldid = np.empty(y.shape[0]) 
        i_split = -1 
        for idx_train, idx_test in folds.split(X,y):
            i_split = i_split+1 
            foldid[idx_test] = scipy.int32(i_split)
            
        foldid = scipy.int32(foldid)
        
        model_ = cvglmnet(x = X.copy(), y = y.copy(), family = 'binomial', ptype = self.scoring, foldid=foldid,
                          nfolds=self.n_splits, n_jobs=self.n_jobs, **self.options) 
        
        self.model_ = model_ # for some reason I have to pass it like that a = funcnet() then self.a = a 
        
        cv_mean_score = model_['cvm'] 
        cv_standard_error = model_['cvsd'] 
        lbd_min_ = model_['lambda_min']  
        lbd_1se_ = model_['lambda_1se'] 
        
        self.cv_mean_score_ = cv_mean_score 
        self.cv_standard_error_ = cv_standard_error         
        
        self.lbd_min_ = lbd_min_
        self.lbd_1se_ = lbd_1se_
        
        self.alpha_ = self.options['alpha'] # for compatibility only 
        
        if self.lbd == 'lambda_1se':
            coef_ = cvglmnetCoef(self.model_, s='lambda_1se')[1:]
        else:
            coef_ = cvglmnetCoef(self.model_, s='lambda_min')[1:] 
        self.coef_ = coef_ 
        
        return self 
    
    def lasso_path(self): 
        cvglmnetPlot(self.model_) 

    def predict(self, X):
        if self.lbd == 'lambda_1se':
            return cvglmnetPredict(self.model_, newx=X, ptype='class', s = 'lambda_1se' ) 
        else:
            return cvglmnetPredict(self.model_, newx=X, ptype='class', s = 'lambda_min' ) 
            
    def predict_proba(self, X, lbd=None): 
        if self.lbd == 'lambda_1se':
            return cvglmnetPredict(self.model_, newx=X, ptype='response', s = 'lambda_1se' ) 
        else:
            return cvglmnetPredict(self.model_, newx=X, ptype='response', s = 'lambda_min' ) 
    
    def score(self, X, y): 
        
        if self.scoring=='class' or self.scoring=='accuracy': 
            y_pred = self.predict(X) 
            return accuracy_score(y, y_pred) 
        if self.scoring=='auc' or self.scoring=='roc_auc' : 
            y_pred = self.predict_proba(X) 
            return roc_auc_score(y, y_pred) 
        if self.scoring=='deviance' or self.scoring=='log_loss': 
            y_pred = self.predict_proba(X) 
            return log_loss(y, y_pred) 
        if self.scoring=='mse': 
            y_pred = self.predict(X) 
            return mean_squared_error(y, y_pred)
        
class logitnetAlphaCV(BaseEstimator, ClassifierMixin): 
    
    def __init__(self, n_alpha=10, n_lambda=100, lbd='lbd_1se', 
                 n_splits=10, fold_type='kfold', scoring='accuracy', 
                 standardize=False, fit_intercept=False, 
                 prescreen=True, f_screen='f_classif', 
                 thresh=1e-4 , maxit=1e6, n_jobs=1, verbose=True): 
        
        opts = dict() 
        opts['nlambda'] = scipy.int32(n_lambda) 
        # opts['lambdau']= scipy.array( -np.sort(-np.logspace(-4, -1, opts['nlambda'])) )  
        # opts['lambdau'] = scipy.array( -np.sort(-np.logspace(np.log(0.5), np.log(0.01), opts['nlambda'], base=np.exp(1)) ) )
        
        opts['standardize'] = standardize 
        opts['intr'] = fit_intercept 
        opts['prescreen'] = prescreen
        opts['f_screen'] = f_screen 
        
        opts['thresh'] = scipy.float64(thresh) 
        opts['maxit'] = scipy.int32(maxit) 
        
        self.options = glmnetSet(opts)                
        
        self.n_alpha = n_alpha 
        self.alpha_path = np.linspace(0.1, 1, n_alpha) 
        
        self.lbd = lbd
        self.n_lambda = n_lambda
        
        self.n_splits = scipy.int32(n_splits) 
        self.fold_type = fold_type 
        
        self.scoring = scoring # 'deviance', 'class', 'auc', 'mse' or 'mae'         
        if 'accuracy' in scoring: 
            self.scoring = 'class'  
        if 'roc_auc' in scoring: 
            self.scoring = 'auc' 
        if 'log_loss' in scoring: 
            self.scoring = 'deviance' 
                    
        self.n_jobs = n_jobs 
        self.verbose = verbose 
        
    def createFolds(self, X, y):
        # fixed seed accross alphas 
        self.random_state = np.random.randint(1e6) 
        
        if self.fold_type=='stratified':
            folds = StratifiedKFold(n_splits=self.n_splits, shuffle=True, random_state=self.random_state) 
        else: 
            folds = KFold(n_splits=self.n_splits, shuffle=True, random_state=self.random_state) 
            
        foldid = np.empty(y.shape[0]) 
        i_split = -1 
        for idx_train, idx_test in folds.split(X,y): 
            i_split = i_split+1 
            foldid[idx_test] = scipy.int32(i_split) 
            
        foldid = scipy.int32(foldid) 
        
        return foldid
    
    def fit(self, X, y):
        
        # generate folds 
        self.foldid = self.createFolds(X,y) 
        
        # fit each model along the alpha path 
        with pg.tqdm_joblib(pg.tqdm(desc='alpha path', total= int(self.n_alpha * (self.n_splits+1) ), disable=not self.verbose) ) as progress_bar: 
            dum = Parallel(n_jobs=self.n_jobs)(delayed(self.fitFixedAlpha)(X, y, i_alpha, self.alpha_path, self.foldid, self.options) 
                                               for i_alpha in range(self.n_alpha) )
        
        self.models_ = scipy.array(dum) 
        if self.verbose:
            print('models', self.models_.shape) 
            
        # compute min score
        with pg.tqdm_joblib(pg.tqdm(desc='cvm min', total= int(self.n_alpha) , disable=not self.verbose ) ) as progress_bar: 
            dum = Parallel(n_jobs=self.n_jobs)(delayed(self.minScoreAlpha)(self.models_[i_alpha]) for i_alpha in range(self.n_alpha) ) 
        self.cvms_min = scipy.array(dum) 
        
        self.idx_alpha_min_ = np.argmin(self.cvms_min) 
        self.alpha_ = self.alpha_path[self.idx_alpha_min_]
        
        self.model_= self.models_[self.idx_alpha_min_]         
        self.lbd_1se_ = self.model_['lambda_1se'] 
        self.lbd_min_ = self.model_['lambda_min'] 
        
        self.non_zero_1se = self.model_['nzero'][ self.model_['lambdau'] == self.model_['lambda_1se'] ]
        self.non_zero_min = self.model_['nzero'][ self.model_['lambdau'] == self.model_['lambda_min'] ]
        
        if self.verbose:
            print('non zero min', self.non_zero_min, 'non zero 1se', self.non_zero_1se) 
        
        if self.lbd == 'lambda_1se':
            coef_ = cvglmnetCoef(self.model_, s='lambda_1se')[1:]
        else:
            coef_ = cvglmnetCoef(self.model_, s='lambda_min')[1:] 
            
        self.coef_ = coef_ 
        
        return self 
    
    def fitFixedAlpha(self, X, y, i_alpha, alpha_path, foldid, options): 
        
        opts = options.copy() 
        opts['alpha'] = alpha_path[i_alpha] 
        
        model_ = cvglmnet(x = X.copy(), y = y.copy(), family = 'binomial', ptype = self.scoring, foldid=foldid,
                          nfolds=self.n_splits, n_jobs=self.n_jobs, **opts) 
        return model_ 
    
    def minScoreAlpha(self, model_): 
        
        idx_lbd_min = model_['lambdau'] == model_['lambda_min'] 
        cvm_min = model_['cvm'][idx_lbd_min] 
        
        return cvm_min 
    
    def lasso_path(self): 
        cvglmnetPlot(self.model_) 
        
    def predict(self, X): 
        if self.lbd == 'lambda_1se': 
            return cvglmnetPredict(self.model_, newx=X, ptype='class', s = 'lambda_1se' ) 
        else: 
            return cvglmnetPredict(self.model_, newx=X, ptype='class', s = 'lambda_min' ) 
            
    def predict_proba(self, X): 
        if self.lbd == 'lambda_1se': 
            return cvglmnetPredict(self.model_, newx=X, ptype='response', s='lambda_1se' ) 
        else: 
            return cvglmnetPredict(self.model_, newx=X, ptype='response', s='lambda_min' ) 
    
    def score(self, X, y): 
        if self.scoring=='class' or self.scoring=='accuracy': 
            y_pred = self.predict(X) 
            return accuracy_score(y, y_pred)
        
        if self.scoring=='auc' or self.scoring=='roc_auc': 
            y_pred = self.predict_proba(X) 
            return roc_auc_score(y, y_pred)
        
        if self.scoring=='deviance' or self.scoring=='log_loss':  
            y_pred = self.predict_proba(X) 
            return log_loss(y, y_pred)
        
        if self.scoring=='mse': 
            y_pred = self.predict(X) 
            return mean_squared_error(y, y_pred) 

class logitnetIterCV(BaseEstimator, ClassifierMixin): 
    
    def __init__(self, n_iter=100, alpha=1, n_lambda=100, lbd='lbd_1se', 
                 n_splits=10, fold_type='kfold', scoring='accuracy',
                 standardize=True, fit_intercept=True,
                 prescreen=True, f_screen='f_classif',
                 thresh=1e-4 , maxit=1e6, n_jobs=1, verbose=False):
        
        opts = dict() 
        opts['nlambda'] = scipy.int32(n_lambda) 
        opts['lambdau'] = scipy.array( -np.sort(-np.logspace(np.log(0.5), np.log(0.01), opts['nlambda'], base=np.exp(1)) ) ) 
        
        opts['alpha'] = scipy.float64(alpha) 
        opts['standardize'] = standardize 
        opts['intr'] = fit_intercept 
        opts['prescreen'] = prescreen 
        opts['f_screen'] = f_screen 
        
        opts['thresh'] = scipy.float64(thresh) 
        opts['maxit'] = scipy.int32(maxit) 
        
        self.alpha_ = alpha
        self.n_iter = n_iter
        self.lbd = lbd
        self.options = glmnetSet(opts) 
        
        self.scoring = scoring # 'deviance', 'class', 'auc', 'mse' or 'mae'         
        if 'accuracy' in scoring: 
            self.scoring = 'class'  
        if 'roc_auc' in scoring: 
            self.scoring = 'auc' 
        if 'log_loss' in scoring: 
            self.scoring = 'deviance' 
            
        self.n_lambda = n_lambda
        
        self.n_splits = scipy.int32(n_splits) 
        self.fold_type = fold_type
        
        self.n_jobs = n_jobs 
        self.verbose = verbose 

    def foldsLoop(self, X, y): 
        self.random_state = np.random.randint(1e6) 
        
        if self.fold_type=='stratified':
            folds = StratifiedKFold(n_splits=self.n_splits, shuffle=True, random_state=self.random_state) 
        else: 
            folds = KFold(n_splits=self.n_splits, shuffle=True, random_state=self.random_state) 
            
        foldid = np.empty(y.shape[0]) 
        i_split = -1 
        for idx_train, idx_test in folds.split(X,y): 
            i_split = i_split+1 
            foldid[idx_test] = i_split
            
        return scipy.int32(foldid)
        
    def fit(self, X, y):
        
        # create foldids for each iteration 
        with pg.tqdm_joblib(pg.tqdm(desc='create foldids', total= int(self.n_iter), disable=not self.verbose ) ) as progress_bar: 
            dum = Parallel(n_jobs=self.n_jobs)(delayed(self.foldsLoop)(X, y) for _ in range(self.n_iter) ) 
        
        foldid = scipy.array(dum) 
        if self.verbose: 
            print(foldid.shape) 
            
        # fit cvglmnet for each iteration and create dataframe 
        with pg.tqdm_joblib(pg.tqdm(desc='iter', total= int(self.n_iter * (self.n_splits+1) ),
                                    disable=not self.verbose ) ) as progress_bar: 
            dum = Parallel(n_jobs=self.n_jobs)(delayed(self.fit_one_iter)(X, y, i_iter, foldid[i_iter], self.options) 
                                               for i_iter in range(self.n_iter) )
        
        df = pd.concat(dum) 
        df = df.groupby(['lambdau'])['cvm', 'cvsd'].mean() 
        
        if self.verbose: 
            print(df)
        
        min_cvm = df['cvm'].min() 
        idx_min_cvm = df['cvm'].idxmin()
        
        if self.verbose: 
            print(idx_min_cvm, min_cvm) 
        
        self.lbd_min_ = scipy.float64( [idx_min_cvm] ) 

        # fit on the entire data 
        model_ = glmnet(x = X.copy(), y = y.copy(), family = 'binomial', **self.options) 
        self.model_ = model_ 
        
        # get coefs for lambda = lambda_min 
        coefs = glmnetCoef(self.model_, s = self.lbd_min_, exact = False) 
        self.intercept_ = coefs[0] 
        self.coef_ = coefs[1:] 
        
        self.non_zero_min = self.model_['df'][ self.model_['lambdau'] == self.lbd_min_ ] 
        
        if self.verbose:
            print('non zero min', self.non_zero_min) 
        
        return self 
    
    def fit_one_iter(self, X, y, i_iter, foldid, options): 
        
        opts = options.copy()         
        model_ = cvglmnet(x = X.copy(), y = y.copy(), family = 'binomial',ptype = self.scoring, foldid=foldid, 
                          nfolds=self.n_splits, n_jobs=self.n_jobs, **opts)
        
        df = pd.DataFrame({ 'i_iter': i_iter,
                            'lambdau': model_['lambdau'],
                            'cvm': model_['cvm'], 
                            'cvsd': model_['cvsd'] } )
        return df 
    
    def lasso_path(self):
        glmnetPlot(self.model_, xvar = 'lambda', label = False);
        
    def predict(self, X):
        return glmnetPredict(self.model_, newx=X, ptype='class', s = self.lbd_min_ ) 
            
    def predict_proba(self, X): 
        return glmnetPredict(self.model_, newx=X, ptype='response', s = self.lbd_min_ ) 
    
    def score(self, X, y): 
        if self.scoring=='class' or self.scoring=='accuracy': 
            y_pred = self.predict(X) 
            return accuracy_score(y, y_pred) 
        if self.scoring=='auc' or self.scoring=='roc_auc' : 
            y_pred = self.predict_proba(X) 
            return roc_auc_score(y, y_pred) 
        if self.scoring=='deviance' or self.scoring=='log_loss': 
            y_pred = self.predict_proba(X) 
            return log_loss(y, y_pred) 
        if self.scoring=='mse': 
            y_pred = self.predict(X) 
            return mean_squared_error(y, y_pred) 
        
class logitnetAlphaIterCV(BaseEstimator, ClassifierMixin): 
    
    def __init__(self, n_iter=100, n_alpha=10, n_lambda=100, lbd='lbd_1se', 
                 n_splits=10, fold_type='kfold', scoring='accuracy',
                 standardize=True, fit_intercept=True,
                 prescreen=True, f_screen='f_classif',
                 thresh=1e-4 , maxit=1e6, n_jobs=1, verbose=False):
        
        opts = dict() 
        opts['nlambda'] = scipy.int32(n_lambda) 
        # opts['lambdau']= scipy.array( -np.sort(-np.logspace(-4, -1, opts['nlambda'])) )  
        opts['lambdau'] = scipy.array( -np.sort(-np.logspace(np.log(0.5), np.log(0.01), opts['nlambda'], base=np.exp(1)) ) ) 
        
        opts['standardize'] = standardize 
        opts['intr'] = fit_intercept 
        opts['prescreen'] = prescreen
        opts['f_screen'] = f_screen 
        
        opts['thresh'] = scipy.float64(thresh) 
        opts['maxit'] = scipy.int32(maxit) 
        
        self.n_iter = n_iter
        self.lbd = lbd
        self.options = glmnetSet(opts)                
        
        self.scoring = scoring # 'deviance', 'class', 'auc', 'mse' or 'mae'         
        if 'accuracy' in scoring: 
            self.scoring = 'class'  
        if 'roc_auc' in scoring: 
            self.scoring = 'auc' 
        if 'log_loss' in scoring: 
            self.scoring = 'deviance' 
            
        self.n_alpha = n_alpha 
        self.alpha_path = np.linspace(.1, 1, n_alpha) 
        self.n_lambda = n_lambda
        
        self.n_splits = scipy.int32(n_splits) 
        self.fold_type = fold_type
        
        self.n_jobs = n_jobs 
        self.verbose = verbose 

    def foldsLoop(self, X, y):
        # fixed seed accross alphas 
        self.random_state = np.random.randint(1e6) 
        
        if self.fold_type=='stratified':
            folds = StratifiedKFold(n_splits=self.n_splits, shuffle=True, random_state=self.random_state) 
        else: 
            folds = KFold(n_splits=self.n_splits, shuffle=True, random_state=self.random_state) 
            
        foldid = np.empty(y.shape[0]) 
        i_split = -1 
        for idx_train, idx_test in folds.split(X,y): 
            i_split = i_split+1 
            foldid[idx_test] = i_split
            
        return scipy.int32(foldid)
        
    def fit(self, X, y):
        
        # create foldids for each iteration 
        with pg.tqdm_joblib(pg.tqdm(desc='create foldids', total= int(self.n_iter), disable=not self.verbose ) ) as progress_bar: 
            dum = Parallel(n_jobs=self.n_jobs)(delayed(self.foldsLoop)(X, y) for _ in range(self.n_iter) ) 
        
        foldid = scipy.array(dum) 
        if self.verbose: 
            print(foldid.shape) 
            
        # fit cvglmnet for each iteration for each alpha and create dataframe
        with pg.tqdm_joblib(pg.tqdm(desc='alpha path', total= int(self.n_alpha * self.n_iter * (self.n_splits+1) ),
                                    disable=not self.verbose ) ) as progress_bar: 
            dum = Parallel(n_jobs=self.n_jobs)(delayed(self.fit_fixed_alpha)(X, y, i_alpha, i_iter, self.alpha_path, foldid[i_iter], self.options) 
                                               for i_alpha in range(self.n_alpha) for i_iter in range(self.n_iter) ) 
        df = pd.concat(dum) 
        df = df.groupby(['i_alpha','lambdau'])['cvm', 'cvsd'].mean() 
        if self.verbose: 
            print(df)
        
        min_cvm = df['cvm'].min() 
        idx_min_cvm = df['cvm'].idxmin() 
        if self.verbose: 
            print(idx_min_cvm, min_cvm) 
        
        self.idx_alpha_min_ = idx_min_cvm[0]
        self.lbd_min_ = scipy.float64( [idx_min_cvm[1]] )
        
        # fit on the entire data for alpha = alpha_min
        self.alpha_ =  self.alpha_path[self.idx_alpha_min_] 
        self.options['alpha'] = self.alpha_path[self.idx_alpha_min_] 
        model_ = glmnet(x = X.copy(), y = y.copy(), family = 'binomial', **self.options) 
        self.model_ = model_ 
        
        # get coefs for lambda = lambda_min
        coefs = glmnetCoef(self.model_, s = self.lbd_min_, exact = False) 
        self.intercept_ = coefs[0] 
        self.coef_ = coefs[1:] 
        
        self.non_zero_min = self.model_['df'][ self.model_['lambdau'] == self.lbd_min_ ] 
        
        if self.verbose:
            print('non zero min', self.non_zero_min, 'non zero 1se', self.non_zero_1se) 
        
        return self 
    
    def fit_fixed_alpha(self, X, y, i_alpha, i_iter, alpha_path, foldid, options): 
        
        opts = options.copy() 
        opts['alpha'] = alpha_path[i_alpha] 
        
        model_ = cvglmnet(x = X.copy(), y = y.copy(), family = 'binomial',ptype = self.scoring, foldid=foldid, grouped=True,
                          nfolds=self.n_splits, n_jobs=self.n_jobs, **opts)
        
        df = pd.DataFrame({ 'i_alpha': i_alpha, 
                            'i_iter': i_iter,
                            'lambdau': model_['lambdau'],
                            'cvm': model_['cvm'], 
                            'cvsd': model_['cvsd'] } )
        return df         
    
    def lasso_path(self):
        glmnetPlot(self.model_, xvar = 'lambda', label = False);
        
    def predict(self, X):
        return glmnetPredict(self.model_, newx=X, ptype='class', s = self.lbd_min_ ) 
            
    def predict_proba(self, X): 
        return glmnetPredict(self.model_, newx=X, ptype='response', s = self.lbd_min_ ) 
    
    def score(self, X, y): 
        if self.scoring=='class' or self.scoring=='accuracy': 
            y_pred = self.predict(X) 
            return accuracy_score(y, y_pred) 
        if self.scoring=='auc' or self.scoring=='roc_auc' : 
            y_pred = self.predict_proba(X) 
            return roc_auc_score(y, y_pred) 
        if self.scoring=='deviance' or self.scoring=='log_loss': 
            y_pred = self.predict_proba(X) 
            return log_loss(y, y_pred) 
        if self.scoring=='mse': 
            y_pred = self.predict(X) 
            return mean_squared_error(y, y_pred) 
