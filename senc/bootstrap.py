from copy import deepcopy
import warnings 
warnings.filterwarnings("ignore") 

import numpy as np

from sklearn.preprocessing import StandardScaler, RobustScaler 
from sklearn.pipeline import Pipeline 
from sklearn.model_selection import KFold, StratifiedKFold 

from joblib import Parallel, delayed 

from sklearn.decomposition import PCA 
from sklearn.feature_selection import SelectKBest, chi2, VarianceThreshold, f_regression, mutual_info_classif, f_classif 

import utils.progressbar as pg 

def preScreenFold(X, y, p_alpha=0.05, scoring=mutual_info_classif ): 
    ''' X is trials x neurons 
    alpha is the level of significance 
    scoring is the statistics, use f_classif or mutual_info_classif 
    '''    
    model = SelectKBest(score_func=scoring, k=X.shape[1])    
    model.fit(X,y) 
    pval = model.pvalues_.flatten() 
    idx_out = np.argwhere(pval>p_alpha) 
    idx_screen = np.argwhere(pval<=p_alpha) 
    X_screen = np.delete(X, idx_out, axis=1) 
    
    # print(X.shape, X_screen.shape, idx_screen.shape)
    return X_screen, idx_out


def get_optimal_number_of_components(X, total_explained_variance=0.9): 
    cov = np.dot(X,X.transpose())/float(X.shape[0]) 
    U,s,v = np.linalg.svd(cov) 
    S_nn = sum(s) 
        
    for num_components in range(0, s.shape[0] ): 
        temp_s = s[0:num_components]
        S_ii = sum(temp_s)
        if (1 - S_ii/float(S_nn)) <= 1 - total_explained_variance: 
            return num_components 
            
    return np.maximum(s.shape[0], 1) 

class crossVal(): 
    def __init__(self, clf, method='stratified', n_splits=10, n_iter=100, scaler='standard', scoring='roc_auc', n_jobs=1, verbose=0): 
        
        if scaler=='standard':
            self.pipe = Pipeline([('scaler', StandardScaler()), ('clf', clf)]) 
        elif scaler=='center':
            self.pipe = Pipeline([('scaler', StandardScaler(with_std=False)), ('clf', clf)])
        elif scaler=='robust':
            self.pipe = Pipeline([('scaler', RobustScaler()), ('clf', clf)]) 
        else: 
            self.pipe = Pipeline([('clf', clf)]) 
            
        self.method = method 
        self.n_iter = n_iter 
        self.n_splits = n_splits 
        self.scoring = scoring 
        self.n_jobs = n_jobs 
        self.verbose = verbose
        
    def loopCV(self, X_t_train, X_t_test, y, idx_train, idx_test): 
        X_train, y_train = X_t_train[idx_train], y[idx_train] 
        X_test, y_test = X_t_test[idx_test], y[idx_test] 
        
        # if self.my_pca is not None: 
        #     scaler = StandardScaler()
        #     X_train = scaler.fit_transform(X_train)
        #     X_test = scaler.transform(X_test) 
        
        #     n_comp = get_optimal_number_of_components(X_train) 
        #     pca = PCA(n_components=n_comp)
        #     X_train = pca.fit_transform(X_train)
        #     X_test = pca.transform(X_test)
        
        pipe_copy = deepcopy(self.pipe) # scaler is inside Pipe 
        pipe_copy.fit(X_train, y_train) # fit decoder on outer train, outer train is splitted into inner train and inner test sets 
        
        try:
            pipe_copy[-1].scoring = self.scoring 
        except:
            pass
        
        cv_score = pipe_copy.score(X_test, y_test) # estimate performance on outer test 
        
        return cv_score 
    
    def cv_splits(self, X_train, y, i_iter):
        # seed = np.random.randint(0,1e6) # fix random seed 
        
        if 'stratified' in self.method: 
            folds = StratifiedKFold(n_splits=self.n_splits, shuffle=True, random_state=None) 
        elif 'loo' in self.method: 
            folds = KFold(n_splits=X_train.shape[0], shuffle=True, random_state=None) 
        else: 
            folds = KFold(n_splits=self.n_splits, shuffle=True, random_state=None) 
        
        return folds.split(X_train, y) 
    
    def get_scores(self, X_train, X_test, y):
        if 'loo' in self.method: 
            self.n_splits = X_train.shape[0]
        
        if self.verbose:
            with pg.tqdm_joblib(pg.tqdm(desc="cross validation", total=int( self.n_splits * self.n_iter) ) ) as progress_bar: 
                cv_scores = Parallel(n_jobs=self.n_jobs)(delayed(self.loopCV)(X_train, X_test, y, idx_train, idx_test) 
                                                         for i_iter in range(self.n_iter) 
                                                         for idx_train, idx_test in self.cv_splits(X_train, y, i_iter) ) 
                
            # self.scores_ = np.asarray(cv_scores).reshape( (self.n_splits, self.n_iter) ).T 
            self.scores_ = np.asarray(cv_scores).reshape( (self.n_iter, int(len(cv_scores) / self.n_iter)) ) 
        else:
            cv_scores = Parallel(n_jobs=self.n_jobs)(delayed(self.loopCV)(X_train, X_test, y, idx_train, idx_test) 
                                                     for i_iter in range(self.n_iter) 
                                                     for idx_train, idx_test in self.cv_splits(X_train, y, i_iter) )
                
            # self.scores_ = np.asarray(cv_scores).reshape( (self.n_splits, self.n_iter) ).T 
            self.scores_ = np.asarray(cv_scores).reshape( (self.n_iter, int( len(cv_scores) / self.n_iter ) ) ) 
        
        self.scores_ = np.mean(self.scores_) 
        # print(self.scores_.shape) 
        return self.scores_ 
        
class bootstrap():
    
    def __init__(self, clf, method='standard', my_pca=False, n_boots=1000, scoring='roc_auc', scaler='standard', n_jobs=1, verbose=0): 

        if scaler=='standard':
            self.pipe = Pipeline([('scaler', StandardScaler()), ('clf', clf)])
        elif scaler=='center':
            self.pipe = Pipeline([('scaler', StandardScaler(with_std=False)), ('clf', clf)])
        elif scaler=='robust':
            self.pipe = Pipeline([('scaler', RobustScaler()), ('clf', clf)]) 
        else:
            self.pipe = Pipeline([('clf', clf)]) 
        
        self.method = method
        self.scoring = scoring 
        self.n_boots = n_boots 
        self.n_jobs = n_jobs 
        self.my_pca = my_pca
        self.verbose = verbose
        
    def scoresLoop(self, X_train, X_test, y):
        
        if self.n_boots==1:
            warnings.warn('n_boots=1, will fit the whole data without resampling')
            X_sample = X_train
            y_sample = y
            
            X_oob = X_test 
            y_oob = y 
        else: 
            if 'standard' in self.method :
                idx_trials = np.random.randint(0, X_train.shape[0], X_train.shape[0]) 
                
            if 'block' in self.method :
                # idx_trials = np.hstack( ( np.random.randint(0, int(X_train.shape[0]/2), int(X_train.shape[0]/2)), 
                #                           np.random.randint(int(X_train.shape[0]/2), X_train.shape[0], int(X_train.shape[0]/2)) ) ) 
                n_0 = np.sum(y==0)
                n_1 = np.sum(y==1)
                
                idx_trials = np.hstack( ( np.random.randint(0, n_0, n_0), 
                                          np.random.randint(n_0, X.shape[0], n_1) ) ) 
            
            X_sample = X_train[idx_trials] 
            y_sample = y[idx_trials] 
            
            array = np.arange(0, X_train.shape[0]) 
            idx_oob = np.delete(array, idx_trials) 
            
            X_oob = X_test[idx_oob] 
            y_oob = y[idx_oob] 
            
        pipe_copy = deepcopy(self.pipe) 
        pipe_copy.fit(X_sample, y_sample)
        
        try:
            pipe_copy[-1].scoring = self.scoring
        except:
            pass
        
        boots_score = pipe_copy.score(X_oob, y_oob) 
                    
        return boots_score 
    
    def get_scores(self, X_train, X_test, y): 

        if self.n_boots>10: 
            bar_name = self.method + ' bootstrap' 
            with pg.tqdm_joblib(pg.tqdm(desc=bar_name, total=self.n_boots)) as progress_bar: 
                boots_score = Parallel(n_jobs=self.n_jobs)(delayed(self.scoresLoop)(X_train, X_test, y) for _ in range(self.n_boots) ) 
            self.scores_ = np.array(boots_score) 
        else: 
            self.scores_ = np.zeros( (self.n_boots) ) # N_boots 
            for i_boot in range(self.n_boots): 
                boots_score = self.scoresLoop(X_train, X_test, y) 
                self.scores_[i_boot] = np.array(boots_score) 
        
        return self.scores_ 

    def coefsLoop(self, X, y):
                
        if self.n_boots==1:
            warnings.warn('n_boots=1, will fit the whole data without resampling')
            X_sample = X
            y_sample = y
        
        else: 
            if 'standard' in self.method :
                idx_trials = np.random.randint(0, X.shape[0], X.shape[0]) 
                
            if 'block' in self.method :
                n_0 = np.sum(y==0)
                n_1 = np.sum(y==1)
                
                idx_trials = np.hstack( ( np.random.randint(0, n_0, n_0), 
                                          np.random.randint(n_0, X.shape[0], n_1) ) ) 
            
            X_sample = X[idx_trials] 
            y_sample = y[idx_trials]
            
        coefs = np.zeros(X_sample.shape[1]) 
        
        # X_sample, idx_screen = preScreenFold(X_sample, y_sample) 
        
        # if self.my_pca :
        # scaler = StandardScaler()
            
        # X_sample = scaler.fit_transform(X_sample) 
        # n_comp = get_optimal_number_of_components(X_sample) 
        # pca = PCA(n_components=n_comp) 
        # X_sample = pca.fit_transform(X_sample)
        
        pipe_copy = deepcopy(self.pipe) 
        pipe_copy.fit(X_sample, y_sample)
                
        dum = pipe_copy[-1].coef_.flatten()
        coefs[0:dum.shape[0]] = dum
        
        return coefs 
    
    def get_coefs(self, X, y): 
        if self.n_boots>10: 
            bar_name = self.method + ' bootstrap'
            if self.verbose:
                with pg.tqdm_joblib(pg.tqdm(desc=bar_name, total=self.n_boots)) as progress_bar: 
                    boots_coef = Parallel(n_jobs=self.n_jobs)(delayed(self.coefsLoop)(X, y) for _ in range(self.n_boots) ) 
            else:
                boots_coef = Parallel(n_jobs=self.n_jobs)(delayed(self.coefsLoop)(X, y) for _ in range(self.n_boots) )
                
            self.coef_ = np.array(boots_coef)
        else: 
            self.coef_ = np.zeros( (self.n_boots, X.shape[1]) ) # N_boots 
            for i_boot in range(self.n_boots): 
                boots_coef = self.coefsLoop(X, y) 
                self.coef_[i_boot] = np.array(boots_coef)
        
        self.coef_ = np.mean(self.coef_, axis=0) # average over boots 
        return self.coef_ 
