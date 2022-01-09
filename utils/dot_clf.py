import numpy as np
import scipy.stats as stats

def unit_vector(vector): 
    norm = np.linalg.norm(vector, axis=0)     
    u = vector / norm     
    return u 

def cos_between(v1, v2): 
    v1_u = unit_vector(v1) 
    v2_u = unit_vector(v2) 
    return np.clip( np.dot(v1_u, v2_u), -1.0, 1.0) 

def get_cos(coefs, X): 
    return cos_between(coefs, X.T) 

def get_proj(coefs, X):
    return np.mean( np.dot(coefs, X.T), axis=0) / np.linalg.norm(coefs) 

def get_norm(coefs):
    return np.linalg.norm(coefs)

class dot_clf(): 
    
    def __init__(self, pval=0.05, scoring='proj'): 
        
        self.pval = pval 
        self.scoring = scoring
        
    def get_X_S1_X_S2(self, X, y):
        n_0 = np.sum(y==0)
        X_S1 = X[:n_0]
        X_S2 = X[n_0:]

        return X_S1, X_S2
    
    def get_coefs(self, X, y): 
        
        X_S1, X_S2 = self.get_X_S1_X_S2(X,y) 
        # # # independant ttest 
        _, p_val = stats.ttest_ind(X_S1, X_S2, equal_var = False, nan_policy='propagate', axis=0) 
        
        X_S1[:, p_val>=self.pval] = 0 
        X_S2[:, p_val>=self.pval] = 0 
        
        self.coef_ = np.median(X_S1, axis=0) - np.median(X_S2, axis=0) 
        # print(self.coef_.shape) 
        
        return self
    
    def fit(self, X, y): 
        self.get_coefs(X,y) 
        return self 
        
    def predict(self, X):         
        y_pred = self.score(X)
        if y_pred>0:
            y_pred=0
        else:
            y_pred=1

        return y_pred
    
    def score(self, X, y=None):
        ''' X is n_samples x n_features, coefs_ is n_features '''
        
        if y is not None : 
            X_S1, X_S2 = self.get_X_S1_X_S2(X,y) 
            score_ = ( get_proj(self.coef_, X_S1) - get_proj(self.coef_, X_S2) ) / 2.0 # mean over samples 
        else: 
            score_ = get_proj(self.coef_, X) # mean over samples
        
        # else:
        #     if y is not None : 
        #         X_S1, X_S2 = self.get_X_S1_X_S2(X,y) 
        #         score_ = ( get_cos(self.coef_, X_S1) - get_cos(self.coef_, X_S2) ) / 2.0 # mean over samples 
        #     else: 
        #         score_ = get_cos(self.coef_, X)
        
        return score_
    
    
