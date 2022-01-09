from sklearn.svm import LinearSVC 
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

from . import constants as gv 
from .options import * 
from .glmnet_wrapper import logitnet, logitnetCV, logitnetIterCV, logitnetAlphaCV, logitnetAlphaIterCV 

from .dot_clf import dot_clf
def get_clf(**kwargs):
    
    # options = set_options(**kwargs)
    # globals().update(options) 

    if 'dot' in kwargs['clf']:
        clf = dot_clf(pval=kwargs['pval']) 
        
    # sklearn    
    if 'LDA' in kwargs['clf']: 
        clf = LinearDiscriminantAnalysis(tol=kwargs['tol'], solver='lsqr', shrinkage=kwargs['shrinkage']) 
    
    # if 'LinearSVC' in gv.clf_name:
    #     clf = LinearSVC(C=C, penalty=penalty, loss=loss, dual=False,
    #                     tol=tol, max_iter=int(max_iter), multi_class='ovr',
    #                     fit_intercept=fit_intercept, intercept_scaling=intercept_scaling,
    #                     class_weight=None, verbose=0, random_state=None) 
    
    if 'LogisticRegressionCV' in kwargs['clf']:
        clf = LogisticRegressionCV(solver=kwargs['solver'], penalty=kwargs['penalty'], l1_ratios=None, 
                                   tol=kwargs['tol'], max_iter=int(kwargs['max_iter']), scoring=kwargs['scoring'], 
                                   fit_intercept=kwargs['fit_intercept'], intercept_scaling=kwargs['intercept_scaling'], 
                                   cv=kwargs['n_splits'], n_jobs=None) 
    
    elif 'LogisticRegression' in kwargs['clf']:
        clf = LogisticRegression(C=kwargs['C'], solver=kwargs['solver'], penalty=kwargs['penalty'],
                                 tol=kwargs['tol'], max_iter=int(kwargs['max_iter']),
                                 fit_intercept=kwargs['fit_intercept'],  intercept_scaling=kwargs['intercept_scaling'],
                                 l1_ratio=kwargs['l1_ratio'], n_jobs=None) 
    
    # glmnet_python
    if 'logitnetAlphaIterCV' in kwargs['clf']:
        clf = logitnetAlphaIterCV(lbd=kwargs['lbd'], n_alpha=kwargs['n_alpha'], n_lambda=kwargs['n_lambda'], 
                              n_splits=kwargs['inner_splits'], fold_type=kwargs['fold_type'], scoring=kwargs['inner_scoring'],
                              standardize= False, fit_intercept=kwargs['fit_intercept'], prescreen=kwargs['prescreen'], 
                              thresh=kwargs['tol'], maxit=kwargs['max_iter'], n_jobs=None, verbose=kwargs['verbose']) 
    
    elif 'logitnetAlphaCV' in kwargs['clf']:
        clf = logitnetAlphaCV(lbd=kwargs['lbd'], n_alpha=kwargs['n_alpha'], n_lambda=kwargs['n_lambda'], 
                              n_splits=kwargs['inner_splits'], fold_type=kwargs['fold_type'], scoring=kwargs['inner_scoring'],
                              standardize= False, fit_intercept=kwargs['fit_intercept'], prescreen=kwargs['prescreen'], 
                              thresh=kwargs['tol'], maxit=kwargs['max_iter'], n_jobs=None, verbose=kwargs['verbose']) 
        
    elif 'logitnetIterCV' in kwargs['clf']:
        clf = logitnetIterCV(lbd=kwargs['lbd'], alpha=kwargs['alpha'], n_lambda=kwargs['n_lambda'], 
                             n_splits=kwargs['inner_splits'], fold_type=kwargs['fold_type'], scoring=kwargs['inner_scoring'],
                             standardize=False, fit_intercept=kwargs['fit_intercept'], prescreen=kwargs['prescreen'], 
                             thresh=kwargs['tol'], maxit=kwargs['max_iter'], n_jobs=None, verbose=kwargs['verbose']) 
        
    elif 'logitnetCV' in kwargs['clf']:
        clf = logitnetCV(lbd=kwargs['lbd'], alpha=kwargs['alpha'], n_lambda=kwargs['n_lambda'], n_splits=kwargs['inner_splits'],
                         standardize=False, fit_intercept=kwargs['fit_intercept'], prescreen=kwargs['prescreen'],
                         fold_type=kwargs['fold_type'], shuffle=False, random_state=None,
                         scoring=kwargs['inner_scoring'], thresh=kwargs['tol'] , maxit=kwargs['max_iter'], n_jobs=None) 
        
    elif 'logitnet' in kwargs['clf']: 
        clf = logitnet(lbd=kwargs['lbd'], alpha=kwargs['alpha'], n_lambda=kwargs['n_lambda'], prescreen=kwargs['prescreen'], 
                       standardize=False, fit_intercept=kwargs['fit_intercept'], 
                       scoring=kwargs['scoring'], thresh=kwargs['tol'] , maxit=kwargs['max_iter']) 
    
    return clf
