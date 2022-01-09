from sklearn.feature_selection import SelectKBest, chi2, VarianceThreshold, f_regression, mutual_info_classif, f_classif 

def feature_selection(X, method='variance'):

    X_avg = np.mean(X[:,:,gv.bins_ED_MD_LD],axis=-1) 

    if 'variance' in method :
        idx = fs.featSel.var_fit_transform(X_avg, threshold=threshold) 
        X_avg = np.delete(X_avg, idx, axis=1) 
        X = np.delete(X, idx, axis=1)
            
    if 'mutual' in method:
        idx = fs.featSel.select_best(X_avg, y, percentage=1-threshold) 
        X_avg = np.delete(X_avg, idx, axis=1) 
        X = np.delete(X, idx, axis=1)
            
    if 'correlation' in method:
        idx = fs.featSel.select_indep(X_avg, threshold=threshold) 
        X_avg = np.delete(X_avg, idx, axis=1) 
        X = np.delete(X, idx, axis=1) 
