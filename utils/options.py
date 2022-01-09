import multiprocessing 
from . import constants as gv 
import numpy as np

def set_globals(**opts):

    gv.code = opts['code']

    gv.laser_on = opts['laser'] 
    
    gv.num_cores = opts['n_jobs']
    gv.IF_SAVE = opts['IF_SAVE']
    gv.SYNTHETIC = opts['SYNTHETIC'] 
    gv.data_type = opts['type']

    gv.first_days = opts['firstDays']
    gv.last_days = opts['lastDays']
    gv.all_days = opts['allDays']
    
    gv.inner_scoring = opts['inner_scoring']
    
    # parameters 
    gv.mouse = gv.mice[opts['i_mice']] 
    
    # if opts['task']=='all': 
    #     gv.tasks = ['all'] 
    #     gv.task = ['all'] 
    # elif opts['task']=='Dual': 
    #     gv.tasks = ['Dual'] 
    #     gv.task = ['Dual']   
    # else:
    #     gv.tasks = opts['tasks'] 
    #     gv.task = gv.tasks[opts['i_task']] 
    
    gv.day = gv.days[opts['i_day']] 
    
    # if gv.code=='memory':
    gv.epochs = [ 'ED', 'MD', 'LD'] 
    # elif gv.code=='sensory':
    # gv.epochs = ['STIM', 'DIST', 'TEST'] 
    # else:
    gv.epochs = opts['epochs']
    
    gv.epoch = gv.epochs[opts['i_epoch']] 

    gv.n_days = opts['n_days'] 
    
    gv.SAME_DAYS = opts['same_days']
    gv.cos_trials = opts['cos_trials']
    gv.scores_trials = opts['scores_trials']
    gv.inter_trials = opts['inter_trials'] 
    
    if not opts['inter_trials']: 
        # gv.pal = ['#ff00ff','#ffff00','#00ffff'] 
        gv.pal = ['black', 'dimgray', 'lightgray'] 
    else: 
        gv.pal = ['r','b','y','k'] 
    
    # preprocessing
    gv.DECONVOLVE= opts['DCV']
    gv.DCV_THRESHOLD = opts['DCV_TH']
    
    gv.F0_THRESHOLD = opts['F0_TH']
    gv.AVG_F0_TRIALS = opts['F0_AVG_TRIALS']
    
    gv.Z_SCORE = opts['Z_SCORE']
    gv.Z_SCORE_BL = opts['Z_SCORE_BL']
    gv.Z_SCORE_TRIALS = opts['Z_SCORE_TRIALS'] 
    gv.NORMALIZE = opts['NORM'] 
    gv.NORMALIZE_TRIALS = opts['NORM_TRIALS'] 

    gv.standardize = opts['scaler']
    
    gv.DETREND = opts['detrend'] # detrend the data 
    gv.DETREND_ORDER = opts['order'] # detrend the data 
    gv.SAVGOL = opts['savgol'] # sav_gol filter 

    gv.T_WINDOW = opts['T_WINDOW'] 
    gv.EDvsLD = opts['EDvsLD'] 
    gv.CONCAT_BINS = opts['concatBins']
    gv.ED_MD_LD = opts['ED_MD_LD'] 
    gv.DELAY_ONLY = 0 

    # feature selection 
    gv.FEATURE_SELECTION = 0 
    gv.LASSOCV = 0 
        
    # bootstrap 
    gv.n_boots = opts['n_boots'] 
    gv.bootstrap_method = opts['bootstrap_method'] 
    gv.bootstrap_cos = opts['boot_cos'] 
    gv.n_cos_boots = opts['n_cos_boots'] 

    gv.correct_trial = opts['correct_trials']
    gv.pair_trials = opts['pair_trials']    
    
    # temporal decoder
    gv.my_decoder = opts['my_decoder']
    gv.fold_type = opts['fold_type']
    gv.n_iter = opts['n_iter']
    
    # classification parameters 
    gv.clf_name = opts['clf'] 
    gv.scoring = opts['scoring'] 
    gv.TIBSHIRANI_TRICK = 0  

    # dimensionality reduction 

    # PCA parameters
    gv.AVG_BEFORE_PCA = 1 
    gv.pca_model = opts['pca_model'] # PCA, sparsePCA, supervisedPCA or None
    gv.explained_variance = opts['exp_var']
    gv.n_components = opts['n_comp']
    gv.list_n_components = None 
    gv.inflection = opts['inflection']

    gv.sparse_alpha = 1 
    gv.ridge_alpha = .01
    
    gv.pca_method = opts['pca_method'] # 'hybrid', 'concatenated', 'averaged' or None
        
    gv.fix_alpha_lbd = opts['fix_alpha_lbd']
    
def set_options(**kwargs): 
    
    opts = dict()
    
    opts['obj'] = 'frac' # 'cos', 'norm'     
    opts['trial_type'] = 'correct'
    opts['trials'] = 'correct' 
    opts['stimulus'] = 'sample'
    opts['sample'] = 'S1' # S1, S2, or S1_S2  
    opts['n_samples'] = 1000 # for permutation test 
    opts['n_shuffles'] = 1000 # for permutation test 
    
    opts['ci']=1 
    opts['shuffle']=1 
    opts['perm_test']=1
    
    opts['mouse_name'] = ['Dumb', 'Alice', 'Bob', 'Charly', 'Dave', 'Eve', 'mPFC', 'ACC'] 
    opts['tasks'] = np.array(['DPA', 'DualGo', 'DualNoGo'])
    opts['code'] = 'memory' 
    opts['pval'] = .05 
    opts['verbose'] = 0 
    opts['type'] = 'raw' 
    opts['n_jobs'] = int(0.9*multiprocessing.cpu_count()) 
    opts['IF_SAVE'] = 1
    opts['add_vlines']=0
    opts['SYNTHETIC'] = 0
    
    opts['fix_alpha_lbd'] = 0
    opts['bins'] = 'ED'
    opts['Delta0']= None 
    
    opts['firstDays'] = 0 
    opts['lastDays'] = 0 
    opts['allDays'] = 0 
    
    opts['stim'] = 0
    opts['day'] = 'all'
    
    # globals 
    opts['i_mice'] = 1
    opts['i_day'] = -1
    opts['i_trial'] = 0  
    opts['i_epoch'] = 0
    opts['i_task'] = 0 
    opts['task'] = 'DPA' # DPA, DualGo, DualNoGo, Dual, or all 
    opts['n_days'] = 6
    
    opts['same_days'] = 1 
    opts['laser']=0 

    opts['feature_sel'] = 'ttest_ind' # 'ttest_ind' or 'lasso' 
    # bootstrap
    opts['boots'] = False 
    opts['n_boots'] = int(1e3) 
    opts['bootstrap_method'] = 'block' # 'bayes', 'bagging', 'standard', 'block' or 'hierarchical' 
    opts['boot_cos'] = 0 
    opts['n_cos_boots'] = int(1e3) 
    
    opts['cos_trials']=0
    opts['correct_trials']=0
    opts['pair_trials']=0

    # temporal decoder 
    opts['inter_trials'] = 1 
    opts['scores_trials'] = 0 
    opts['n_iter'] = 1 
    opts['my_decoder'] = 0 
    opts['fold_type'] = 'stratified' 
    
    # preprocessing parameters 
    opts['T_WINDOW'] = 0.5 
    opts['EDvsLD'] = 1 # average over epochs ED, MD and LD 
    
    opts['epochs'] = ['ED', 'MD', 'LD']
    # opts['epochs'] = ['STIM', 'DIST', 'LD'] 
    
    opts['concatBins'] = ''    
    opts['ED_MD_LD'] = 0     
    opts['savgol'] = 0 # sav_gol filter 
    
    opts['DCV']=0 
    opts['DCV_TH']=0.5 
    
    opts['F0_TH']=None 
    opts['F0_AVG_TRIALS'] = 0
    
    opts['Z_SCORE'] = 0 
    opts['Z_SCORE_BL'] = 0 
    opts['Z_SCORE_TRIALS'] = 0 
    opts['NORM'] = 0 
    opts['NORM_TRIALS'] = 0 
    
    opts['detrend'] = 0 # detrend the data 
    opts['order'] = 3
    
    opts['scaler_BL']= 'standard' # standard, robust, center
    opts['center_BL']= None 
    opts['scale_BL']= None 
    
    opts['scaler']= 'standard' # standard, robust, center 
    opts['center']= None 
    opts['scale']= None 
    opts['return_center_scale'] = 0 
    
    opts['avg_mean']=0 
    opts['avg_noise']=0 
    opts['unit_var']=0 
    
    # PCA parameters 
    opts['pca_model'] = None # PCA, sparsePCA, supervisedPCA or None
    opts['pca_method'] = 'hybrid' # 'hybrid', 'concatenated', 'averaged' or None
    opts['exp_var'] = 0.90 
    opts['n_comp'] = None
    opts['inflection'] = False 
    
    # classification parameters 
    opts['clf']='logitnetAlphaCV' 
    # opts['clf'] = 'LogisticRegression' 
    opts['scoring'] = 'roc_auc' # 'accuracy', 'f1', 'roc_auc'
    opts['inner_scoring'] = 'deviance' # 'accuracy', 'f1', 'roc_auc' or 'neg_log_loss' 'r2' 
    opts['inner_splits'] = 5 
    
    # sklearn LogisticRegression, LogisticRegressionCV 
    opts['C']=1e2 
    opts['Cs']=10 
    opts['penalty']='l2' 
    opts['solver']='liblinear' # liblinear or saga 
    opts['l1_ratio'] = 0.5
    
    # LDA
    opts['loss']='lsqr' 
    opts['shrinkage']='auto'

    # LassoLarsIC
    opts['criterion']='bic'

    opts['fit_intercept'] = False
    opts['intercept_scaling']=1e2
    
    # for glmnet only 
    opts['n_splits'] = 5
    opts['alpha'] = 0.5 
    opts['n_alpha'] = 10 
    opts['n_lambda'] = 10 
    opts['alpha_path']= None # -np.sort(-np.logspace(-4, -2, opts['Cs'])) 
    opts['min_lambda_ratio'] = 1e-4 
    opts['prescreen'] = False 
    
    opts['lbd'] = 'lambda_1se' 
    
    opts['off_diag']=True     
    opts['lambda_path']= None # -np.sort(-np.linspace(-3, -1, opts['Cs'])) 
    opts['cut_point']=1 
    
    # opts['shuffle'] = True 
    opts['random_state'] = None 
    opts['tol']=1e-4
    opts['max_iter']= int(1e3) 
    
    opts.update(kwargs) 
    # if opts['concatBins']==1:
    #     opts['EDvsLD']=0
    
    return opts 
