import numpy as np 
from scipy.io import loadmat 

from . import constants as gv 

def get_n_trials():
    if gv.mouse in [gv.mice[0]]: 
        gv.n_trials = 40 
    elif gv.mouse in [gv.mice[4]] or gv.mouse in [gv.mice[5]]: 
        gv.n_trials = 64
    else:
        gv.n_trials = 32 
            
def get_days():
    if (gv.mouse=='JawsM15') | (gv.mouse=='ChRM04') | (gv.mouse=='JawsM18'):
        if gv.n_days == 9:
            gv.days = np.arange(1, gv.n_days) 
        else: 
            gv.days = np.arange(1, gv.n_days+1 ) 
    else: 
        gv.days = [1,2,3,4,5] 
        
def get_delays_times():
    if((gv.mouse=='ChRM04') | ('Jaws' in gv.mouse) | ('ACC' in gv.mouse) ): 
        gv.t_ED = [3, 4.5] 
        gv.t_MD = [5.5, 6.5] 
        gv.t_LD = [7.5, 9] 
    else: 
        gv.t_ED = [3, 6] 
        gv.t_MD = [7, 8] 
        gv.t_LD = [9, 12] 

def get_stimuli_times():
    if gv.mouse=='C57_2_DualTask' : 
        gv.t_DIST = [6, 7] 
        gv.t_CUE = [8, 8.5] 
        gv.t_RWD = [8.5, 9] 
        gv.t_TEST = [12, 13] 
    else:
        gv.t_DIST = [4.5, 5.5] 
        gv.t_CUE = [6.5, 7] 
        gv.t_RWD = [7, 7.5] 
        gv.t_TEST = [9, 10] 
        
def get_fluo_data():
    
    get_days()
    # print('days', gv.days)
    
    if gv.SAME_DAYS: 
        # print('same neurons accross days') 
        if 'ACC' in gv.mouse:
            # print(gv.path + '/data/' + gv.mouse + '/SamedROI/' + gv.mouse + '_day_' + str(gv.day) + '.mat' )
            data = loadmat(gv.path + '/data/' + gv.mouse + '/SamedROI/' + gv.mouse + '_day_' + str(gv.day) + '.mat' ) 
        else:
            data = loadmat(gv.path + '/data/' + gv.mouse + '/SamedROI_0%dDays/' % gv.n_days + gv.mouse + '_day_' + str(gv.day) + '.mat' )
                
    else: 
        data = loadmat(gv.path + '/data/' + gv.mouse + '/' + gv.mouse +'_day_' + str(gv.day) + '.mat') 
            
    if 'raw' in gv.data_type:
        # print('raw') 
        X_data = np.rollaxis(data['Cdf_Mice'],1,0) 
    else: 
        # print('dF') 
        X_data = np.rollaxis(data['dff_Mice'],1,0) 
            
    y_labels = data['Events'].transpose()
    
    if 'ACC' in gv.mouse:
        # print('mouse', gv.mouse, 'days', gv.days, 'type', gv.data_type, 'all data: X', X_data.shape,'y', y_labels.shape)
        
        X_data = X_data.reshape( (5, int(X_data.shape[0]/5), X_data.shape[1], X_data.shape[2]) )
        y_labels = y_labels.T.reshape( (5, int(y_labels.T.shape[0]/5), y_labels.T.shape[1]) ) 

        X_data = X_data[gv.day-1]
        y_labels = y_labels[gv.day-1].T 
        
    gv.frame_rate = 6 
    
    # print(data.keys()) 
    # print(y_labels[4])
    
    gv.duration = X_data.shape[2]/gv.frame_rate 
    gv.time = np.linspace(0, gv.duration, X_data.shape[2]); 
    gv.bins = np.arange(0, len(gv.time)) 
    gv.n_neurons = X_data.shape[1] 
    gv.trial_size = X_data.shape[2] 
    
    get_stimuli_times() 
    get_delays_times() 
    get_n_trials() 
    get_bins() 
    
    print('mouse', gv.mouse, 'day', gv.day, 'type', gv.data_type, 'all data: X', X_data.shape,'y', y_labels.shape) 
    # print(gv.n_neurons)
    # print(y_labels[4])
    return X_data, y_labels

def correct_incorrect_labels(y_labels):
    bool_correct = ( y_labels[2]==1 ) | ( y_labels[2]==4 ) 
    bool_incorrect = ( y_labels[2]==2 ) | ( y_labels[2]==3 ) 
    
    y_correct = np.argwhere( bool_correct ).flatten()
    y_incorrect = np.argwhere( bool_incorrect ).flatten()
    
    return y_correct, y_incorrect

def which_trials(y_labels): 
    y_trials = []
    
    if gv.laser_on:
        bool_ND = (y_labels[4]==0) & (y_labels[8]!=0) 
        bool_D1 = (y_labels[4]==13) & (y_labels[8]!=0) 
        bool_D2 = (y_labels[4]==14) & (y_labels[8]!=0) 
    else: 
        bool_ND = (y_labels[4]==0) & (y_labels[8]==0) 
        bool_D1 = (y_labels[4]==13) & (y_labels[8]==0) 
        bool_D2 = (y_labels[4]==14) & (y_labels[8]==0) 
    
    bool_S1 = (y_labels[0]==17) 
    bool_S2 = (y_labels[0]==18) 
    
    bool_T1 = (y_labels[1]==11) 
    bool_T2 = (y_labels[1]==12) 
    
    bool_pair = ( (y_labels[0]==17) & (y_labels[1]==11) ) | ( (y_labels[0]==18) & (y_labels[1]==12) ) 
    bool_unpair = ( (y_labels[0]==17) & (y_labels[1]==12) ) | ( (y_labels[0]==18) & (y_labels[1]==11) ) 
    
    if 'all' in gv.task: 
        bool_trial = 1
    elif 'DPA' in gv.task: 
        bool_trial = bool_ND 
    elif 'DualGo' in gv.task: 
        bool_trial = bool_D1 
    elif 'DualNoGo' in gv.task: 
        bool_trial = bool_D2 
    elif 'Dual' in gv.task:
        bool_trial = bool_D1 | bool_D2        
    
    print(gv.task)
    
    if 'incorrect' in gv.task:
        bool_incorrect = ( y_labels[2]==2 ) | ( y_labels[2]==3 ) 
        bool_trial = bool_trial & bool_incorrect 
    elif 'correct' in gv.task: 
        bool_correct = ( y_labels[2]==1 ) | ( y_labels[2]==4 ) 
        bool_trial = bool_trial & bool_correct 
    
    if 'pair' in gv.task: 
        bool_trial = bool_trial & bool_pair 
        # y_trials = np.argwhere( bool_trial & bool_pair ).flatten() 
    elif 'unpair' in gv.task: 
        bool_trial = bool_trial & bool_unpair 
        # y_trials = np.argwhere( bool_trial & bool_unpair ).flatten() 
        
    if 'S1' in gv.task: 
        y_trials = np.argwhere( bool_trial & bool_S1 ).flatten() 
    elif 'S2' in gv.task: 
        y_trials = np.argwhere( bool_trial & bool_S2 ).flatten() 
                
    elif 'D1' in gv.task:
        y_trials = np.argwhere( bool_trial & bool_D1 ).flatten() 
    elif 'D2' in gv.task: 
        y_trials = np.argwhere( bool_trial & bool_D2 ).flatten() 
    
    elif 'T1' in gv.task: 
        y_trials = np.argwhere( bool_trial & bool_T1 ).flatten() 
    elif 'T2' in gv.task: 
        y_trials = np.argwhere( bool_trial & bool_T2 ).flatten() 
    else:
        y_trials = np.argwhere( bool_trial & bool_S1 & bool_S2 ).flatten()
    
    return y_trials 

def get_pair_trials(X_data, y_labels): 
    
    task = gv.task 
    gv.task = task + "_pair" 
    y_pair = which_trials(y_labels) 
    
    task = gv.task 
    gv.task = task + "_unpair" 
    y_unpair = which_trials(y_labels) 
    
    gv.task = task 
    X_pair = X_data[y_pair] 
    X_unpair = X_data[y_unpair] 
    
    # print('X_pair', X_pair.shape, 'X_unpair', X_unpair.shape) 
    return X_pair, X_unpair 
    
def get_S1_S2_trials(X_data, y_labels): 
    
    task = gv.task
    gv.task = task + "_S1" 
    y_S1 = which_trials(y_labels) 
    
    gv.task = task + "_S2" 
    y_S2 = which_trials(y_labels) 
    
    gv.task = task 
    X_S1 = X_data[y_S1] 
    X_S2 = X_data[y_S2] 
    
    # print('X_S1', X_S1.shape, 'X_S2', X_S2.shape)
    
    return X_S1, X_S2 

def get_X_S1_S2(X_data, y_labels, stimulus='sample'): 
    
    X_S1_S2 = np.zeros( (len(gv.tasks), len(gv.samples), int(gv.n_trials/len(gv.samples)), gv.n_neurons, gv.trial_size) ) 
    y_trials = np.zeros( (len(gv.tasks), 3, len(gv.samples), int(gv.n_trials/len(gv.samples)) ) )
    
    if stimulus == 'sample': 
        dum_str = 'S'
    if stimulus=='test':
        dum_str = 'T'
    if stimulus=='distractor':
        dum_str = 'D'
    
    _task = gv.task
        
    for i_task, gv.task in enumerate(gv.tasks):
        task = gv.task 
        gv.task = task + "_%s1" % dum_str 
        y_S1 = which_trials(y_labels) 
        
        gv.task = task + "_%s2" % dum_str 
        y_S2 = which_trials(y_labels) 

        gv.task = task + "_%s1_correct" % dum_str
        y_S1_correct = which_trials(y_labels) 
        
        gv.task = task + "_%s1_incorrect" % dum_str
        y_S1_incorrect = which_trials(y_labels) 
        
        gv.task = task + "_%s2_correct" % dum_str
        y_S2_correct = which_trials(y_labels) 

        gv.task = task + "_%s2_incorrect" % dum_str
        y_S2_incorrect = which_trials(y_labels) 
        
        y_S1_S2 = np.stack((y_S1, y_S2))
        print(y_S1_S2.shape)
        
        y_correct = np.empty((2, y_S1_S2.shape[1])) * np.nan 
        y_correct[0,0:y_S1_correct.shape[0]] = y_S1_correct            
        y_correct[1,0:y_S2_correct.shape[0]] = y_S2_correct
        
        y_incorrect = np.empty((2, y_S1_S2.shape[1])) * np.nan 
        y_incorrect[0,0:y_S1_incorrect.shape[0]] = y_S1_incorrect 
        y_incorrect[1,0:y_S2_incorrect.shape[0]] = y_S2_incorrect 
        
        y_trials[i_task] = np.stack( (y_S1_S2, y_correct, y_incorrect) ) 
        
        gv.task = task 
        X_S1 = X_data[y_S1] 
        X_S2 = X_data[y_S2] 
        
        X_S1_S2[i_task, 0, 0:X_S1.shape[0]] = X_S1 
        X_S1_S2[i_task, 1, 0:X_S2.shape[0]] = X_S2 
        
    print('X_S1', X_S1.shape, 'X_S2', X_S2.shape, 'X_S1_S2', X_S1_S2.shape) 
    gv.task = _task 
    
    return X_S1_S2, y_trials 

def get_X_S1_X_S2_task(X_data, y_labels, stimulus='sample', task='all', trials = 'correct'): 
        
    if stimulus == 'sample': 
        dum_str = 'S'
    if stimulus=='test':
        dum_str = 'T'
    if stimulus=='distractor':
        dum_str = 'D'
    
    gv.task = task + "_%s1_" % dum_str + trials
    y_S1 = which_trials(y_labels) 
    
    gv.task = task + "_%s2_" % dum_str + trials 
    y_S2 = which_trials(y_labels) 
    
    gv.task = task 
        
    X_S1 = X_data[y_S1] 
    X_S2 = X_data[y_S2] 
    
    return X_S1, X_S2 

def get_bins():

    if(gv.T_WINDOW==0): 
        gv.bins_BL = [ bin for bin in gv.bins if (gv.time[bin]>=gv.t_BL[0]) and (gv.time[bin]<=gv.t_BL[1])] 
    
        gv.bins_STIM = [ bin for bin in gv.bins if (gv.time[bin]>=gv.t_STIM[0]) and (gv.time[bin]<=gv.t_STIM[1]) ] 
    
        gv.bins_ED = [ bin for bin in gv.bins if (gv.time[bin]>=gv.t_ED[0]) and (gv.time[bin]<=gv.t_ED[1]) ] 
                
        gv.bins_DIST = [ bin for bin in gv.bins if (gv.time[bin]>=gv.t_DIST[0]) and (gv.time[bin]<=gv.t_DIST[1]) ]
        
        gv.bins_MD = [ bin for bin in gv.bins if (gv.time[bin]>=gv.t_MD[0]) and (gv.time[bin]<=gv.t_MD[1]) ]
        
        gv.bins_CUE = [ bin for bin in gv.bins if (gv.time[bin]>=gv.t_CUE[0]) and (gv.time[bin]<=gv.t_CUE[1]) ]
        
        gv.bins_LD = [ bin for bin in gv.bins if (gv.time[bin]>=gv.t_LD[0]) and (gv.time[bin]<=gv.t_LD[1]) ] 
        
        gv.bins_RWD = [ bin for bin in gv.bins if (gv.time[bin]>=gv.t_RWD[0]) and (gv.time[bin]<=gv.t_RWD[1]) ]
        
        gv.bins_TEST = [ bin for bin in gv.bins if (gv.time[bin]>=gv.t_TEST[0]) and (gv.time[bin]<=gv.t_TEST[1]) ] 

        gv.bins_DELAY = [ bin for bin in gv.bins if (gv.time[bin]>=gv.t_ED[0]) and (gv.time[bin]<=gv.t_LD[1]) ] 
        
    else:
        gv.bins_BL = [ bin for bin in gv.bins if (gv.time[bin]>=gv.t_BL[0]) and (gv.time[bin]<=gv.t_BL[1] ) ] 
        
        gv.bins_STIM = [ bin for bin in gv.bins if (gv.time[bin]>=gv.t_STIM[1]-gv.T_WINDOW) and (gv.time[bin]<=gv.t_STIM[1]) ] 
    
        gv.bins_ED = [ bin for bin in gv.bins if (gv.time[bin]>=gv.t_ED[1]-gv.T_WINDOW) and (gv.time[bin]<=gv.t_ED[1]) ] 
        
        gv.bins_DIST = [ bin for bin in gv.bins if (gv.time[bin]>=gv.t_DIST[1]-gv.T_WINDOW) and (gv.time[bin]<=gv.t_DIST[1] ) ] 
        
        gv.bins_MD = [ bin for bin in gv.bins if (gv.time[bin]>=gv.t_MD[1]-gv.T_WINDOW) and (gv.time[bin]<=gv.t_MD[1] ) ]
        
        gv.bins_CUE = [ bin for bin in gv.bins if (gv.time[bin]>=gv.t_CUE[1]-gv.T_WINDOW) and (gv.time[bin]<=gv.t_CUE[1]) ]
        
        gv.bins_LD = [ bin for bin in gv.bins if (gv.time[bin]>=gv.t_LD[1]-gv.T_WINDOW) and (gv.time[bin]<=gv.t_LD[1] ) ] 

        gv.bins_RWD = [ bin for bin in gv.bins if (gv.time[bin]>=gv.t_RWD[1]-gv.T_WINDOW) and (gv.time[bin]<=gv.t_RWD[1] ) ] 
        
        gv.bins_TEST = [ bin for bin in gv.bins if (gv.time[bin]>=gv.t_TEST[1]-gv.T_WINDOW) and (gv.time[bin]<=gv.t_TEST[1]) ] 
        
        gv.bins_DELAY = [ bin for bin in gv.bins if (gv.time[bin]>=gv.t_ED[0]) and (gv.time[bin]<=gv.t_LD[1]) ] 
        
