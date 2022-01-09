import numpy as np 

from . import constants as gv 
from . import get_data as data 
from . import preprocessing as pp

def get_X_y_day(day=1, stimulus='sample'): 
    ''' get X (fluo) and y (labels) for a given day 
    inputs: - day (int)
    outputs: - X_trials, y_trials 
    '''
    
    # print('day', day) 
    gv.day = day 
    
    X_all, y_all = data.get_fluo_data() 
    X_trials, y_trials = data.get_X_S1_S2(X_all, y_all, stimulus) 
    # print('X_trials', X_trials.shape, 'y_trials', y_trials.shape) 
    
    return X_trials, y_trials 

def get_X_S1_X_S2_day_task(day=1, stimulus='sample', task='all', trials='correct'): 
    ''' get X (fluo) and y (labels) for a given day 
    inputs: - day (int)
    outputs: - X_trials, y_trials 
    '''    
    gv.day = day 
    
    X_all, y_all = data.get_fluo_data() 
    X_S1, X_S2 = data.get_X_S1_X_S2_task(X_all, y_all, stimulus, task, trials=trials)    
    print('X_S1', X_S1.shape, 'X_S2', X_S2.shape) 
    # X_S1 = pp.preprocess_X(X_S1) 
    # X_S2 = pp.preprocess_X(X_S2)
    
    # X_S1_S2 = np.vstack( (X_S1, X_S2) ) 
    # X_scale = pp.preprocess_X(X_S1_S2) 
    # X_S1 = X_scale[:X_S1.shape[0]]
    # X_S2 = X_scale[X_S1.shape[0]:]
    
    return X_S1, X_S2 

def get_X_y_days(day='all', stimulus='sample'): 
    ''' Returns X, y for days in day 
    inputs: - day 
    outputs: - X_days, y_days 
    '''
    if day=='first': 
        day_list = gv.days[:3] 
    elif day=='last': 
        day_list = gv.days[3:] 
    elif day=='all': 
        day_list = gv.days 
    elif isinstance(day, int): 
        day_list = [day] 
    elif day.isnumeric():
        day_list = [int(day)] 
    
    print('days', day_list)
    
    dum=0
    for i_day in day_list:
        X_trials, y_trials = get_X_y_day(day=i_day, stimulus=stimulus) 

        # print(X_trials.shape, y_trials.shape)
        
        y_trials = y_trials +  dum * 192 # important to distinguish between days correct and incorrect trials 
        if dum==0:
            X_days = X_trials[np.newaxis] 
            y_days = y_trials[np.newaxis] 
        else: 
            X_days = np.concatenate((X_days, X_trials[np.newaxis])) 
            y_days = np.concatenate((y_days, y_trials[np.newaxis])) 
        
        dum = dum +1
        
    # print('X_days', X_days.shape, 'y_days', y_days.shape) 
    X_days = np.swapaxes(X_days, 2, 3) 
    X_days = np.hstack(X_days) 
    X_days = np.swapaxes(X_days, 1, 2) 

    y_days = np.swapaxes(y_days, 2, 4) 
    y_days = np.hstack(y_days) 
    y_days = np.swapaxes(y_days, 1, 3) 
    # print('X_days', X_days.shape, 'y_days', y_days.shape) 
    
    return X_days, y_days 

def get_X_S1_X_S2_days_task(day='all', stimulus='sample', task='all', trials='correct'): 
    ''' Returns X, y for days in day 
    inputs: - day 
    outputs: - X_days, y_days 
    '''
    
    if day=='first': 
        day_list = gv.days[:3] 
    elif day=='last': 
        day_list = gv.days[3:] 
    elif day=='all': 
        day_list = gv.days 
    elif isinstance(day, int): 
        day_list = [day] 
    elif day.isnumeric():
        day_list = [int(day)] 
    
    print('days', day_list) 
    dum=0
    for i_day in day_list:
        if(dum==0):
            X_S1_days, X_S2_days = get_X_S1_X_S2_day_task(day=i_day, stimulus=stimulus, task=task, trials=trials) 
            dum=1 
        else: 
            X_S1_day, X_S2_day = get_X_S1_X_S2_day_task(day=i_day, stimulus=stimulus, task=task, trials=trials) 
            
            X_S1_days = np.vstack( (X_S1_days, X_S1_day) )
            X_S2_days = np.vstack( (X_S2_days, X_S2_day) )
        
        # print(X_S1_days.shape, X_S2_days.shape) 
    
    return X_S1_days, X_S2_days 
