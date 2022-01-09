import seaborn as sns
import numpy as np 
import multiprocessing 

global clf, clf_name
clf = ''
clf_name = ''

global eps
eps = np.finfo(float).eps

global num_cores
num_cores = int(0.9*multiprocessing.cpu_count()) 

global path, scriptdir, figdir, filedir 
path = '/homecentral/alexandre.mahrach/IDIBAPS/python/data_analysis' 
figdir = path + '/figs' 
filedir = path + '/data' 

global mouse, mice
mouse = 'ChRM04' 
mice = ['C57_2_DualTask','ChRM04','JawsM15','JawsM18','ACCM03','ACCM04', 'mPFC', 'ACC'] 

global day, days, n_days
day=1
n_days = 6

days = np.arange(1, n_days+1) 

global task, tasks 
task = 'DPA'
tasks = ['DPA', 'DualGo', 'DualNoGo'] 

global epoch_str, task_str 
epoch_str = ['Early', 'Middle', 'Late'] 
task_str = ['DPA', 'Dual Go', 'Dual NoGo'] 

global SAME_DAYS
SAME_DAYS = 1

global code
code = 'sensory' # 'memory', 'sensory', 'decision'

global IF_SAVE
IF_SAVE = 1 

global laser_on 
laser_on = 0

global epochs
epoch = 'STIM'
# epochs = ['Baseline','Stim','ED','Dist','MD','CUE','LD','Test'] 
# epochs = ['ED','MD','LD'] 
epochs = ['STIM', 'DIST', 'TEST'] 

global t_ED, t_MD, t_LD
t_ED = []
t_MD = []
t_LD = []

global frame_rate, n_bin, duration, time 
frame_rate = []
n_bin = []
duration = []
time = []

global t_BL, t_STIM, t_TEST, t_DIST, t_CUE, t_RWD 
t_BL = [0,2]
t_STIM = [2,3]
t_TEST = []
t_DIST = []
t_CUE = []
t_RWD = []

global bins, bins_BL, bins_STIM, bins_ED, bins_DIST, bins_MD, bins_LD, bins_CUE, bins_RWD, bins_TEST
bins = []
bins_BL = []
bins_STIM=[]
bins_ED=[]
bins_DIST=[]
bins_MD=[]
bins_CUE = []
bins_RWD = [] 
bins_LD=[]
bins_TEST = []


global  n_neurons, n_trials, trial_size 
n_neurons = [] 
n_trials= [] 
trial_size = [] 

global stimulus, samples
stimulus = 'SAMPLE' # 'SAMPLE' or 'TEST'
samples = ['S1', 'S2'] 

global data_type 
data_type = 'raw' # 'raw' or 'dF'

global correct_trial 
correct_trial = 0

global bin_start, t_start
bin_start = np.array(0)
t_start = np.array(0)

global CONCAT_BINS
CONCAT_BINS=''

global pal
# pal = ['r','b','y']
pal = [sns.color_palette('colorblind')[2], sns.color_palette('colorblind')[0], sns.color_palette('colorblind')[1]] 

global T_WINDOW
T_WINDOW = 0 

global SAVGOL, SAVGOL_ORDER
SAVGOL=0
SAVGOL_ORDER=1

global Z_SCORE, Z_SCORE_BL, DECONVOLVE, DCV_THRESHOLD, bins_z_score, NORMALIZE, Z_SCORE_TRIALS
Z_SCORE_BL=0 
Z_SCORE=0
Z_SCORE_TRIALS = 0 
bins_z_score = 0
DECONVOLVE=0
DCV_THRESHOLD=.1
NORMALIZE=0
NORMALIZE_TRIALS=0 

global SYNTHETIC
SYNTHETIC=0

global standardize
standardize=True

global F0_THRESHOLD, AVG_F0_TRIALS
F0_THRESHOLD=None
AVG_F0_TRIALS=0

global inter_trials
inter_trials=1

global first_days, last_days, all_days
first_days=0
last_days=0 
all_days=0
