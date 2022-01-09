import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import pickle

def SetPlotParams(x=3.5):

    plt.style.use('ggplot')
    
    golden_ratio = (5 ** 0.5 - 1) / 2  

    # sns.set_style('dark') # darkgrid, white grid, dark, white and ticks
    plt.rcParams['axes.spines.right'] = False
    plt.rcParams['axes.spines.top'] = False
    sns.color_palette('deep')
    
    plt.style.use('ggplot')
    
    fig_width = x # width in inches 
    fig_height = x * golden_ratio  # height in inches 
    
    fig_size =  [fig_width,fig_height]
    plt.rcParams['figure.figsize'] = fig_size
    plt.rcParams['figure.autolayout'] = True
    
    plt.rcParams['lines.linewidth'] = 1.0
    plt.rcParams['lines.markeredgewidth'] = 0.3
    plt.rcParams['lines.markersize'] = 2.5
    plt.rcParams['font.size'] = 12
    plt.rcParams['legend.fontsize'] = 12
    plt.rcParams['axes.facecolor'] = '1'
    plt.rcParams['axes.edgecolor'] = '0'
    plt.rcParams['axes.linewidth'] = '0.7'
    
    plt.rcParams['axes.labelcolor'] = '0' 
    plt.rcParams['axes.labelsize'] = 12 
    plt.rcParams['axes.titlesize'] = 12 
    plt.rcParams['xtick.labelsize'] = 10 
    plt.rcParams['ytick.labelsize'] = 10 
    plt.rcParams['xtick.color'] = '0' 
    plt.rcParams['ytick.color'] = '0' 
    plt.rcParams['xtick.major.size'] = 2 
    plt.rcParams['ytick.major.size'] = 2 
    plt.rcParams["font.family"] = "Times New Roman"    
    # plt.rcParams['font.sans-serif'] = 'Arial' 

def SetPlotDim(x,y):
    
    fig_width = x # width in inches
    fig_height = y # height in inches
    fig_size = [fig_width,fig_height]
    plt.rcParams['figure.figsize'] = fig_size
    plt.rcParams['figure.autolayout'] = True
    
    
#### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### 
#### Store and retrieve objects

def Store (obj, name, path):
    
    f = open (path+name, 'wb')
    pickle.dump(obj, f)
    f.close()

def Retrieve (name, path):

    f = open(path+name, 'rb')
    obj = pickle.load(f)
    f.close()

    return obj
