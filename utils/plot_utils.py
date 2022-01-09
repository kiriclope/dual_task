import os, sys

import numpy as np
import pickle
import matplotlib
matplotlib.use('GTK3cairo')

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.lines import Line2D
from mpl_toolkits.mplot3d import Axes3D 

from datetime import date

from . import constants as gv 

shade_alpha = 0.1 
lines_alpha = 0.8

def set_size(width, fraction=1):
    """ Set aesthetic figure dimensions to avoid scaling in latex.

    Parameters
    ----------
    width: float
            Width in pts
    fraction: float
            Fraction of the width which you wish the figure to occupy

    Returns
    -------
    fig_dim: tuple
            Dimensions of figure in inches
    """
    # Width of figure
    fig_width_pt = width * fraction

    # Convert from pt to inches
    inches_per_pt = 1 / 72.27

    # Golden ratio to set aesthetic figure height
    golden_ratio = (5 ** 0.5 - 1) / 2

    # Figure width in inches
    fig_width_in = fig_width_pt * inches_per_pt
    # Figure height in inches
    fig_height_in = fig_width_in * golden_ratio

    return fig_width_in, fig_height_in 

def figDir():

    gv.figdir = gv.path + '/figs' 
    gv.filedir = gv.path + '/data' 
    
    today = date.today() 
    today = today.strftime("/%y-%m-%d")
        
    gv.figdir = gv.figdir + today 
    
    if gv.laser_on: 
        gv.figdir = gv.figdir + '/laser_on'
    else:
        gv.figdir = gv.figdir + '/laser_off'
            
    if gv.SYNTHETIC :
        gv.figdir = gv.figdir + '/synthetic' 
    
    if gv.F0_THRESHOLD is not None: 
        gv.figdir = gv.figdir + '/F0_thresh_%.2f' % gv.F0_THRESHOLD 
        if gv.AVG_F0_TRIALS:
            gv.figdir = gv.figdir + '_avg_trials'
            
    elif gv.DECONVOLVE:
        gv.figdir = gv.figdir + '/deconvolve_th_%.2f' % gv.DCV_THRESHOLD
        
    elif gv.data_type=='dF':
        gv.figdir = gv.figdir + '/dF'
        
    else:
        gv.figdir = gv.figdir + '/rawF' 
                
    if gv.CONCAT_BINS: 
        gv.figdir = gv.figdir + '/concat_bins' 
                
    if gv.T_WINDOW!=0 :
        gv.figdir = gv.figdir + '/t_window_%.1f' % gv.T_WINDOW        
        
    if gv.SAVGOL :
        gv.figdir = gv.figdir + '/savgol' 

    if gv.DETREND: 
        gv.figdir = gv.figdir + '/detrend'          
        
    if gv.Z_SCORE :        
        gv.figdir = gv.figdir + '/z_score'
        
    elif gv.Z_SCORE_BL :        
        gv.figdir = gv.figdir + '/z_score_bl'

    elif gv.Z_SCORE_TRIALS :
        gv.figdir = gv.figdir + '/z_score_trials'
        
    elif gv.NORMALIZE : 
        gv.figdir = gv.figdir + '/norm'     

    if gv.standardize is not None: 
        gv.figdir = gv.figdir + '/%s_scaler' % gv.standardize 
        
    if not os.path.isdir(gv.figdir):
        os.makedirs(gv.figdir)

    print(gv.figdir)
    
def vlines_delay(ax):
    
    ax.axvline(gv.t_ED[0]-2, color='k', ls='--') 
    ax.axvline(gv.t_ED[-1]-2, color='k', ls='--') 

    ax.axvline(gv.t_MD[0]-2, color='r', ls='--')
    ax.axvline(gv.t_MD[-1]-2, color='r', ls='--')

    ax.axvline(gv.t_LD[0]-2, color='k', ls='--')
    ax.axvline(gv.t_LD[-1]-2, color='k', ls='--') 

def vlines_all(ax):
    
    ax.axvline(gv.t_STIM[0]-2, color='k', ls='-')
    ax.axvline(gv.t_DIST[0]-2, color='k', ls='-')    
    ax.axvline(gv.t_TEST[0]-2, color='k', ls='-') 
    
    ax.axvline(gv.t_ED[0]-2, color='k', ls='--') 
    # ax.axvline(gv.t_ED[-1]-2, color='k', ls='--') 
    
    ax.axvline(gv.t_MD[0]-2, color='k', ls='--') 
    ax.axvline(gv.t_MD[-1]-2, color='k', ls='--') 

    ax.axvline(gv.t_LD[0]-2, color='k', ls='--')
    # ax.axvline(gv.t_LD[-1]-2, color='k', ls='--') 
    
def hlines_delay(ax):
    
    ax.axhline(gv.t_ED[0]-2, color='k', ls='--')
    ax.axhline(gv.t_ED[-1]-2, color='k', ls='--')

    ax.axhline(gv.t_MD[0]-2, color='r', ls='--')
    ax.axhline(gv.t_MD[-1]-2, color='r', ls='--')

    ax.axhline(gv.t_LD[0]-2, color='k', ls='--')
    ax.axhline(gv.t_LD[-1]-2, color='k', ls='--')

def add_orientation_legend(ax):
    custom_lines = [Line2D([0], [0], color=gv.pal[k], lw=4) for
                    k in range(len(gv.trials))]
    labels = [t for t in gv.trials]
    ax.legend(custom_lines, labels,
              frameon=False, loc='center left', bbox_to_anchor=(1, 0.5))
    plt.tight_layout(rect=[0,0,0.9,1])    

def save_fig(figname):
    plt.figure(figname)
    if not os.path.isdir(gv.figdir):
        os.makedirs(gv.figdir)
    
    # if gv.IF_SAVE:
    plt.savefig(gv.figdir + '/' + figname +'.svg',format='svg', dpi=300) 
    print('save fig to', gv.figdir)
    print('figname', figname)
        
def save_dat(array, filename):
    if not os.path.isdir(gv.filedir):
        os.makedirs(gv.filedir)
        
    with open(gv.filedir + '/' + filename + '.pkl','wb') as f:
        pickle.dump(array, f) 
        print('saved to', gv.filedir + '/' + filename + '.pkl' )

def open_dat(filename):
    if not os.path.isdir(gv.filedir):
        os.makedirs(gv.filedir)
        
    with open(gv.filedir + '/' + filename + '.pkl','rb') as f:
        print('opening', gv.filedir + '/' + filename + '.pkl' )
        return pickle.load(f) 

def add_vlines():
    # plt.axvline(gv.t_STIM[0], c='k', ls='--') # sample onset
    # plt.axvline(gv.t_STIM[1], c='k', ls='--') # sample onset
    
    plt.axvspan(gv.t_STIM[0], gv.t_STIM[1], alpha=shade_alpha, color='b') 
    plt.axvspan(gv.t_DIST[0], gv.t_DIST[1], alpha=shade_alpha, color='b') 
    plt.axvspan(gv.t_MD[1], gv.t_LD[0], alpha=shade_alpha, color='g') 
    plt.axvspan(gv.t_TEST[0], gv.t_TEST[1], alpha=shade_alpha, color='b') 
    
    # plt.axvspan(gv.t_ED[0], gv.t_ED[1], alpha=shade_alpha, color='#ff00ff')
    # plt.axvspan(gv.t_MD[0], gv.t_MD[1], alpha=shade_alpha, color='#ffff00')
    # plt.axvspan(gv.t_LD[0], gv.t_LD[1], alpha=shade_alpha, color='#00ffff') 
    
    # plt.axvline(gv.t_MD[1], c='k', ls='--') 
    # plt.axvline(gv.t_LD[0], c='k', ls='--') 
    
    # plt.axvline(gv.t_DIST[0], color='k', ls='--')
    # plt.axvline(gv.t_DIST[1], color='k', ls='--')
    
    # plt.axvline(gv.t_MD[0], c='g', ls='--') #DRT delay
    # plt.axvline(gv.t_MD[1], c='g', ls='--') 
        
    # plt.axvline(gv.t_LD[0], c='b', ls='--')
    # plt.axvline(gv.t_LD[1], c='b', ls='--') # DPA late delay

    # plt.axvline(gv.t_test[0], color='k', ls='--')
    # plt.axvline(gv.t_test[1], color='k', ls='--')
    
    
def add_hlines():
    plt.axhline(gv.t_STIM[0], c='k', ls='-') # sample onset

    plt.axhline(gv.t_ED[0], c='k', ls='--') 
    plt.axhline(gv.t_ED[1], c='k', ls='--') # DPA early delay

    plt.axhline(gv.t_MD[0], c='r', ls='--') #DRT delay
    plt.axhline(gv.t_MD[1], c='r', ls='--') 
        
    plt.axhline(gv.t_LD[0], c='k', ls='--')
    plt.axhline(gv.t_LD[1], c='k', ls='--') # DPA late delay 
