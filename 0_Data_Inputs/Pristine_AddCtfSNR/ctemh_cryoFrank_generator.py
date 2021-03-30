import os
import matplotlib
from matplotlib.figure import Figure
from matplotlib.widgets import Slider, Button
import mpl_toolkits.axes_grid1
import matplotlib.path as pltPath
import matplotlib.image as mpimg
from matplotlib import rc
import matplotlib.pyplot as plt
import numpy as np
from pylab import plot, loadtxt, imshow, show, xlabel, ylabel
import sys
import numpy as np
import math
from scipy.fftpack import ifftshift
from scipy.fftpack import fft2
from scipy.fftpack import ifft2

"""
function y = ctemh_cryoFrank(k,params)
% from Kirkland, adapted for cryo (EMAN1) by P. Schwander
% Version V 1.1
% Copyright (c) UWM, Peter Schwander 2010 MATLAB version
% '''
% Copyright (c) Columbia University Hstau Liao 2018 (python version)   
% Modified by Evan Seitz, Columbia University 2021 (python version)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Here, the damping envelope is characterized by a single parameter B 
% see J. Frank
% params(1)   Cs in mm
% params(2)   df in Angstrom, a positive value is underfocus
% params(3)   Electron energy in keV
% params(4)   B in A-2
% params(5)   Amplitude contrast as fraction in [0,1]
% Note: we assume |k| = s
"""

def create_grid(N,N2):
    if N%2 == 1: 
        a = np.arange(-(N-1)/2,(N-1)/2+1)
    else:        
        a = np.arange(-N2,N/2)
    X, Y = np.meshgrid(a,a)
    Q = (1./(N/2.))*np.sqrt(X**2+Y**2)
    return Q

def op(k,params):
   Cs = params[0]*1.0e7
   df = params[1]
   kev = params[2]
   B = params[3]
   ampc = params[4]
   mo = 511.0
   hc = 12.3986
   wav = (2*mo)+kev
   wav = hc/np.sqrt(wav*kev)
   w1 = np.pi*Cs*wav*wav*wav
   w2 = np.pi*wav*df
   k2 = k*k
   sigm = B/math.sqrt(2*math.log(2)) # B is Gaussian Env. Halfwidth
   wi = np.exp(-k2/(2*sigm**2))
   wr = (0.5*w1*k2-w2)*k2 # gam = (pi/2)Cs lam^3 k^4 - pi lam df k^2 
   y = (np.sin(wr)-ampc*np.cos(wr))*wi
   return y

def gen_ctf(N, px, Cs, df, volt, B, ampc): #N:box size; pix_size: pixel size [A]; s
    # ========================================================================
    # Inputs:
    # ========================================================================
        # N: box size [A]
        # px: pixel size [A]
        # Cs: spherical aberattion [mm]
        # df: defocus [A]
        # volt: voltage [keV]
        # B: Gaussian envelope halfwidth
        # ampc: amplitude contrast [fraction]
    # ========================================================================
    # Usage: e.g., [320, 1, 2.7, 11962, 300, np.inf, 0.1]
    # ========================================================================
    N2 = N/2.
    Q = create_grid(N,N2)
    k = Q / (2 * px)
    tmp = op(k, [Cs, df, volt, B, ampc])
    ctf = ifftshift(tmp)
    return ctf