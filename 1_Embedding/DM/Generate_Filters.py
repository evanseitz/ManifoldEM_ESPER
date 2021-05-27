import numpy as np
import math
from scipy.fftpack import ifftshift

# =============================================================================
# Function y = ctemh_cryoFrank(k,params)
# From Kirkland, adapted for cryo (EMAN1) by P. Schwander
# Version V 1.1
# Copyright (c) UWM, Peter Schwander 2010 MATLAB version
# History:   See ManifoldEM Matlab repository for syntax parallels. As well, a similar...
#            ...workflow will be publically released via the ManifoldEM Python suite...
#            ...(estimated 2021) with that code translated from Matlab to Python...
#            ...by H. Liao (CU, 2019) and modified therein by E. Seitz (CU, 2020).
# =============================================================================
# NOTES:
# Here, the damping envelope is characterized by a single parameter B 
# see J. Frank
# params(1)   Cs in mm
# params(2)   df in Angstrom, a positive value is underfocus
# params(3)   Electron energy in keV
# params(4)   B in A-2
# We assume |k| = s
# =============================================================================

def create_grid(N, N2):
    if N%2 == 1: 
        a = np.arange(-(N-1)/2,(N-1)/2+1)
    else:        
        a = np.arange(-N2,N/2)
    X, Y = np.meshgrid(a,a)
    Q = (1./(N/2.))*np.sqrt(X**2+Y**2)
    return Q

def create_filter(filter_type, NN, Qc, Q):
    if filter_type == 'Gauss':
        G = np.exp(-(np.log(2)/2.)*(Q/Qc)**2)
    elif filter_type == 'Butter':
        G = np.sqrt(1./(1+(Q/Qc)**(2*NN))) 
    return G

def op(k, params):
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

def gen_ctf(N, px, Cs, df, volt, B, ampc):
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
    CTF = ifftshift(tmp)
    
    # Additionally, create Butterworth filter for images:
    G = create_filter('Butter', 8, 0.5, Q)
    G = ifftshift(G)
    
    return np.array([CTF, G], dtype='object')
