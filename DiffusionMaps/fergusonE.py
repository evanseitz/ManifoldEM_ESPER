import numpy as np, logging
from scipy.optimize import curve_fit
from scipy.optimize import OptimizeWarning
import warnings
warnings.simplefilter(action='ignore', category=OptimizeWarning)
import time
"""
%--------------------------------------------------------------------------
% function ferguson(D,s)
% D: Distance matrix
% logEps: Range of values to try
% Adapted from Chuck, 2011
%-------------------------
% example:
% logEps = [-10:0.2:10];
% ferguson(sqrt(yVal),logEps,1*(rand(4,1)-.5))
%--------------------------------------------------------------------------
Copyright (c) UWM, Ali Dashti 2016 (original matlab version)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
Copyright (c) Columbia University Hstau Liao 2018 (python version)    
%
"""
def fun(xx,aa0,aa1,aa2,aa3):
    F = aa3 + aa2*np.tanh(aa0*xx+aa1)
    #aa3 vertical location (large=raise)
    #aa2 vertical spread (large=shrink)
    #aa0 curvature (large=less)
    return F

def find_thres(logEps,D2):
    eps = np.exp(logEps)
    d = 1. / (2. * np.max(eps)) * D2
    sg = np.sort(d)
    ss = np.sum(np.exp(-sg))
    thr = max(-np.log(0.01*ss/len(D2)),10)  # taking 1% of the average, or 10
    return thr

def op(D,logEps,a0):
    # % Range of values to try
    logSumWij = np.zeros(len(logEps)) #% Initialize
    D2 = D*D
    thr = find_thres(logEps,D2)
    #print "thr=",thr
    for k in xrange(len(logEps)):
        eps = np.exp(logEps[k])
        d = 1. / (2. * eps) * D2
        d = -d[d < thr]
        Wij = np.exp(d)   # % See Coifman 2008
        # A_ij = csr_matrix(A_ij)
        logSumWij[k] = np.log(sum(Wij))

    # % curve fitting of a tanh()
    resnorm = np.inf
    cc = 0
    while (resnorm>100):
        cc += 1
        a, pcov = curve_fit(fun, logEps, logSumWij, p0=a0)

        resnorm = sum(np.sqrt(np.fabs(np.diag(pcov))))

        a0 = 1 * (np.random.rand(4, 1) - .5)

    return (a, logSumWij,resnorm)
