import os
import numpy as np
from scipy.optimize import curve_fit
import matplotlib
from matplotlib import rc
from matplotlib.figure import Figure
import matplotlib.pyplot as plt
from matplotlib import cm


def op(U, groundTruth, CM_idx, v1, v2, figIdx, outDir):
    # =========================================================================
    # Absolute-value fit (robust alternative to conics):
    # =========================================================================
    xlist = np.arange(0,np.pi,.01) #.01: ~300 points; .001: ~3000 points
    def AbsoluteFit(x, a, b, c): #absolute-value function
        return a*np.abs(x-b) + c
    guess_a = 1.
    guess_b = np.pi/2.
    guess_c = 1.
    p1 = [guess_a, guess_b, guess_c]
    coeffs = curve_fit(AbsoluteFit, U[:,0], U[:,1], p1, maxfev=10000)
    Pfit = AbsoluteFit(U[:,0], *coeffs[0])

    # Calculate coefficient of determination:
    SE_Gfit = 0
    for i in range(len(Pfit)):
        SE_Gfit += (U[:,1][i] - Pfit[i])**2
    ybar = np.mean(U[:, 1])
    SE_ybar = 0
    for i in range(len(Pfit)):
        SE_ybar += (U[:,1][i] - ybar)**2  
    R2 = 1 - (SE_Gfit / SE_ybar)
    
    # =========================================================================
    # Collect parameters from absolute value fit:     
    # =========================================================================
    if coeffs[0][0] > 0: #upward-facing V
        face = 'up'
        absVertex = np.amin(AbsoluteFit(xlist, *coeffs[0]))
        absIdx = np.argmin(AbsoluteFit(xlist, *coeffs[0]))
    else: #downward-facing V
        face = 'down'
        absVertex = np.amax(AbsoluteFit(xlist, *coeffs[0]))
        absIdx = np.argmax(AbsoluteFit(xlist, *coeffs[0]))
    abs1 = xlist[absIdx]
    abs2 = AbsoluteFit(xlist, *coeffs[0])[absIdx]

    if 1: #plot absolute-value fit
        if groundTruth is True:
            color=iter(cm.tab20(np.linspace(1, 0, np.shape(CM_idx)[0])))
            for b in range(np.shape(CM_idx)[0]):
                c=next(color)
                plt.scatter(U[:,0][CM_idx[b]], U[:,1][CM_idx[b]], color=c, s=15, edgecolor='k', linewidths=.1, zorder=1)
            plt.plot(xlist, AbsoluteFit(xlist, *coeffs[0]), c='k', linewidth=2, zorder=2)
            plt.scatter(abs1, abs2, c='k', zorder=3) #vertex from absolute value fit 
        else:
            plt.scatter(U[:,0], U[:,1], s=20, c='white', linewidth=.5, edgecolor='k')
            plt.plot(xlist, AbsoluteFit(xlist, *coeffs[0]), c='r', linewidth=2, zorder=2)
            plt.scatter(abs1, abs2, c='r', zorder=3) #vertex from absolute value fit 

        plt.xlabel(r'$\Phi_{%s}$' % (int(v1)+1), labelpad=5, fontsize=14)
        plt.ylabel(r'$\Phi_{%s}$' % (int(v2)+1), labelpad=7.5, fontsize=14)
        plt.title(r'R$^{2}$=%.2f' % R2, fontsize=12)
        plt.axis('scaled')
        fig = plt.gcf()
        fig.savefig(os.path.join(outDir,'Fig_%s_AbsVal.png' % figIdx), dpi=200)
        #plt.show()
        plt.clf()
        
    # =========================================================================
    # Use fit parameters to delimit points in manifold:  
    # =========================================================================  
    split_LHS_idx = []
    split_RHS_idx = []
    for i in range(len(U[:,0])):
        if U[:,0][i] <= abs1:
            split_LHS_idx.append(i)
        else:
            split_RHS_idx.append(i)        

    fit_LHS_x = []
    fit_LHS_y = []
    fit_RHS_x = []
    fit_RHS_y = []
    for i in xlist:
        if i <= abs1:
            fit_LHS_x.append(i)
            fit_LHS_y.append(AbsoluteFit(i, *coeffs[0]))
        else:
            fit_RHS_x.append(i)
            fit_RHS_y.append(AbsoluteFit(i, *coeffs[0]))
    
    return np.array([fit_LHS_x, fit_LHS_y, fit_RHS_x, fit_RHS_y, split_LHS_idx, split_RHS_idx, face], dtype='object')