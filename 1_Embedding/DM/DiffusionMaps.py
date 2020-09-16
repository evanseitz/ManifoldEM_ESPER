import sys, os, re
import numpy as np
from numpy import linalg as LA
from scipy.spatial.distance import pdist, cdist, squareform
import pandas as pd
import matplotlib
from matplotlib import rc
from matplotlib.figure import Figure
import matplotlib.pyplot as plt
from pylab import imshow, show, loadtxt, axes
import scipy
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.ticker import MaxNLocator
import GaussianBandwidth

# =============================================================================
# Embed distance files for each PD via diffusion maps framework
# =============================================================================
# SETUP: First, make sure the input file path is correct for your dataset...
#   ...via the 'Dist' variable below. You may also want to edit output names...
#   ...such as those that include `tau` (here, set to 5 as default)
# RUNNING: To run a series of PDs at once: first edit `2_DM_Batch.sh`...
#   ...for the total number of PDs requested; e.g., {1...5} for 5 PDs...
#   ...or {1...1} for only the first PD;
#   ...then start batch processing via `sh 2_DM_Batch.sh`
# =============================================================================
# Author:    E. Seitz @ Columbia University - Frank Lab - 2020
# Contact:   evan.e.seitz@gmail.com
# =============================================================================

def op(pyDir, PD):
    dataDir = os.path.join(pyDir, 'Data_Distances')
    outDir = os.path.join(pyDir, 'Data_Manifolds')
    if not os.path.exists(outDir):
        os.mkdir(outDir)
    Dist = np.load(os.path.join(dataDir, 'PD%s_tau5_SNR_dist.npy' % PD))

    # =========================================================================
    # Distances matrix analysis
    # =========================================================================
    m = np.shape(Dist)[0]
    if 0: #if smaller subset of states wanted for investigation
        m = 20
        Dist = Dist[0:m, 0:m] #for CM2 only (if using example dataset)
        #Dist = Dist[::20, ::20] #for CM1 only (if using example dataset)
    
    if 0: #plot Distance matrix
        imshow(Dist, cmap='jet', origin='lower', interpolation='nearest')
        plt.title('Distances', fontsize=20)
        plt.xlabel('State')
        plt.ylabel('State')
        plt.colorbar()
        plt.tight_layout()
        plt.show()
        
    if 0: #plot distances of state_01_01 to all others
        plt.scatter(np.linspace(1,m,m), Dist[0,:])
        plt.show()
    
    # =========================================================================
    # Method to estimate optimal epsilon; see Ferguson SI (2010)
    # =========================================================================
    logEps = np.arange(-20,20.2,0.2)
    a0 = 1*(np.random.rand(4,1)-.5)
    popt, logSumWij, resnorm, R2 = GaussianBandwidth.op(Dist,logEps,a0)
    
    def fun(xx, aa0, aa1, aa2, aa3): #fit tanh()
        F = aa3 + aa2 * np.tanh(aa0 * xx + aa1)
        return F
    
    plt.scatter(logEps, logSumWij, s=1, c='C0', edgecolor='#1f77b4', zorder=.1, label='data')
    plt.plot(logEps, fun(logEps, popt[0], popt[1], popt[2], popt[3]), c='C1', linewidth=.5, zorder=.2, label='tanh(x)')
    plt.plot(logEps, popt[0]*popt[2]*(logEps+popt[1]/popt[0]) + popt[3], c='C2', linewidth=.5, zorder=.3, label='slope')
    plt.axvline(x=-(popt[1] / popt[0]), c='C3', linewidth=.5, label='epsilon')
    
    plt.legend(loc='best')
    plt.xlabel(r'$\mathrm{ln \ \epsilon}$', fontsize=16)
    plt.ylabel(r'$\mathrm{ln \ \sum_{i,j} \ A_{i,j}}$', fontsize=18, rotation=90)
    plt.ylim(np.amin(fun(logEps, popt[0], popt[1], popt[2], popt[3]))-1, np.amax(fun(logEps, popt[0], popt[1], popt[2], popt[3]))+1)

    slope = popt[0]*popt[2] #slope of tanh
    eps = -(popt[1] / popt[0]) #x-axis line through center of tanh
    print('Coefficient of Determination: %s' % R2)
    print('Slope: %s' % slope) 
    print('ln(epsilon): %s; epsilon: %s' % (eps, np.exp(eps)))
    if 0: #plot Gaussian Bandwidth
        plt.show()
    if 0: #save Gaussian Bandwidth plot to file
        np.save('GaussianBandwidth_PD%s.npy' % PD, [logEps, logSumWij])
    
    # =========================================================================
    # Generate optimal Gaussian kernel for Similarity Matrix (A)
    # =========================================================================
    A = np.exp(-1.*((Dist**2.) / (2.*eps))) #similarity matrix
    
    if 0: #plot similarity matrix A
        imshow(A, cmap='jet', origin='lower')
        plt.title(r'Gaussian Kernel, $\mathit{\epsilon}$=%s' % eps, fontsize=20)
        plt.colorbar()
        plt.tight_layout()
        plt.show()
        if 0:
            rowSums = np.sum(A, axis=1)
            print('minimum affinity:', np.where(rowSums == np.amin(rowSums)))
            plt.scatter(np.linspace(1,m,m), rowSums)
            plt.xlim(1, m)
            plt.ylim(np.amin(rowSums), np.amax(rowSums))
            plt.show()
        if 0: #similarity of state_01_01 to all others
            plt.scatter(np.linspace(1,m,m), A[0,:])
            plt.show()
    
    # =========================================================================
    # Construction of Markov Transition Matrix (M):        
    # =========================================================================
    D = np.ndarray(shape=(m,m), dtype=float) #diagonal matrix D
    for i in range(0,m):
        for j in range(0,m):
            if i == j:
                D[i,j] = np.sum(A[i], axis=0)
            else:
                D[i,j] = 0
                
    Dinv = np.linalg.inv(D)
    M = np.matmul(A, Dinv) #Markov matrix via normalization of A to right stochastic form
    if 0: #check M is right (row) stochastic
        print(np.sum(M, axis=0)) #should be all 1's
        
    # =========================================================================
    # cite: 'Systematic determination of order parameters for chain dynamics...
    #       ...using diffusion maps'; PNAS, Ferguson (2010), SI
    # =========================================================================
    Dhalf = scipy.linalg.sqrtm(D)
    Dinv_half = scipy.linalg.sqrtm(Dinv)
    if 0: #sanity-check for D^-1/2 
        print(Dinv_half.dot(D).dot(Dinv_half)) #should be Identity matrix
    Ms = Dinv_half.dot(M).dot(Dhalf) #M is adjoint to symmetric matrix Ms
        
    def is_symmetric(A):
        # for positive semidefinite, need to additionally check that all...
        # ...eigenvalues are (significantly) non-negative
        if np.allclose(A, A.T, rtol=1e-08, atol=1e-08):
            print('Matrix is symmetric')    
        else:
            print('Matrix is not symmetric')
    is_symmetric(Ms) #check if matrix is symmetric
    
    # =========================================================================
    # Eigendecomposition
    # =========================================================================
    U, sdiag, vh = np.linalg.svd(Ms) #vh = U.T
    sdiag = sdiag**(2.) #eigenvalues given by s**2
    
    np.save(os.path.join(outDir, 'PD%s_tau5_val.npy' % PD), sdiag)
    np.save(os.path.join(outDir, 'PD%s_tau5_vec.npy' % PD), U)
        
    # =========================================================================
    # Analysis of diffusion map
    # =========================================================================
    if 0: #orthogonality check
        print(np.linalg.det(U)) #should be +- 1
        print(np.sum(U[:,0]*U[:,0])) #should be 1
        print(np.sum(U[:,1]*U[:,1])) #should be 1
        print(np.sum(U[:,1]*U[:,2])) #should be ~0
        print(np.sum(U[:,1]*U[:,3])) #should be ~0
        print(np.sum(U[:,2]*U[:,3])) #should be ~0
    
    if 0: #plot eignevalue spectrum
        x = range(1,len(sdiag[1:])+1)
        plt.scatter(x, sdiag[1:])
        plt.title('Eigenvalue Spectrum, $\mathit{\epsilon}$=%s' % eps)
        plt.xlabel(r'$\mathrm{\Psi}$')
        plt.ylabel(r'$\mathrm{\lambda}$', rotation=0)
        plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
        plt.xlim([0,15])
        plt.ylim(-sdiag[1]/8., sdiag[1]+sdiag[1]/8.)
        plt.locator_params(nbins=15)
        plt.axhline(y=0, color='k', alpha=.5, linestyle='--', linewidth=1)
        plt.show()
    
    if 0: #2d diffusion map; sets of higher-order eigenfunction combinations     
        enum = np.arange(1,m+1)
        s = 20
        lw = .5
        cmap = 'nipy_spectral' #'gist_rainbow' 
        fig = plt.figure()
        dimRows = 6
        dimCols = 9
        idx = 1
        for v1 in range(1,dimRows+1):
            for v2 in range(v1+1, v1+dimCols+1):
                plt.subplot(dimRows, dimCols, idx)
                plt.scatter(U[:,v1], U[:,v2], c=enum, cmap=cmap, s=s, linewidths=lw, edgecolor='k') #gist_rainbow, nipy_spectral
                #plt.plot(U[:,v1], U[:,v2], zorder=-1, color='black', alpha=.25)
                plt.xlabel(r'$\Psi_{%s}$' % v1, fontsize=12, labelpad=5)
                plt.ylabel(r'$\Psi_{%s}$' % v2, fontsize=12, labelpad=2.5)
                plt.ticklabel_format(style='sci', axis='x', scilimits=(0,0))
                plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0)) 
                plt.rc('font', size=6)
                if 1:
                    frame = plt.gca()
                    frame.axes.xaxis.set_ticklabels([])
                    frame.axes.yaxis.set_ticklabels([])
                    plt.gca().set_xticks([])
                    plt.xticks([])
                    plt.gca().set_yticks([])
                    plt.yticks([])
                else:
                    plt.tick_params(axis="x", labelsize=6)
                    plt.tick_params(axis="y", labelsize=6) 
    
                plt.xlim(np.amin(U[:,v1])*1.1, np.amax(U[:,v1])*1.1)
                plt.ylim(np.amin(U[:,v2])*1.1, np.amax(U[:,v2])*1.1)
                idx += 1 
        plt.tight_layout()
        plt.subplots_adjust(left=0.02, right=0.99, bottom=0.05, top=0.99, wspace=0.26, hspace=0.23)
        plt.show()
        
if __name__ == '__main__':
    path1 = os.path.splitext(sys.argv[0])[0]
    path2, tail = os.path.split(path1)
    op(path2, sys.argv[1])
        
