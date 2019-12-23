import sys, os
import numpy as np
from numpy import linalg as LA
from itertools import permutations, combinations
from scipy.spatial.distance import pdist, cdist, squareform
import pandas as pd
import matplotlib
from matplotlib import rc
#matplotlib.rc('text', usetex = True)
from matplotlib.figure import Figure
import matplotlib.pyplot as plt
from pylab import imshow, show, loadtxt, axes
import scipy
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.ticker import MaxNLocator
import mrcfile
import fergusonE

'''
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
Copyright (c) CU, Evan Seitz 2019-2020
Contact: evan.e.seitz@gmail.com
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
'''

pyDir = os.path.dirname(os.path.abspath(__file__)) #python file location
projName = 'p2_noNorm'
m = 400 #number of images

#####################
# generate (if 1) or load (if 0) distances:
if 0:
    stack = mrcfile.open(os.path.join(pyDir, '9_GenSNR_pyRelion/Hsp2D_noNorm.mrcs'))
    states = range(0,m)
    RMSD = np.ndarray(shape=(m,m), dtype=float)
    
    if 1: #manual distance calculation
        p = 2 #Minkowski distance metric: p1=Manhattan, p2=Euclidean, ..., pInf=Chebyshev
        for i in states:
            for j in states:
                #RMSD[i,j] = (np.sum(np.abs((stack.data[i]-stack.data[j]))**2))**(1./p) / stack.data[i].size #kernel
                RMSD[i,j] = (np.sum(np.abs((stack.data[i]-stack.data[j]))**p) / stack.data[i].size)**(1./p) #equivalent of 2D-RMSD
        
    else: #use scipy library for distance metric
        stack2 = np.ndarray(shape=(m,250**2), dtype=float) #if boxSize=250
        for i in states:
            stack2[i] = stack.data[i].flatten()
        
        RMSD = cdist(stack2, stack2, 'euclidean')
        #RMSD = cdist(stack2, stack2, 'sqeuclidean')
        #RMSD = cdist(stack2, stack2, 'minkowski', p=2.)
        #RMSD = cdist(stack2, stack2, 'cityblock')
        #RMSD = cdist(stack2, stack2, 'cosine')
        #RMSD = cdist(stack2, stack2, 'correlation')
        #RMSD = cdist(stack2, stack2, 'chebyshev')
        #RMSD = cdist(stack2, stack2, 'canberra')
        #RMSD = cdist(stack2, stack2, 'braycurtis')
        
    if 0: #save distance matrix for subsequent use
        np.save('Dist_%s.npy' % (projName), RMSD)
    
else:
    if 0: #RMSD from 2D images of 3D structures:
        RMSD = np.load('Dist_%s.npy' % (projName)) #best eps=0.1?
    else: #RMSD from 3D structures:
        RMSD = np.load('Dist_3D_RMSD.npy') #best eps=.0001

if 0:
    imshow(RMSD, cmap='jet', origin='lower')
    plt.title('Distances', fontsize=20)
    plt.colorbar()
    plt.tight_layout()
    plt.show()
    
if 1: #distances of state_01_01 to all others
    plt.scatter(np.linspace(1,400,400), RMSD[0,:])
    plt.show()

#####################
# Ferguson to find optimal epsilon:

if 0:
    logEps = np.arange(-30,30.2,0.2)
    a0 = 1*(np.random.rand(4,1)-.5)
    popt, logSumWij, resnorm = fergusonE.op(RMSD,logEps,a0)
    
    def fun(xx, aa0, aa1, aa2, aa3): #fit tanh()
        F = aa3 + aa2 * np.tanh(aa0 * xx + aa1)
        return F
    
    eps0 = np.exp(-(popt[1] / popt[0]))
    print('Optimal ln(eps):', -(popt[1] / popt[0]))
    
    if 1:
        plt.scatter(logEps, logSumWij, s=1, c='#1f77b4', edgecolor='#1f77b4', zorder=.1)
        plt.plot(logEps, fun(logEps, popt[0], popt[1], popt[2],popt[3]), c='red', linewidth=.5, zorder=.2)
        plt.title('Gaussian Kernel Bandwidth',fontsize=14)
        plt.xlabel(r'$\mathrm{ln \ \epsilon}$', fontsize=16)
        plt.ylabel(r'$\mathrm{ln \ \sum_{i,j} \ A_{i,j}}$', fontsize=18, rotation=90) 
        plt.show()

###########################
# generate optimal kernel:
eps = .001#.05#10000 #.5 #change to eps0 for "optimal" fit found above
A = np.exp(-(RMSD**2 / 2*eps)) #Gaussian kernel
alpha = 1 #currently unassigned (default alpha=1)
    # alpha = 1.0: Laplace-Beltrami operator
    # alpha = 0.5: Fokker-Planck diffusion
    # alpha = 0.0: graph Laplacian normalization

if 0:
    imshow(A, cmap='jet', origin='lower')
    plt.title(r'Gaussian Kernel, $\mathit{\epsilon}$=%s' % eps, fontsize=20)
    plt.colorbar()
    plt.tight_layout()
    show()
    
    rowSums = np.sum(A,axis=1)    
    if 0:
        print('minimum affinity:', np.where(rowSums == np.amin(rowSums)))
        plt.scatter(np.linspace(1,400,400),rowSums)
        plt.xlim(1,400)
        plt.ylim(np.amin(rowSums),np.amax(rowSums))
        plt.show()
        
    if 1: #kernel of state_01_01 to all others
        plt.scatter(np.linspace(1,400,400), A[0,:])
        plt.show()
        
###########################################
# normalized graph laplacian construction:

D = np.ndarray(shape=(m,m), dtype=float) #diagonal matrix
for i in range(0,m):
    for j in range(0,m):
        if i == j:
            D[i,j] = np.sum(A[i], axis=0)
        else:
            D[i,j] = 0    

if 0:
    print(D.diagonal())
    imshow(D, cmap='jet', origin='lower')
    plt.title(r'$\mathit{M}, \mathit{\alpha}$=%s' % alpha, fontsize=20)
    plt.colorbar()
    plt.tight_layout()
    show()
    
D_inv = scipy.linalg.fractional_matrix_power(D, -1)
M = np.matmul(A, D_inv)

if 0:
    print(np.sum(M, axis=0)) #should be all 1's
    imshow(M, cmap='jet', origin='lower')
    plt.title('Markov Transition Matrix', fontsize=20)
    plt.colorbar()
    plt.tight_layout()
    show()
    
    MrowSums = np.sum(M,axis=0)
    
    if 0:
        print('minimum affinity:', np.where(MrowSums == np.amin(MrowSums)))
        plt.scatter(np.linspace(1,400,400),MrowSums)
        plt.xlim(1,400)
        plt.ylim(np.amin(MrowSums),np.amax(MrowSums))
        plt.show()

#####################
# SVD decompisition:
    
def tidyUp(D,EV):
    order = np.argsort(D)[::-1]
    D = np.sort(D)[::-1]
    EV = EV[:,order]
    sqrtD = np.sqrt(D)
    S = np.diag(sqrtD)
    invS = np.diag(1./sqrtD)
    return (D,EV,S,invS)

D,U = np.linalg.eigh(np.matmul(M,M.T))
D,U,S,invS = tidyUp(D,U)
V = np.matmul(M.T,np.matmul(U,invS))
sdiag = np.diag(S)

if 0:
    if 0:
        print('U:', U)
        print('S:', sdiag)
        print('V:', V)
        print(sdiag[0], sdiag[1])
        print(U[0], U[1])
    if 1: #eignevalue spectrum
        x = range(1,len(sdiag[1:])+1)
        plt.scatter(x, sdiag[1:])
        plt.title('Eigenvalue Spectrum, $\mathit{\epsilon}$=%s' % eps)
        plt.xlabel(r'$\mathrm{\Psi}$')
        plt.ylabel(r'$\mathrm{\lambda}$', rotation=0)
        plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
        plt.xlim([0,15])
        #plt.ylim(np.amin(sdiag[1:]), np.amax(sdiag[1:]))
        plt.locator_params(nbins=15)
        plt.axhline(y=0, color='k', alpha=.5, linestyle='--', linewidth=1)
        plt.show()
    
if 1:
    if 1: #2d diffusion map
        plt.scatter(U[:,1]*sdiag[1], U[:,2]*sdiag[2], c=U[:,1]*sdiag[1])
        plt.title(r'2D Embedding, $\mathit{\epsilon}$=%s' % eps)
        plt.xlabel(r'$\psi_1$')
        plt.ylabel(r'$\psi_2$')
        plt.ticklabel_format(style='sci', axis='x', scilimits=(0,0))
        plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
        plt.colorbar()
        plt.xlim(np.amin(U[:,1])*sdiag[1], np.amax(U[:,1])*sdiag[1])
        plt.ylim(np.amin(U[:,2])*sdiag[1], np.amax(U[:,2])*sdiag[1])
        plt.show()

    if 1: #3d diffusion map
        fig = plt.figure()
        ax = fig.gca(projection='3d')
        ax.scatter(U[:,1]*sdiag[1], U[:,2]*sdiag[2], U[:,3]*sdiag[3], c=U[:,1]*sdiag[1], cmap='jet')
        plt.title(r'3D Embedding, $\mathit{\epsilon}$=%s' % eps)
        ax.set_xlabel(r'$\psi_1$')
        ax.set_ylabel(r'$\psi_2$')
        ax.set_zlabel(r'$\psi_3$')
        plt.ticklabel_format(style='sci', axis='x', scilimits=(0,0))
        plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
        plt.ticklabel_format(style='sci', axis='z', scilimits=(0,0))
        plt.show()
        
if 0: #save diffusion map
    np.save('Dist_%s_DM.npy' % (projName), U)
    
if 1: #plot n-dimensional Euclidean distance from any given reference point
    dists = []
    ref = 0 #reference point
    for i in range(0,m):
        dn = 0
        for n in range(1,399): #number of dimensions to consider
            if sdiag[n] > 0: #only count eigenfunctions with non-zero eigenvalues
                dn += (sdiag[n]*U[:,n][ref] - sdiag[n]*U[:,n][i])**2
        dists.append((dn)**(1/2.))
    plt.scatter(np.linspace(1,400,400), dists)
    plt.xlim(-1,m+1)
    plt.ylim(min(dists),max(dists)) 
    plt.show()
    
if 1: #plot geodesic distance between neighboring states
    dists = []
    ref = 0 #reference point
    for i in [1]:
        dn = 0
        for n in range(1,399): #number of dimensions to consider
            if sdiag[n] > 0: #only count eigenfunctions with non-zero eigenvalues
                dn += (sdiag[n]*U[:,n][ref] - sdiag[n]*U[:,n][i])**2
        dists.append((dn)**(1/2.))
    print(dists)
    #plt.scatter(np.linspace(1,400,400), dists)
    #plt.xlim(-1,m+1)
    #plt.ylim(min(dists),max(dists)) 
    #plt.show()
    
if 0: #temporary, for plotting only (presentation); 3D RMSD
    e = [.0001, .001, .01, .1, 1, 10, 100, 1000]
    d = [1.103e-5, .0001046, .00082, .008035, .0928, 1.02, 1.414213557, 1.41421356]
    plt.scatter(np.log(e),d)
    plt.axhline(y=0, color='gray', alpha=.5, linestyle='--', linewidth=1)
    plt.title(r'$n$-dim geodesic distance between neighboring states (0, 1)',fontsize=14)
    plt.xlabel(r'$\mathrm{ln \ \epsilon}$', fontsize=18)
    plt.ylabel('Geodesic distance', fontsize=14, rotation=90) 
    plt.show()

if 1: #temporary, for plotting only (presentation); 2D RMSD
    e = [.001, .01, .1, 1, 10, 100, 1000]
    d = [1.867e-5, .0001889, .002122, .03277, .555, 1.4138, 1]
    plt.scatter(np.log(e),d)
    plt.axhline(y=0, color='gray', alpha=.5, linestyle='--', linewidth=1)
    plt.axvline(x=np.log(10), color='red', alpha=.5, linestyle='-', linewidth=1)
    plt.title(r'$n$-dim geodesic distance between neighboring states (0, 1)',fontsize=14)
    plt.xlabel(r'$\mathrm{ln \ \epsilon}$', fontsize=18)
    plt.ylabel('Geodesic distance', fontsize=14, rotation=90) 
    plt.show()    
        
        
        
    
