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

# ====================================================================================
# Diffusion Maps (run via 'python DM_SyntheticContinuum.py') 
# Author:    E. Seitz @ Columbia University - Frank Lab - 2019-2020
# Contact:   evan.e.seitz@gmail.com
# ====================================================================================

pyDir = os.path.dirname(os.path.abspath(__file__)) #python file location
dataDir = os.path.join(pyDir, 'GenDists')

if 1: #Times font for all figures
    rc('text', usetex=True)
    rc('font', family='serif')
    
# =============================================================================
# Load in previously generated distance matrix (choose 1)
# =============================================================================
    
### From atomic-coordinates (PDBs):
#Dist = np.load(os.path.join(dataDir,'Dist_SS2_3dRMSD.npy')) #distances from PDB files (2 degrees of freedom)
#Dist = np.load(os.path.join(dataDir, 'Dist_SS3_3dRMSD_small.npy')) #distances from PDB files (3 degrees of freedom)
#Dist = np.load(os.path.join(dataDir, 'Dist_SS3_3dRMSD_large.npy')) #distances from PDB files (3 degrees of freedom)
        
### From 3D Coulomb potential maps (MRCs); 250**3 error correction due to previous output:   
Dist = np.load(os.path.join(dataDir, 'Dist_SS2_Volumes.npy'))*(250**3) #distances from MRC files
#Dist = np.load(os.path.join(dataDir, 'Dist_SS3_Volumes_small.npy'))*(250**3) #distances from MRC files
#Dist = np.load(os.path.join(dataDir, 'Dist_SS3_Volumes_large.npy'))*(250**3) #distances from MRC files

### From 2D projections of 3D Coulomb potential maps (PDs):
#Dist = np.load(os.path.join('Dist_SS2_PD0.npy')) #distances from projections of MRC files
# =============================================================================
    
m = np.shape(Dist)[0] #number of states to consider from distance matrix; e.g., m=20 for 1D motion from SS2

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

# =============================================================================
# Method to estimate optimal epsilon; Ferguson (2010)
# =============================================================================
if 1:
    logEps0 = np.arange(-50,50.2,0.1)
    logSumA0 = np.zeros(len(logEps0))
    for k in xrange(len(logEps0)):
        eps0 = np.exp(logEps0[k]) #temporary kernel bandwidth
        A0 = np.exp(-1.*(Dist**2. / 2.*eps0)) #temporary similarity matrix        
        logSumA0[k] = np.log(np.sum(A0))
        
    if 1: #plot of Ferguson method
        plt.scatter(logEps0, logSumA0, s=1, c='#1f77b4', edgecolor='#1f77b4', zorder=.1)
        #plt.axvline(np.log(1e-18), c='red')
        #plt.axvline(np.log(1e-7), c='red')
        plt.xlabel(r'$\mathrm{ln \ \epsilon}$', fontsize=16)
        plt.ylabel(r'$\mathrm{ln \ \sum_{i,j} \ A_{i,j}}$', fontsize=18, rotation=90)
        plt.show()

    if 0: #save Ferguson plot to file
        np.save('Ferg_SS2_MRC.npy', [logEps0, logSumA0])

# =============================================================================
# Choose optimal Gaussian kernel bandwidth (choose 1)
# =============================================================================
        
#eps = 1e-4 #best for 'Dist_3D_RMSD.npy'; optimal range: [1e-13, 1e-4]
eps = 1e-7#1e-7 #best for 'Dist_MRCs.npy'; optimal range: [1e-18, 1e-7]
#eps = .01 #best for 'Dist_PD_0.npy'; optimal range: [1e-11, 1e1]

# =============================================================================
# Generate optimal Gaussian kernel for Similarity Matrix (A)
# =============================================================================

A = np.exp(-1.*(Dist**2. / 2.*eps)) #similarity matrix
'''
# alpha = 1.0: Laplace-Beltrami operator (default)
# alpha = 0.5: Fokker-Planck diffusion
# alpha = 0.0: graph Laplacian normalization
'''
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

# =============================================================================
# Construction of Markov Transition Matrix (M):        
# =============================================================================
        
D = np.ndarray(shape=(m,m), dtype=float) #diagonal matrix D
for i in range(0,m):
    for j in range(0,m):
        if i == j:
            D[i,j] = np.sum(A[i], axis=0)
        else:
            D[i,j] = 0
    
Dinv = scipy.linalg.fractional_matrix_power(D, -1)
M = np.matmul(A, Dinv) #Markov transition matrix via normalization of A to be row stochastic
if 0: #check Markov transition matrix is row stochastic
    print(np.sum(M, axis=0)) #should be all 1's

if 0: #constructing the density invariant Graph Laplacian:
    # ================================================================================
    # cite: Graph Laplacian Tomography From Unknown Random Projections; Coifman (2008)
    # ================================================================================
    L = M - np.identity(len(M)) #negatively defined normalized graph Laplacian
    W_Dinv = np.matmul(Dinv, A)
    Wtilda = np.matmul(W_Dinv, Dinv)
    Dtilda = np.ndarray(shape=(m,m), dtype=float) #diagonal matrix
    for i in range(0,m):
        for j in range(0,m):
            if i == j:
                Dtilda[i,j] = np.sum(Wtilda[i], axis=0)
            else:
                Dtilda[i,j] = 0
    Dtildainv = scipy.linalg.fractional_matrix_power(Dtilda, -1)
    Mtilda = np.matmul(Wtilda, Dtildainv)
    Ltilda = Mtilda - np.identity(len(Wtilda)) #density invariant graph Laplacian

# ==============================================================================
# cite: Systematic Determination... for Chain Dynamics Using DM; Ferguson (2010)
# ==============================================================================
    
Dinv_half = scipy.linalg.sqrtm(Dinv)
Dhalf = scipy.linalg.sqrtm(D)
Ms0 = np.matmul(Dinv_half, M)
Ms = np.matmul(Ms0, Dhalf) #note that M is adjoint to symmetric matrix Ms   
    
def check_symmetric(a, rtol=1e-08, atol=1e-08, equal_nan=True):
    return np.allclose(a, a.T, rtol=rtol, atol=atol)
print('Hermitian:', check_symmetric(Ms))

# =============================================================================
# Eigendecomposition
# =============================================================================

### choose 1 of the following 2 eigendecomposition methods:
if 0: #np.linalg.eigh() version
    def tidyUp(D,EV):
        order = np.argsort(D)[::-1]
        D = np.sort(D)[::-1]
        EV = EV[:,order]
        sqrtD = np.sqrt(D)
        S = np.diag(sqrtD)
        invS = np.diag(1./sqrtD)
        return (D,EV,S,invS)
    d,U = np.linalg.eigh(np.matmul(Ms,Ms.T))
    d,U,S,invS = tidyUp(d,U)
    V = np.matmul(Ms.T,np.matmul(U,invS))
    sdiag = np.diag(S)
    sdiag = sdiag**(2.) #eigenvalues given by s**2 
else: #np.linalg.svd() version; computationally same result as above
    U, sdiag, vh = np.linalg.svd(Ms) #vh = U.T
    sdiag = sdiag**(2.) #eigenvalues given by s**2

if 0:
    np.save('Eig_SS2_MRC.npy', sdiag)
    np.save('DM_SS2_MRC.npy', U)
    
if 0: #orthogonality check
    print(np.sum(U[:,0]*U[:,0])) #should be 1
    print(np.sum(U[:,1]*U[:,1])) #should be 1
    print(np.sum(U[:,1]*U[:,2])) #should be ~0
    print(np.sum(U[:,1]*U[:,3])) #should be ~0
    print(np.sum(U[:,2]*U[:,3])) #should be ~0

if 1: #eignevalue spectrum
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
    
# =============================================================================
# Analysis of diffusion map
# =============================================================================
    
enum = np.arange(1,m+1)

if 1: #plot 2d diffusion map
    v1 = 1 #eigenfunction to plot on first axis
    v2 = 2 #eigenfunction to plot on second axis
    plt.scatter(U[:,v1]*sdiag[v1], U[:,v2]*sdiag[v2], c=enum, cmap='gist_rainbow')
    enum = np.arange(1,m+1)
    if 1: #annotate points in plot with indices of each state
        for i, txt in enumerate(enum):
            plt.annotate(txt, (U[i,v1]*sdiag[v1], U[i,v2]*sdiag[v2]), fontsize=12, zorder=1, color='gray')
    plt.title(r'2D Embedding, $\mathit{\epsilon}$=%s' % eps)
    plt.xlabel(r'$\psi_1$')
    plt.ylabel(r'$\psi_2$')
    plt.ticklabel_format(style='sci', axis='x', scilimits=(0,0))
    plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
    plt.colorbar()
    plt.xlim(np.amin(U[:,v1])*sdiag[v1]*1.1, np.amax(U[:,v1])*sdiag[v1]*1.1)
    plt.ylim(np.amin(U[:,v2])*sdiag[v2]*1.1, np.amax(U[:,v2])*sdiag[v2]*1.1)
    plt.show()
    
if 0: #2d diffusion map; several higher-order eigenfunction combinations
    if 1:
        fig = plt.figure()
        dim = 9
        idx = 0
        for v1 in range(1,dim+2):
            for v2 in range(1,dim+2):
                if v2 > v1:
                    if v1 > 1:
                        plt.subplot(dim, dim, idx-v1*2+2)
                    else:
                        plt.subplot(dim, dim, idx)
    
                    plt.scatter(U[:,v1]*sdiag[v1], U[:,v2]*sdiag[v2], c=enum, cmap='gist_rainbow')
                    #plt.plot(U[:,v1]*sdiag[v1], U[:,v2]*sdiag[v2], zorder=-1, color='black', alpha=.25)
                    plt.xlabel(r'$\Psi_%s$' % v1, fontsize=14)
                    plt.ylabel(r'$\Psi_%s$' % v2, fontsize=14, labelpad=-1)
                    plt.ticklabel_format(style='sci', axis='x', scilimits=(0,0))
                    plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0)) 
                    plt.rc('font', size=6)
                    plt.tick_params(axis="x", labelsize=6)
                    plt.tick_params(axis="y", labelsize=6)
                    plt.xlim(np.amin(U[:,v1])*sdiag[v1]*1.1, np.amax(U[:,v1])*sdiag[v1]*1.1)
                    plt.ylim(np.amin(U[:,v2])*sdiag[v2]*1.1, np.amax(U[:,v2])*sdiag[v2]*1.1)
                idx += 1 
        plt.tight_layout()
        plt.show()
    
    if 0:
        v1 = 1
        s=35
        
        plt.subplot(2, 3, 1)
        plt.scatter(U[:,v1]*sdiag[v1], U[:,0]*sdiag[0], c=enum, cmap='gist_rainbow', s=s)
        plt.plot(U[:,v1]*sdiag[v1], U[:,0]*sdiag[0], zorder=-1, color='black', alpha=.25)
        plt.xlabel(r'$\lambda_%s\Psi_%s$' % (v1,v1), fontsize=20, labelpad=10)
        plt.ylabel(r'$\lambda_0\Psi_0$', fontsize=20, labelpad=-1)
        plt.ticklabel_format(style='sci', axis='x', scilimits=(0,0))
        plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
        plt.xlim(np.amin(U[:,v1])*sdiag[v1]*1.1, np.amax(U[:,v1])*sdiag[v1]*1.1)
        plt.ylim(np.amin(U[:,0])*sdiag[0]-.00001, np.amax(U[:,0])*sdiag[0]+.00001)
        
        plt.subplot(2, 3, 2)
        plt.scatter(U[:,v1]*sdiag[v1], U[:,1]*sdiag[1], c=enum, cmap='gist_rainbow', s=s)
        plt.plot(U[:,v1]*sdiag[v1], U[:,1]*sdiag[1], zorder=-1, color='black', alpha=.25)
        plt.xlabel(r'$\lambda_%s\Psi_%s$' % (v1,v1), fontsize=20, labelpad=10)
        plt.ylabel(r'$\lambda_1\Psi_1$', fontsize=20, labelpad=-1)
        plt.ticklabel_format(style='sci', axis='x', scilimits=(0,0))
        plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
        plt.xlim(np.amin(U[:,v1])*sdiag[v1]*1.1, np.amax(U[:,v1])*sdiag[v1]*1.1)
        plt.ylim(np.amin(U[:,1])*sdiag[1]*1.1, np.amax(U[:,1])*sdiag[1]*1.1)
    
        plt.subplot(2, 3, 3)
        plt.scatter(U[:,v1]*sdiag[v1], U[:,2]*sdiag[2], c=enum, cmap='gist_rainbow', s=s)
        plt.plot(U[:,v1]*sdiag[v1], U[:,2]*sdiag[2], zorder=-1, color='black', alpha=.25)
        plt.xlabel(r'$\lambda_%s\Psi_%s$' % (v1,v1), fontsize=20, labelpad=10)
        plt.ylabel(r'$\lambda_2\Psi_2$', fontsize=20, labelpad=-1)
        plt.ticklabel_format(style='sci', axis='x', scilimits=(0,0))
        plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
        plt.xlim(np.amin(U[:,v1])*sdiag[v1]*1.1, np.amax(U[:,v1])*sdiag[v1]*1.1)
        plt.ylim(np.amin(U[:,2])*sdiag[2]*1.1, np.amax(U[:,2])*sdiag[2]*1.1)
    
        plt.subplot(2, 3, 4)
        plt.scatter(U[:,v1]*sdiag[v1], U[:,3]*sdiag[3], c=enum, cmap='gist_rainbow', s=s)
        plt.plot(U[:,v1]*sdiag[v1], U[:,3]*sdiag[3], zorder=-1, color='black', alpha=.25)
        plt.xlabel(r'$\lambda_%s\Psi_%s$' % (v1,v1), fontsize=20, labelpad=10)
        plt.ylabel(r'$\lambda_3\Psi_3$', fontsize=20, labelpad=-1)
        plt.ticklabel_format(style='sci', axis='x', scilimits=(0,0))
        plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
        plt.xlim(np.amin(U[:,v1])*sdiag[v1]*1.1, np.amax(U[:,v1])*sdiag[v1]*1.1)
        plt.ylim(np.amin(U[:,3])*sdiag[3]*1.1, np.amax(U[:,3])*sdiag[3]*1.1) 
        
        plt.subplot(2, 3, 5)
        plt.scatter(U[:,v1]*sdiag[v1], U[:,4]*sdiag[4], c=enum, cmap='gist_rainbow', s=s)
        plt.plot(U[:,v1]*sdiag[v1], U[:,4]*sdiag[4], zorder=-1, color='black', alpha=.25)
        plt.xlabel(r'$\lambda_%s\Psi_%s$' % (v1,v1), fontsize=20, labelpad=10)
        plt.ylabel(r'$\lambda_4\Psi_4$', fontsize=20, labelpad=-1)
        plt.ticklabel_format(style='sci', axis='x', scilimits=(0,0))
        plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
        plt.xlim(np.amin(U[:,v1])*sdiag[v1]*1.1, np.amax(U[:,v1])*sdiag[v1]*1.1)
        plt.ylim(np.amin(U[:,4])*sdiag[4]*1.1, np.amax(U[:,4])*sdiag[4]*1.1)
        
        plt.subplot(2, 3, 6)
        plt.scatter(U[:,v1]*sdiag[v1], U[:,5]*sdiag[5], c=enum, cmap='gist_rainbow', s=s)
        plt.plot(U[:,v1]*sdiag[v1], U[:,5]*sdiag[5], zorder=-1, color='black', alpha=.25)
        plt.xlabel(r'$\lambda_%s\Psi_%s$' % (v1,v1), fontsize=20, labelpad=10)
        plt.ylabel(r'$\lambda_5\Psi_5$', fontsize=20, labelpad=-1)
        plt.ticklabel_format(style='sci', axis='x', scilimits=(0,0))
        plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
        plt.xlim(np.amin(U[:,v1])*sdiag[v1]*1.1, np.amax(U[:,v1])*sdiag[v1]*1.1)
        plt.ylim(np.amin(U[:,5])*sdiag[5]*1.1, np.amax(U[:,5])*sdiag[5]*1.1)
    
        plt.tight_layout()
        plt.show()

if 1: #3d diffusion map
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    
    if 1:
        ax.scatter(U[:,1]*sdiag[1], U[:,2]*sdiag[2], U[:,3]*sdiag[3], c=enum, cmap='gist_rainbow')
    else: #annotate indices
        for i in range(m): #plot each point + it's index as text above
            ax.scatter(U[i,1]*sdiag[1], U[i,2]*sdiag[2], U[i,3]*sdiag[3])
            ax.text(U[i,1]*sdiag[1], U[i,2]*sdiag[2], U[i,3]*sdiag[3], '%s' % (str(i)), size=10, zorder=1, color='gray') 
        
    plt.title(r'3D Embedding, $\mathit{\epsilon}$=%s' % eps)
    ax.set_xlabel(r'$\psi_1$')
    ax.set_ylabel(r'$\psi_2$')
    ax.set_zlabel(r'$\psi_3$')
    ax.view_init(elev=-90, azim=90)
    plt.ticklabel_format(style='sci', axis='x', scilimits=(0,0))
    plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
    plt.ticklabel_format(style='sci', axis='z', scilimits=(0,0))
    plt.show()
    
# =============================================================================
# Calculate diffusion map geodesics
# =============================================================================
    
if 0: #plot n-dimensional Euclidean distance from any given reference point
    dists = []
    ref = 0 #reference point
    for i in range(0,m):
        dn = 0
        for n in range(1,m-1): #number of dimensions to consider
            if sdiag[n] > 0: #only count eigenfunctions with non-zero eigenvalues
                dn += (sdiag[n]**2)*(U[:,n][ref] - U[:,n][i])**2
        dists.append((dn)**(1/2.))
    
    fig = plt.figure() #the following two plots should have the same shape if the correct eps was chosen
    
    plt.subplot(1, 3, 1)
    plt.title(r'Distance matrix (State 01)', wrap=1, fontsize=8)
    plt.scatter(np.linspace(1,m,m), Dist[0,:])
    plt.xlim(-1,m+1)
    plt.ylim(min(Dist[0,:])*1.1,max(Dist[0,:])*1.1)

    plt.subplot(1, 3, 2) 
    plt.title(r'DM Distances (State 01), S, $\mathit{\epsilon}$=%s' % eps, wrap=1, fontsize=8)
    plt.scatter(np.linspace(1,m,m), dists)
    plt.xlim(-1,m+1)
    plt.ylim(min(dists)*1.1,max(dists)*1.1)
    
    # Note: the below relationship to the initial distance matrix is remarkable, but as of yet unexplained:
    distsRt = []
    sdiagRt = sdiag**(1/4.)
    ref = 0 #reference point
    for i in range(0,m):
        dn = 0
        for n in range(1,m-1): #number of dimensions to consider
            if sdiagRt[n] > 0: #only count eigenfunctions with non-zero eigenvalues
                dn += (sdiagRt[n]**2)*(U[:,n][ref] - U[:,n][i])**2
        distsRt.append((dn)**(1/2.))
        
    plt.subplot(1, 3, 3) 
    plt.title(r'DM Distances (State 01), S$^{1/4}$, $\mathit{\epsilon}$=%s' % eps, wrap=1, fontsize=8)
    plt.scatter(np.linspace(1,m,m), distsRt)
    plt.xlim(-1,m+1)
    plt.ylim(min(distsRt)*1.1,max(distsRt)*1.1)
    
    plt.show()
 
if 0: #plot geodesic distance between neighboring states
    dists = []
    ref = 0 #reference point
    for i in [1]:
        dn = 0
        for n in range(1,4): #number of dimensions to consider
            if sdiag[n] > 0: #only count eigenfunctions with non-zero eigenvalues
                dn += (sdiag[n]*U[:,n][ref] - sdiag[n]*U[:,n][i])**2
        dists.append((dn)**(1/2.))
    if 1:
        plt.scatter(np.linspace(1,m,m), dists)
        plt.xlim(-1,m+1)
        plt.ylim(min(dists),max(dists)) 
        plt.show()
        