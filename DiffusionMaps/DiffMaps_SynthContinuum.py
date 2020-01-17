import sys, os, re
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
import imageio
import mrcfile

'''
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
Copyright (c) CU, Evan Seitz 2019-2020
Contact: evan.e.seitz@gmail.com
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
'''

pyDir = os.path.dirname(os.path.abspath(__file__)) #python file location

# =============================================================================
# generate distances (if 1) or load previously-computed distances (if 0):
# =============================================================================

if 0: #generate distances for image stack of projections for a given projection direction ('PD', as chosen above)
    dataDir = os.path.join(pyDir, 'projection_stacks') #folder with all m .mrcs files
    PD = 0 #projection direction
    boxSize = 250 #dimensions of image; e.g, '250' for 250x250
    
    def natural_sort(l): #sort filenames from directory in correct order (leading zeros)
        convert = lambda text: int(text) if text.isdigit() else text.lower() 
        alphanum_key = lambda key: [ convert(c) for c in re.split('([0-9]+)', key) ] 
        return sorted(l, key = alphanum_key)
    
    dataPaths0 = [] #load in all .mrcs files
    for root, dirs, files in os.walk(dataDir):
        for file in sorted(files):
            if not file.startswith('.'): #ignore hidden files
                if file.endswith(".mrcs"):
                    dataPaths0.append(os.path.join(root, file))
    dataPaths = natural_sort(dataPaths0) 
    
    m = len(dataPaths) #number of images to consider from distance matrix; m <= len(dataPaths); e.g., m=20 for 1D motion
    RMSD = np.ndarray(shape=(m,m), dtype=float)
    
    frames = np.ndarray(shape=(m,boxSize,boxSize))
    new_m = 0
    for i in range(m):
    #for i in [1,5,9,11,13,17,19]: #to compare manifold of "full" state space with partial (for experimenting)
        print(dataPaths[i])
        stack = mrcfile.open(dataPaths[i])
        frames[new_m] = stack.data[PD]
        if 0: #plot each frame sequentially
            if i < 2: #number of frames to plot
                plt.imshow(frames[i])
                plt.show()
                plt.hist(frames[i],bins=100)
                plt.show()
                print('image mean:', np.mean(frames[i]))
                print('image std:', np.std(frames[i]))
        stack.close()
        new_m += 1
    m = new_m #`new_m` will only change if comparing to partial state space (above)
        
    if 0: #save gif
        imageio.mimsave('movie_PD_%s.gif' % PD, frames)
    
    if 1: #manual distance calculation
        p = 2 #Minkowski distance metric: p1=Manhattan, p2=Euclidean, etc.
        for i in range(0,m):
            for j in range(0,m):
                #RMSD[i,j] = (np.sum(np.abs((stack.data[i]-stack.data[j]))**2))**(1./p) / stack.data[i].size #kernel
                RMSD[i,j] = (np.sum(np.abs((frames[i]-frames[j]))**p) / frames[i].size)**(1./p) #equivalent of 2D-RMSD
        
    else: #or use scipy library for distance metric:
        stack2 = np.ndarray(shape=(m,boxSize**2), dtype=float)
        for i in range(0,m):
            stack2[i] = frames[i].flatten()
        
        RMSD = cdist(stack2, stack2, 'euclidean')
        # other options: ('sqeuclidean'), (minkowski', p=2.), ('cityblock'), ('cosine'), ('correlation'),
                        #('chebyshev'), (canberra'), ('braycurtis')
       
    if 0: #save distance matrix for subsequent use
        np.save('Dist_PD_%s.npy' % PD, RMSD)
        
else: #or load in previously-generated distance matrix
    RMSD = np.load('Dist_2DoF_3dRMSD.npy') #distances from PDB files (2 degrees of freedom)
    #RMSD = np.load('Dist_3DoF_3dRMSD.npy') #distances from PDB files (3 degrees of freedom)
    #RMSD = np.load('Dist_2DoF_Volumes.npy') #distances from MRC files
    #RMSD = np.load('Dist_2DoF_PD0.npy') #distances from projections of MRC files
    
    m = np.shape(RMSD)[0]#number of states to consider from distance matrix; m <= len(dataPaths); e.g., m=20 for 1D motion
    
RMSD = RMSD[0:m,0:m] #needed if m <= len(dataPaths), as defined above 

if 1: #plot distance matrix
    imshow(RMSD, cmap='jet', origin='lower', interpolation='nearest')
    plt.title('Distances', fontsize=20)
    plt.xlabel('State (PDB)')
    plt.ylabel('State (PDB)')
    plt.colorbar()
    plt.tight_layout()
    plt.show()
    
if 1: #plot distances of state_01_01 to all others
    plt.scatter(np.linspace(1,m,m), RMSD[0,:])
    plt.show()

# =============================================================================
# ferguson method to find optimal epsilon:
# =============================================================================

if 0:
    import fergusonE
    logEps = np.arange(-50,50.2,0.1)
    a0 = 1*(np.random.rand(4,1)-.5)
    popt, logSumWij, resnorm = fergusonE.op(RMSD,logEps,a0)
    
    def tanh(xx, aa0, aa1, aa2, aa3): #fit for tanh() fun
        F = aa3 + aa2 * np.tanh(aa0 * xx + aa1)
        return F
    
    # compute slope of linear portion of tanh():
    xi = -(popt[1] / popt[0]) - np.abs(popt[3]-popt[2])*.01
    xf = -(popt[1] / popt[0]) + np.abs(popt[3]-popt[2])*.01
    yi = tanh(xi, popt[0], popt[1], popt[2], popt[3])
    yf = tanh(xf, popt[0], popt[1], popt[2], popt[3])
    slope = (yf-yi)/(xf-xi)
    print('Effective Dimensionality:', 2.*slope) #via Coifman/Ferguson

    eps = np.exp(-(popt[1] / popt[0]))
    print('Optimal ln(eps):', -(popt[1] / popt[0]))
    print('Optimal eps:', eps)
    
    if 1: #plot of Ferguson method
        plt.scatter(logEps, logSumWij, s=1, c='#1f77b4', edgecolor='#1f77b4', zorder=.1)
        plt.plot(logEps, tanh(logEps, popt[0], popt[1], popt[2], popt[3]), c='red', linewidth=.5, zorder=.2)
        plt.title('Gaussian Kernel Bandwidth', fontsize=14)
        plt.xlabel(r'$\mathrm{ln \ \epsilon}$', fontsize=16)
        plt.ylabel(r'$\mathrm{ln \ \sum_{i,j} \ A_{i,j}}$', fontsize=18, rotation=90)
        plt.show()
        
else:
    eps = .0001 #best for 'Dist_3D_RMSD.npy'
    #eps = 1000 #best for 'Dist_MRCs.npy'
    #eps = .01 #best for 'Dist_PD_0.npy'

# =============================================================================
# generate optimal kernel:
# =============================================================================

A = np.exp(-(RMSD**2 / 2*eps)) #similarity matrix
alpha = 1 #currently unassigned (always alpha=1)
    # alpha = 1.0: Laplace-Beltrami operator
    # alpha = 0.5: Fokker-Planck diffusion
    # alpha = 0.0: graph Laplacian normalization

if 0: #plot similarity matrix, A
    imshow(A, cmap='jet', origin='lower')
    plt.title(r'Gaussian Kernel, $\mathit{\epsilon}$=%s' % eps, fontsize=20)
    plt.colorbar()
    plt.tight_layout()
    show()
    
    rowSums = np.sum(A,axis=1)
    if 0:
        print('minimum affinity:', np.where(rowSums == np.amin(rowSums)))
        plt.scatter(np.linspace(1,m,m),rowSums)
        plt.xlim(1,m)
        plt.ylim(np.amin(rowSums),np.amax(rowSums))
        plt.show()
        
    if 0: #kernel of state_01_01 to all others
        plt.scatter(np.linspace(1,m,m), A[0,:])
        plt.show()

# =============================================================================
# normalized graph laplacian construction:        
# =============================================================================

D = np.ndarray(shape=(m,m), dtype=float) #diagonal matrix
for i in range(0,m):
    for j in range(0,m):
        if i == j:
            D[i,j] = np.sum(A[i], axis=0)
        else:
            D[i,j] = 0    

if 0: #plot diagonal matrix, D
    print(D.diagonal())
    imshow(D, cmap='jet', origin='lower')
    plt.title(r'$\mathit{M}, \mathit{\alpha}$=%s' % alpha, fontsize=20)
    plt.colorbar()
    plt.tight_layout()
    show()
    
D_inv = scipy.linalg.fractional_matrix_power(D, -1)
M = np.matmul(A, D_inv)

if 0: #plot Markov Transition matrix, M
    print(np.sum(M, axis=0)) #should be all 1's
    imshow(M, cmap='jet', origin='lower')
    plt.title('Markov Transition Matrix', fontsize=20)
    plt.colorbar()
    plt.tight_layout()
    show()
    
    MrowSums = np.sum(M,axis=0)
    
    if 0:
        print('minimum affinity:', np.where(MrowSums == np.amin(MrowSums)))
        plt.scatter(np.linspace(1,m,m),MrowSums)
        plt.xlim(1,m)
        plt.ylim(np.amin(MrowSums),np.amax(MrowSums))
        plt.show()

# note: M is adjoint to symmetric matrix Ms:     
D_inv_half = scipy.linalg.sqrtm(D_inv)
D_half = scipy.linalg.sqrtm(D)
Ms0 = np.matmul(D_inv_half, M)
Ms = np.matmul(Ms0, D_half)

def check_symmetric(a, rtol=1e-08, atol=1e-08, equal_nan=True):
    return np.allclose(a, a.T, rtol=rtol, atol=atol)
print('Hermitian:', check_symmetric(Ms))

# =============================================================================
# eigendecomposition:
# =============================================================================

if 0: #np.linalg.eigh() version
    def tidyUp(D,EV):
        order = np.argsort(D)[::-1]
        D = np.sort(D)[::-1]
        EV = EV[:,order]
        sqrtD = np.sqrt(D)
        S = np.diag(sqrtD)
        invS = np.diag(1./sqrtD)
        return (D,EV,S,invS)
    
    d,U = np.linalg.eigh(np.matmul(M,M.T))
    d,U,S,invS = tidyUp(d,U)
    V = np.matmul(M.T,np.matmul(U,invS))
    sdiag = np.diag(S)
    sdiag = sdiag**(1/2.) #eigenvalues given by s**2
    
else: #np.linalg.svd() version; computationally same result as above
    U, sdiag, vh = np.linalg.svd(Ms) #vh = U.T
    sdiag = sdiag**(1/2.) #eigenvalues given by s**2
    
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
# analysis of diffusion map:
# =============================================================================

if 1: #plot 2d diffusion map
    v1 = 1 #eigenfunction to plot on first axis
    v2 = 2 #eigenfunction to plot on second axis
    plt.scatter(U[:,v1]*sdiag[v1], U[:,v2]*sdiag[v2],c=U[:,v1]*sdiag[v1])
    plt.title(r'2D Embedding, $\mathit{\epsilon}$=%s' % eps)
    plt.xlabel(r'$\psi_1$')
    plt.ylabel(r'$\psi_2$')
    plt.ticklabel_format(style='sci', axis='x', scilimits=(0,0))
    plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
    plt.colorbar()
    plt.xlim(np.amin(U[:,v1])*sdiag[v1]*1.1, np.amax(U[:,v1])*sdiag[v1]*1.1)
    plt.ylim(np.amin(U[:,v2])*sdiag[v2]*1.1, np.amax(U[:,v2])*sdiag[v2]*1.1)
    plt.show()
    
if 1: #2d diffusion map; several higher-order eigenfunction combinations
    fig = plt.figure()
    fig.suptitle(r'2D Embedding, $\mathit{\epsilon}$=%s' % eps)
    
    plt.subplot(2, 3, 1)
    plt.scatter(U[:,1]*sdiag[1], U[:,0]*sdiag[0],c=U[:,1]*sdiag[1])
    plt.xlabel(r'$\psi_1$', fontsize=20)
    plt.ylabel(r'$\psi_0$', fontsize=20)
    plt.ticklabel_format(style='sci', axis='x', scilimits=(0,0))
    plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
    plt.xlim(np.amin(U[:,1])*sdiag[1]*1.1, np.amax(U[:,1])*sdiag[1]*1.1)
    plt.ylim(np.amin(U[:,0])*sdiag[0], np.amax(U[:,0])*sdiag[0])
    
    plt.subplot(2, 3, 2)
    plt.scatter(U[:,1]*sdiag[1], U[:,1]*sdiag[1],c=U[:,1]*sdiag[1])
    plt.xlabel(r'$\psi_1$', fontsize=20)
    plt.ylabel(r'$\psi_1$', fontsize=20)
    plt.ticklabel_format(style='sci', axis='x', scilimits=(0,0))
    plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
    plt.xlim(np.amin(U[:,1])*sdiag[1]*1.1, np.amax(U[:,1])*sdiag[1]*1.1)
    plt.ylim(np.amin(U[:,1])*sdiag[1]*1.1, np.amax(U[:,1])*sdiag[1]*1.1)

    plt.subplot(2, 3, 3)
    plt.scatter(U[:,1]*sdiag[1], U[:,2]*sdiag[2],c=U[:,1]*sdiag[1])
    plt.xlabel(r'$\psi_1$', fontsize=20)
    plt.ylabel(r'$\psi_2$', fontsize=20)
    plt.ticklabel_format(style='sci', axis='x', scilimits=(0,0))
    plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
    plt.xlim(np.amin(U[:,1])*sdiag[1]*1.1, np.amax(U[:,1])*sdiag[1]*1.1)
    plt.ylim(np.amin(U[:,2])*sdiag[2]*1.1, np.amax(U[:,2])*sdiag[2]*1.1)

    plt.subplot(2, 3, 4)
    plt.scatter(U[:,1]*sdiag[1], U[:,3]*sdiag[3],c=U[:,1]*sdiag[1])
    plt.xlabel(r'$\psi_1$', fontsize=20)
    plt.ylabel(r'$\psi_3$', fontsize=20)
    plt.ticklabel_format(style='sci', axis='x', scilimits=(0,0))
    plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
    plt.xlim(np.amin(U[:,1])*sdiag[1]*1.1, np.amax(U[:,1])*sdiag[1]*1.1)
    plt.ylim(np.amin(U[:,3])*sdiag[3]*1.1, np.amax(U[:,3])*sdiag[3]*1.1) 
    
    plt.subplot(2, 3, 5)
    plt.scatter(U[:,1]*sdiag[1], U[:,4]*sdiag[4],c=U[:,1]*sdiag[1])
    plt.xlabel(r'$\psi_1$', fontsize=20)
    plt.ylabel(r'$\psi_4$', fontsize=20)
    plt.ticklabel_format(style='sci', axis='x', scilimits=(0,0))
    plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
    plt.xlim(np.amin(U[:,1])*sdiag[1]*1.1, np.amax(U[:,1])*sdiag[1]*1.1)
    plt.ylim(np.amin(U[:,4])*sdiag[4]*1.1, np.amax(U[:,4])*sdiag[4]*1.1)
    
    plt.subplot(2, 3, 6)
    plt.scatter(U[:,1]*sdiag[1], U[:,5]*sdiag[5],c=U[:,1]*sdiag[1])
    plt.xlabel(r'$\psi_1$', fontsize=20)
    plt.ylabel(r'$\psi_5$', fontsize=20)
    plt.ticklabel_format(style='sci', axis='x', scilimits=(0,0))
    plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
    plt.xlim(np.amin(U[:,1])*sdiag[1]*1.1, np.amax(U[:,1])*sdiag[1]*1.1)
    plt.ylim(np.amin(U[:,5])*sdiag[5]*1.1, np.amax(U[:,5])*sdiag[5]*1.1)
    
    plt.tight_layout()
    plt.show()

if 1: #3d diffusion map
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    
    if 0: #preliminary (ad-hoc) fit for testing
        def Legendre(x, v0):
            return 1/2. * (3.*((1.25*x)**2.) - v0)
        curve_x = []
        curve_y = []
        scale = 1e-7
        #print(U[:,1])
        idx = 0
        for i in U[:,1]:#np.linspace(-1,1,100):
            curve_x.append(i*sdiag[1])
            curve_y.append(Legendre(i, U[idx,0]) * sdiag[1])
            idx += 1
        ax.plot(curve_x, curve_y)

    ax.scatter(U[:,1]*sdiag[1], U[:,2]*sdiag[2], U[:,3]*sdiag[3],c=U[:,1]*sdiag[1], cmap='jet')
    plt.title(r'3D Embedding, $\mathit{\epsilon}$=%s' % eps)
    ax.set_xlabel(r'$\psi_1$')
    ax.set_ylabel(r'$\psi_2$')
    ax.set_zlabel(r'$\psi_3$')
    ax.view_init(elev=-90, azim=90)
    plt.ticklabel_format(style='sci', axis='x', scilimits=(0,0))
    plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
    plt.ticklabel_format(style='sci', axis='z', scilimits=(0,0))
    plt.show()
        
if 0: #save diffusion map to file
    np.save('DM_PD%s_eps0.npy' % PD, U)
    np.save('Eig_PD%s_eps0.npy' % PD, sdiag)
    
# =============================================================================
# calculate diffusion map geodesics:
# =============================================================================
    
if 1: #plot n-dimensional Euclidean distance from any given reference point
    dists = []
    ref = 0 #reference point
    for i in range(0,m):
        dn = 0
        for n in range(1,m-1): #number of dimensions to consider
            if sdiag[n] > 0: #only count eigenfunctions with non-zero eigenvalues
                dn += (sdiag[n]*U[:,n][ref] - sdiag[n]*U[:,n][i])**2
        dists.append((dn)**(1/2.))
    
    fig = plt.figure() #the following two plots should have the same shape if the correct eps was chosen
    
    plt.subplot(1, 2, 1)
    plt.title(r'Distance matrix distance from state_1 to all others', wrap=1, fontsize=8)
    plt.scatter(np.linspace(1,m,m), RMSD[0,:])
    plt.xlim(-1,m+1)
    plt.ylim(min(RMSD[0,:])*1.1,max(RMSD[0,:])*1.1)

    plt.subplot(1, 2, 2) 
    plt.title(r'n-dim DM distance from state_1 to all others, $\mathit{\epsilon}$=%s' % eps, wrap=1, fontsize=8)
    plt.scatter(np.linspace(1,m,m), dists)
    plt.xlim(-1,m+1)
    plt.ylim(min(dists)*1.1,max(dists)*1.1)
    
    plt.show()
    
if 1: #plot distances of state_1 to all others
    plt.show()
    
if 1: #plot geodesic distance between neighboring states
    dists = []
    ref = 0 #reference point
    for i in [1]:
        dn = 0
        for n in range(1,4): #number of dimensions to consider
            if sdiag[n] > 0: #only count eigenfunctions with non-zero eigenvalues
                dn += (sdiag[n]*U[:,n][ref] - sdiag[n]*U[:,n][i])**2
        dists.append((dn)**(1/2.))
    if 0:
        plt.scatter(np.linspace(1,m,m), dists)
        plt.xlim(-1,m+1)
        plt.ylim(min(dists),max(dists)) 
        plt.show()
