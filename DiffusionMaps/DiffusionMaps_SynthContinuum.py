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
import imageio
import mrcfile

'''
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
Copyright (c) CU, Evan Seitz 2019-2020
Contact: evan.e.seitz@gmail.com
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
'''

if 1: #Times font for all figures
    rc('text', usetex=True)
    rc('font', family='serif')
    
pyDir = os.path.dirname(os.path.abspath(__file__)) #python file location

# =============================================================================
# Generate distances (if 1) or load previously-computed distances (if 0)
# =============================================================================

if 0: #generate distances for image stack of projections for a given projection direction ('PD', as chosen above)
    dataDir = os.path.join(pyDir, 'projection_stacks')#_SS3_small') #folder with all m .mrcs files
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
    Dist = np.ndarray(shape=(m,m), dtype=float)
    
    frames = np.ndarray(shape=(m,boxSize,boxSize))
    new_m = 0
    for i in range(m):
    #for i in [7,8,9,10,11,12,13]: #to compare manifold of "full" state space with partial (for experimenting)
        print(dataPaths[i])
        stack = mrcfile.open(dataPaths[i])
        frames[new_m] = stack.data[PD]
        if 0:
            if i == m-1:
                np.save('PD_Images/PD%s/State_%s.npy' % (PD,i), frames[i])
            if 0:
                import cv2 as cv
                normalizedImg = np.zeros((250, 250))
                norm = cv.normalize(frames[i], normalizedImg, 0, 255, cv.NORM_MINMAX)
                plt.imshow(norm)#frames[i])
                plt.savefig('PD_Images/PD%s/State_%s.png' % (PD,i), bbox_inches='tight')
                #plt.show()

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
        imageio.mimsave('movie_PD_%s_SS3.gif' % PD, frames)
    
    if 1: #manual distance calculation
        p = 2 #Minkowski distance metric: p1=Manhattan, p2=Euclidean, etc.
        for i in range(0,m):
            for j in range(0,m):
                #Dist[i,j] = (np.sum(np.abs((stack.data[i]-stack.data[j]))**2))**(1./p) / stack.data[i].size #kernel
                Dist[i,j] = (np.sum(np.abs((frames[i]-frames[j]))**p) / frames[i].size)**(1./p) #equivalent of 2D-Dist
        
    else: #or use scipy library for distance metric:
        stack2 = np.ndarray(shape=(m,boxSize**2), dtype=float)
        for i in range(0,m):
            stack2[i] = frames[i].flatten()
        
        Dist = cdist(stack2, stack2, 'euclidean')
        # other options: ('sqeuclidean'), (minkowski', p=2.), ('cityblock'), ('cosine'), ('correlation'),
                        #('chebyshev'), (canberra'), ('braycurtis')
       
    if 0: #save distance matrix for subsequent use
        np.save('Dist_2DoF_PD%s.npy' % PD, Dist)
        
else: #or load in previously-generated distance matrix
    #Dist = np.load('Dist_2DoF_3dRMSD.npy') #distances from PDB files (2 degrees of freedom)
    #Dist = np.load('Dist_3DoF_3dRMSD_small.npy') #distances from PDB files (3 degrees of freedom)
    #Dist = np.load('Dist_3DoF_3dRMSD_large.npy') #distances from PDB files (3 degrees of freedom)
    Dist = np.load('Dist_2DoF_Volumes.npy')*(250**3) #distances from MRC files
    #Dist = np.load('Dist_3DoF_Volumes_small.npy')*(250**3) #distances from MRC files
    #Dist = np.load('Dist_3DoF_Volumes_large.npy')*(250**3) #distances from MRC files
    #Dist = np.load('Dist_2DoF_PD0.npy') #distances from projections of MRC files
    
    m = np.shape(Dist)[0]#number of states to consider from distance matrix; m <= len(dataPaths); e.g., m=20 for 1D motion

Dist = Dist[0:m,0:m] #needed if m <= len(dataPaths), as defined above 

if 0: #plot distance matrix
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
# Ferguson method to find optimal epsilon
# =============================================================================

if 0:
    import fergusonE
    #import fergusonLogistic
    logEps = np.arange(-50,50.2,0.1)

    popt, logSumWij, resnorm = fergusonE.op(Dist,logEps)
    #popt, logSumWij, resnorm = fergusonLogistic.op(Dist,logEps)

    def tanh(xx, aa0, aa1, aa2, aa3): #fit for tanh() fun
        F = aa3 + aa2 * np.tanh(aa0 * xx + aa1)
        return F
    
    def logistic(xx, ymin, ymax, a1, a2, b1, b2):
        F = ymin + ((ymax - ymin) / (1+(a1*np.exp(-b1*xx))+(a2*np.exp(-b2*xx))))
        return F
    
    print('popt:', popt)
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

    if 0: #save Ferguson plot to file
        np.save('FERG_SS3_PDB_small.npy', [logEps, logSumWij])

else:
    #eps = 1e-4 #best for 'Dist_3D_RMSD.npy'; optimal range: [1e-13, 1e-4]
    eps = 1e-7#1e-7 #best for 'Dist_MRCs.npy'; optimal range: [1e-18, 1e-7]
    #eps = .01 #best for 'Dist_PD_0.npy'; optimal range: [1e-11, 1e1]

# =============================================================================
# Generate optimal Gaussian kernel for Similarity Matrix (A)
# =============================================================================

A = np.exp(-(Dist**2 / 2*eps)) #similarity matrix

'''
# alpha = 1.0: Laplace-Beltrami operator (default)
# alpha = 0.5: Fokker-Planck diffusion
# alpha = 0.0: graph Laplacian normalization
'''

if 0: #plot similarity matrix, A
    imshow(A, cmap='jet', origin='lower')
    plt.title(r'Gaussian Kernel, $\mathit{\epsilon}$=%s' % eps, fontsize=20)
    plt.colorbar()
    plt.tight_layout()
    show()
    
    rowSums = np.sum(A, axis=1)
    if 0:
        print('minimum affinity:', np.where(rowSums == np.amin(rowSums)))
        plt.scatter(np.linspace(1,m,m), rowSums)
        plt.xlim(1, m)
        plt.ylim(np.amin(rowSums), np.amax(rowSums))
        plt.show()
        
    if 0: #similarity of state_01_01 to all others
        plt.scatter(np.linspace(1,m,m), A[0,:])
        plt.show()

# =============================================================================
# Markov Transition Matrix (M) construction:        
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
    
Dinv = scipy.linalg.fractional_matrix_power(D, -1)
M = np.matmul(A, Dinv) #Markov transition matrix; normalization of A to be row stochastic
if 0: #check Markov transition matrix is row stochastic
    print(np.sum(M, axis=0)) #should be all 1's

if 0: # constructing the density invariant Graph Laplacian:
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
        np.save('Eig_SS3_PD1.npy', sdiag)
        np.save('DM_SS3_PD1.npy', U)
    
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
        