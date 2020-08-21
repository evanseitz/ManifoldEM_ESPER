import sys, os, re
import numpy as np
from numpy import linalg as LA
import matplotlib
from matplotlib import rc
from matplotlib.figure import Figure
import matplotlib.pyplot as plt
from pylab import imshow, show, loadtxt, axes
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.ticker import MaxNLocator
from matplotlib.pyplot import cm

# =============================================================================
# Data viewer for PCA manifolds (run via 'python PCA_Viewer.py') 
# Used for analysis (viewing) manifold characteristics only
# Author:    E. Seitz @ Columbia University - Frank Lab - 2020
# Contact:   evan.e.seitz@gmail.com
# =============================================================================

pyDir = os.path.dirname(os.path.abspath(__file__)) #python file location
maniDir = os.path.join(pyDir, 'Data_Manifolds')
distDir = os.path.join(pyDir, 'Data_Distances')

if 0: #Times font for all figures
    rc('text', usetex=True)
    rc('font', family='serif')

# =============================================================================
# Import data into arrays
# =============================================================================
Dist = np.load(os.path.join(distDir, 'PD001_tau1_dist.npy')) #distances from projections of MRC files  
m = np.shape(Dist)[0]  
U = np.load(os.path.join(maniDir,'PD001_tau1_vec.npy')) #eigenvectors
sdiag = np.load(os.path.join(maniDir,'PD001_tau1_val.npy')) #eigenvalues

# =============================================================================
# Analysis of embedding
# =============================================================================
enum = np.arange(1,m+1)
s = 20
lw = .5
cmap = 'nipy_spectral' #'gist_rainbow'

if 1: #plot 2d diffusion map
    v1 = 1 #eigenfunction to plot on first axis
    v2 = 2 #eigenfunction to plot on second axis
    plt.scatter(U[:,v1]*sdiag[v1], U[:,v2]*sdiag[v2], c=enum, cmap=cmap, s=s, linewidths=lw, edgecolor='k')
    enum = np.arange(1,m+1)
    if 0: #annotate points in plot with indices of each state
        for i, txt in enumerate(enum):
            plt.annotate(txt, (U[i,v1]*sdiag[v1], U[i,v2]*sdiag[v2]), fontsize=12, zorder=1, color='gray')
    plt.title(r'2D Embedding')
    plt.xlabel(r'$\psi_1$')
    plt.ylabel(r'$\psi_2$')
    plt.ticklabel_format(style='sci', axis='x', scilimits=(0,0))
    plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
    plt.colorbar()
    plt.xlim(np.amin(U[:,v1])*sdiag[v1]*1.1, np.amax(U[:,v1])*sdiag[v1]*1.1)
    plt.ylim(np.amin(U[:,v2])*sdiag[v2]*1.1, np.amax(U[:,v2])*sdiag[v2]*1.1)
    plt.show()
    
if 1: #2d diffusion map; sets of higher-order eigenfunction combinations       
    if 1: #plot of all eigenfunctions v1 (up to 'dimRows/Cols') vs all others (v1+i)
        fig = plt.figure()
        dimRows = 6
        dimCols = 9
        idx = 1
        for v1 in range(1,dimRows+1):
            for v2 in range(v1+1, v1+dimCols+1):
                plt.subplot(dimRows, dimCols, idx)
                plt.scatter(U[:,v1]*sdiag[v1], U[:,v2]*sdiag[v2], c=enum, cmap=cmap, s=s, linewidths=lw, edgecolor='k') #gist_rainbow, nipy_spectral
                #plt.plot(U[:,v1]*sdiag[v1], U[:,v2]*sdiag[v2], zorder=-1, color='black', alpha=.25)
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

                plt.xlim(np.amin(U[:,v1])*sdiag[v1]*1.1, np.amax(U[:,v1])*sdiag[v1]*1.1)
                plt.ylim(np.amin(U[:,v2])*sdiag[v2]*1.1, np.amax(U[:,v2])*sdiag[v2]*1.1)
                idx += 1 
        plt.tight_layout()
        plt.subplots_adjust(left=0.02, right=0.99, bottom=0.05, top=0.99, wspace=0.26, hspace=0.23)

        plt.show()

if 1: #3d diffusion map
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    v1 = 1
    v2 = 2
    v3 = 3
    if 1:
        ax.scatter(U[:,v1]*sdiag[v1], U[:,v2]*sdiag[v2], U[:,v3]*sdiag[v3], c=enum, cmap='gist_rainbow', linewidths=lw, s=s, edgecolor='k')
    else: #annotate indices
        for i in range(m): #plot each point + it's index as text above
            ax.scatter(U[i,v1]*sdiag[v1], U[i,v2]*sdiag[v2], U[i,v3]*sdiag[v3])
            ax.text(U[i,v1]*sdiag[v1], U[i,v2]*sdiag[v2], U[i,v3]*sdiag[v3], '%s' % (str(i)), size=10, zorder=1, color='gray') 
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

if 1: #plot n-dimensional Euclidean distance from any given reference point
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
    plt.scatter(np.linspace(1,m,m), Dist[0,:], s=s, linewidths=lw, edgecolor='k')
    plt.xlim(-1,m+1)
    plt.ylim(min(Dist[0,:])*1.1,max(Dist[0,:])*1.1)

    plt.subplot(1, 3, 2) 
    plt.title(r'DM Distances (State 01), S', wrap=1, fontsize=8)
    plt.scatter(np.linspace(1,m,m), dists, s=s, linewidths=lw, edgecolor='k')
    plt.xlim(-1,m+1)
    plt.ylim(min(dists)*1.1,max(dists)*1.1)
    
    # Note: below relationship to initial distance matrix is empirically derived
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
    plt.title(r'DM Distances (State 01), S$^{1/4}$', wrap=1, fontsize=8)
    plt.scatter(np.linspace(1,m,m), distsRt, s=s, linewidths=lw, edgecolor='k')
    plt.xlim(-1,m+1)
    plt.ylim(min(distsRt)*1.1,max(distsRt)*1.1)
    plt.show()
 
if 1: #DM distance between neighboring states 
    refs = range(0,19)#m-1) #reference points
    distsND = []
    for r in refs:
        dnND = 0
        for n in range(1,len(sdiag)): #number of dimensions to consider
            if sdiag[n] > 0: #only count eigenfunctions with non-zero eigenvalues
                dnND += (sdiag[n]**2)*(U[:,n][r] - U[:,n][r+1])**2
        distsND.append((dnND)**(1/2.))
        
    plt.subplot(1,1,1)
    plt.scatter(refs, distsND, s=15, linewidths=.5, edgecolor='k')
    plt.plot(refs, distsND, zorder=-1, color='black', alpha=.25)
    plt.ylim(np.amin(distsND) - np.amax(distsND)*.02, np.amax(distsND) + np.amax(distsND)*.02)
    plt.xlim(-1,19)
    plt.ylabel('DM Distance', fontsize=16, labelpad=10)
    positions = (refs)
    labels = ('1-2','2-3','3-4','4-5','5-6','6-7','7-8','8-9','9-10','10-11',
              '11-12','12-13','13-14','14-15','15-16','16-17','17-18','18-19','19-20')
    plt.xticks(positions, labels)
    plt.xticks(rotation=90)
    plt.xlabel(r'CM$_2$ Edges', labelpad=10, fontsize=16) 
    plt.tick_params(axis='both', which='major', labelsize=8)
    plt.tick_params(axis='both', which='minor', labelsize=6)
    plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
    plt.show()