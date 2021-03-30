import sys, os, re
import numpy as np
from numpy import linalg as LA
import matplotlib
from matplotlib import rc
from matplotlib.figure import Figure
import matplotlib.pyplot as plt
from pylab import imshow, show, loadtxt, axes
#from mpl_toolkits.mplot3d import Axes3D
#from matplotlib.ticker import MaxNLocator
from matplotlib.pyplot import cm

# =============================================================================
# Data viewer for DM manifolds (run via 'python DM_Viewer.py')
# =============================================================================
# USAGE: for viewing manifold characteristics only. Manifolds must first be...
#   ...generated via previous 'DiffusionMaps.py' workflow. Alter parameters...
#   ...below to match requested PD and file locations. If using custom synthetic...
#   ...data, may need to alter colormap indexing to match data construction.
# =============================================================================
# Author:    E. Seitz @ Columbia University - Frank Lab - 2020
# Contact:   evan.e.seitz@gmail.com
# =============================================================================

pyDir = os.path.dirname(os.path.abspath(__file__)) #python file location
parDir1 = os.path.abspath(os.path.join(pyDir, os.pardir))
parDir2 = os.path.abspath(os.path.join(parDir1, os.pardir))
maniDir = os.path.join(pyDir, 'Data_Manifolds')
distDir = os.path.join(pyDir, 'Data_Distances')

if 1: #Times font for all figures
    rc('text', usetex=True)
    rc('font', family='serif')

# =============================================================================
# User parameters; import data into arrays:
# =============================================================================
PD = '064'
groundTruth = True #use GT indices for visualizations; see '0_Data_Inputs/GroundTruth_Indices'
viewCM = 1 #{1,2, etc.}; if using ground-truth, CM reference frame to use for color map indices
Dist = np.load(os.path.join(distDir, 'PD_%s_dist.npy' % PD)) #distance files
U = np.load(os.path.join(maniDir,'PD_%s_vec.npy' % PD)) #eigenvectors
sdiag = np.load(os.path.join(maniDir,'PD_%s_val.npy' % PD)) #eigenvalues
   
if groundTruth is True:
    if viewCM == 1: #view in reference frame of CM1
        CM_idx = np.load(os.path.join(parDir2, '0_Data_Inputs/GroundTruth_Indices/CM1_Indices.npy'), allow_pickle=True)
    elif viewCM == 2: #view in reference frame of CM2
        CM_idx = np.load(os.path.join(parDir2, '0_Data_Inputs/GroundTruth_Indices/CM2_Indices.npy'), allow_pickle=True)

# =============================================================================
# Analysis of embedding:
# =============================================================================
m = np.shape(Dist)[0]
enum = np.arange(1,m+1)
s = 20
lw = .5
cmap = 'nipy_spectral' #'gist_rainbow'

if 0: #view a single 2D subspace
    v1 = 1 #eigenfunction to plot on first axis
    v2 = 3 #eigenfunction to plot on second axis
    if groundTruth is True:
        #plt.scatter(U[:,v1], U[:,v2], c=enum, cmap=cmap, s=s, linewidths=lw, edgecolor='k')
        color=iter(cm.tab20(np.linspace(1, 0, np.shape(CM_idx)[0])))
        for b in range(np.shape(CM_idx)[0]):
            c=next(color)
            plt.scatter(U[:,v1][CM_idx[b]], U[:,v2][CM_idx[b]], color=c, s=15, edgecolor='k', linewidths=.1, zorder=1)
    else:
        plt.scatter(U[:,v1], U[:,v2], c='white', s=20, linewidths=.5, edgecolor='k', zorder=0)
        
    enum = np.arange(1,m+1)
    if 0: #annotate points in plot with indices of each state
        for i, txt in enumerate(enum):
            plt.annotate(txt, (U[i,v1], U[i,v2]), fontsize=12, zorder=1, color='gray')
    plt.title(r'2D Embedding')
    plt.xlabel(r'$\psi_1$')
    plt.ylabel(r'$\psi_2$')
    plt.ticklabel_format(style='sci', axis='x', scilimits=(0,0))
    plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
    plt.axis('scaled')
    plt.show()
    
if 1: #view an organized array of 2D subspaces   
    fig = plt.figure()
    dimRows = 6
    dimCols = 9
    idx = 1
    for v1 in range(1,dimRows+1):
        for v2 in range(v1+1, v1+dimCols+1):
            plt.subplot(dimRows, dimCols, idx)
            if groundTruth is True:
                color=iter(cm.tab20(np.linspace(1, 0, np.shape(CM_idx)[0])))
                for b in range(np.shape(CM_idx)[0]):
                    c=next(color)
                    plt.scatter(U[:,v1][CM_idx[b]], U[:,v2][CM_idx[b]], color=c, s=15, edgecolor='k', linewidths=.1, zorder=1)
                #plt.scatter(U[:,v1], U[:,v2], c=np.arange(1,m+1), cmap=cmap, s=s, linewidths=lw, edgecolor='k') #cmap: 'nipy_spectral', 'gist_rainbow'
            else:
                plt.scatter(U[:,v1], U[:,v2], c='white', s=20, linewidths=.5, edgecolor='k', zorder=0)
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

if 0: #view a single 3D subspace
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    v1 = 1
    v2 = 2
    v3 = 3
    if 1:
        ax.scatter(U[:,v1], U[:,v2], U[:,v3], c=enum, cmap='gist_rainbow', linewidths=lw, s=s, edgecolor='k')
    else: #annotate indices
        for i in range(m): #plot each point + it's index as text above
            ax.scatter(U[i,v1], U[i,v2], U[i,v3])
            ax.text(U[i,v1], U[i,v2], U[i,v3], '%s' % (str(i)), size=10, zorder=1, color='gray') 
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
 
if 0: #DM distance between neighboring states 
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