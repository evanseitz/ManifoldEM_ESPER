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
import Generate_Nd_Rot

# =============================================================================
# Data viewer for eigenfunction rotations (run via 'python RotationViewer.py') 
# Used for viewing eigenfunction rotation characteristics only.
# =============================================================================
# Author:    E. Seitz @ Columbia University - Frank Lab - 2020
# Contact:   evan.e.seitz@gmail.com
# =============================================================================

if 1: #Times font for all figures
    rc('text', usetex=True)
    rc('font', family='serif')

pyDir = os.path.dirname(os.path.abspath(__file__)) #python file location 
parDir = os.path.abspath(os.path.join(pyDir, os.pardir))

# =============================================================================
# User parameters
# =============================================================================
PD = '033'
groundTruth = True #use GT indices for visualizations; see '0_Data_Inputs/GroundTruth_Indices'
viewCM = 1 #{1,2, etc.}; if using ground-truth, CM reference frame to use for color map indices
PCA = False #specify if manifolds from PCA or DM folder {if False, DM is True}

# =============================================================================
# Import data into arrays
# =============================================================================
if PCA is True: #change end of file path to match name of your PCA outputs
    maniPath = os.path.join(parDir, '1_Embedding/PCA/Data_Manifolds/PD_%s_vec.npy' % PD) 
else: #change end of file path to match name of your DM outputs
    maniPath = os.path.join(parDir, '1_Embedding/DM/Data_Manifolds/PD_%s_vec.npy' % PD)
print('Manifold Info:', np.shape(np.load(maniPath)))
U0 = np.load(maniPath) #eigenvectors
ss = np.shape(U0)[0]

if groundTruth is True:
    if viewCM == 1: #view in reference frame of CM1
        CM_idx = np.load(os.path.join(parDir, '0_Data_Inputs/GroundTruth_Indices/CM1_Indices.npy'), allow_pickle=True)
    elif viewCM == 2: #view in reference frame of CM2
        CM_idx = np.load(os.path.join(parDir, '0_Data_Inputs/GroundTruth_Indices/CM2_Indices.npy'), allow_pickle=True)

# =============================================================================
# Import PD rotation info:
# =============================================================================
rotMatrices = os.path.join(pyDir, 'Data_Rotations/PD%s/PD%s_RotMatrices.npy' % (PD,PD))
rotEigenfunctions = os.path.join(pyDir, 'Data_Rotations/PD%s/PD%s_RotEigenfunctions.npy' % (PD,PD))
rotParameters = os.path.join(pyDir, 'Data_Rotations/PD%s/PD%s_RotParameters.npy' % (PD,PD))
rotMatrix = np.load(rotMatrices)
rotEigs = np.load(rotEigenfunctions)
rotParams = np.load(rotParameters)
dim = rotParams[0]
CM = 0 #change this to match CM of interest (zero indexing)

v1_list = []
v2_list = []
for i in range(len(rotEigs)-1):
    if rotEigs[i] != 0:
        v1_list.append(i+1)
        v2_list.append(int(rotEigs[i]+1))
thetas = rotMatrix[CM]

# =============================================================================
# Normalize manifold range and initiate subspace
# =============================================================================
def normalize(_d, to_sum=False, copy=True): #normalize all eigenfunctions
    # d is a (n x dimension) np array
    d = _d if not copy else np.copy(_d)
    d -= np.min(d, axis=0)
    d /= (np.sum(d, axis=0) if to_sum else np.ptp(d, axis=0)) #normalize btw [0,1]
    d = 2.*(d - np.min(d))/np.ptp(d)-1 #scale btw [-1,1]
    return d

U = normalize(U0) #rescale manifold between -1 and +1 (all axes)
if PCA is True:
    U_init = U[:,0:dim] #manifold subspace (for 'dim' dimensions)
else: #if DM, don't use steady-state eigenvector (v_0)
    U_init = U[:,1:dim+1]

# =============================================================================
# Visualization parameters
# =============================================================================
cmap = 'nipy_spectral' #'gist_rainbow'
s = 20
lw = .5
enum = np.arange(1,ss+1) #for colormap numbering

for f in [0,1]:
    if f == 1: #plot rotated subspaces
        v1 = v1_list[CM]-1 #2D subspace's 1st eigenvector
        v2 = v2_list[CM]-1 #2D subspace's 2nd eigenvector
        # =====================================================================
        # Use previously-defined optimal theta to rotate N-dim manifold into place
        # =====================================================================
        R = Generate_Nd_Rot.genNdRotations(dim, thetas)
        U_rot = np.matmul(R, U_init.T)
        U = U_rot.T
    else: #no rotations applied
        U = U_init

    # View an organized array of 2D subspaces:
    plt.clf()
    fig = plt.gcf()
    dimRows, dimCols = dim-1, dim
    idx = 1
    plt.rc('font', size=6)
    for v1 in range(1,dimRows+1):
        for v2 in range(v1+1, v1+dimCols+1):
            plt.subplot(dimRows, dimCols, idx)
            try:
                if groundTruth is True:                
                    color=iter(cm.tab20(np.linspace(1, 0, np.shape(CM_idx)[0])))
                    for b in range(np.shape(CM_idx)[0]):
                        c=next(color)
                        plt.scatter(U[:,v1-1][CM_idx[b]], U[:,v2-1][CM_idx[b]], color=c, s=15, edgecolor='k', linewidths=.1, zorder=1)
                    #plt.scatter(U[:,v1-1], U[:,v2-1], c=enum, cmap='nipy_spectral', s=20, linewidths=.5, edgecolor='k')
                else:
                    plt.scatter(U[:,v1-1], U[:,v2-1], c='white', s=20, linewidths=.5, edgecolor='k')
            except:
                plt.scatter(0,0, c='k')
            if PCA is True:
                plt.xlabel(r'$PC_{%s}$' % (int(v1)), fontsize=8, labelpad=5)
                plt.ylabel(r'$PC_{%s}$' % (int(v2)), fontsize=8, labelpad=2.5)
            else:
                plt.xlabel(r'$\Psi_{%s}$' % (int(v1)), fontsize=8, labelpad=5)
                plt.ylabel(r'$\Psi_{%s}$' % (int(v2)), fontsize=8, labelpad=2.5)
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
            idx+=1
    plt.tight_layout()
    plt.show()
    #plt.subplots_adjust(left=0.02, right=0.99, bottom=0.05, top=0.99, wspace=0.26, hspace=0.23)
    
