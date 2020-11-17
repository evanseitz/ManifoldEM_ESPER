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
# Used for analysis (viewing) eigenfunction rotations characteristics only
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
groundTruth = True #optional, for comparing outputs with ground-truth knowledge
PCA = False #specify if manifolds from PCA or DM folder {if False, DM is True}
PD = '003'

# =============================================================================
# Import data into arrays
# =============================================================================
if PCA is True: #change end of file path to match name of your PCA outputs
    if 0:
        maniPath = os.path.join(parDir, '1_Embedding/PCA/Data_Manifolds/PD%s_tau5_SNR_vec.npy' % PD) 
    else: #example 126 PD great circle provided in repository
        maniPath = os.path.join(parDir, '1_Embedding/PCA/Data_Manifolds_126/PD%s_SS2_SNRpt1_tau5_vec.npy' % PD)
else: #change end of file path to match name of your DM outputs
    maniPath = os.path.join(parDir, '1_Embedding/DM/Data_Manifolds/PD%s_tau5_SNR_vec.npy' % PD)
print('Manifold Info:', np.shape(np.load(maniPath)))
U0 = np.load(maniPath) #eigenvectors
ss = np.shape(U0)[0]

# =============================================================================
# Import PD rotation info:
# =============================================================================
if PCA is True:
    rotPath = os.path.join(parDir, '2_Manifold_Rotations/Data_Rotations_PCA/PD%s/PD%s_CM_Rot.npy' % (PD,PD))
else:
    rotPath = os.path.join(parDir, '2_Manifold_Rotations/Data_Rotations_DM/PD%s/PD%s_CM_Rot.npy' % (PD,PD))
rotInfo = np.load(rotPath)
dim = 5 #ensure this is set equal to `dim` used in `Rotation_Histograms.py`

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
        CMrotInfo = rotInfo[0] #change to match CM of interest
        optAngle = CMrotInfo[1]
        v1 = int(CMrotInfo[3]-1) #2D subspace's 1st eigenvector (one-indexing)
        v2 = int(CMrotInfo[4]-1) #2D subspace's 2nd eigenvector (one-indexing)
        # =====================================================================
        # Use previously-defined optimal theta to rotate N-dim manifold into place
        # =====================================================================
        theta_total = int(float(dim)*(float(dim)-1.)/2.)
        thetas = np.zeros(shape=(theta_total,1), dtype=float)
        thetas[dim-1] = optAngle*(np.pi/180)
        R = Generate_Nd_Rot.genNdRotations(dim, thetas)
        U_rot = np.matmul(R, U_init.T)
        U = U_rot.T
    else: #no rotations applied
        U = U_init

    if 1: #view an organized array of 2D subspaces
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
                        plt.scatter(U[:,v1-1], U[:,v2-1], c=enum, cmap='nipy_spectral', s=20, linewidths=.5, edgecolor='k')
                    else:
                        plt.scatter(U[:,v1-1], U[:,v2-1], c='white', s=20, linewidths=.5, edgecolor='k')
                except:
                    plt.scatter(0,0)
                if PCA is True:
                    plt.xlabel(r'$PC_{%s}$' % (int(v1)), fontsize=8, labelpad=5)
                    plt.ylabel(r'$PC_{%s}$' % (int(v2)), fontsize=8, labelpad=2.5)
                else:
                    plt.xlabel(r'$\Psi_{%s}$' % (int(v1)), fontsize=8, labelpad=5)
                    plt.ylabel(r'$\Psi_{%s}$' % (int(v2)), fontsize=8, labelpad=2.5)
                plt.ticklabel_format(style='sci', axis='x', scilimits=(0,0))
                plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0)) 
                idx+=1
        plt.tight_layout()
        plt.show()
        #plt.subplots_adjust(left=0.02, right=0.99, bottom=0.05, top=0.99, wspace=0.26, hspace=0.23)
    
