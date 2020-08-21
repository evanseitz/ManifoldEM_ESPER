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
dataDir = os.path.join(pyDir, 'Data_Manifolds')
if 0: #Times font for all figures
    rc('text', usetex=True)
    rc('font', family='serif')

# =============================================================================
# Import data into arrays
# =============================================================================
E = np.load(os.path.join(dataDir,'PD001_tau5_val.npy')) #eigenvalues
Y = np.load(os.path.join(dataDir,'PD001_tau5_vec.npy')) #eigenvectors
m = np.shape(Y)[0] #number of images for colormap (number of initial states*tau)

# =============================================================================
# Analysis of embedding
# =============================================================================
enum = np.arange(1,m+1)
cmap = 'nipy_spectral' #'gist_rainbow'
s = 20
lw = .5

if 1: #view eigenvalue spectrum
    x = range(1,len(E)+1)
    plt.scatter(x, E)
    plt.title('Eigenvalue Spectrum')
    plt.xlabel(r'$PC$')
    plt.ylabel(r'$\mathrm{\lambda}$', rotation=0)
    plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
    plt.xlim([0,15])
    plt.ylim(-E[0]/8., E[0]+E[0]/8.)
    plt.locator_params(nbins=15)
    plt.axhline(y=0, color='k', alpha=.5, linestyle='--', linewidth=1)
    plt.show()

if 1: #view a single 2D subspace
    fig, ax = plt.subplots()
    ax.scatter(Y[:,0], Y[:,1], c=enum, cmap=cmap, s=s, linewidths=lw, edgecolor='k')
    if 0:
        for i, txt in enumerate(enum):
            ax.annotate(txt, (Y[:,0][i], Y[:,1][i]))
    plt.xlabel(r'$PC_1$', fontsize=20)
    plt.ylabel(r'$PC_2$', fontsize=20)
    plt.xlim(np.amin(Y[:,0])-np.amax(Y[:,0])*.1, np.amax(Y[:,0])+np.amax(Y[:,0])*.1)
    plt.ylim(np.amin(Y[:,1])-np.amax(Y[:,1])*.1, np.amax(Y[:,1])+np.amax(Y[:,1])*.1)
    plt.tight_layout()
    plt.show()
    
if 1: #view an organized array of 2D subspaces
    fig = plt.figure()
    dimRows, dimCols = 4, 5
    idx = 1
    plt.rc('font', size=6)

    for v1 in range(1,dimRows+1):
        for v2 in range(v1+1, v1+dimCols+1):
            plt.subplot(dimRows, dimCols, idx)
            plt.scatter(Y[:,v1-1], Y[:,v2-1], c=enum, cmap=cmap, s=s, linewidths=lw, edgecolor='k')
            plt.xlabel(r'$PC_{%s}$' % (int(v1)), fontsize=8, labelpad=5)
            plt.ylabel(r'$PC_{%s}$' % (int(v2)), fontsize=8, labelpad=2.5)
            plt.ticklabel_format(style='sci', axis='x', scilimits=(0,0))
            plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0)) 
            idx+=1
    plt.tight_layout()
    plt.show()

if 1: #view a single 3D subspace
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    color=cm.gist_rainbow(np.linspace(0,1,m+1))
    for i,c in zip(range(m),color):
        ax.scatter(Y[i,0], Y[i,1], Y[i,2], c=c, cmap=cmap, s=s, linewidths=lw, edgecolor='k')
    ax.set_xlabel('$PC_1$')
    ax.set_ylabel('$PC_2$')
    ax.set_zlabel('$PC_3$')
    plt.show()