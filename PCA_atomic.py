import sys, os, re
import numpy as np
from numpy import linalg as LA
import pandas as pd
import matplotlib
from matplotlib import rc
#matplotlib.rc('text', usetex = True)
from matplotlib.figure import Figure
import matplotlib.pyplot as plt
from pylab import imshow, show, loadtxt, axes
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.ticker import MaxNLocator
from matplotlib.pyplot import cm
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from Bio.PDB import PDBParser

# ====================================================================================
# Diffusion Maps (run via 'python DM_SyntheticContinuum.py') 
# Authors:    E. Seitz @ Columbia University - Frank Lab - 2020
#             H. Liao @ Columbia University - Frank Lab - 2020 
# Contact:   evan.e.seitz@gmail.com
# ====================================================================================

pyDir = os.path.dirname(os.path.abspath(__file__)) #python file location
dataDir = os.path.join(pyDir, 'Datasets')

# =============================================================================
# Import data into array
# =============================================================================

p = PDBParser()
N = 20
for i in range(N):
    for j in range(N):
        ind = i*N+j
        print('ind:',ind)
        name = os.path.join(dataDir, '1_Atomic_2D/state_{:02d}_{:02d}.pdb'.format(i+1,j+1))
        struc = p.get_structure('X', name)
        k = 0
        if ind == 0:  # only at the beginning
            for atom in struc.get_atoms():
                k+=1
            data = np.zeros((N*N, 3*k))
            dataSize = 3*k
            print('atoms:',k)
        # iterating:
        k = 0
        for atom in struc.get_atoms():
            data[ind, k:k+3] = atom.get_coord()
            k = k + 1

X_std = StandardScaler().fit_transform(data)

# =============================================================================
# Eigendecomposition
# =============================================================================
    
print('Computing SVD...')
u,s,v = np.linalg.svd(X_std.T, full_matrices=False)
print('SVD complete')
eig_vals = s**2
eig_vecs = u
print('Eigenvector shape:', np.shape(eig_vecs)) #i.e., [k*3, ind+1]

if 1:
    fig = plt.subplots()
    plt.subplot(1,1,1)
    plt.scatter(range(len(s)), s**2)
    plt.title('Eigenvalue spectrum')
    plt.xlim(0,15)
    plt.show()

# =============================================================================
# Project data into principal components
# =============================================================================

dim = 9 #number of dimensions to consider
W = np.hstack((eig_vecs[:,i].reshape(dataSize,1) for i in range(dim)))
print('W shape:',np.shape(W))
Y = X_std.dot(W)

if 0:
    np.save('PCA_val_atom.npy', eig_vals)
    np.save('PCA_vec_atom.npy', eig_vecs)

# =============================================================================
# Analysis of embedding
# =============================================================================

enum = np.arange(1,ind+1)

if 1:
    fig, ax = plt.subplots()
    ax.scatter(Y[:,0], Y[:,1], c=enum, cmap='gist_rainbow')
    for i, txt in enumerate(enum):
        ax.annotate(txt, (Y[:,0][i], Y[:,1][i]))
    plt.xlabel('PC 1', fontsize=20)
    plt.ylabel('PC 2', fontsize=20)
    plt.xlim(np.amin(Y[:,0])-np.amax(Y[:,0])*.1, np.amax(Y[:,0])+np.amax(Y[:,0])*.1)
    plt.ylim(np.amin(Y[:,1])-np.amax(Y[:,1])*.1, np.amax(Y[:,1])+np.amax(Y[:,1])*.1)
    plt.tight_layout()
    plt.show()
    
if 1:
    v1 = 0
    fig = plt.figure()
    
    plt.subplot(3, 3, 1)
    plt.scatter(Y[:,v1], Y[:,0], c=enum, cmap='gist_rainbow')
    plt.xlabel('PC %s' % (v1+1))
    plt.ylabel('PC 1')    
    plt.xlim(np.amin(Y[:,v1])-np.amax(Y[:,v1])*.1, np.amax(Y[:,v1])+np.amax(Y[:,v1])*.1)
    plt.ylim(np.amin(Y[:,0])-np.amax(Y[:,0])*.1, np.amax(Y[:,0])+np.amax(Y[:,0])*.1)
    
    plt.subplot(3, 3, 2)
    plt.scatter(Y[:,v1], Y[:,1], c=enum, cmap='gist_rainbow')
    plt.xlabel('PC %s' % (v1+1))
    plt.ylabel('PC 2')
    plt.xlim(np.amin(Y[:,v1])-np.amax(Y[:,v1])*.1, np.amax(Y[:,v1])+np.amax(Y[:,v1])*.1)
    plt.ylim(np.amin(Y[:,1])-np.amax(Y[:,1])*.1, np.amax(Y[:,1])+np.amax(Y[:,1])*.1)

    plt.subplot(3, 3, 3)
    plt.scatter(Y[:,v1], Y[:,2], c=enum, cmap='gist_rainbow')
    plt.xlabel('PC %s' % (v1+1))
    plt.ylabel('PC 3')
    plt.xlim(np.amin(Y[:,v1])-np.amax(Y[:,v1])*.1, np.amax(Y[:,v1])+np.amax(Y[:,v1])*.1)
    plt.ylim(np.amin(Y[:,2])-np.amax(Y[:,2])*.1, np.amax(Y[:,2])+np.amax(Y[:,2])*.1)

    plt.subplot(3, 3, 4)
    plt.scatter(Y[:,v1], Y[:,3], c=enum, cmap='gist_rainbow')
    plt.xlabel('PC %s' % (v1+1))
    plt.ylabel('PC 4')
    plt.xlim(np.amin(Y[:,v1])-np.amax(Y[:,v1])*.1, np.amax(Y[:,v1])+np.amax(Y[:,v1])*.1)
    plt.ylim(np.amin(Y[:,3])-np.amax(Y[:,3])*.1, np.amax(Y[:,3])+np.amax(Y[:,3])*.1)

    plt.subplot(3, 3, 5)
    plt.scatter(Y[:,v1], Y[:,4], c=enum, cmap='gist_rainbow')
    plt.xlabel('PC %s' % (v1+1))
    plt.ylabel('PC 5')
    plt.xlim(np.amin(Y[:,v1])-np.amax(Y[:,v1])*.1, np.amax(Y[:,v1])+np.amax(Y[:,v1])*.1)
    plt.ylim(np.amin(Y[:,4])-np.amax(Y[:,4])*.1, np.amax(Y[:,4])+np.amax(Y[:,4])*.1)

    plt.subplot(3, 3, 6)
    plt.scatter(Y[:,v1], Y[:,5], c=enum, cmap='gist_rainbow')
    plt.xlabel('PC %s' % (v1+1))
    plt.ylabel('PC 6')
    plt.xlim(np.amin(Y[:,v1])-np.amax(Y[:,v1])*.1, np.amax(Y[:,v1])+np.amax(Y[:,v1])*.1)
    plt.ylim(np.amin(Y[:,5])-np.amax(Y[:,5])*.1, np.amax(Y[:,5])+np.amax(Y[:,5])*.1)

    plt.subplot(3, 3, 7)
    plt.scatter(Y[:,v1], Y[:,6], c=enum, cmap='gist_rainbow')
    plt.xlabel('PC %s' % (v1+1))
    plt.ylabel('PC 7')
    plt.xlim(np.amin(Y[:,v1])-np.amax(Y[:,v1])*.1, np.amax(Y[:,v1])+np.amax(Y[:,v1])*.1)
    plt.ylim(np.amin(Y[:,6])-np.amax(Y[:,6])*.1, np.amax(Y[:,6])+np.amax(Y[:,6])*.1)

    plt.subplot(3, 3, 8)
    plt.scatter(Y[:,v1], Y[:,7], c=enum, cmap='gist_rainbow')
    plt.xlabel('PC %s' % (v1+1))
    plt.ylabel('PC 8')
    plt.xlim(np.amin(Y[:,v1])-np.amax(Y[:,v1])*.1, np.amax(Y[:,v1])+np.amax(Y[:,v1])*.1)
    plt.ylim(np.amin(Y[:,7])-np.amax(Y[:,7])*.1, np.amax(Y[:,7])+np.amax(Y[:,7])*.1)
    
    plt.subplot(3, 3, 9)
    plt.scatter(Y[:,v1], Y[:,8], c=enum, cmap='gist_rainbow')
    plt.xlabel('PC %s' % (v1+1))
    plt.ylabel('PC 9')
    plt.xlim(np.amin(Y[:,v1])-np.amax(Y[:,v1])*.1, np.amax(Y[:,v1])+np.amax(Y[:,v1])*.1)
    plt.ylim(np.amin(Y[:,8])-np.amax(Y[:,8])*.1, np.amax(Y[:,8])+np.amax(Y[:,8])*.1)
    
    plt.tight_layout()
    plt.show()

if 1:
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    color=cm.gist_rainbow(np.linspace(0,1,ind+1))
    for i,c in zip(range(ind+1),color):
        ax.scatter(Y[i,0], Y[i,1], Y[i,2], c=c, cmap='gist_rainbow')
    ax.set_xlabel('PC 1')
    ax.set_ylabel('PC 2')
    ax.set_zlabel('PC 3')
    plt.show()
    

    
