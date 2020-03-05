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
import mrcfile
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from Bio.PDB import PDBParser

# ====================================================================================
# Diffusion Maps (run via 'python DM_SyntheticContinuum.py') 
# Authors:    E. Seitz @ Columbia University - Frank Lab - 2020
# Contact:   evan.e.seitz@gmail.com
# ====================================================================================

pyDir = os.path.dirname(os.path.abspath(__file__)) #python file location
dataDir = os.path.join(pyDir, 'Datasets')

# =============================================================================
# Import data into array (images from PD)
# =============================================================================

m = 400 #number of states
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
m = len(dataPaths) #number of images to consider

frames = np.ndarray(shape=(m,boxSize,boxSize))
for i in range(m):
    print(dataPaths[i])
    stack = mrcfile.open(dataPaths[i])
    frames[i] = stack.data[PD]
    if 0: #plot each frame sequentially
        if i < 2: #number of frames to plot
            plt.imshow(frames[i], cmap = plt.get_cmap(name = 'gray'))
            plt.show()
    stack.close()

stack2 = np.ndarray(shape=(m,boxSize**2), dtype=float)
for i in range(0,m):
    stack2[i] = frames[i].flatten()

print('image stack dim:', np.shape(stack2))
X_std = StandardScaler().fit_transform(stack2)

# =============================================================================
# OPTION 1: covariance matrix -> np.linalg.eig()
# =============================================================================

## MUCH SLOWER!
if 0:
    #cov = np.cov(stack2.T) #not normalized
    covNorm = np.cov(X_std.T) #normalized
    
    if 0:
        fig = plt.subplots()
        plt.subplot(1,2,1)
        plt.imshow(covNorm)
        plt.title('Covariance matrix')    

    print('Computing eigendecomposition...')
    eig_vals, eig_vecs = np.linalg.eig(covNorm)
    print('Eigendecomposition complete')
    
    if 1:
        fig = plt.subplots()
        plt.subplot(1,1,1)
        plt.scatter(range(len(eig_vals)), eig_vals)
        plt.title('Eigenvalue spectrum')
        plt.xlim(0,15)

# =============================================================================
# OPTION 2: SVD via np.linalg.svd()
# =============================================================================
   
else:
    print('Computing SVD...')
    u,s,v = np.linalg.svd(X_std.T, full_matrices=False)
    print('SVD complete')
    eig_vals = s**2
    eig_vecs = u

    if 1:
        idx = 0
        for i in eig_vals:
            if idx < 10:
                print(i)
            idx += 1
            
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
W = np.hstack((eig_vecs[:,i].reshape(boxSize**2,1) for i in range(dim)))
Y = X_std.dot(W)

if 0:
    np.save('PCA_val_image.npy', eig_vals)
    np.save('PCA_vec_image.npy', eig_vecs)

# =============================================================================
# Analysis of embedding
# =============================================================================

enum = np.arange(1,m+1)

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
    color=cm.gist_rainbow(np.linspace(0,1,m+1))
    for i,c in zip(range(m),color):
        ax.scatter(Y[i,0], Y[i,1], Y[i,2], c=c, cmap='gist_rainbow')
    ax.set_xlabel('PC 1')
    ax.set_ylabel('PC 2')
    ax.set_zlabel('PC 3')
    plt.show()