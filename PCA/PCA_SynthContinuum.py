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
import imageio
import mrcfile
from sklearn.preprocessing import StandardScaler

'''
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
Copyright (c) CU, Evan Seitz 2019-2020
Contact: evan.e.seitz@gmail.com
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
'''

pyDir = os.path.dirname(os.path.abspath(__file__)) #python file location
parDir1 = os.path.abspath(os.path.join(pyDir, os.pardir))
parDir2 = os.path.abspath(os.path.join(parDir1, os.pardir))
parDir3 = os.path.abspath(os.path.join(parDir2, os.pardir))

# =============================================================================
# choose dataset:
# =============================================================================

#dataDir = os.path.join(parDir, 'Data/1_SS2_PDBs')
#dataDir = os.path.join(parDir, 'Data/2_SS2_MRC')

# =============================================================================
# Import data into array
# =============================================================================

if 1: #Hsp90 synthetic data (PDs)
    dataDir = os.path.join(pyDir, 'projection_stacks')
    m = 400
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
        
# =============================================================================
# Prepare data
# =============================================================================

#stack2 = np.ndarray(shape=(m,boxSize**2), dtype=float)
#for i in range(0,m):
    #stack2[i] = stack[i].flatten()
    
print('image stack dim:', np.shape(stack2))
X_std = StandardScaler().fit_transform(stack2)

# =============================================================================
# OPTION 1: covariance matrix -> np.linalg.eig()
# =============================================================================

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
        fig = plt.subplots()
        plt.subplot(1,1,1)
        plt.scatter(range(len(s)), s**2)
        plt.title('Eigenvalue spectrum')
        plt.xlim(0,15)
        plt.show()

# =============================================================================
# Sort [eigenvalue, eigenvector] tuples from high to low
# =============================================================================
  
if 0: #sanity check, may show False if `full_matrices=False` above
    for ev in eig_vecs:
        np.testing.assert_array_almost_equal(1.0, np.linalg.norm(ev))
    print('All eigenvectors have same unit length 1')
    
eig_pairs = [(np.abs(eig_vals[i]), u[:,i]) for i in range(len(s))]
eig_pairs.sort()
eig_pairs.reverse()
idx = 0
for i in eig_pairs:
    if idx < 10:
        print(i[0])
    idx += 1
    
# =============================================================================
# Project data into principal components
# =============================================================================

dim = 9 #number of dimensions to consider
W = np.hstack((eig_pairs[i][1].reshape(boxSize**2,1) for i in range(dim)))
Y = X_std.dot(W)

# =============================================================================
# Analysis of embedding
# =============================================================================

enum = np.arange(1,m+1)

if 1:
    fig, ax = plt.subplots()
    ax.scatter(Y[:,0], Y[:,2], c=enum, cmap='gist_rainbow')
    for i, txt in enumerate(enum):
        ax.annotate(txt, (Y[:,0][i], Y[:,2][i]))
    plt.xlabel('PC 1')
    plt.ylabel('PC 3')  
    plt.tight_layout()
    plt.show()
    
if 1:
    v1 = 0
    fig = plt.figure()
    
    plt.subplot(3, 3, 1)
    plt.scatter(Y[:,v1], Y[:,0], c=enum, cmap='gist_rainbow')
    plt.xlabel('PC %s' % (v1+1))
    plt.ylabel('PC 1')  
    
    plt.subplot(3, 3, 2)
    plt.scatter(Y[:,v1], Y[:,1], c=enum, cmap='gist_rainbow')
    plt.xlabel('PC %s' % (v1+1))
    plt.ylabel('PC 2') 

    plt.subplot(3, 3, 3)
    plt.scatter(Y[:,v1], Y[:,2], c=enum, cmap='gist_rainbow')
    plt.xlabel('PC %s' % (v1+1))
    plt.ylabel('PC 3') 

    plt.subplot(3, 3, 4)
    plt.scatter(Y[:,v1], Y[:,3], c=enum, cmap='gist_rainbow')
    plt.xlabel('PC %s' % (v1+1))
    plt.ylabel('PC 4')

    plt.subplot(3, 3, 5)
    plt.scatter(Y[:,v1], Y[:,4], c=enum, cmap='gist_rainbow')
    plt.xlabel('PC %s' % (v1+1))
    plt.ylabel('PC 5') 

    plt.subplot(3, 3, 6)
    plt.scatter(Y[:,v1], Y[:,5], c=enum, cmap='gist_rainbow')
    plt.xlabel('PC %s' % (v1+1))
    plt.ylabel('PC 6') 

    plt.subplot(3, 3, 7)
    plt.scatter(Y[:,v1], Y[:,6], c=enum, cmap='gist_rainbow')
    plt.xlabel('PC %s' % (v1+1))
    plt.ylabel('PC 7')

    plt.subplot(3, 3, 8)
    plt.scatter(Y[:,v1], Y[:,7], c=enum, cmap='gist_rainbow')
    plt.xlabel('PC %s' % (v1+1))
    plt.ylabel('PC 8')
    
    plt.subplot(3, 3, 9)
    plt.scatter(Y[:,v1], Y[:,8], c=enum, cmap='gist_rainbow')
    plt.xlabel('PC %s' % (v1+1))
    plt.ylabel('PC 9')
    
    plt.tight_layout()
    plt.show()

if 1:
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    plt.scatter(Y[:,0], Y[:,1], Y[:2], c=enum, cmap='gist_rainbow')
    ax.set_xlabel('PC 1')
    ax.set_ylabel('PC 2')
    ax.set_zlabel('PC 3')
    plt.show()
    

    
