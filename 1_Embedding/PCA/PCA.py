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
import mrcfile
from sklearn.preprocessing import StandardScaler

# =============================================================================
# Embed each PD image stack via PCA framework
# =============================================================================
# SETUP: First, make sure the input file path is correct for your dataset...
#   ...via the 'dataDir' and 'dataPath' variables below. You may also want to...
#   ...edit output names or the total number of dimensions to consider.
#   ...As well, note that this script has only been constructed to handle...
#   ...data without CTF. For data with CTF, see the DM framework.
# RUNNING: To run a series of PDs at once: first edit '1_PCA_Batch.sh'...
#   ...for the total number of PDs requested; e.g., {1...5} for 5 PDs...
#   ...or {1...1} for only the first PD.
#   ...Then start batch processing via 'sh 1_PCA_Batch.sh'
# =============================================================================
# Author:    E. Seitz @ Columbia University - Frank Lab - 2020
# Contact:   evan.e.seitz@gmail.com
# =============================================================================

def op(pyDir, PD):
    parDir1 = os.path.abspath(os.path.join(pyDir, os.pardir))
    parDir2 = os.path.abspath(os.path.join(parDir1, os.pardir))
    dataDir = os.path.join(parDir2, '0_Data_Inputs/Pristine_2D')
    outDir = os.path.join(pyDir, 'Data_Manifolds')
    if not os.path.exists(outDir):
        os.makedirs(outDir)
    print('PD:',PD)
    
    # =========================================================================
    # Import image stack per PD and standardize
    # =========================================================================
    dataPath = os.path.join(dataDir, 'PD_%s.mrcs' % PD)
    init_stack = mrcfile.mmap(dataPath)
    ss, box, box = init_stack.data.shape
        
    flat_stack = np.ndarray(shape=(ss, box**2), dtype=float)  
    for i in range(0,ss):
        flat_stack[i] = init_stack.data[i].flatten()
        
    print('Flattened stack dim:', np.shape(flat_stack))
    X_std = StandardScaler().fit_transform(flat_stack)
    
    # The two options below perform identically for noisy datasets; if the...
    # ...second option is used, will need to ignore the steady-state...
    # ...(leading) eigenvector, as is similarly done in DM:
    if 1: #standardize each pixel-column individually
        X_std = StandardScaler().fit_transform(flat_stack)
        print(np.mean(X_std))
        print(np.std(X_std))
        
    else: #standardize each image-row individually
        for i in range(ss):
            mu, sigma = flat_stack[i,:].mean(), flat_stack[i,:].std()
            flat_stack[i,:] -= mu
            flat_stack[i,:] /= sigma
        print(np.mean(flat_stack))
        print(np.std(flat_stack))
        X_std = flat_stack
        
    print('Computing SVD...')
    u,s,v = np.linalg.svd(X_std.T, full_matrices=False)
    print('SVD complete')
    eig_vals = s**2
    eig_vecs = u
       
    # =========================================================================
    # Project data into principal components
    # =========================================================================
    dim = 15 #number of dimensions to consider
    W = np.hstack([eig_vecs[:,i].reshape(box**2,1) for i in range(dim)])
    Y = X_std.dot(W)
    
    np.save(os.path.join(outDir, 'PD_%s_vec.npy' % PD), Y)
    np.save(os.path.join(outDir, 'PD_%s_val.npy' % PD), eig_vals)
        
    init_stack.close()

if __name__ == '__main__':
    path1 = os.path.splitext(sys.argv[0])[0]
    path2, tail = os.path.split(path1)
    op(path2, sys.argv[1])