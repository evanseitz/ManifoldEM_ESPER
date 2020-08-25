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
#   ...via the 'dataPath' variable below. You may also want to edit output...
#   ...names such as those that include `tau` (here, set to 5 as default)
# RUNNING: To run a series of PDs at once: first edit `1_PCA_Batch.sh`...
#   ...for the total number of PDs requested; e.g., {1...5} for 5 PDs...
#   ...or {1...1} for only the first PD;
#   ...then start batch processing via `sh 1_PCA_Batch.sh`
# =============================================================================
# Author:    E. Seitz @ Columbia University - Frank Lab - 2020
# Contact:   evan.e.seitz@gmail.com
# =============================================================================

def op(pyDir, PD):
    parDir1 = os.path.abspath(os.path.join(pyDir, os.pardir))
    parDir2 = os.path.abspath(os.path.join(parDir1, os.pardir))
    dataDir = os.path.join(parDir2, '0_Data_Inputs/3_PDs_2D_SNR_tau')
    outDir = os.path.join(pyDir, 'Data_Manifolds')
    print('PD:',PD)
    
    # =========================================================================
    # Import image stack per PD and standardize
    # =========================================================================
    dataPath = os.path.join(dataDir, 'PD%s_SNR_tau5_stack.mrcs' % PD)
    init_stack = mrcfile.mmap(dataPath)
    ss, box, box = init_stack.data.shape
        
    flat_stack = np.ndarray(shape=(ss, box**2), dtype=float)  
    for i in range(0,ss):
        flat_stack[i] = init_stack.data[i].flatten()
        
    print('Flattened stack dim:', np.shape(flat_stack))
    X_std = StandardScaler().fit_transform(flat_stack)
        
    print('Computing SVD...')
    u,s,v = np.linalg.svd(X_std.T, full_matrices=False)
    print('SVD complete')
    eig_vals = s**2
    eig_vecs = u
       
    # =========================================================================
    # Project data into principal components
    # =========================================================================
    dim = 15 #number of dimensions to consider
    W = np.hstack((eig_vecs[:,i].reshape(box**2,1) for i in range(dim)))
    Y = X_std.dot(W)
    
    np.save(os.path.join(outDir, 'PD%s_tau5_vec.npy' % (PD), Y)
    np.save(os.path.join(outDir, 'PD%s_tau5_val.npy' % (PD), eig_vals)
        
    init_stack.close()

if __name__ == '__main__':
    path1 = os.path.splitext(sys.argv[0])[0]
    path2, tail = os.path.split(path1)
    op(path2, sys.argv[1])