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
import mrcfile
from sklearn.preprocessing import StandardScaler

# =============================================================================
# Calculate pairwise Euclidean distances between all images in each PD
# =============================================================================
# SETUP: First, make sure all input file paths are correct for your datasets...
#   ...in 'dataDir' and 'dataPath' variables below. You may want to edit...
#   ...output names such as those that include `tau` (here, set to 5 as default)
# RUNNING: To run a series of PDs at once: first edit `1_Dist_Batch.sh`...
#   ...for the total number of PDs requested; e.g., {1...5} for 5 PDs...
#   ...or {1...1} for only the first PD;
#   ...then start batch processing via `sh 1_Dist_Batch.sh`
# =============================================================================
# Author:    E. Seitz @ Columbia University - Frank Lab - 2020
# Contact:   evan.e.seitz@gmail.com
# =============================================================================

def op(pyDir, PD):
    parDir1 = os.path.abspath(os.path.join(pyDir, os.pardir))
    parDir2 = os.path.abspath(os.path.join(parDir1, os.pardir))
    #dataDir = os.path.join(parDir2, '0_Data_Inputs/2_PDs_2D') #for noiselss datasets
    dataDir = os.path.join(parDir2, '0_Data_Inputs/3_PDs_2D_SNR_tau')
    outDir = os.path.join(pyDir, 'Data_Distances')
    
    # =========================================================================
    # Import image stack per PD and standardize
    # =========================================================================
    #dataPath = os.path.join(dataDir, 'PD_%s.mrcs' % PD) #for noiseless datasets
    dataPath = os.path.join(dataDir, 'PD%s_SNR_tau5_stack.mrcs' % PD)
    init_stack = mrcfile.mmap(dataPath)
    ss, box, box = init_stack.data.shape
    
    # Note: mean image subtraction is enough (center the values to 0)...
    #...since values have the same scale to begin with (0-255):
    norm_stack = np.ndarray(shape=(ss, box, box), dtype=float)
    for i in range(0,ss):
        image = init_stack.data[i]/1.
        image -= image.mean()
        #image /= image.std() #see note above
        norm_stack[i,:,:] = image
    
    # =========================================================================
    # Generate distances from images in PD
    # =========================================================================
    print('Computing Distances...')
    Dist = np.ndarray(shape=(ss,ss), dtype=float)
    
    p = 2 #Minkowski distance metric: p1=Manhattan, p2=Euclidean, etc.
    for i in range(0,ss):
        for j in range(0,ss):
            print(i,j)
            Dist[i,j] = (np.sum(np.abs((norm_stack[i,:,:]-norm_stack[j,:,:]))**p) / norm_stack[i,:,:].size)**(1./p)
       
    np.save(os.path.join(outDir, 'PD%s_tau5_dist.npy' % PD), Dist)
        
    if 0:
        plt.imshow(Dist, origin='lower', interpolation='nearest', cmap='jet')
        plt.colorbar()
        plt.tight_layout()
        plt.show()
        
if __name__ == '__main__':
    path1 = os.path.splitext(sys.argv[0])[0]
    path2, tail = os.path.split(path1)
    op(path2, sys.argv[1])
    