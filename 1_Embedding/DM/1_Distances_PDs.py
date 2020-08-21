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
# Euclidean Distance Calculator (run via 'python PDs_Distance_Calculator.py') 
# Author:    E. Seitz @ Columbia University - Frank Lab - 2019-2020
# Contact:   evan.e.seitz@gmail.com
# =============================================================================
PD = '001'
tau = 1 #needs to match value used in inputs

pyDir = os.path.dirname(os.path.abspath(__file__)) #python file location
parDir1 = os.path.abspath(os.path.join(pyDir, os.pardir))
parDir2 = os.path.abspath(os.path.join(parDir1, os.pardir))
dataDir = os.path.join(parDir2, '0_Data_Inputs/2_PDs_2D')
outDir = os.path.join(pyDir, 'Data_Distances')

# =============================================================================
# Import image stack per PD and standardize
# =============================================================================
dataPath = os.path.join(dataDir, 'PD_%s.mrcs' % PD)
init_stack = mrcfile.mmap(dataPath)
ss, box, box = init_stack.data.shape

# Note: mean image subtraction is enough (center the values to 0)...
#...since values have the same scale to begin with (0-255):
norm_stack = np.ndarray(shape=(ss, box, box), dtype=float)
for i in range(0,ss):
    image = init_stack.data[i]/1.
    image -= image.mean()
    #image /= image.std() 
    norm_stack[i,:,:] = image

# =============================================================================
# Generate distances from images in PD
# =============================================================================
print('Computing Distances...')
Dist = np.ndarray(shape=(ss,ss), dtype=float)

p = 2 #Minkowski distance metric: p1=Manhattan, p2=Euclidean, etc.
for i in range(0,ss):
    for j in range(0,ss):
        print(i,j)
        Dist[i,j] = (np.sum(np.abs((norm_stack[i,:,:]-norm_stack[j,:,:]))**p) / norm_stack[i,:,:].size)**(1./p)
   
np.save(os.path.join(outDir, 'PD%s_tau%s_dist.npy' % (PD, tau)), Dist)
    
if 1:
    plt.imshow(Dist, origin='lower', interpolation='nearest', cmap='jet')
    plt.colorbar()
    plt.tight_layout()
    plt.show()
    