import sys, os, re
import numpy as np
from numpy import linalg as LA
from scipy.spatial.distance import pdist, cdist, squareform
import pandas as pd
import matplotlib
from matplotlib import rc
from matplotlib.figure import Figure
import matplotlib.pyplot as plt
from pylab import imshow, show, loadtxt, axes
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.ticker import MaxNLocator
import imageio
import mrcfile

# ====================================================================================
# Python Image Stack Distance Calculator (run via 'python PDs_Distance_Calculator.py') 
# Author:    E. Seitz @ Columbia University - Frank Lab - 2019-2020
# Contact:   evan.e.seitz@gmail.com
# ====================================================================================

pyDir = os.path.dirname(os.path.abspath(__file__)) #python file location
parDir = os.path.abspath(os.path.join(pyDir, os.pardir))
dataDir = os.path.join(parDir, 'Datasets/3_Projections_1D') #folder with all .mrcs files
sys.path.insert(0, parDir)
import GenNoise

# =============================================================================
# Load in images from image stack for a given projection direction (PD)
# =============================================================================
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

m = len(dataPaths) #number of images to consider from distance matrix; m <= len(dataPaths); e.g., m=20 for 1D motion
Dist = np.ndarray(shape=(m,m), dtype=float)

frames = np.ndarray(shape=(m,boxSize,boxSize))
new_m = 0
for i in range(m):
#for i in [1,2,3]: #to compare manifold of "full" state space with partial (for experimenting)
    print(dataPaths[i])
    stack = mrcfile.open(dataPaths[i])
    
    if 0: #optional: add noise to each image
        frames[new_m] = GenNoise.op(stack.data[PD], 0.1) #SNR = 0.1 for experimental regime
    else:
        frames[new_m] = stack.data[PD]
        
    if 0:
        if i == m-1: #save last image from stack to file
            np.save('PD_%s_frame_%s.npy' % (PD,i+1), frames[i])
            matplotlib.image.imsave('PD_%s_frame_%s.png' % (PD,i+1), frames[i])

    if 0: #plot each frame sequentially
        if i < 2: #number of frames to plot
            plt.imshow(frames[i], cmap='gray')
            plt.show()
            if 0:
                plt.hist(frames[i],bins=100)
                plt.show()
                print('image mean:', np.mean(frames[i]))
                print('image std:', np.std(frames[i]))
    stack.close()
    new_m += 1  
m = new_m #new_m will only change if comparing to partial state space (above)
    
if 0: #save gif
    imageio.mimsave('SS1_PD_%s.gif' % PD, frames)
    
# =============================================================================
# Generate distances from images in PD
# =============================================================================
if 1: #manual distance calculation
    p = 2 #Minkowski distance metric: p1=Manhattan, p2=Euclidean, etc.
    for i in range(0,m):
        for j in range(0,m):
            Dist[i,j] = (np.sum(np.abs((frames[i]-frames[j]))**p))# / frames[i].size)**(1./p)
    
else: #or use scipy library for distance metric:
    stack2 = np.ndarray(shape=(m,boxSize**2), dtype=float)
    for i in range(0,m):
        stack2[i] = frames[i].flatten()
    Dist = cdist(stack2, stack2, 'euclidean')
    # other options: ('sqeuclidean'), (minkowski', p=2.), ('cityblock'), ('cosine'), ('correlation'),
                    #('chebyshev'), (canberra'), ('braycurtis')
   
if 1: #save distance matrix for subsequent use
    np.save('Dist_SS1_PD%s_test.npy' % PD, Dist)
    
if 0:
    plt.imshow(Dist, origin='lower', interpolation='nearest', cmap='jet')
    plt.colorbar()
    plt.tight_layout()
    plt.show()
