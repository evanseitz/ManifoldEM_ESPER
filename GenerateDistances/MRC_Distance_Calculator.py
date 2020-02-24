import sys, os, re
import numpy as np
from numpy import linalg as LA
import mrcfile
import matplotlib
from matplotlib import rc
#matplotlib.rc('text', usetex = True)
from matplotlib.figure import Figure
import matplotlib.pyplot as plt
from pylab import imshow, show, loadtxt, axes
import scipy
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.ticker import MaxNLocator

# =============================================================================
# PyMOL RMSD Calculator (run via 'python MRC_Distance_Calculator.py') 
# Author:    E. Seitz @ Columbia University - Frank Lab - 2019-2020 
# =============================================================================

def calc_dist(vol1, vol2):
    dist = LA.norm(vol1 - vol2)
    #return np.divide(dist, np.shape(vol1)[0]**3)
    return dist
    
def natural_sort(l): 
    convert = lambda text: int(text) if text.isdigit() else text.lower() 
    alphanum_key = lambda key: [ convert(c) for c in re.split('([0-9]+)', key) ] 
    return sorted(l, key = alphanum_key)

pyDir = os.getcwd() #python file location, place in '2_GenStates_CM2' (if Hsp2D synthetic data)
CM_dir = os.path.join(pyDir, 'MRCs') #location of 400 MRCs (i.e., entire state space)

MRC_paths0 = []
for root, dirs, files in os.walk(CM_dir):
    for file in sorted(files):
        if not file.startswith('.'): #ignore hidden files
            if file.endswith(".mrc"):
                MRC_paths0.append(os.path.join(root, file))               
MRC_paths = natural_sort(MRC_paths0)

m = 400 #total number of states
states = range(0,m)
D = np.zeros(shape=(m,m), dtype=float)

for i in states:
    for j in states:
        if i < j:
            print('Index [%s, %s]:' % (i,j))
            vol1 = mrcfile.open(MRC_paths[i])
            vol2 = mrcfile.open(MRC_paths[j])
            D[i,j] = float(calc_dist(vol1.data, vol2.data))
            print('\t%s' % D[i,j])
            vol1.close()
            vol2.close()
            
D = D + D.T

if 1: #save to file
    np.save('Dist_2DoF_Volumes.npy', D)
        
if 1:
    plt.imshow(D, origin='lower', interpolation='nearest', cmap='jet')
    plt.colorbar()
    plt.tight_layout()
    plt.show()
        

    
    
