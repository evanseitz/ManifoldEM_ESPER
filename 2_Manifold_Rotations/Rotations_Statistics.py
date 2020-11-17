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
# Data viewer for rotations statistics (run via 'python Rotation_Statistics.py') 
# Used for analysis of distribution of manifold rotation angles once outputs...
# of `Rotation_Histograms.py` have already been generated)
# =============================================================================
# Author:    E. Seitz @ Columbia University - Frank Lab - 2020
# Contact:   evan.e.seitz@gmail.com
# =============================================================================

pyDir = os.path.dirname(os.path.abspath(__file__)) #python file location

if 1: #render with LaTeX font for figures
    rc('text', usetex=True)
    rc('font', family='serif')

totalPDs = 126 #total number of PDs
PCA = True #specify if manifolds from PCA or DM folder {if False, DM is True}
CM = 1 #choose which CM distribution to view; e.g., choose {1,2,3}
radians = False #choose to display angles in radians {True} or degrees {False}

rotations = []
for pd in range(1, totalPDs+1):
    if PCA is True:
        rotPath = os.path.join(pyDir, 'Data_Rotations_PCA/PD%.03d/PD%.03d_CM_Rot.npy' % (pd,pd))
    else:
        rotPath = os.path.join(pyDir, 'Data_Rotations_DM/PD%.03d/PD%.03d_CM_Rot.npy' % (pd,pd))
    rotInfo = np.load(rotPath) #load in rotations to view distribution of angles for all PDs
    
    CMrotInfo = rotInfo[CM-1]
    optAngle = CMrotInfo[1]
    if radians is False:
        rotations.append(optAngle)
    elif radians is True:
        rotations.append(optAngle*(np.pi/180))
         
if 1:
    print('R mean:', np.mean(rotations))
    print('R std:', np.std(rotations))
    print('R range:', np.amax(rotations) - np.amin(rotations))
    print('R max:', np.amax(rotations))
    print('R min:', np.amin(rotations))
    plt.hist(rotations, bins=12 ,edgecolor='k', linewidth=1, color='C0')
    if radians is False:
        plt.xlabel('Angle (degrees)', fontsize=16, labelpad=10)
    elif radians is True:
        plt.xlabel('Angle (radians)', fontsize=16, labelpad=10)
    plt.ylabel('Frequency', fontsize=16, labelpad=10)
    plt.tight_layout()
    plt.show()