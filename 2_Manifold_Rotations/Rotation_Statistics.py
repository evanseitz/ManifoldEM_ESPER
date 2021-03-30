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

totalPDs = 40 #total number of PDs
CM = 0 #change this to match CM of interest (zero indexing)
dim = 6 #must match 'dim' assignment used in 'Manifold_Binning.py'
theta_total = int(float(dim)*(float(dim)-1.)/2.)

Rij_stats = []
for Rs in range(theta_total):
    Rij_stats.append([])    
    
for pd in range(1, totalPDs+1):
    pd = "{0:0=3d}".format(pd)
    rotMatrices = os.path.join(pyDir, 'Data_Rotations/PD%s/PD%s_RotMatrices.npy' % (pd,pd))
    rotEigenfunctions = os.path.join(pyDir, 'Data_Rotations/PD%s/PD%s_RotEigenfunctions.npy' % (pd,pd))
    rotParameters = os.path.join(pyDir, 'Data_Rotations/PD%s/PD%s_RotParameters.npy' % (pd,pd))
    rotMatrix = np.load(rotMatrices)
    rotEigs = np.load(rotEigenfunctions)
    rotParams = np.load(rotParameters)
    if dim != rotParams[0]:
        print('Dimensionality conflict detected.') #see 'dim' comment above
    v1_list = []
    v2_list = []
    for i in range(len(rotEigs)-1):
        if rotEigs[i] != 0:
            v1_list.append(i+1)
            v2_list.append(int(rotEigs[i]+1))
    thetas = rotMatrix[CM]
    for t in range(theta_total):
        Rij_stats[t].append(thetas[t]*(180/np.pi))

    
for t in range(theta_total):
    plt.subplot(3,int(theta_total/3),t+1) #will need to update figure dimensions and labels based on 'dim' used
    plt.hist(Rij_stats[t], bins=60, edgecolor='k', linewidth=1, color='C0')
    if t > 9:
        plt.xlabel('Angle (degrees)', fontsize=14, labelpad=10)
    #plt.xticks([-45,-30,-15,0,15,30,45])
    plt.title('R$_{%s}$' % (t+1), fontsize=16)
    if t%5 == 0:
        plt.ylabel('Frequency', fontsize=14, labelpad=15)
    plt.tight_layout()

plt.subplots_adjust(left=0.05, bottom=0.075, right=0.95, top=0.95, wspace=0.25, hspace=0.25)
plt.show()