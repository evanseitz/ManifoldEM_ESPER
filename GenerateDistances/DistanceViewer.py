import os, os.path, sys
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from pylab import imshow, show, loadtxt

pyDir = os.path.dirname(os.path.abspath(__file__)) #python file directory

D = np.load('Dist_2DoF_3dRMSD.npy')
plt.imshow(D, origin='lower', interpolation='nearest')
plt.title('RMSD Distances')
plt.xlabel('State (PDB)')
plt.ylabel('State (PDB)')
plt.colorbar()
plt.tight_layout()
plt.show()
