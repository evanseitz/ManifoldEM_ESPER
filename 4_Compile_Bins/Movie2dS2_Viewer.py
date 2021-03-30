import os, sys
import mrcfile
import matplotlib
from matplotlib import rc
#matplotlib.rc('text', usetex = True)
from matplotlib.figure import Figure
import matplotlib.pyplot as plt
from pylab import imshow, show, loadtxt, axes
import numpy as np
from matplotlib.ticker import MaxNLocator
import imageio

# =============================================================================
# Render movie of a CM as seen across all PDs (run via 'python Gen2DMovieS2.py') 
# Movies can only be generated after `1_Compile_Bins.py` has been run...
# ...change CM variable below under user parameters.
# =============================================================================
# Author:    E. Seitz @ Columbia University - Frank Lab - 2020
# Contact:   evan.e.seitz@gmail.com
# =============================================================================

pyDir = os.path.dirname(os.path.abspath(__file__))

# =============================================================================
# User parameters
# =============================================================================
totalPDs = 126
box = 250
bins = 20
CM = 2

stackDir = os.path.join(pyDir, 'S2_bins/CM%s' % CM)
final_stack = np.ndarray(shape=(126*20,box,box))
stackPaths = []
for file in os.listdir(stackDir):
    if file.endswith(".mrcs"):
        stackPaths.append(os.path.join(stackDir, file))

idx = 0
for pd in range(0,totalPDs):
    print('PD:', pd+1)
    for b in range(0,bins):
        init_stack = mrcfile.mmap(stackPaths[b])
        final_stack[idx] = init_stack.data[pd]
        if 0:
            plt.imshow(final_stack[idx])
            plt.show()
        idx += 1

imageio.mimsave(os.path.join(pyDir,'S2_bins/CM%s_S2.gif' % CM), final_stack)