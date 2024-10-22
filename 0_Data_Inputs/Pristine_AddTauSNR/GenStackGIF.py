import os
import matplotlib
from matplotlib import rc
from matplotlib.figure import Figure
import matplotlib.pyplot as plt
from pylab import imshow, show
import imageio
import mrcfile

# =============================================================================
# Generate GIF of image stack
# Author:    E. Seitz @ Columbia University - Frank Lab - 2020
# Contact:   evan.e.seitz@gmail.com
# =============================================================================

parDir = os.path.abspath(os.path.join(pyDir, os.pardir))
pyDir = os.path.dirname(os.path.abspath(__file__)) #python file location
dataDir = os.path.join(parDir, 'Pristine_2D') #location of input stacks
PD = '001'

dataPath = os.path.join(dataDir, 'PD_%s.mrcs' % PD)
init_stack = mrcfile.mmap(dataPath)
ss, box, box = init_stack.data.shape
total = ss
print('Snapshots: %s; Box Size: %s, New Total: %s' % (ss, box, total))

if 0: #check input data
    plt.imshow(init_stack.data[0], cmap=plt.get_cmap(name='gray'))
    plt.show()
        
if 1: #save gif
    imageio.mimsave(os.path.join(dataDir, 'PD%s_movie.gif' % PD), init_stack.data)

init_stack.close()
    
