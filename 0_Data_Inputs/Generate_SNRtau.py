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
import imageio
import mrcfile
import AddNoise

# =============================================================================
# Generate noisy duplicates of images (run via 'sh 3_SNRtau_Batch.py') 
# Author:    E. Seitz @ Columbia University - Frank Lab - 2020
# Contact:   evan.e.seitz@gmail.com
# =============================================================================

def op(pyDir, PD):
    dataDir = os.path.join(pyDir, '2_PDs_2D') #location of input stacks
    outStackDir = os.path.join(pyDir, '3_PDs_2D_SNR_tau')
    
    # =============================================================================
    # User parameters
    # =============================================================================
    apply_noise = True #if True, make sure to set SNR immediatey below (default 0.1)
    SNR = 0.1 #use for emulating experimental cryo-EM regime
    tau = 5 #number of noisy (or non-noisy) duplicates requested per state
    
    # =============================================================================
    # Import images and create stack with noisy duplicates (or without noisy...
    # ...duplicates, but still standardize all images and save in new stack)
    # =============================================================================
    dataPath = os.path.join(dataDir, 'PD_%s.mrcs' % PD)
    init_stack = mrcfile.mmap(dataPath)
    ss, box, box = init_stack.data.shape
    total = ss*tau
    print('Snapshots: %s; Box Size: %s, New Total: %s' % (ss, box, total))
    
    if 0: #check input data
        plt.imshow(init_stack.data[0], cmap=plt.get_cmap(name='gray'))
        plt.show()
    
    if apply_noise is True:
        stackOut = os.path.join(outStackDir, 'PD%s_SNR_tau%s_stack.mrcs' % (PD, tau))
    else:
        stackOut = os.path.join(outStackDir, 'PD%s_tau%s_stack.mrcs' % (PD, tau))
    if os.path.exists(stackOut):
        os.remove(stackOut)
    final_stack = mrcfile.new_mmap(stackOut, shape=(total,box,box), mrc_mode=2, overwrite=True) #mrc_mode 2: float32
    
    imgIdx = 0
    for i in range(ss):
        for j in range(tau):
            print('img:',imgIdx)
            if apply_noise is True: #generate noisy duplicates with unique noise for each image
                final_stack.data[imgIdx] = AddNoise.op(init_stack.data[i], SNR)
            else: #no noise added, but tau-duplicate still created
                final_stack.data[imgIdx] = init_stack.data[i]
            imgIdx += 1
            
    if 1: #save gif
        if apply_noise is True:
            imageio.mimsave(os.path.join(outStackDir, 'PD%s_SNR_tau%s_movie.gif' % (PD, tau)), final_stack.data)
        else:
            imageio.mimsave(os.path.join(outStackDir, 'PD%s_tau%s_movie.gif' % (PD, tau)), final_stack.data)
    
    init_stack.close()
    
if __name__ == '__main__':
    path1 = os.path.splitext(sys.argv[0])[0]
    path2, tail = os.path.split(path1)
    op(path2, sys.argv[1])