import sys, os
import matplotlib
from matplotlib import rc
from matplotlib.figure import Figure
import matplotlib.pyplot as plt
from pylab import imshow, show
import imageio
import mrcfile
import AddNoise

# =============================================================================
# Generate noisy duplicates of images (run via 'sh 1_TauSNR_Batch.sh')
# =============================================================================
# USAGE: before running, change number of PDs in '1_TauSNR_Batch.sh'.
#   ...Next, alter user parameters below to generate datasets as desired.
# Author:    E. Seitz @ Columbia University - Frank Lab - 2020
# Contact:   evan.e.seitz@gmail.com
# =============================================================================

def op(pyDir, PD):
    parDir = os.path.abspath(os.path.join(pyDir, os.pardir))
    dataDir = os.path.join(parDir, 'Pristine_2D') #location of input stacks
    outStackDir = os.path.join(pyDir, 'Modified_Stacks')
    if not os.path.exists(outStackDir):
        os.mkdir(outStackDir)
    
    # =========================================================================
    # User parameters
    # =========================================================================
    apply_noise = True #if True, make sure to set SNR immediatey below (default 0.1)
    SNR = 0.1 #use for emulating experimental cryo-EM regime
    tau = 10 #number of noisy (or non-noisy) duplicates requested per state
    
    # =========================================================================
    # Import images and create stack with noisy duplicates (or without noisy...
    # ...duplicates, but still standardize all images and save in new stack)
    # =========================================================================
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
            
    if 0: #save gif
        if apply_noise is True:
            imageio.mimsave(os.path.join(outStackDir, 'PD%s_SNR_tau%s_movie.gif' % (PD, tau)), final_stack.data)
        else:
            imageio.mimsave(os.path.join(outStackDir, 'PD%s_tau%s_movie.gif' % (PD, tau)), final_stack.data)
    
    init_stack.close()
    
if __name__ == '__main__':
    path1 = os.path.splitext(sys.argv[0])[0]
    path2, tail = os.path.split(path1)
    op(path2, sys.argv[1])