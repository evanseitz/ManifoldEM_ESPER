import os, os.path, sys
import numpy as np
import itertools
import matplotlib
import matplotlib.pyplot as plt
from pylab import imshow, show
import mrcfile
import pandas as pd
from decimal import Decimal
from scipy.fftpack import fft2
from scipy.fftpack import ifft2
import ctemh_cryoFrank_generator

# =============================================================================
# Add CTF and noise (SNR) to pristine dataset (run via 'sh 1_GenCtfSnr_Batch.sh')
# =============================================================================
# USAGE: before running, change number of PDs in '1_GenCtfSnr_Batch.sh'.
#   ...Next, alter user parameters below to generate datasets as desired.
#   ...An ideal application of this data is to first generate stacks with...
#   ...non-noisy duplicates first via the 'Pristine_AddTauSNR' folder, and then...
#   ...use this script to apply CTF followed by noise (SNR) on those new stacks.
#   ...This scipt will additionally generate alignment files with all microscopy...
#   ...parameters used for CTF construction and angles for PDs. We have supplied...
#   ...some example great cirlce angular trajectories within this repository...
#   ...which can be used to generate custom PDs and read in here for alignment files. 
# Author:    E. Seitz @ Columbia University - Frank Lab - 2021
# Contact:   evan.e.seitz@gmail.com
# =============================================================================
    
def op(pyDir, PD):
    #pyDir = os.path.dirname(os.path.abspath(__file__)) #python file directory
    #PD='001'
    parDir = os.path.dirname(pyDir) #parent directory
    stackPath = os.path.join(parDir, 'Pristine_2D/PD_%s.mrcs' % PD) #location of .mrcs for current PD
    PD_stack = mrcfile.open(stackPath)

    # =========================================================================
    # User parameters
    # =========================================================================
    matlab = False
    fname = 'Hsp2D_5k15k_PD_%s' % PD
    #occFile = os.path.join(parDir, '5_GenOcc_python/Occ2D_4k.npy') #if energy landscape used for occupancy assignments
    # =========================================================================
    
    stackOut = os.path.join(pyDir, '%s.mrcs' % fname)
    alignOut = os.path.join(pyDir, '%s.star' % fname)
    if os.path.exists(stackOut):
        os.remove(stackOut)
    if os.path.exists(alignOut):
        os.remove(alignOut)
    
    if matlab:
        binaryOut = os.path.join(pyDir, '%s.dat' % fname)
        spiderOut = os.path.join(pyDir, '%s.spi' % fname)
        if os.path.exists(spiderOut):
            os.remove(spiderOut)
        if os.path.exists(binaryOut):
            os.remove(binaryOut)
        
    snapshots, box, box = PD_stack.data.shape
    print('Box size:', box)
    print('Snapshots:',snapshots)

    #occ = np.load(occFile)
    #snapshots = int(np.sum(occ))
    
    # =========================================================================
    # Import angles:
    # =========================================================================
    S2 = pd.read_csv(os.path.join(parDir, 'EulerCoordinates/hGC_126_v1.txt'), header=None, delim_whitespace=True) #projections
    rot = S2[1][int(PD)-1]
    tilt = S2[2][int(PD)-1]
    psi = S2[3][int(PD)-1]
    print('Euler:', rot, tilt, psi)
    # =========================================================================
        
    # =========================================================================
    # FUNCTIONS:
    # ========================================================================= 
    def find_SNR(image):
        IMG_2D = []
        for pix in image:
            IMG_2D.append(pix)
        IMG_1D = list(itertools.chain.from_iterable(IMG_2D))
        img_mean = np.mean(IMG_1D)
        img_var = np.var(IMG_1D)
        img_std = np.sqrt(img_var)
        if 0: #to illustrate signal only, must be OFF during stack generation
            SIG_1D = np.ndarray(shape=np.shape(IMG_1D)) #empty array for signal-pixels only
            idx = 0
            for pix in IMG_1D:
                #if -img_std*.25 < pix < img_std*.25:
                if (img_mean-img_std*.5) < pix < (img_mean+img_std*.5):
                    SIG_1D[idx] = -100 #arbitrarily large for illustration
                else:
                    SIG_1D[idx] = IMG_1D[idx]
                idx += 1
            SIG_2D = np.asarray(SIG_1D).reshape(250, 250)
            plt.imshow(SIG_2D)
            plt.show()
        else:
            SIG_1D = []
            for pix in IMG_1D:
                if -img_std*1. < pix < img_std*1.: #originally *0.25; 1. better behaved for CTF
                    pass
                else:
                    SIG_1D.append(pix) #only grab signal
        sig_mean = np.mean(SIG_1D)
        sig_var = np.var(SIG_1D)
        noise_var = sig_var / 0.1 #experimental regime
        noise_std = np.sqrt(noise_var)
        return sig_mean, noise_std
    
    def add_noise(mean, std, image):
        row, col = image.shape
        gauss = np.random.normal(mean, std, (row, col))
        gauss = gauss.reshape(row, col)
        noisy = image + gauss
        return noisy
    
    def normalize(image):
        bg = []
        h, w = image.shape[:2]
        center = [int(w/2), int(h/2)]
        radius = int(h/2)
        Y, X = np.ogrid[:h,:w]
        dist_from_center = np.sqrt((X - center[0])**2 + (Y - center[1])**2)
        mask = dist_from_center <= radius
        masked_img = image.copy()
        if 0: #for visualization of mask only, keep off (0)
            masked_img[~mask] = 0
            plt.imshow(masked_img, cmap='gray')
            plt.show()
        bg = masked_img[~mask]
        bg_mean = np.mean(bg)
        bg_std = np.std(bg)
        img_norm = (image - bg_mean) / bg_std
        if 0: #NORM CHECK, for testing only
            print('bg_mean:', bg_mean)
            print('bg_std:', bg_std)
        return img_norm
    
    # =========================================================================
    # Initiate alignment file:
    # =========================================================================
    alignFile = open(alignOut, 'w')
    alignFile.write('\ndata_ \
                    \n \
                    \nloop_ \
                    \n \
                    \n_rlnAngleRot #1 \
                    \n_rlnAngleTilt #2 \
                    \n_rlnAnglePsi #3 \
                    \n_rlnOriginX #4 \
                    \n_rlnOriginY #5 \
                    \n_rlnDefocusU #6 \
                    \n_rlnDefocusV #7 \
                    \n_rlnVoltage #8 \
                    \n_rlnSphericalAberration #9 \
                    \n_rlnAmplitudeContrast #10 \
                    \n_rlnDefocusAngle #11 \
                    \n_rlnCtfBfactor #12 \
                    \n_rlnPhaseShift #13 \
                    \n_rlnPixelSize #14 \
                    \n_rlnImageName #15 \
                    \n')
    
    # Initiate spider file:
    if matlab:
        spiderFile = open(spiderOut, 'w')
    
    # Create empty arrays:
    img_array = mrcfile.new_mmap(stackOut, shape=(snapshots,box,box), mrc_mode=2, overwrite=True) #mrc_mode 2: float32
    if matlab:
        binary_array = np.memmap(binaryOut, dtype='float32', mode='w+', shape=(snapshots,box,box))#, offset=4*box**2)
    
    # =========================================================================
    # Main loop:
    # =========================================================================       
    for img_idx in range(snapshots):
        img_orig = PD_stack.data[img_idx]
        df = np.random.randint(low=5000, high=15000, size=1)[0]
        
        angleRot = Decimal(rot)
        angleTilt = Decimal(tilt)
        anglePsi = Decimal(psi)
        origX = 0
        origY = 0
        dfX = float(df)
        dfY = float(df)
        volt = 300
        Cs = 2.7
        ampc = 0.1
        dfAng = 0
        Bfact = 0
        pShift = 0
        pixSize = 1.0
        
        # Calculate and add CTF:
        ctf = ctemh_cryoFrank_generator.gen_ctf(box, pixSize, Cs, dfX, volt, np.inf, ampc)
        img_ctf = ifft2(fft2(img_orig)*ctf).real
        img_ctf = -1.*img_ctf
        
        sig_mean, noise_std = find_SNR(img_ctf) #find SNR
        img_noise = add_noise(sig_mean, noise_std, img_ctf) #apply noise
        img_norm = normalize(img_noise) #normalize
        
        img_array.data[img_idx] = img_norm
        if matlab:
            binary_array[img_idx] = img_norm
            
        if 0:
            print('norm check:')
            normalize(img_norm) #NORM CHECK
        
        # Update alignment file:
        alignFile.write('%.6f\t%.6f\t%.6f\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s@%s.mrcs\n' \
            % (angleRot, angleTilt, anglePsi, origX, origY, dfX, dfY, volt, Cs, ampc, dfAng, Bfact, pShift, pixSize, int(img_idx+1), str(fname)))
	
    	# Update spider file (index, number of params, psi, theta [0,180], phi [-180,180], class, x, y, u, v, uv angle):
        if matlab:
            spiderFile.write('%s\t%s\t%.6f\t%.6f\t%.6f\t%s\t%s\t%s\t%s\t%s\t%s\n' % (img_idx+1, 9, origX, anglePsi, angleTilt-180, 1, 0, 0, dfX, volt, 0))
        
        print('%s / %s' % (img_idx, snapshots))

    PD_stack.close()
        
    alignFile.close()
    if matlab:
        spiderFile.close()
        
    print('PD Complete')
    
if __name__ == '__main__':
    path1 = os.path.splitext(sys.argv[0])[0]
    path2, tail = os.path.split(path1)
    op(path2, sys.argv[1])