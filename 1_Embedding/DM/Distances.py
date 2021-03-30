import sys, os
import numpy as np
import mrcfile
from scipy.fftpack import fft2
from scipy.fftpack import ifft2
import Read_Alignment
import Generate_Filters

# =============================================================================
# Calculate pairwise Euclidean distances between all images in each PD
# =============================================================================
# NOTICE: This workflow is designed such that images belonging to Euler coordinates...
#   ...on the 2-sphere have already been partitioned into projection directions (PDs)...
#   ...such that each stack/alignment file cooresponds to a single PD.
# SETUP: First, make sure all input file paths are correct for your datasets...
#   ...in 'dataDir' and 'dataPath' variables in 'op' function below. You may want...
#   ...to edit output names. As well, set the 'CTF' variable to T/F according to...
#   ...whether your data has been generated with (or naturally has) CTF or not.
# RUNNING: To run a series of PDs at once: first edit '1_Dist_Batch.sh'...
#   ...for the total number of PDs requested; e.g., {1...5} for 5 PDs...
#   ...or {1...1} for only the first PD;
#   ...then start batch processing via 'sh 1_Dist_Batch.sh'
# =============================================================================
# Authors:   E. Seitz @ Columbia University - Frank Lab - 2021
# History:   H. Liao (CU, 2019); P. Schwander (UWM, 2019); A. Dashti (UWM, 2016)
#            See ManifoldEM Python/Matlab 2020-2021: i.e., GetDistances.py
# Contact:   evan.e.seitz@gmail.com
# =============================================================================


def op(pyDir, PD):
    parDir1 = os.path.abspath(os.path.join(pyDir, os.pardir))
    parDir2 = os.path.abspath(os.path.join(parDir1, os.pardir))
    
    # =========================================================================
    # User parameters:
    # =========================================================================
    CTF = True #if using data modified with CTF
    #dataDir = os.path.join(parDir2, '0_Data_Inputs/Pristine_2D') #for noiselss datasets
    dataDir = os.path.join(parDir2, '0_Data_Inputs/CTF5k15k_SNRpt1_ELS_2D')
    outDir = os.path.join(pyDir, 'Data_Distances')
    if not os.path.exists(outDir):
        os.mkdir(outDir)
    
    # =========================================================================
    # Import image stack per PD and standardize:
    # =========================================================================
    if CTF is False:
        print('CTF mode disabled.')
        #stackPath = os.path.join(dataDir, 'PD_%s.mrcs' % PD) #for noiseless datasets
        stackPath = os.path.join(dataDir, 'PD%s_SNR_tau5_stack.mrcs' % PD)
    elif CTF is True:
        print('CTF mode enabled.')
        stackPath = os.path.join(dataDir, 'Hsp2D_5k15k_PD_%s.mrcs' % PD)
        alignPath = os.path.join(dataDir, 'Hsp2D_5k15k_PD_%s.star' % PD)
        wienerOut = os.path.join(dataDir, 'Hsp2D_5k15k_PD_%s_filtered.mrcs' % PD) #new stack for CTF correction
        # Read in microscopy parameters from alignment file:
        starFile = Read_Alignment.parse_star(alignPath)
        df = starFile['rlnDefocusU'].values
        volt = starFile['rlnVoltage'].values
        Cs = starFile['rlnSphericalAberration'].values
        ampc = starFile['rlnAmplitudeContrast'].values
        px = starFile['rlnPixelSize'].values
        
    PD_stack = mrcfile.mmap(stackPath, 'r')
    nS, N, N = PD_stack.data.shape #number of snapshots; boxsize; boxsize

    # =========================================================================
    # Generate distances from images in PD:
    # =========================================================================
    print('Preparing images...')
    D = np.zeros((nS,nS)) #distance matrix
    imgAll = np.zeros((nS,N,N))
    wiener_stack = mrcfile.new_mmap(wienerOut, shape=(nS,N,N), mrc_mode=2, overwrite=True) #mrc_mode 2: float32

    if CTF is False:
        # Note: if using synthetic data, mean image subtraction is enough (center...
        # ...the values to 0) since values have the same scale to begin with (0-255).
        for i in range(0,nS):
            image = PD_stack.data[i]/1.
            image -= image.mean()
            image /= image.std() #see note above
            imgAll[i,:,:] = image
            
        # Compute distances:
        print('Computing distances...')
        p = 2. #Minkowski distance metric: p1=Manhattan, p2=Euclidean, etc.
        for i in range(0,nS):
            for j in range(0,nS):
                if i > j:
                    print(i,j)
                    D[i,j] = (np.sum(np.abs((imgAll[i,:,:]-imgAll[j,:,:]))**p) / imgAll[i,:,:].size)**(1./p)
                elif i == j:
                    D[i,j] = 0
        D = D + D.T - np.diag(np.diag(D))

        
    elif CTF is True:
        y = np.zeros((N**2, nS)) #each row is a flattened image
        fy = complex(0)*np.ones((nS,N,N)) #each (i,:,:) is a Fourier image
        CTF = np.zeros((nS,N,N)) #each (i,:,:) is the CTF
        
        print('Generating filters...')
        for iS in range(nS):
            tmp = PD_stack.data[iS]
            tmp = (tmp - tmp.mean())/tmp.std()
            y[:,iS] = tmp.flatten('F') #normalized image
            # Create CTF and Butterworth filter:
            CTF[iS,:,:], G = Generate_Filters.gen_ctf(N, px[iS], Cs[iS], df[iS], volt[iS], np.inf, ampc[iS])
            G = G.astype(float)
            # Apply Butterworth filter:
            image = y[:,iS].reshape(-1,N).transpose()
            image = ifft2(fft2(image)*G).real
            # Round up files:
            y[:,iS] = image.real.flatten('F')
            fy[iS,:,:] = fft2(y[:,iS].reshape(-1,N))
            imgAll[iS,:,:] = y[:,iS].reshape(-1,N).transpose()
        
        print('Performing CTF correction...')
        #imgAvg = 0
        wiener_dom = -gen_wiener(CTF)
        for iS in range(nS):
            img =  imgAll[iS,:,:]
            img_f = fft2(img)
            CTF_i = CTF[iS,:,:]
            img_f_wiener = img_f*(CTF_i/wiener_dom)
            wiener_stack.data[iS] = ifft2(img_f_wiener).real
            #imgAvg = imgAvg + ifft2(img_f_wiener).real

        # Compute distances using defocus-tolerant kernel:
        print('Computing distances...')
        fy = fy.reshape(nS,N**2)
        CTF = CTF.reshape(nS,N**2)
        CTFfy = CTF.conj()*fy
        D = np.dot((abs(CTF)**2), (abs(fy)**2).T)
        D = D + D.T - 2*np.real(np.dot(CTFfy, CTFfy.conj().transpose()))
        D -= np.diag(D)
        D = np.sqrt(D)
     
    np.save(os.path.join(outDir, 'PD_%s_dist.npy' % PD), D)
    print('Distances complete.')
    
    PD_stack.close()
    wiener_stack.close()


def gen_wiener(CTF1):
    SNR = 5
    wiener_dom = 0.
    for i in range(CTF1.shape[0]):
        wiener_dom = wiener_dom + CTF1[i, :, :]**2
    wiener_dom = wiener_dom + 1. / SNR
    return (wiener_dom)

        
if __name__ == '__main__':
    path1 = os.path.splitext(sys.argv[0])[0]
    path2, tail = os.path.split(path1)
    op(path2, sys.argv[1])
    