import os, sys
import mrcfile
import matplotlib
from matplotlib import rc
#matplotlib.rc('text', usetex = True)
from matplotlib.figure import Figure
import matplotlib.pyplot as plt
from pylab import imshow, show
import numpy as np
from matplotlib.ticker import MaxNLocator
import json
pyDir = os.path.dirname(os.path.abspath(__file__)) #python file location
parDir = os.path.abspath(os.path.join(pyDir, os.pardir))
readDir = os.path.join(parDir, '1_Embedding/DM')
sys.path.append(readDir)
import Read_Alignment
    
# =============================================================================
# Compile 2D CM bins across all PDs as obtained in previous step (`Manifold_Binning.py`)
# =============================================================================
# SETUP: First, make sure all user parameters are correct for your dataset...
#   ...below via the 'User parameters' section. As well, ensure that the...
#   ...'dataDir' and related path names are correct.
#   Additionally, check the EMAN/RELION conversion protocol at the end of...
#   ...this script to ensure the correct conventions are being applied.
# IMPORTANT:
#   Before running, senses and CM indices must be assigned for each PD in the...
#   ...`3_Spline_Binning/bins_" directory. An example workflow, would include...
#   ...manually watching each of the 2D CM movies per PD and...
#   ...assigning those CM movies corresponding to true CM motions an index...
#   ...in the corresponding `CMs.txt` file as well as a sense in the corresponding...
#   ...`Senses.txt` file. For example, if the movie for CM1 in PD001 represented...
#   ...your molecule's primary motion (arbitrarily defined by the user as "primary")...
#   ...then alter the first entry in `CMs.txt` for that PD to ['1', ...], then...
#   ...arbitrarily define that primary motion with a directionality (i.e., "sense")...
#   ...of either forward (['F', ...]) or reverse (['R', ...]), making sure to keep...
#   ...this arbitrary assignment consistent for all PDs (and all CMs therein).
#   If a movie is not a valid motion, just leave it as an 'X' in both lists.
# RUNNING: Initiate batch processing via `python 2_Compile_Bins_2D.py`
# =============================================================================
# Author:    E. Seitz @ Columbia University - Frank Lab - 2020-2021
# Contact:   evan.e.seitz@gmail.com
# =============================================================================

if 1: #render with LaTeX font for figures
    rc('text', usetex=True)
    rc('font', family='serif')
    
outDir = os.path.join(pyDir, 'S2_bins')
if not os.path.exists(outDir):
    os.makedirs(outDir)
    
# =============================================================================
# User parameters
# =============================================================================
totalPDs = 126 #total number of projection directions
# Choose the two CMs to consider:
CM1 = 1 #first CM
CM2 = 2 #second CM
bins = 20 #needs to match choice previously used in 'Manifold_Binning.py'
box = 320 #image dimensions (i.e., box size)
groundTruth = False #optional, for comparing final outputs with ground-truth knowldege
R2_skip = True #optional: skip PDs based on the current CM's parabolic fit score
R2_thresh = 0.7 #active if 'R2_skip' is True
dataDir = os.path.join(parDir, '0_Data_Inputs/_CTF5k15k_SNRpt1_ELS_GC1_2D') #also check paths below

if groundTruth is True: #analyze total True Positives
    occmap2D_GT = np.load(os.path.join(parDir, '0_Data_Inputs/GroundTruth_Indices/Occ2D_4k.npy'))
    occmap2D_GT.astype(int)
    occmap2D_accuracy = np.zeros(shape=(bins,bins), dtype=float)
    CM1_idx = np.load(os.path.join(parDir, '0_Data_Inputs/GroundTruth_Indices/CM1_Indices.npy'), allow_pickle=True) #view in reference frame of CM1
    CM2_idx = np.load(os.path.join(parDir, '0_Data_Inputs/GroundTruth_Indices/CM2_Indices.npy'), allow_pickle=True) #view in reference frame of CM2

# =============================================================================
# Initiate empty 2D occupancy map:
# =============================================================================
cmDir = os.path.join(outDir, 'CM%s_CM%s' % (CM1, CM2))
if not os.path.exists(cmDir):
    os.makedirs(cmDir)
occmap2D = np.zeros(shape=(bins,bins), dtype=int) #initiate empty occupancy map; note that CM1 and CM2 have nonzero indexing
states_list = np.arange(1,bins+1,1)

for CM_i in states_list:
    for CM_j in states_list:
        print('CM%s_%s, CM%s_%s' % (CM1, CM_i, CM2, CM_j))
        CM_i_idx = "{0:0=2d}".format(CM_i)
        CM_j_idx = "{0:0=2d}".format(CM_j)
        
        # =============================================================================
        # Initiate each STAR file:
        # =============================================================================
        alignOutPath = os.path.join(cmDir, 'CM%s_%s_CM%s_%s.star' % (CM1, CM_i_idx, CM2, CM_j_idx))
        if os.path.exists(alignOutPath):
            os.remove(alignOutPath)
        binAlign_out = open(alignOutPath, 'w')
        binAlign_out.write('# RELION; version 3.0.8\
                        \n \
                        \ndata_images \
                        \n \
                        \nloop_ \
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
                        \n_rlnDetectorPixelSize #14 \
                        \n_rlnMagnification #15 \
                        \n_rlnImageName #16 \
                        \n')

        for pd in range(1,totalPDs+1):
            PD = '{0:0=3d}'.format(pd)
            pdDir = os.path.join(parDir, '3_Manifold_Binning/bins/PD%.03d' % pd)

            # =============================================================
            # Read in initial angles and microscopy parameters used to generate PDs:
            # =============================================================
            origAlignPath = os.path.join(dataDir, 'Hsp2D_5k15k_PD_%s.star' % PD) #update for user location of raw data
            orig_align = Read_Alignment.parse_star(origAlignPath)
            # Read in microscopy parameters from PD alignment file (may need to modify below parameters to match inputs):
            angRot = orig_align['rlnAngleRot'].values
            angTilt = orig_align['rlnAngleTilt'].values
            angPsi = orig_align['rlnAnglePsi'].values
            origX = orig_align['rlnOriginX'].values
            origY = orig_align['rlnOriginY'].values
            dfX = orig_align['rlnDefocusU'].values
            dfY = orig_align['rlnDefocusV'].values
            volt = orig_align['rlnVoltage'].values
            Cs = orig_align['rlnSphericalAberration'].values
            ampc = orig_align['rlnAmplitudeContrast'].values
            dfAng = orig_align['rlnDefocusAngle'].values
            Bfact = orig_align['rlnCtfBfactor'].values
            pShift = orig_align['rlnPhaseShift'].values
            px = orig_align['rlnPixelSize'].values
            imgName = orig_align['rlnImageName'].values

            # Setup to match proper CM/Sense within each PD directory: 
            with open(os.path.join(pdDir,'CMs.txt'), 'r') as read_file:
                CM_idx = read_file.read().replace('\n', '') 
            cm_idx_list = []
            for idx in CM_idx:
                if idx.isnumeric():
                    cm_idx_list.append(int(idx))
                if idx == 'X':
                    cm_idx_list.append('X')
            try:
                cm_i_idx = cm_idx_list.index(CM1)
            except ValueError:
                print('Specified CM_%s not found in PD_%s CMs.txt' % (CM1, pd))
                break
            try:
                cm_j_idx = cm_idx_list.index(CM2)
            except ValueError:
                print('Specified CM_%s not found in PD_%s CMs.txt' % (CM2, pd))
                break
            
            with open(os.path.join(pdDir,'Senses.txt'), 'r') as read_file:
                sense = read_file.read().replace('\n', '')
            sense_idx_list = []
            for idx in sense:
                if idx == 'F' or idx == 'R':
                    sense_idx_list.append(idx)
                if idx == 'X':
                    sense_idx_list.append('X')
            sense_CM_i = sense_idx_list[cm_i_idx]
            sense_CM_j = sense_idx_list[cm_j_idx]
 
            # =================================================================
            # Reorganize CM_i based on sense:
            # =================================================================
            binPaths_CM_i = []
            binDir_CM_i = os.path.join(pdDir, 'CM%s' % (cm_i_idx+1))
            for file in os.listdir(binDir_CM_i):
                if not file.endswith("Hist.txt"):
                    if not file.endswith("R2.txt"):
                        if file.endswith(".txt"):
                            binPaths_CM_i.append(os.path.join(binDir_CM_i, file))
            if sense_CM_i == 'R':
                binPaths_CM_i = binPaths_CM_i[::-1] #reverse order to match sense    
                
            # Grab coefficient of determination (R^2):
            for file in os.listdir(binDir_CM_i):
                if file.endswith('R2.txt'):
                    R2Path = os.path.join(binDir_CM_i, file)
            cm_i_R2 = np.genfromtxt(R2Path, unpack=True).T
            
            # =================================================================
            # Reorganize CM2 based on sense:
            # =================================================================
            binPaths_CM_j = []
            binDir_CM_j = os.path.join(pdDir, 'CM%s' % (cm_j_idx+1))
            for file in os.listdir(binDir_CM_j):
                if not file.endswith("Hist.txt"):
                    if not file.endswith("R2.txt"):
                        if file.endswith(".txt"):
                            binPaths_CM_j.append(os.path.join(binDir_CM_j, file))
            if sense_CM_j == 'R':
                binPaths_CM_j = binPaths_CM_j[::-1] #reverse order to match sense 
                
            # Grab coefficient of determination (R^2):
            for file in os.listdir(binDir_CM_j):
                if file.endswith('R2.txt'):
                    R2Path = os.path.join(binDir_CM_j, file)
            cm_j_R2 = np.genfromtxt(R2Path, unpack=True).T
                            
            # =================================================================
            # Find intersection of CM_i, CM_j bins:
            # =================================================================
            with open(binPaths_CM_i[CM_i-1], "r") as read_file:
                binFile_CM_i = json.load(read_file)
            
            with open(binPaths_CM_j[CM_j-1], "r") as read_file:
                binFile_CM_j = json.load(read_file)
            
            intersect = list(set(binFile_CM_i) & set(binFile_CM_j))
            occmap2D[CM_j-1, CM_i-1] += len(intersect)
            
            if groundTruth is True:
                intersect_GT = list(set(CM1_idx[CM_i-1]) & set(CM2_idx[CM_j-1]))                            
                intersect_ESPER_GT = list(set(intersect) & set(intersect_GT))                
                occmap2D_accuracy[CM_j-1, CM_i-1] += len(intersect_ESPER_GT)

            if R2_skip is True: #only affects occupancies while preserving all images
                if cm_i_R2 < R2_thresh or cm_j_R2 < R2_thresh:
                    occmap2D[CM_j-1, CM_i-1] -= len(intersect)
                    
            # =================================================================
            # Propogate STAR file:
            # =================================================================         
            for img in intersect:
                if 1: #EMAN2 to RELION conversion
                    angRot[img] -= 90
                    angPsi[img] += 90
                # Update alignment file:
                imgNumber, imgLocation = imgName[img].split('@')
                imgDir = imgNumber + '@' + dataDir + '/' + imgLocation                 
                binAlign_out.write('%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
                    % (angRot[img], angTilt[img], angPsi[img], origX[img], origY[img], dfX[img], dfY[img], volt[img], Cs[img], ampc[img], dfAng[img], Bfact[img], pShift[img], px[img], 10000., str(imgDir)))

        binAlign_out.close()
        
# sanity-check:
print('CM%s: %s' % (CM1, np.sum(occmap2D, axis=0)))
print('CM%s: %s' % (CM2, np.sum(occmap2D, axis=1)))
print('CM%s_CM%s: %s' % (CM1, CM2, np.sum(occmap2D)))
print('Mean:', np.mean(occmap2D))
print('Std:', np.std(occmap2D))

# =============================================================================
# Plot 2D occupancy map:
# =============================================================================
plt.imshow(occmap2D, origin='lower', interpolation='nearest', aspect='equal', extent=[.5,20.5,.5,20.5], cmap='viridis')
plt.xlabel('CM 1', fontsize=14, labelpad=10)
plt.ylabel('CM 2', fontsize=14, labelpad=10)
plt.xticks([1,5,10,15,20])
plt.yticks([1,5,10,15,20])
plt.tight_layout()
plt.colorbar()
plt.clim(0,np.amax(occmap2D))
plt.title('Occupancy Map', fontsize=16)
plt.tight_layout()
fig = plt.gcf()
fig.savefig(os.path.join(cmDir,'Occupancy_Map.png'), dpi=200)
np.save(os.path.join(cmDir,'Occupancy_Map.npy'), occmap2D)
#plt.show()

if groundTruth is True:
    if 0: #convert to ratios
        occmap2D_accuracy = occmap2D_accuracy / (occmap2D_GT*126.)
    plt.clf()
    plt.imshow(occmap2D_accuracy, origin='lower', interpolation='nearest', aspect='equal', extent=[.5,20.5,.5,20.5], cmap='viridis')
    plt.xlabel('CM 1', fontsize=14, labelpad=10)
    plt.ylabel('CM 2', fontsize=14, labelpad=10)
    plt.xticks([1,5,10,15,20])
    plt.yticks([1,5,10,15,20])
    plt.tight_layout()
    plt.colorbar()
    plt.clim(0,np.amax(occmap2D_accuracy))
    plt.tight_layout()
    plt.show()