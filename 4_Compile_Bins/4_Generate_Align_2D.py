import os, sys
import json
import numpy as np
import matplotlib
from matplotlib import rc
#matplotlib.rc('text', usetex = True)
from matplotlib.figure import Figure
import matplotlib.pyplot as plt
from pylab import imshow, show, loadtxt, axes

if 1: #nice font for figures
    rc('text', usetex=True)
    rc('font', family='serif')

# =============================================================================
# Create alignment files for images in each 2D occupancy bin (via intersection)
# =============================================================================
# SETUP: First, make sure all user parameters are correct for your dataset below...
#   ...in the initial sections (correct input file names via directories, if altered).
#   For complete instructions on readying data, see previous script '1_Compile_Bins.py'.
# RUNNING: Run via `python 4_Generate_Align_2D.py`
# =============================================================================
# Authors:    E. Seitz @ Columbia University - Frank Lab - 2020
#             F. A. Reyes @ Columbia University - Frank Lab - 2020
# Contact:   evan.e.seitz@gmail.com
# =============================================================================

pyDir = os.path.dirname(os.path.abspath(__file__)) #python file location
parDir = os.path.abspath(os.path.join(pyDir, os.pardir))

# =============================================================================
# User parameters
# =============================================================================
totalPDs = 126
bins = 20
# Choose the two CMs to consider:
CM1 = 1 #first CM
CM2 = 2 #second CM
R2_thresh = 0.825 #optional: skip PDs based on the current CM's parabolic fit score

#inDir = os.path.join(parDir, '3_Spline_Binning/bins_PCA')
inDir = os.path.join(parDir, '3_Spline_Binning/bins_PCA_GC3_v5')
outDir = os.path.join(pyDir, 'S2_bins_GC3_v5/CM1_CM2')
if not os.path.exists(outDir):
    os.makedirs(outDir)

# =============================================================================
# Read in angles corresponding to each PD:    
# =============================================================================
projDir = os.path.join(parDir, '0_Data_Inputs_126gc')
for file in os.listdir(projDir):
    if file.endswith('Euler_v3.txt'):
        projPath = os.path.join(projDir, file)
proj = np.genfromtxt(projPath, unpack=True).T

# =============================================================================
# Initiate empty 2D occupancy map:
# =============================================================================
occmap2D = np.zeros(shape=(bins,bins), dtype=int) #initiate empty occupancy map
# note that CM1, CM2 have nonzero-indexing:
states_list = np.arange(1,bins+1,1)

for CM_i in states_list:
    for CM_j in states_list:
        print('CM%s_%s, CM%s_%s' % (CM1, CM_i, CM2, CM_j))
        CM_i_idx = "{0:0=2d}".format(CM_i)
        CM_j_idx = "{0:0=2d}".format(CM_j)
        
        # =============================================================================
        # Initiate each STAR file:
        # =============================================================================
        alignOut = os.path.join(outDir, 'CM%s_%s_CM%s_%s.star' % (CM1, CM_i_idx, CM2, CM_j_idx))
        if os.path.exists(alignOut):
            os.remove(alignOut)
        alignFile = open(alignOut, 'w')
        alignFile.write('# RELION; version 3.0.8\
                        \n \
                        \ndata_images\
                        \n \
                        \nloop_\
                        \n_rlnCoordinateX #1 \
                        \n_rlnCoordinateY #2 \
                        \n_rlnImageName #3 \
                        \n_rlnAnglePsi #4 \
                        \n_rlnAngleRot #5 \
                        \n_rlnAngleTilt #6 \
                        \n_rlnAmplitudeContrast #7 \
                        \n_rlnVoltage #8 \
                        \n_rlnDetectorPixelSize #9 \
                        \n_rlnMagnification #10 \
                        \n')
            
        for pd in range(1,totalPDs+1):
            pdDir = os.path.join(inDir, 'PD%03d' % pd)
            
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
            #cm_i_R2_all.append(cm_i_R2)
            
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
            #cm_j_R2_all.append(cm_i_R2)
                            
            # =================================================================
            # Find intersection of CM_i, CM_j bins:
            # =================================================================
            with open(binPaths_CM_i[CM_i-1], "r") as read_file:
                binFile_CM_i = json.load(read_file)
            
            with open(binPaths_CM_j[CM_j-1], "r") as read_file:
                binFile_CM_j = json.load(read_file)
                
                '''TO DO: if 1: #optional: skip PDs based on the current CM's parabolic fit score
                if cm_i_R2 < R2_thresh:
                    if b == 0:
                        totalPDs_thresh -= 1
                    continue'''
            
            intersect = list(set(binFile_CM_i) & set(binFile_CM_j))
            occmap2D[CM_j-1, CM_i-1] += len(intersect)
            
            # =================================================================
            # Propogate STAR file:
            # =================================================================         
            #fname = 'stacks/PD%03d_SS2_SNRpt1_tau10_stack.mrcs' % pd
            fname = 'stacks_tau10_GC3/PD%03d_SNR_tau10_stack.mrcs' % pd
            fDir = os.path.join(projDir,fname)
            projPD = proj[pd-1] #grab angle corresponding to current PD
            for ii in intersect:
                alignFile.write('%s\t%s\t%s@%s\t%.6f\t%.6f\t%.6f\t%s\t%s\t%s\t%s\n' \
                    % (0., 0., ii+1, fDir, float(projPD[3]+90), \
                    (float(projPD[1])-90), float(projPD[2]), .1, 300., 1., 10000.))        
        
        alignFile.close()
        
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
fig.savefig(os.path.join(outDir,'Occupancy_Map.png'), dpi=200)
plt.show()