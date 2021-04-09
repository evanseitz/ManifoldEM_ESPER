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
# Compile 1D CM bins across all PDs as obtained in previous step (`Manifold_Binning.py`)
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
# RUNNING: Initiate batch processing via `python 1_Compile_Bins_1D.py`
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
totalCMs = 2 #total number of conformational motions to consider
bins = 20 #needs to match choice previously used in 'Manifold_Binning.py'
box = 320 #image dimensions (i.e., box size)
groundTruth = True #optional, for comparing final outputs with ground-truth knowldege
R2_skip = True #optional: skip PDs based on the current CM's parabolic fit score
R2_thresh = 0.71 #active if 'R2_skip' is True
printFigs = True #show figures of outputs throughout framework
dataDir = os.path.join(parDir, '0_Data_Inputs/CTF5k15k_SNRpt1_ELS_2D') #also check stackPath and alignPath below

if groundTruth is True:
    CM1_idx = np.load(os.path.join(parDir, '0_Data_Inputs/GroundTruth_Indices/CM1_Indices.npy'), allow_pickle=True) #view in reference frame of CM1
    CM2_idx = np.load(os.path.join(parDir, '0_Data_Inputs/GroundTruth_Indices/CM2_Indices.npy'), allow_pickle=True) #view in reference frame of CM2

# =============================================================================
# Main loop to compile CM bins across all PDs
# =============================================================================
for CM in range(totalCMs): #CMs to consider
    binAcc = np.zeros(shape=(bins,bins))
    if CM == 0:
        binsActual = CM1_idx
    elif CM == 1:
        binsActual = CM2_idx #(etc.)
        
    R2_all = []
    totalPDs_thresh = totalPDs #if PDs thresholded by R^2 below (used for uniform distributions only)
    occmapAll = np.zeros(bins)
    #occmapPDs = []
    #for b in range(bins):
        #occmapPDs.append([])
    for b in range(bins):
        B = "{0:0=2d}".format(b+1)
        print('CM_%s, Bin_%s' % ((CM+1),B))
        cmDir = os.path.join(outDir, 'CM%s' % (CM+1))
        if not os.path.exists(cmDir):
            os.makedirs(cmDir)
             
        # Initiate alignment file per bin:
        alignOutPath = os.path.join(cmDir, 'CM%s_bin%s.star' % ((CM+1),B))
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
            #print(PD_%s' % PD)
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
            cm_idx = cm_idx_list.index(CM+1)
            
            with open(os.path.join(pdDir,'Senses.txt'), 'r') as read_file:
                sense = read_file.read().replace('\n', '')
            sense_idx_list = []
            for idx in sense:
                if idx == 'F' or idx == 'R':
                    sense_idx_list.append(idx)
                if idx == 'X':
                    sense_idx_list.append('X')
            sense_idx = sense_idx_list[cm_idx]
            
            stackDir = os.path.join(pdDir, 'CM%s' % (cm_idx+1))
            
            # Grab coefficient of determination (R^2):
            for file in os.listdir(stackDir):
                if file.endswith('R2.txt'):
                    R2Path = os.path.join(stackDir, file)
            R2 = np.genfromtxt(R2Path, unpack=True).T
            R2_all.append(R2)
            # Grab occupancy information:
            for file in os.listdir(stackDir):
                if file.endswith('Hist.txt'):
                    occmapPath = os.path.join(stackDir, file)
            occmap = np.genfromtxt(occmapPath, unpack=True).T
            
            # Grab stack of CM images (movie):
            for file in os.listdir(stackDir):
                if file.endswith('.mrcs'):
                    moviePath = os.path.join(stackDir, file)
            movie_stack = mrcfile.mmap(moviePath)
            # Arbitrary order; valid so long as it's consistent across all PDs:
            if sense_idx == 'F':
                fix=0 #keep index order
            elif sense_idx == 'R':
                fix=1 #reverse index order
                    
            # Combine integrated images (as output in 'Manifold_Binning.py') across all PDs to new stack per bin...
            # ...will need create binStack_out mrcfile first to utilize this step:
            ###binStack_out.data[pd-1] = movie_stack.data[b-fix*(2*b+1)]

            if R2_skip: #optional: skip PDs based on the current CM's parabolic fit score
                if R2 > R2_thresh:
                    occmapAll[b] += occmap[b-fix*(2*b+1)]
            else:
                occmapAll[b] += occmap[b-fix*(2*b+1)]
            
            # =================================================================
            # Retrieve image indices within each bin:
            # =================================================================
            binPaths = []
            for file in os.listdir(stackDir):
                if file.endswith('.txt'):
                    if not file.endswith('Hist.txt'):
                        if not file.endswith('R2.txt'):
                            binPaths.append(os.path.join(stackDir, file))
            with open(binPaths[b-fix*(2*b+1)], 'r') as read_file:
                binFile = json.load(read_file)
            
            # =================================================================
            # Append alignment file:
            # =================================================================
            for img in binFile:
                if 1: #EMAN2 to RELION conversion
                    angRot[img] -= 90
                    angPsi[img] += 90
                # Update alignment file:
                imgNumber, imgLocation = imgName[img].split('@')
                imgDir = imgNumber + '@' + dataDir + '/' + imgLocation                 
                binAlign_out.write('%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
                    % (angRot[img], angTilt[img], angPsi[img], origX[img], origY[img], dfX[img], dfY[img], volt[img], Cs[img], ampc[img], dfAng[img], Bfact[img], pShift[img], px[img], 10000., str(imgDir)))
        
            movie_stack.close() #close up stack before next PD

            # =================================================================
            # Compare number of elements in trajectory-bin vs ground-truth:
            # =================================================================
            if groundTruth is True:
                skips = np.arange(1, (bins*2)+1, 2)
                if R2_skip:
                    if R2 > R2_thresh:
                        for i in range(0,bins):
                            binAcc[i-fix*skips[i],b] += sum(el in binFile for el in binsActual[i-fix*skips[i]]) #accuracy of single state
                    else:
                        if b == 0:
                            totalPDs_thresh -=1
                else:
                    for i in range(0,bins):
                        binAcc[i-fix*skips[i],b] += sum(el in binFile for el in binsActual[i-fix*skips[i]]) #accuracy of single state
                    
         
    # =========================================================================
    # Save each CM occupancy map to file:
    # =========================================================================
    np.savetxt(os.path.join(cmDir,'Occupancy_Map.txt'), occmapAll, delimiter=',', fmt='%i')
    print('Total Occupancy:',int(np.sum(occmapAll))) #sanity check
            
    # Save bar chart (standard form):
    plt.bar(range(1,bins+1),occmapAll)
    plt.xlabel(r'CM$_{%s}$ State' % (CM+1), fontsize=16, labelpad=10)
    plt.ylabel(r'Occupancy', fontsize=16, labelpad=10)
    plt.xlim(0.25, bins+.75)
    plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))
    plt.gca().yaxis.set_major_locator(MaxNLocator(integer=True))
    plt.gca().xaxis.set_major_locator(plt.MultipleLocator(1))
    #plt.axhline(y=tau*20*totalPDs_thresh, color='r', linewidth=2)
    plt.tight_layout()
    fig = plt.gcf()
    fig.savefig(os.path.join(cmDir,'Occupancy_Sum.png'), dpi=200)
    #plt.show()
    plt.clf()
    plt.cla()
    plt.close()
    
    # Save R2 histogram:
    plt.hist(np.asarray(R2_all), bins=np.linspace(0,1,40), edgecolor='black')
    plt.xlabel(r'Coefficient of Determination ($R^2$)')
    plt.ylabel('PDs')
    plt.tight_layout()
    fig = plt.gcf()
    fig.savefig(os.path.join(cmDir,'R2_Hist.png'), dpi=200)
    #plt.show()
    plt.clf()
    plt.cla()
    plt.close()
        
    if groundTruth is True and printFigs is True: #bar chart (stacked); strictly for checking accuracy with ground-truth (if available)
        fig = plt.figure()
        ax = plt.gca()
        x_pos = np.arange(1,bins+1)
        barWidth = 1
        cmap = matplotlib.cm.get_cmap('tab20') #{viridis, jet, spectral}
        crange = np.linspace(0,1,20)
        # Choose (arbitrary) order to display states:
        if 0: #forward order
            bar0x1 = np.add(binAcc[19,:],binAcc[18,:]).tolist()
            bar0x2 = np.add(bar0x1,binAcc[17,:]).tolist()
            bar0x3 = np.add(bar0x2,binAcc[16,:]).tolist()
            bar0x4 = np.add(bar0x3,binAcc[15,:]).tolist()
            bar0x5 = np.add(bar0x4,binAcc[14,:]).tolist()
            bar0x6 = np.add(bar0x5,binAcc[13,:]).tolist()
            bar0x7 = np.add(bar0x6,binAcc[12,:]).tolist()
            bar0x8 = np.add(bar0x7,binAcc[11,:]).tolist()
            bar0x9 = np.add(bar0x8,binAcc[10,:]).tolist()
            bar0x10 = np.add(bar0x9,binAcc[9,:]).tolist()
            bar0x11 = np.add(bar0x10,binAcc[8,:]).tolist()
            bar0x12 = np.add(bar0x11,binAcc[7,:]).tolist()
            bar0x13 = np.add(bar0x12,binAcc[6,:]).tolist()
            bar0x14 = np.add(bar0x13,binAcc[5,:]).tolist()
            bar0x15 = np.add(bar0x14,binAcc[4,:]).tolist()
            bar0x16 = np.add(bar0x15,binAcc[3,:]).tolist()
            bar0x17 = np.add(bar0x16,binAcc[2,:]).tolist()
            bar0x18 = np.add(bar0x17,binAcc[1,:]).tolist()
            plt.bar(x_pos, binAcc[19,:], color=cmap(crange[0]), edgecolor='white', width=barWidth, label='S20')
            plt.bar(x_pos, binAcc[18,:], bottom=binAcc[19,:], color=cmap(crange[1]), edgecolor='white', width=barWidth, label='S19')
            plt.bar(x_pos, binAcc[17,:], bottom=bar0x1, color=cmap(crange[2]), edgecolor='white', width=barWidth, label='S18')
            plt.bar(x_pos, binAcc[16,:], bottom=bar0x2, color=cmap(crange[3]), edgecolor='white', width=barWidth, label='S17')
            plt.bar(x_pos, binAcc[15,:], bottom=bar0x3, color=cmap(crange[4]), edgecolor='white', width=barWidth, label='S16')
            plt.bar(x_pos, binAcc[14,:], bottom=bar0x4, color=cmap(crange[5]), edgecolor='white', width=barWidth, label='S15')
            plt.bar(x_pos, binAcc[13,:], bottom=bar0x5, color=cmap(crange[6]), edgecolor='white', width=barWidth, label='S14')
            plt.bar(x_pos, binAcc[12,:], bottom=bar0x6, color=cmap(crange[7]), edgecolor='white', width=barWidth, label='S13')
            plt.bar(x_pos, binAcc[11,:], bottom=bar0x7, color=cmap(crange[8]), edgecolor='white', width=barWidth, label='S12')
            plt.bar(x_pos, binAcc[10,:], bottom=bar0x8, color=cmap(crange[9]), edgecolor='white', width=barWidth, label='S11')
            plt.bar(x_pos, binAcc[9,:], bottom=bar0x9, color=cmap(crange[10]), edgecolor='white', width=barWidth, label='S10')
            plt.bar(x_pos, binAcc[8,:], bottom=bar0x10, color=cmap(crange[11]), edgecolor='white', width=barWidth, label='S09')
            plt.bar(x_pos, binAcc[7,:], bottom=bar0x11, color=cmap(crange[12]), edgecolor='white', width=barWidth, label='S08')
            plt.bar(x_pos, binAcc[6,:], bottom=bar0x12, color=cmap(crange[13]), edgecolor='white', width=barWidth, label='S07')
            plt.bar(x_pos, binAcc[5,:], bottom=bar0x13, color=cmap(crange[14]), edgecolor='white', width=barWidth, label='S06')
            plt.bar(x_pos, binAcc[4,:], bottom=bar0x14, color=cmap(crange[15]), edgecolor='white', width=barWidth, label='S05')
            plt.bar(x_pos, binAcc[3,:], bottom=bar0x15, color=cmap(crange[16]), edgecolor='white', width=barWidth, label='S04')
            plt.bar(x_pos, binAcc[2,:], bottom=bar0x16, color=cmap(crange[17]), edgecolor='white', width=barWidth, label='S03')
            plt.bar(x_pos, binAcc[1,:], bottom=bar0x17, color=cmap(crange[18]), edgecolor='white', width=barWidth, label='S02')
            plt.bar(x_pos, binAcc[0,:], bottom=bar0x18, color=cmap(crange[19]), edgecolor='white', width=barWidth, label='S01')
        else: #reverse order
            bar0x1 = np.add(binAcc[0,:],binAcc[1,:]).tolist()
            bar0x2 = np.add(bar0x1,binAcc[2,:]).tolist()
            bar0x3 = np.add(bar0x2,binAcc[3,:]).tolist()
            bar0x4 = np.add(bar0x3,binAcc[4,:]).tolist()
            bar0x5 = np.add(bar0x4,binAcc[5,:]).tolist()
            bar0x6 = np.add(bar0x5,binAcc[6,:]).tolist()
            bar0x7 = np.add(bar0x6,binAcc[7,:]).tolist()
            bar0x8 = np.add(bar0x7,binAcc[8,:]).tolist()
            bar0x9 = np.add(bar0x8,binAcc[9,:]).tolist()
            bar0x10 = np.add(bar0x9,binAcc[10,:]).tolist()
            bar0x11 = np.add(bar0x10,binAcc[11,:]).tolist()
            bar0x12 = np.add(bar0x11,binAcc[12,:]).tolist()
            bar0x13 = np.add(bar0x12,binAcc[13,:]).tolist()
            bar0x14 = np.add(bar0x13,binAcc[14,:]).tolist()
            bar0x15 = np.add(bar0x14,binAcc[15,:]).tolist()
            bar0x16 = np.add(bar0x15,binAcc[16,:]).tolist()
            bar0x17 = np.add(bar0x16,binAcc[17,:]).tolist()
            bar0x18 = np.add(bar0x17,binAcc[18,:]).tolist()
            plt.bar(x_pos, binAcc[0,:], color=cmap(crange[0]), edgecolor='white', width=barWidth, label='S1')
            plt.bar(x_pos, binAcc[1,:], bottom=binAcc[0,:], color=cmap(crange[1]), edgecolor='white', width=barWidth, label='S2')
            plt.bar(x_pos, binAcc[2,:], bottom=bar0x1, color=cmap(crange[2]), edgecolor='white', width=barWidth, label='S3')
            plt.bar(x_pos, binAcc[3,:], bottom=bar0x2, color=cmap(crange[3]), edgecolor='white', width=barWidth, label='S4')
            plt.bar(x_pos, binAcc[4,:], bottom=bar0x3, color=cmap(crange[4]), edgecolor='white', width=barWidth, label='S5')
            plt.bar(x_pos, binAcc[5,:], bottom=bar0x4, color=cmap(crange[5]), edgecolor='white', width=barWidth, label='S6')
            plt.bar(x_pos, binAcc[6,:], bottom=bar0x5, color=cmap(crange[6]), edgecolor='white', width=barWidth, label='S7')
            plt.bar(x_pos, binAcc[7,:], bottom=bar0x6, color=cmap(crange[7]), edgecolor='white', width=barWidth, label='S8')
            plt.bar(x_pos, binAcc[8,:], bottom=bar0x7, color=cmap(crange[8]), edgecolor='white', width=barWidth, label='S9')
            plt.bar(x_pos, binAcc[9,:], bottom=bar0x8, color=cmap(crange[9]), edgecolor='white', width=barWidth, label='S10')
            plt.bar(x_pos, binAcc[10,:], bottom=bar0x9, color=cmap(crange[10]), edgecolor='white', width=barWidth, label='S11')
            plt.bar(x_pos, binAcc[11,:], bottom=bar0x10, color=cmap(crange[11]), edgecolor='white', width=barWidth, label='S12')
            plt.bar(x_pos, binAcc[12,:], bottom=bar0x11, color=cmap(crange[12]), edgecolor='white', width=barWidth, label='S13')
            plt.bar(x_pos, binAcc[13,:], bottom=bar0x12, color=cmap(crange[13]), edgecolor='white', width=barWidth, label='S14')
            plt.bar(x_pos, binAcc[14,:], bottom=bar0x13, color=cmap(crange[14]), edgecolor='white', width=barWidth, label='S15')
            plt.bar(x_pos, binAcc[15,:], bottom=bar0x14, color=cmap(crange[15]), edgecolor='white', width=barWidth, label='S16')
            plt.bar(x_pos, binAcc[16,:], bottom=bar0x15, color=cmap(crange[16]), edgecolor='white', width=barWidth, label='S17')
            plt.bar(x_pos, binAcc[17,:], bottom=bar0x16, color=cmap(crange[17]), edgecolor='white', width=barWidth, label='S18')
            plt.bar(x_pos, binAcc[18,:], bottom=bar0x17, color=cmap(crange[18]), edgecolor='white', width=barWidth, label='S19')
            plt.bar(x_pos, binAcc[19,:], bottom=bar0x18, color=cmap(crange[19]), edgecolor='white', width=barWidth, label='S20')
            
        Box = ax.get_position()
        ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.11), fancybox=True, shadow=False, ncol=10)
        plt.xlabel(r'CM$_{%s}$ State' % (CM+1), fontsize=18, labelpad=10)
        plt.ylabel('Occupancy', fontsize=18, labelpad=10)
        plt.xlim(0.25, bins+.75)
        plt.ylim(0, (np.amax(occmapAll)+.02*np.amax(occmapAll)))
        plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))
        plt.gca().yaxis.set_major_locator(MaxNLocator(integer=True))
        plt.gca().xaxis.set_major_locator(plt.MultipleLocator(1))
        #plt.axhline(y=tau*20*totalPDs_thresh, color='k', linewidth=1.5)
        plt.subplots_adjust(left=0.075, right=0.415, bottom=0.075, top=0.45, wspace=0.2, hspace=0.2)
        plt.show()
        plt.clf()