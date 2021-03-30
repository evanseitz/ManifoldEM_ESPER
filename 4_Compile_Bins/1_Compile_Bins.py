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
import json

if 1: #render with LaTeX font for figures
    rc('text', usetex=True)
    rc('font', family='serif')
    
# =============================================================================
# Compile CM bins across all PDs as obtained in previous step (`Manifold_Binning.py`)
# =============================================================================
# SETUP: First, make sure all user parameters are correct for your dataset...
#   ...below via the 'User parameters' section ('PCA' for embedding type...
#   ...and correct input file names for directories, if altered)
# IMPORTANT:
#   Before running, senses and CM indices must be assigned for each PD in the...
#   ...`3_Spline_Binning/bins_" directory. An example workflow, would include...
#   ...manually watching each of the CM movies (6, by default, per PD) and...
#   ...assigning those CM movies corresponding to true CM motions an index...
#   ...in the corresponding `CMs.txt` file as well as a sense in the corresponding...
#   ...`Senses.txt` file. For example, if the movie for CM1 in PD001 represented...
#   ...your molecule's primary motion (arbitrarily defined by the user as "primary")...
#   ...then alter the first entry in `CMs.txt` for that PD to ['1', ...], then...
#   ...arbitrarily define that primary motion with a directionality (i.e., "sense")...
#   ...of either forward (['F', ...]) or reverse (['R', ...]), making sure to keep...
#   ...this arbitrary assignment consistent for all PDs (and all CMs therein).
#   If a movie is not a valid motion, just leave it as an 'X' in both lists.
# RUNNING: To run a series of PDs at once: edit PD range() below`...
#   ...for the total number of PDs requested; e.g., range(1,6) for 5 PDs...
#   ...then initiate batch processing via `python 1_Compile_Bins.py`
# =============================================================================
# Author:    E. Seitz @ Columbia University - Frank Lab - 2020
# Contact:   evan.e.seitz@gmail.com
# =============================================================================

pyDir = os.path.dirname(os.path.abspath(__file__)) #python file location
parDir = os.path.abspath(os.path.join(pyDir, os.pardir))
outDir = os.path.join(pyDir, 'S2_bins_GC3_v5')
if not os.path.exists(outDir):
    os.makedirs(outDir)
    
# =============================================================================
# User parameters
# =============================================================================
PCA = True #specify if manifolds from PCA or DM folder {if False, DM is True}
totalPDs = 126
bins = 20
box = 250 #image dimensions
tau = 10 #number of noisy duplicates used to create stacks (synthetic data only)
groundTruth = True #optional, for comparing final outputs with ground-truth knowldege
R2_thresh = 0.825 #optional: skip PDs based on the current CM's parabolic fit score

# =============================================================================
# Main loop to compile CM bins across all PDs
# =============================================================================
total_CMs = 2 #total number of conformational motions to consider
for CM in range(total_CMs): #CMs to consider
    
    if groundTruth is True: #optional, for comparing with ground-truth
        states = 20
        ss = (states**2)*tau
        binAcc = np.zeros(shape=(bins,bins))
        binsActual = []
        if CM == 0:
            low = 0
            high = states*tau
            for idx in range(bins):
                binsActual.append(np.arange(low,high))
                low += states*tau
                high += states*tau
        elif CM == 1:
            Idx = 0
            for s in range(0,states):
                state_list = []
                for r in range(0,states):
                    state_list.append(np.arange(Idx,Idx+tau))
                    Idx+=(states*tau)
                binsActual.append([item for sublist in state_list for item in sublist])
                Idx-=(ss-tau)
     
    R2_all = []
    totalPDs_thresh = totalPDs #if PDs thresholded by R^2 below
    occmapAll = np.zeros(bins)
    occmapPDs = []
    for i in range(bins):
        occmapPDs.append([])
    for b in range(bins):
        B = "{0:0=2d}".format(b+1)
        print('CM_%s, Bin_%s' % ((CM+1),B))
        cmDir = os.path.join(outDir, 'CM%s' % (CM+1))
        if not os.path.exists(cmDir):
            os.makedirs(cmDir)
        stackOut = os.path.join(cmDir, 'CM%s_bin%s.mrcs' % ((CM+1),B))
        if os.path.exists(stackOut):
            os.remove(stackOut)
        # Stack of images across S2 for each bin:
        bin_stack = mrcfile.new_mmap(stackOut, shape=(totalPDs,box,box), mrc_mode=2, overwrite=True)

        #totalPDs = 5 #ZULU
        for pd in range(1,totalPDs+1):
            PD = '{0:0=3d}'.format(pd)
            #print(PD_%s' % PD)
            pdDir = os.path.join(parDir, '3_Spline_Binning/bins_PCA_GC3_v5/PD%.03d' % pd)
 
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
            #print(pd, stackDir)
            
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
                    stackPath = os.path.join(stackDir, file)
            init_stack = mrcfile.mmap(stackPath)
            # Arbitrary order; valid so long as it's consistent across all PDs
            if sense_idx == 'F':
                fix=0 #keep index order
            elif sense_idx == 'R':
                fix=1 #reverse index order
                        
            bin_stack.data[pd-1] = init_stack.data[b-fix*(2*b+1)]
            
            if 1: #optional: skip PDs based on the current CM's parabolic fit score
                if R2 < R2_thresh:
                    if b == 0:
                        totalPDs_thresh -= 1
                    continue
            
            occmapAll[b] += occmap[b-fix*(2*b+1)]
            occmapPDs[b].append(occmap[b-fix*(2*b+1)])
            
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
            # Compare number of elements in trajectory-bin vs ground-truth:
            # =================================================================
            if groundTruth is True:
                skips = np.arange(1, (bins*2)+1, 2)
                for i in range(0,bins):
                    binAcc[i-fix*skips[i],b] += sum(el in binFile for el in binsActual[i-fix*skips[i]]) #accuracy of single state
                                
            init_stack.close() #close up stack before next PD
            
        if 0: #plot of same-state images across S2
            for pd in range(0,10):#totalPDs):
                plt.imshow(bin_stack.data[pd], cmap='gray')
                plt.title('PD%.03d' % (pd+1))
                plt.show()
            
        bin_stack.close() #close up stack before next bin
    
    # =========================================================================
    # Save each CM occupancy map to file:
    # =========================================================================
    np.savetxt(os.path.join(cmDir,'Occupancy_Map.txt'), occmapAll, delimiter=',', fmt='%i')
    #print('Total Occupancy:',int(np.sum(occmapAll))) #sanity check
        
    # Save bar chart (standard form):
    plt.bar(range(1,bins+1),occmapAll)
    plt.xlabel(r'CM$_{%s}$ State' % (CM+1), fontsize=16, labelpad=10)
    plt.ylabel(r'Occupancy', fontsize=16, labelpad=10)
    plt.xlim(0.25, bins+.75)
    plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))
    plt.gca().yaxis.set_major_locator(MaxNLocator(integer=True))
    plt.gca().xaxis.set_major_locator(plt.MultipleLocator(1))
    plt.axhline(y=tau*20*totalPDs_thresh, color='r', linewidth=2)
    plt.tight_layout()
    fig = plt.gcf()
    fig.savefig(os.path.join(cmDir,'Occupancy_Sum.png'), dpi=200)
    #plt.show()
    plt.clf()
    plt.cla()
    plt.close()
    
    # Save R2 histogram:
    plt.hist(np.asarray(R2_all), bins=np.linspace(0,1,40), edgecolor='black')
    plt.xlabel('Coefficient of Determination')
    plt.ylabel('PDs')
    plt.tight_layout()
    fig = plt.gcf()
    fig.savefig(os.path.join(cmDir,'R2_Hist.png'), dpi=200)
    #plt.show()
    plt.clf()
    plt.cla()
    plt.close()
        
    if groundTruth is True: #bar chart (stacked); strictly for checking accuracy with ground-truth (if available)
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
        plt.xlabel(r'CM$_{%s}$ State' % (CM+1), fontsize=16, labelpad=10)
        plt.ylabel('Occupancy', fontsize=16, labelpad=10)
        plt.xlim(0.25, bins+.75)
        plt.ylim(0, (np.amax(occmapAll)+.01*np.amax(occmapAll)))
        plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))
        plt.gca().yaxis.set_major_locator(MaxNLocator(integer=True))
        plt.gca().xaxis.set_major_locator(plt.MultipleLocator(1))
        plt.axhline(y=tau*20*totalPDs, color='k', linewidth=1.5)
        plt.show()
        plt.clf()