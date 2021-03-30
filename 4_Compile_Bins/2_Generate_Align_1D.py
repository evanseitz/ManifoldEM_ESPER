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
# Generate alignment files (STAR format) for each of the compiled bins
# =============================================================================
# SETUP: `1_Compile_Bins.py` must be run before initiating this script.
#   Additionally, the original angles used to generate each PD (for ground-truth)
#   ...must be available (an example has been provided in this repository...
#   ...within the `0_Data_Inputs_126gc` folder). Alter the `projDir` variable...
#   ...below to match the location of your PD angles.
# RUNNING: Initiate batch processing via `python 2_Generate_Align_1D.py`
# POST-PROCESSING: After running this script, run the next step via...
#   ...`sh 3_Volumes_Bins_1D.sh`, which will call RELION for all STARs via: 
#   ....`relion_reconstruct --i {angles}.star --o {relion_reconstruct}.mrc`
# =============================================================================
# Author:    E. Seitz @ Columbia University - Frank Lab - 2020
# Contact:   evan.e.seitz@gmail.com
# =============================================================================

pyDir = os.path.dirname(os.path.abspath(__file__)) #python file directory
parDir = os.path.abspath(os.path.join(pyDir, os.pardir))
bins = 20

for CM in [1,2]: #CMs to consider
    cmDir = os.path.join(pyDir, 'S2_bins/CM%s' % CM)

    for b in range(bins):
        B = "{0:0=2d}".format(b+1)
        print('CM_%s, Bin_%s' % (CM,B))
        
        # =============================================================================
        # Read in initial image stack:
        # =============================================================================
        fname = 'CM%s_bin%s.mrcs' % (CM, B)
        stack_file = os.path.join(cmDir, fname)
        init_stack = mrcfile.mmap(stack_file)
        snapshots, box, box = init_stack.data.shape
        init_stack.close()

        # =============================================================================
        # Initiate alignment file:
        # =============================================================================
        alignOut = os.path.join(cmDir, 'CM%s_bin%s.star' % (CM, B))
        if os.path.exists(alignOut):
            os.remove(alignOut)

        # =============================================================================
        # Read in initial angles used to generate PDs:
        # =============================================================================
        projDir = os.path.join(parDir, '0_Data_Inputs_126gc')
        for file in os.listdir(projDir):
            if file.endswith('Euler.txt'):
                projPath = os.path.join(projDir, file)
        proj = np.genfromtxt(projPath, unpack=True).T

        # =============================================================================
        # Initialize final data structures:
        # =============================================================================
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

        ###################################################
        # EMAN2 to RELION conventions:                    #
        # =============================================== #
        # az - 90 = AngleRot (a1); [-180,180] in RELION   #
        # alt = AngleTilt (a2); [0, 180] in RELION        #
        # phi + 90 = AnglePsi (0); [-180, 180] in RELION  #
        ###################################################
                
        # =============================================================================
        # Initiate main loop:
        # =============================================================================
        idx = 0
        for pd in range(snapshots):
            #print('Particle:', pd+1)
            projPD = proj[pd] 
            # update alignment file:
            alignFile.write('%s\t%s\t%s@%s\t%.6f\t%.6f\t%.6f\t%s\t%s\t%s\t%s\n' \
                % (0., 0., idx+1, fname, float(projPD[3]+90), \
                (float(projPD[1])-90), float(projPD[2]), .1, 300., 1., 10000.))        
            idx += 1
           
        alignFile.close()
