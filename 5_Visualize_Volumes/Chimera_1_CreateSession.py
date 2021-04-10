import os
from chimera import runCommand as rc

# =============================================================================
# Generate Chimera Session (run via `chimera 1_CreateSession.py`)
# =============================================================================
# SETUP: First, make sure the data directory below is correct for your dataset...
#   ...via the 'stackDir' variable. As well, decide on what states to view...
#   ...for either 1D or 2D frameworks.
# RUNNING: After opening with Chimera, you can optionally change the camera...
#   ...view. When finalized, save the Chimera session via 'Save Session As'.
# =============================================================================
# Author:    E. Seitz @ Columbia University - Frank Lab - 2020-2021
# Contact:   evan.e.seitz@gmail.com
# =============================================================================


pyDir = os.path.dirname(os.path.abspath(__file__)) #python file location
parDir = os.path.abspath(os.path.join(pyDir, os.pardir))

if 0: #examine 3D movies for 1 degree of freedom
    stackDir = os.path.join(parDir, '4_Compile_Bins/S2_bins/CM1')
    states=['01','02','03','04','05','06','07','08','09','10','11','12','13','14','15','16','17','18','19','20']
    fnames = []
    for i in states:
        fnames.append(os.path.join(pyDir, 'CM1_bin%s.mrc' % i))
        
else: #examine 3D movies for 2 degrees of freedom
    stackDir = os.path.join(parDir, '4_Compile_Bins/S2_bins/CM1_CM2')
    fnames = []
    for file in os.listdir(stackDir):
        if file.endswith('.mrc'):
            if file.startswith('CM1_05'): #select custom paths in state space
                fnames.append(os.path.join(stackDir, file))   

idx = 0	
for f in fnames:
    rc('open' + f)
    rc('volume #%d step 1' % idx)
    rc('sop hideDust #%s size 100' % idx)
    rc("volume #%d hide" % idx)
    idx += 1
