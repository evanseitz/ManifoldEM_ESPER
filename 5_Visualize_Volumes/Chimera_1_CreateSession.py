#run via: chimera 1_CreateSession.py

################################################################################
# GENERATE CHIMERA SESSION #
################################################################################

import os
from chimera import runCommand as rc

pyDir = os.path.dirname(os.path.abspath(__file__)) #python file location

states=['01','02','03','04','05','06','07','08','09','10','11','12','13','14','15','16','17','18','19','20']

fnames = []
for i in states:
    fnames.append(os.path.join(pyDir, 'CM1_bin%s.mrc' % i))     

idx = 0	
for f in fnames:
    rc('open' + f)
    rc('volume #%d step 1' % idx)
    rc('sop hideDust #%s size 100' % idx)
    rc("volume #%d hide" % idx)
    idx += 1
