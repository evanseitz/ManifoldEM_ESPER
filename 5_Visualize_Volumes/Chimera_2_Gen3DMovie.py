import os
from chimera import runCommand as rc

# =============================================================================
# Generate 3D Movies in Chimera (run via `chimera 2_GenMovie.py`)
# =============================================================================
# SETUP: First, make sure your session has been saved via the previous script...
#   ...'Chimera_1_CreateSession.py'. Next, input that session name for the...
#   ...'session' variable below.
# RUNNING: After opening with Chimera, you can optionally change the camera...
#   ...view. When finalized, save the Chimera session via 'Save Session As'.
# =============================================================================
# Author:    E. Seitz @ Columbia University - Frank Lab - 2020-2021
# Contact:   evan.e.seitz@gmail.com
# =============================================================================


session = 'view1' #name of session saved in Chimera (e.g., 'view1' for view1.py)

pyDir = os.path.dirname(os.path.abspath(__file__)) #python file location
projDir = os.path.join(pyDir, '%s' % session)
outDir = os.path.join(pyDir, 'views/%s/%s' % (session, session)) #folder where files will be written
if not os.path.exists(os.path.join(pyDir, 'views')):
    os.mkdir(os.path.join(pyDir, 'views'))
if not os.path.exists(os.path.join(pyDir, 'views/%s' % session)):
    os.mkdir(os.path.join(pyDir, 'views/%s' % session))

wait_time=1
states=(1,21)

rc('open %s.py' % projDir)
rc('movie record')
for i in xrange(*states):
    rc('background solid black')
    rc('set projection orthographic')
    #rc('unset depthCue')
    #if 0: #SURFACE REPRESENTATION
        #rc('volume #%d show style surface step 1 level 0.008 color white' % (int(i-1)))
    #if 1: #SOLID REPRESENTATION
        #rc('volume #%s show style solid step 1 color gray' % (int(i-1)))
    #rc('sop hideDust #%s size 100' % (int(i-1)))
    rc('volume #%d show color white' % (int(i-1)))
    rc('wait %d' % wait_time)
    rc('copy file ' + outDir + '_%s.png' % (i))
    rc('volume #%d hide' % (int(i-1)))
##rc('close all')

# create MOV:
rc('movie stop')
rc('movie encode output %s.mov' % outDir)
