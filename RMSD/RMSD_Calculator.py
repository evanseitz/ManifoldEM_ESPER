import sys, os
from pymol import stored
from pymol import cmd
import numpy as np

## PyMOL RMSD Calculator (run via 'pymol RMSD_Calculator.py')
## Authors: F. Acosta-Reyes @ Columbia University - Frank Lab - 2019
##          E. Seitz @ Columbia University - Frank Lab - 2019

def calc_rmsd(sel1, sel2):
    stored.rms1=[]
    stored.o1=[]
    stored.o2=[]    
    # store coordinates of the models
    cmd.iterate_state( 1, selector.process(sel1), "stored.o1.append([x,y,z])")
    cmd.iterate_state( 1, selector.process(sel2), "stored.o2.append([x,y,z])")
    rmst = 0
    for i in range(len(stored.o1)):
        if 0: #3d RMSD
            rmst += (np.square(np.abs(stored.o1[i][0]-stored.o2[i][0])) + np.square(np.abs(stored.o1[i][1]-stored.o2[i][1])) + np.square(np.abs(stored.o1[i][2]-stored.o2[i][2]))) #x,y,z 
        else: #2d RMSD
            rmst += (np.square(np.abs(stored.o1[i][0]-stored.o2[i][0])) + np.square(np.abs(stored.o1[i][1]-stored.o2[i][1]))) #x,y
            #rmst += (np.square(np.abs(stored.o1[i][0]-stored.o2[i][0])) + np.square(np.abs(stored.o1[i][2]-stored.o2[i][2]))) #x,z
            #rmst += (np.square(np.abs(stored.o1[i][1]-stored.o2[i][1])) + np.square(np.abs(stored.o1[i][2]-stored.o2[i][2]))) #y,z
    return np.sqrt(np.divide(rmst,len(stored.o1)))

pyDir = os.getcwd() #python file location, place in '2_GenStates_CM2' (if Hsp2D synthetic data)
CC_dir = os.path.join(pyDir, 'Generate_CC2')

CC_paths = []
for root, dirs, files in os.walk(CC_dir):
    for file in sorted(files):
        if not file.startswith('.'): #ignore hidden files
            if file.endswith(".pdb"):
                CC_paths.append(os.path.join(root, file))

for CC in CC_paths:
    name1 = os.path.basename(CC)
    name2 = os.path.splitext(name1)[0]
    cmd.load(filename=CC, object=name2)
    
states = range(1,21)
states = [str(item).zfill(2) for item in states] #leading zeros
D = np.ndarray(shape=(400,400), dtype=float)

count1 = 0
count2 = 0

for i in states:
    for j in states:
        for k in states:
            for l in states:
                print(i,j,k,l,'counts=',count1,count2)                
                #D[int(count1),int(count2)] = float(cmd.rms_cur('state_%s_%s' % (i,j), 'state_%s_%s' % (k,l))) #PyMOL equivalent (3D only); slower
                D[int(count1),int(count2)] = float(calc_rmsd('state_%s_%s' % (i,j), 'state_%s_%s' % (k,l)))
                print(D[int(count1),int(count2)])
                count2 += 1
                if int(k)*int(l) == 400:
                    count1 += 1
                    count2 = 0
  
if 1: #view matrix
    plt.title('RMSD Matrix for 2D Atomic Projections (XY)')
    plt.xlabel('State (PDB)')
    plt.ylabel('State (PDB)')
    plt.colorbar()
    plt.tight_layout()
    plt.show()
    
if 1: #save to file
    np.save('RMSD_2D_XY.npy', D)