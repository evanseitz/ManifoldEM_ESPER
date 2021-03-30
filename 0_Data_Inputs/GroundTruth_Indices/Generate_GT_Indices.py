import os
import numpy as np

# =============================================================================
# Generate ground-truth indices (if using synthetic data):
# =============================================================================
# SETUP: This workflow is applicable in the case of using synthetic data...
#   ...as similarly constructed via our previously published repository:
#   https://github.com/evanseitz/cryoEM_synthetic_continua
#   If using this workflow, place the occupancy map (.npy) generated in step...
#   ...'5_GenOcc_python' into this directory (or change path below). This...
#   ...code will output the indices of images as conveniently ordered according...
#   ...to your ground-truth conformational reaction coordinates; i.e., one...
#   ...sequence of indices for each degree of freedom in your dataset.
#   ...These files can later be read into subsequent scripts to color code...
#   ...manifold visualizations, so long as the parameter 'GroundTruth' in those...
#   ...scripts is enabled, with paths linked back to this location. If more...
#   ...than two degrees of freedom were used, this code will need to be altered.
# RUNNING: This script can be run via 'python Generate_GT_Indices.py'
# =============================================================================
# Author:    E. Seitz @ Columbia University - Frank Lab - 2021
# Contact:   evan.e.seitz@gmail.com
# =============================================================================


pyDir = os.path.dirname(os.path.abspath(__file__)) #python file location

if 1: #use this branch if occupancies assigned from previously-generated occupancy map
    occPath = os.path.join(pyDir, 'Occ2D_4k.npy') #location of ground-truth occupancy map
    occFile = np.load(occPath)
    occFile.astype(int)
    
    states_gt = 20 #ground-truth number of states along each degree of freedom (assumed symmetric)
    ss_gt = 4000 #total number of images in dataset
    occ = []
    for i in range(1, states_gt+1): #[1,20]
        for j in range(1, states_gt+1): #[1,20]
            occ.append(int(occFile[j-1][i-1]))
            
    CM1_sums = np.sum(occFile, axis=0)
    CM2_sums = np.sum(occFile, axis=1)
    CM1_idx = []
    CM2_idx = []
    for i in range(states_gt):
        CM1_idx.append([])
        CM2_idx.append([])
    
    # ========================================================================
    # Generate CM1 ground-truth indexing:
    # ========================================================================
    step = 0
    for i in range(states_gt):
        for j in range(int(CM1_sums[i])):
            CM1_idx[i].append(step+int(j))
        step += int(CM1_sums[i])
        
    np.save(os.path.join(pyDir, 'CM1_Indices.npy'), CM1_idx)
    if 0:
        CM1_final = np.load('CM1_Indices.npy', allow_pickle=True)
        print(CM1_final)
                
    # ========================================================================
    # Generate CM2 ground-truth indexing:
    # ========================================================================
    step = 0
    for i in range(states_gt):
        if i > 0:
            step = CM2_idx[i-1][int(occFile[i-1,0]-1)]
        for j in range(states_gt):
            for k in range(int(occFile[i,j])):
                step+=1
                CM2_idx[i].append(step)
            if j < 19:
                step += int((np.sum(occFile[:,:], axis=0)[j]) - np.sum(occFile[:i+1,:], axis=0)[j] + np.sum(occFile[:i,:], axis=0)[j+1])
    #zero-indexing:
    for i in range(states_gt):
        for j in range(len(CM2_idx[i])):
            CM2_idx[i][j] -= 1
            
    np.save(os.path.join(pyDir, 'CM2_Indices.npy'), CM2_idx)
    if 0:
        CM2_final = np.load('CM2_Indices.npy', allow_pickle=True)
        print(CM2_final)
    
else: #use this branch for more simple tau*N uniform formulation (all states sampled tau times)   
    states_gt = 20 #ground-truth number of states along each degree of freedom (assumed symmetric)
    tau = 10 #see '0_Data_Inputs/Pristin_addTauSNR' for more information
    ss_gt = 4000 #total number of images in dataset
    CM1_idx = np.ndarray(shape=(states_gt, states_gt*tau), dtype=int)  
    CM2_idx = np.ndarray(shape=(states_gt, states_gt*tau), dtype=int)  

    # ========================================================================
    # Generate CM1 ground-truth indexing:
    # ========================================================================
    idx=0
    shift=0
    Idx=0
    for i in range(ss_gt):
        if Idx*tau <= i < (Idx+states_gt)*tau:
            CM1_idx[shift, idx-Idx*tau] = i
            idx+=1
            if idx%(states_gt*tau) == 0:
                Idx+=states_gt
                shift+=1
    
    np.save(os.path.join(pyDir, 'CM1_Indices_tau.npy'), CM1_idx)
    if 0:
        CM1_final = np.load('CM1_Indices_tau.npy', allow_pickle=True)
        print(CM1_final)
                
    # ========================================================================
    # Generate CM2 ground-truth indexing:
    # ======================================================================== 
    binsActual = []
    Idx = 0
    for s in range(0,states_gt):
        state_list = []
        for r in range(0,states_gt):
            state_list.append(np.arange(Idx,Idx+tau))
            Idx+=(states_gt*tau)
        binsActual.append([item for sublist in state_list for item in sublist])
        Idx-=(ss_gt-tau)
    
    for i in range(states_gt):
        for j in range(states_gt*tau):
            CM2_idx[i,j] = binsActual[i][j]
            
    np.save(os.path.join(pyDir, 'CM2_Indices_tau.npy'), CM2_idx)            
    if 0:
        CM1_final = np.load('CM2_Indices_tau.npy', allow_pickle=True)
        print(CM1_final)
  
