import sys, os
import numpy as np
import matplotlib
from matplotlib import rc
from matplotlib.figure import Figure
import matplotlib.pyplot as plt
from matplotlib import cm
from pylab import imshow, show
import matplotlib.gridspec as gridspec
import Generate_Nd_Rot
pyDir = os.path.dirname(os.path.abspath(__file__)) #python file location
parDir = os.path.abspath(os.path.join(pyDir, os.pardir))
fitDir = os.path.join(parDir, '3_Manifold_Binning') #ZULU change
sys.path.append(fitDir)
import ConicFit_Parabola

if 0: #render with LaTeX font for figures
    rc('text', usetex=True)
    rc('font', family='serif')

# =============================================================================
# Find optimal d-dimensional rotation for each manifold via 2D histogram...
# ...method as well as isoalte most-probable CM sub-manifolds:
# =============================================================================
# USAGE: This script is optimized for the isolation of the first two leading...
#   ...conformational motions. It will eventually need to be modified to...
#   ...properly handle the more complex scenario of n>2 degrees of freedom.
#   Example DM outputs have been provided in the 'Data_Manifolds_126' folder...
#   ...within this directory for experimental use, which were calculated from...
#   ....images on half of a great circle (126 PDs) in SS_2 (2 CMs) with each...
#   ...image generated with CTF=[5k,15k] and SNR=0.1.
# SETUP: First, make sure all user parameters are correct for your dataset...
#   ...below via the 'User parameters' section ('PCA' for embedding type...
#   ...and correct input file names via 'maniPath' variable). You may also...
#   ...want to adjust the number of dimensions 'dim' to consider. See comments...
#   ...for other parameter choices if errors arise.
# RUNNING: To run a series of PDs at once: first edit `1_RotHist_Batch.sh`...
#   ...for the total number of PDs requested; e.g., {1...5} for 5 PDs...
#   ...or {1...1} for only the first PD.
#   ...Then start batch processing via `sh 1_RotHist_Batch.sh`
# =============================================================================
# Author:    E. Seitz @ Columbia University - Frank Lab - 2020
# Contact:   evan.e.seitz@gmail.com
# =============================================================================

def op(pyDir, PD):
    #pyDir = os.path.dirname(os.path.abspath(__file__)) #python file location
    #PD = '001' #for use if not calling batch via .sh    
    parDir = os.path.abspath(os.path.join(pyDir, os.pardir))

    # =========================================================================
    # User parameters:
    # =========================================================================
    groundTruth = True #use GT indices for visualizations; see '0_Data_Inputs/GroundTruth_Indices'
    viewCM = 1 #{1,2, etc.}; if using ground-truth, CM reference frame to use for color map indices
    PCA = False #specify if manifolds from PCA or DM folder {if False, DM is True}
    CTF = True #if CTF protocols previously used in DM
    
    outDir = os.path.join(pyDir, 'Data_Rotations')
    if not os.path.exists(outDir):
        os.mkdir(outDir)
    pdDir = os.path.join(outDir, 'PD%s' % PD)
    if not os.path.exists(pdDir):
        os.mkdir(pdDir)
        
    if groundTruth is True:
        if viewCM == 1: #view in reference frame of CM1
            CM_idx = np.load(os.path.join(parDir, '0_Data_Inputs/GroundTruth_Indices/CM1_Indices.npy'), allow_pickle=True)
        elif viewCM == 2: #view in reference frame of CM2
            CM_idx = np.load(os.path.join(parDir, '0_Data_Inputs/GroundTruth_Indices/CM2_Indices.npy'), allow_pickle=True)
        
    if PCA is True: #change end of file path to match name of your PCA outputs
        maniPath = os.path.join(parDir, '1_Embedding/PCA/Data_Manifold/PD_%s_vec.npy' % PD)
    else: #change end of file path to match name of your DM outputs
        maniPath = os.path.join(parDir, '1_Embedding/DM/Data_Manifolds/PD_%s_vec.npy' % PD)
    print('Manifold Info:', np.shape(np.load(maniPath)))
    U0 = np.load(maniPath) #eigenvectors

    # =========================================================================
    # Import data into arrays and setup parameters:
    # =========================================================================
    m = np.shape(U0)[0] #number of images for colormap (number of initial states*tau)
    enum = np.arange(1,m+1)
    dim = 6 #analyze 'dim'-dimensional manifold subspace with 'dim'('dim'-1)/2 rotation operators
    theta_total = int(dim*(dim-1)/2) #total number of rotation operators

    # =========================================================================
    # Visualization parameters:
    # =========================================================================
    cmap = 'nipy_spectral' #'gist_rainbow'
    s = 20 #scatter point size
    lw = .5 #scatter point line width
    
    # =========================================================================
    # Normalize manifold range and initiate subspace:
    # =========================================================================
    def normalize(_d, to_sum=False, copy=True): #normalize all eigenfunctions
        # d is a (n x dimension) np array
        d = _d if not copy else np.copy(_d)
        d -= np.min(d, axis=0)
        d /= (np.sum(d, axis=0) if to_sum else np.ptp(d, axis=0)) #normalize btw [0,1]
        d = 2.*(d - np.min(d))/np.ptp(d)-1 #scale btw [-1,1]
        return d
    
    U = normalize(U0) #rescale manifold between -1 and +1 (all axes)
    if PCA is True:
        U_init = U[:,0:dim] #manifold subspace (for 'dim' dimensions)
    else: #if DM, don't use steady-state eigenvector (v_0)
        U_init = U[:,1:dim+1]
        
    if 1: #render an organized array of pre-rotated 2D subspaces
        fig = plt.figure()
        dimRows, dimCols = dim-1, dim-1
        idx = 1
        plt.rc('font', size=6)
        for v1 in range(1,dimRows+1):
            for v2 in range(v1+1, v1+dimCols+1):
                plt.subplot(dimRows, dimCols, idx)
                try:
                    if groundTruth is True:
                        color=iter(cm.tab20(np.linspace(1, 0, np.shape(CM_idx)[0])))
                        for b in range(np.shape(CM_idx)[0]):
                            c=next(color)
                            plt.scatter(U_init[:,v1-1][CM_idx[b]], U_init[:,v2-1][CM_idx[b]], color=c, s=15, edgecolor='k', linewidths=.1, zorder=1)
                    else:
                        plt.scatter(U_init[:,v1-1], U_init[:,v2-1], c='white', s=20, linewidths=.5, edgecolor='k', zorder=0)
                except:
                    plt.scatter(0,0, c='k')
                plt.title(r'$\Psi_{%s}$, $\Psi_{%s}$' % (int(v1), int(v2)), fontsize=6)
                plt.ticklabel_format(style='sci', axis='x', scilimits=(0,0))
                plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0)) 
                if 1:
                    frame = plt.gca()
                    frame.axes.xaxis.set_ticklabels([])
                    frame.axes.yaxis.set_ticklabels([])
                    plt.gca().set_xticks([])
                    plt.xticks([])
                    plt.gca().set_yticks([])
                    plt.yticks([])
                plt.axis('scaled')
                idx+=1
        plt.tight_layout()
        fig = plt.gcf()
        fig.savefig(os.path.join(pdDir,'00_Initial_Subspaces.png'), dpi=200)
        plt.clf()
                    
    # ========================================================================
    # Fit each 2D subspace to identify parabola in each row:
    # ========================================================================
    def findParabolas(dim, U_init, itIdx):
        figFits = plt.figure(constrained_layout=True)
        figFitsSpec = gridspec.GridSpec(ncols=dim+1, nrows=dim, figure=figFits)
        plt.rc('font', size=6)
        dim_start = 1
        R2_best_vals = np.zeros(dim-1, dtype=float)
        R2_best_psi = np.zeros(dim-1, dtype=int)
        for v1 in range(dim-1):
            R2_best = 0
            for v2 in range(dim_start,dim):      

                if CTF is False:
                    cF, R2 = ConicFit_Parabola.fit3(U_init, v1, v2) #parabolic fit, no cross-terms
                    disc = -1 #dummy variable to compensate for CTF logic
                elif CTF is True: #need to compensate for inward-curling of manifolds due to defocus-tolerant kernel
                    cF1, cF, Theta2D, R2 = ConicFit_Parabola.fit2(U_init, v1, v2) #general conic with cross-terms
                    disc = cF[1]**2 - 4.*cF[0]*cF[2] #discriminant
                
                if R2 > R2_best and v2 != R2_best_psi[0]:#and disc < .01 #may need to be changed to disallow for certain cases of hyperbolas (via 'disc')...
                    R2_best = R2
                    R2_best_vals[v1] = R2
                    R2_best_psi[v1] = v2 #index of psi is minus one
                    
                if 1: #plot least square fits on each 2D subspace
                    figFits.add_subplot(figFitsSpec[v1,v2])
                    x_range = np.linspace(-1.1, 1.1, 1000)
                    XX, YY = np.meshgrid(x_range, x_range)
                    if CTF is False: #if ConicFit_Parabola.fit3 used above
                        plt.plot(x_range, cF[0]*x_range**2 + cF[1]*x_range + cF[2], c='C3', zorder=1, linewidth=1) 
                    elif CTF is True: #if ConicFit_Parabola.fit2 used above
                        Conic = cF[0]*XX**2 + cF[1]*XX*YY + cF[2]*YY**2 + cF[3]*XX + cF[4]*YY + cF[5] 
                        plt.contour(XX, YY, Conic, [0], colors='C3', zorder=1)
                    plt.scatter(U_init[:,v1], U_init[:,v2], s=5, c='white', linewidth=lw, edgecolor='lightgray', zorder=-1) #c=enum, cmap=cmap
                    plt.title('R$^{2}$=%.3f; %.3f' % (R2, disc), fontsize=4)
                    plt.ticklabel_format(style='sci', axis='x', scilimits=(0,0))
                    plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
                    frame = plt.gca()
                    frame.axes.xaxis.set_ticklabels([])
                    frame.axes.yaxis.set_ticklabels([])
                    plt.gca().set_xticks([])
                    plt.xticks([])
                    plt.gca().set_yticks([])
                    plt.yticks([])
                    plt.axis('scaled')
            dim_start += 1
        figFits.savefig(os.path.join(pdDir,'01_Fitted_Subspaces_%s.png' % itIdx), dpi=200)
        plt.clf()
        
        # Exception (rare, still experimenting with) if manifold is initially rotated such that no parabolas reasonably detected:
        if (R2_best_vals[0] < 0.2) and (itIdx == 1): #value may need to be adjusted based on dataset
            itIdx += 1            
            thetas = np.zeros(shape=(theta_total,1), dtype=float)
            thetas[0] = 45*np.pi/180 #Rij operator may need to be experimented with
            R = Generate_Nd_Rot.genNdRotations(dim, thetas)
            U_rot = np.matmul(R, U_init.T)
            U_init = U_rot.T
            R2_best_psi, U_init, itIdx = findParabolas(dim, U_init, itIdx)
            return np.array([R2_best_psi, U_init, itIdx], dtype=object)
        else:
            return np.array([R2_best_psi, U_init, itIdx], dtype=object)
        
    itIdx = 1 #for determining if above exception (in 'findParabolas') is required
    R2_best_psi, U_init, itIdx = findParabolas(dim, U_init, itIdx)
        
    comboList = [] #discovered parabolic subspaces
    saveEigs = np.zeros(dim-1) #keep track of parabolic subspaces for subsequent scripts
    catch_harmonics = [] #list to ensure lowest-order parabolic harmonics are not stored as potential CM subspaces
    for i in range(dim-1):
        if i not in catch_harmonics:
            if R2_best_psi[i] < i: #exception for scenario above when all discriminants are hyperbolic
                comboList.append((i,i+1))
                saveEigs[i] = i+1
            else:
                comboList.append((i,R2_best_psi[i]))
                saveEigs[i] = R2_best_psi[i]
            catch_harmonics.append(i)
            catch_harmonics.append(R2_best_psi[i])
    print('Parabolic subspaces:', comboList) #zero-indexing
    
    comboFinal = [] #find proper combination for d-dim rotation operators
    for c in range(1,dim-1):
        c1, c2 = comboList[0]
        if c1 in comboList[c] or c2 in comboList[c]: #avoids harmonics
            pass
        else:
            comboFinal.append((c1, comboList[c][0]))
            comboFinal.append((c1, comboList[c][1]))
            comboFinal.append((c2, comboList[c][0]))
            comboFinal.append((c2, comboList[c][1]))
            break
    print('Rij:', comboFinal)
    
    Rij_1 = 1
    Rij_2 = 2
    Rij_idx = 1 #keep track of which rows (eigenfunctions) are being operated on in rotation matrix
    Rij_final = [] #zero-index

    for Rij in range(theta_total): #index for different rotation sub-matrices
        # Note for R_ij: index for different rotation sub-matrices; e.g., for dim=5...
        # ...there exists 10 sub-matrices; of which R_ij=0 rotates {Psi_1, Psi_2};
        # ...R_ij=1 rotates {Psi_1, Psi_3}; R_ij=2 rotates {Psi_1, Psi_4}; ...;
        # ...R_ij=4 rotates {Psi_2, Psi_3}; R_ij=7 rotates {Psi_3, Psi_4}, etc.        
        Rij_list = [] #store leading Rij's
        d_idx = 0
        d_iter = 0
        for d in range(dim-1):
            if d == 0:
                pass
            else:
                d_iter += (dim-d_idx)
                Rij_list.append(d_iter)
            d_idx += 1    
        
        if (Rij_1 == comboFinal[0][0]+1 and Rij_2 == comboFinal[0][1]+1) or (Rij_2 == comboFinal[0][0]+1 and Rij_1 == comboFinal[0][1]+1):
            Rij_final.append(Rij)
        if (Rij_1 == comboFinal[1][0]+1 and Rij_2 == comboFinal[1][1]+1) or (Rij_2 == comboFinal[1][0]+1 and Rij_1 == comboFinal[1][1]+1):
            Rij_final.append(Rij)
        if (Rij_1 == comboFinal[2][0]+1 and Rij_2 == comboFinal[2][1]+1) or (Rij_2 == comboFinal[2][0]+1 and Rij_1 == comboFinal[2][1]+1):
            Rij_final.append(Rij)
        if (Rij_1 == comboFinal[3][0]+1 and Rij_2 == comboFinal[3][1]+1) or (Rij_2 == comboFinal[3][0]+1 and Rij_1 == comboFinal[3][1]+1):
            Rij_final.append(Rij)
  
        if Rij_2 < dim:
            Rij_2 += 1
        else:
            Rij_1 += 1
            Rij_2 = 2 + Rij_idx
            Rij_idx += 1 
    print('Rij indices:', Rij_final)
                
    # =========================================================================
    # Perform series of 'dim'-dimensional rotations iteratively on set of 2D subspaces...
    # ...then bin each subspace via 2D histograms, and calculate number of nozero...
    # ...bins therein (for each 2D subspace) as a funciton of theta
    # =========================================================================
    thetas_all = np.zeros(shape=(theta_total,len(comboList))) #optimal set of Rij for each set of eigenfunctions in comboList
    # Note for thetas_all: each row will be filled with the set of optimal thetas for each rotation submatrix for each 2D subspace
    start, stop, step = -40, 40, .5 #range and density of thetas for n-dimensional rotations
    steps = np.abs(int((start - stop) / step))
    
    comboListIdx = 0
    for v1, v2 in comboList:
        #print('Psi %s, Psi %s' % (v1,v2))
        thetas = np.zeros(shape=(theta_total,1), dtype=float)

        for Rij in Rij_final:
            meas = np.zeros(steps)              
            step_i = 0
            theta_list = np.arange(start, stop, step)
            for i in theta_list:
                thetas[Rij] = i*np.pi/180
                R = Generate_Nd_Rot.genNdRotations(dim, thetas)
                U_rot = np.matmul(R, U_init.T)
                U_rot = U_rot.T
    
                bins = 51
                proj = np.vstack((U_rot[:,v1], U_rot[:,v2])).T
                H, edges = np.histogramdd(proj, bins=bins, normed=False, range=((-1.5,1.5),(-1.5,1.5)))
                NumberOfZeros = 100-(np.count_nonzero(H)/(float(bins**2)))*100 #as percentage of total bins
                meas[step_i] = NumberOfZeros
                if 0: #for viewing intermediate results only
                    if v1 == 1 and v2 == 4:
                        if 0: #2D histogram 
                            plt.imshow(H.T)
                        else:
                            plt.scatter(U_rot[:,v1], U_rot[:,v2], c=enum, cmap=cmap, s=s, linewidths=lw, edgecolor='k', zorder=0)
                        plt.title('%.3f, %.3f' % (NumberOfZeros, i))
                        fig = plt.gcf()
                        fig.savefig(os.path.join(pdDir,'temp_%s_%s.png' % (Rij, step_i)))
                        #plt.show()
                        plt.clf()        
                step_i += 1 

            meas_maxIdx = np.argmax(meas)
            theta_min = theta_list[meas_maxIdx]
            thetas[Rij] = theta_min*np.pi/180
            print('Rij_%s theta: %s' % (Rij, thetas[Rij]))

        # =====================================================================
        # Plot each fully-rotated (and fitted) 2D subspace and save image to file:
        # =====================================================================
        R = Generate_Nd_Rot.genNdRotations(dim, thetas)
        U_rot = np.matmul(R, U_init.T)
        U_rot = U_rot.T
        
        plt.subplot(1,1,1)
        if groundTruth is True:
            color=iter(cm.tab20(np.linspace(1, 0, np.shape(CM_idx)[0])))
            for b in range(np.shape(CM_idx)[0]):
                c=next(color)
                plt.scatter(U_rot[:,v1][CM_idx[b]], U_rot[:,v2][CM_idx[b]], color=c, s=15, edgecolor='k', linewidths=.15, zorder=1)
        else:
            plt.scatter(U_rot[:,v1], U_rot[:,v2], c='white', s=20, linewidths=.5, edgecolor='k', zorder=0)
            
        plt.title(r'$\Psi_{%s}$ $\times$ $\Psi_{%s}$' % (int(v1)+1, int(v2)+1), fontsize=12)
        plt.axis('scaled')
        plt.xlim(-1.1, 1.1)
        plt.ylim(-1.1, 1.1)
        fig = plt.gcf()
        fig.savefig(os.path.join(pdDir,'%s_Subspace_v%s_v%s.png' % (v1+1,v1+1,v2+1)), dpi=200)
        plt.clf()
        
        # =====================================================================
        # Update angles:
        # =====================================================================
        if itIdx > 1:
            thetas[0] += 45*np.pi/180
        
        thetas_all[:,comboListIdx] = thetas.T
        comboListIdx += 1
    thetas_all = thetas_all.T

    # =========================================================================
    # Save subspace and rotation information to file:
    # =========================================================================    
    np.save(os.path.join(pdDir, 'PD%s_RotMatrices.npy' % PD), thetas_all)
    np.save(os.path.join(pdDir, 'PD%s_RotEigenfunctions.npy' % PD), saveEigs)
    np.save(os.path.join(pdDir, 'PD%s_RotParameters' % PD), np.array([dim]))
    print('')
    

if __name__ == '__main__':
    path1 = os.path.splitext(sys.argv[0])[0]
    path2, tail = os.path.split(path1)
    op(path2, sys.argv[1])