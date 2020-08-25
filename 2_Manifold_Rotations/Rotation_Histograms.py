import sys, os, re
import numpy as np
import itertools
import matplotlib
from matplotlib import rc
from matplotlib.figure import Figure
import matplotlib.pyplot as plt
from pylab import imshow, show, loadtxt, axes
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.ticker import MaxNLocator
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.ticker
from matplotlib.lines import Line2D
from matplotlib.ticker import FuncFormatter
import matplotlib.gridspec as gridspec
from scipy.optimize import curve_fit
import Generate_Nd_Rot

if 0: #render with LaTeX font for figures
    rc('text', usetex=True)
    rc('font', family='serif')

# =============================================================================
# Find optimal N-dimensional rotation for each manifold via 2D histogram...
# ...method as well as most-probable CM submanifolds ranked in descended order 
# =============================================================================
# SETUP: First, make sure all user parameters are correct for your dataset...
#   ...below via the 'User parameters' section ('PCA' for embedding type...
#   ...and correct input file names via 'maniPath' variable)
# RUNNING: To run a series of PDs at once: first edit `1_RotHist_Batch.sh`...
#   ...for the total number of PDs requested; e.g., {1...5} for 5 PDs...
#   ...or {1...1} for only the first PD;
#   ...then start batch processing via `sh 1_RotHist_Batch.sh`
# =============================================================================
# Author:    E. Seitz @ Columbia University - Frank Lab - 2020
# Contact:   evan.e.seitz@gmail.com
# =============================================================================

def op(pyDir, PD):
    parDir = os.path.abspath(os.path.join(pyDir, os.pardir))
    outDir = os.path.join(pyDir, 'Data_Rotations')
    if not os.path.exists(outDir):
        os.mkdir(outDir)
    # =========================================================================
    # User parameters
    # =========================================================================
    pdDir = os.path.join(outDir, 'PD%s' % PD)
    if not os.path.exists(pdDir):
        os.mkdir(pdDir)
    PCA = True #specify if manifolds from PCA or DM folder {if False, DM is True}
    if PCA is True: #change end of file path to match name of your PCA outputs
        if 0:
            maniPath = os.path.join(parDir, '1_Embedding/PCA/Data_Manifolds/PD%s_tau5_SNR_vec.npy' % PD) 
        else: #example 126 PD great circle provided in repository
            maniPath = os.path.join(parDir, '1_Embedding/PCA/Data_Manifolds_126/PD%s_SS2_SNRpt1_tau5_vec.npy' % PD) 
    else: #change end of file path to match name of your DM outputs
        maniPath = os.path.join(parDir, '1_Embedding/DM/Data_Manifolds/PD%s_tau5_vec.npy' % PD)
    print('Manifold Info:', np.shape(np.load(maniPath)))
    
    # =========================================================================
    # Import data into arrays and setup parameters
    # =========================================================================
    U0 = np.load(maniPath) #eigenvectors
    m = np.shape(U0)[0] #number of images for colormap (number of initial states*tau)
    enum = np.arange(1,m+1)
    dim = 5 #analyze 5D manifold subspace with 5D rotation operator
    bins = 51 #bin parameter for 2D histogram
    
    # =========================================================================
    # Normalize manifold range and initiate subspace
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
    
    # =========================================================================
    # Create array of combinations for indexing 2D subspaces
    # =========================================================================
    comboSetup = []
    for d in range(dim):
        comboSetup.append(d+1)
    combo = itertools.combinations_with_replacement(comboSetup, 2)
    comboList = []
    for v1, v2 in list(combo):
        if v1 != v2:
            comboList.append((v1,v2)) #setup all combinations of 2D subspaces (up to dim)
    # Note: 'comboList' = [(1,2), (1,3),..., (1,dim), (2,3), (2,4), ..., (2,dim), etc.]
    # ...thus, 'dim'-1 leading Psi 1's, 'dim'-2 leading Psi 2's, etc.
            
    # =========================================================================
    # Perform series of 'dim'-dimensional rotations iteratively on set of 2D subspaces...
    # ...then bin each subspace via 2D histograms, and calculate number of nozero...
    # ...bins therein (for each 2D subspace) as a funciton of theta
    # =========================================================================
    start, stop, step = 0, 360, 1 #range of theta for n-dimensional rotations
    steps = np.abs(int((start - stop) / step))
    nonZ = np.zeros(shape=(steps,len(comboList))) #initiate list of 'theta vs. nonzero' for each 2D subspace
    step_i = 0
    theta_list = np.arange(start, stop, step)
    for i in theta_list:
        theta_total = int(float(dim)*(float(dim)-1.)/2.)
        thetas = np.zeros(shape=(theta_total,1), dtype=float)
        # genNdRotations.py is set up s.t. correct rotation submatrix has index dim-1:
        thetas[dim-1] = i*np.pi/180
        R = Generate_Nd_Rot.genNdRotations(dim, thetas)
        U_rot = np.matmul(U_init, R)
        idx = 0
        for v1,v2 in comboList:
            proj = np.vstack((U_rot[:,v1-1], U_rot[:,v2-1])).T
            H, edges = np.histogramdd(proj, bins=bins, normed=False, range=((-1.5,1.5),(-1.5,1.5)))
            nonzero = (np.count_nonzero(H)/(float(bins**2)))*100 #as percentage of total bins
            nonZ[:,idx][step_i] = nonzero
            idx+=1
        step_i += 1
    
    if 1: #plot 'theta vs. nonzeros' for set of 2D subspaces and save figure to file
        fig1 = plt.figure(constrained_layout=False) #prepare figure, part 1
        spec1 = gridspec.GridSpec(ncols=dim-1, nrows=dim-1, figure=fig1) #prepare figure, part 2
        idx=0
        for v1,v2 in comboList:
            f1_ax1 = fig1.add_subplot(spec1[int(v1-1), int(v2-2)])            
            plt.plot(nonZ[:,idx], linewidth=1)
            plt.title('Psi %s, Psi %s' % (v1,v2), fontsize=9)
            plt.gca().tick_params(axis='both', which='major', labelsize=7)
            plt.gca().tick_params(axis='both', which='minor', labelsize=7)        
            idx += 1
        plt.tight_layout()
        fig1.savefig(os.path.join(pdDir,'HistRot.png'))
        #plt.show()
        
    # =========================================================================
    # Determine out-of-phase duplicate 2D subspaces via correlation with 90 degree shift
    # =========================================================================
    CM_list = [] #list of tuples to locate each perfectly-correlated pair of 2D subspaces
    CM_idx = [] #list of single indices for above tuples
    for i in range(len(comboList)):
        for j in range(len(comboList)):
            if i < j and i != j: #no repeat or nontrivial combinations
                if np.std(nonZ[:,i]) > 1e-3 and np.std(np.roll(nonZ[:,j], 90)) > 1e-3:            
                    corr = np.corrcoef(nonZ[:,i], np.roll(nonZ[:,j], 90))[0,1]
                else:
                    corr = 0
                #print(comboList[i], comboList[j], corr)
                if corr > .9999: #find all perfectly correlated pairs
                    #print('Match found:',comboList[i], comboList[j], corr)
                    CM_list.append([comboList[i]]) #only need to keep track of one (i.e., not comboList[j]])
                    CM_idx.append(i)
                    #CM_idx.append(j)
    
    # For heightened precision, keep track of optimal angle for each possible CM:
    CM_info = np.ndarray(shape=(len(CM_idx),5))
    for i in range(len(CM_list)): 
        CM_info[i,0] = CM_idx[i] #1st column: 2D subspace location
        CM_info[i,1] = np.argmin(nonZ[:,CM_idx[i]]) #2nd column: optimal theta
        #note: 3rd column is supplied in next step (coefficient of determination, R^2)
        CM_info[i,3] = CM_list[i][0][0] #keep 2D subspace's 1st eigenvector index
        CM_info[i,4] = CM_list[i][0][1] #keep 2D subspace's 2nd eigenvector index
        
    # =========================================================================
    # Rank each potential 2D subspace via parabola fitting
    # =========================================================================  
    def ParabolicFit(x, a, b, c, d):
        return d + b*(a*x + c)**2
    
    xlist1 = np.arange(-1,1,.001)
    idx = 0
    for V in CM_list:
        v1 = V[0][0]-1
        v2 = V[0][1]-1
    
        thetas = np.zeros(shape=(theta_total,1), dtype=float)
        thetas[dim-1] = CM_info[idx,1]*(np.pi/180) #using lowest-indexed eigenvector, but any row will do
        R = Generate_Nd_Rot.genNdRotations(dim, thetas)
        U_rot = np.matmul(U_init, R)
    
        guess_a = 1.
        guess_b = 0.5
        guess_c = 0.
        guess_d = 0.
        p0=[guess_a, guess_b, guess_c, guess_d]
        coeffs = curve_fit(ParabolicFit, U_rot[:,v1], U_rot[:,v2], p0)
        Pfit = ParabolicFit(U_rot[:,v1], *coeffs[0])
        
        # calculate coefficient of determination:
        SE_Gfit = 0
        for i in range(len(Pfit)):
            SE_Gfit += (U_rot[:,v2][i] - Pfit[i])**2
        ybar = np.mean(U_rot[:, v2])
        SE_ybar = 0
        for i in range(len(Pfit)):
            SE_ybar += (U_rot[:,v2][i] - ybar)**2  
        R2 = 1 - (SE_Gfit / SE_ybar)
        print('Global Fit R2:', R2)
        CM_info[idx,2] = R2
        
        idx += 1
        
        if 0: #plot each 2D subspace's fit with corresponding R^2 value:
            lw = 1.5
            plt.subplot(1,1,1)
            plt.title(r'R$^{2}$=%.3f' % R2, fontsize=12)
            plt.scatter(U_rot[:,v1], U_rot[:,v2], s=1, c=enum, cmap=cmap)
            plt.plot(xlist1, ParabolicFit(xlist1, *coeffs[0]),c='k',linewidth=lw)
            plt.xlabel(r'v$_%s$' % (int(v1)+1), labelpad=10)
            plt.ylabel(r'v$_%s$' % (int(v2)+1), labelpad=0)
            plt.xlim(-1.25,1.25)
            plt.ylim(-1.25,1.25)
            plt.show()
    
    # Order potential CM subspaces by decreasing R^2 value:
    CM_info = CM_info[CM_info[:,2].argsort()][::-1]
    
    # =============================================================================
    # Use optimal theta to rotate N-dim manifold into place and save PD info to file
    # =============================================================================
    cmap = 'nipy_spectral'
    s = 20
    lw = .5
    idx = 1
    plt.rc('font', size=6)
    for i in range(np.shape(CM_info)[0]):
        # perform unique rotation, as required, for each 2D subspace:
        thetas = np.zeros(shape=(theta_total,1), dtype=float)
        thetas[dim-1] = CM_info[i,1]*(np.pi/180)
        R = Generate_Nd_Rot.genNdRotations(dim, thetas)
        U_rot = np.matmul(U_init, R)
        # plot final data in descending order of CM probability:
        plt.subplot(1,np.shape(CM_info)[0],i+1)
        plt.scatter(U_rot[:,int(CM_info[i,3]-1)], U_rot[:,int(CM_info[i,4]-1)], c=enum, cmap=cmap, s=s, linewidths=lw, edgecolor='k')
        plt.xlabel(r'$v_{%s}$' % int(CM_info[i,3]), fontsize=8, labelpad=5, color='k')
        plt.ylabel(r'$v_{%s}$' % int(CM_info[i,4]), fontsize=8, labelpad=5, color='k')
        plt.ticklabel_format(style='sci', axis='x', scilimits=(0,0))
        plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
        plt.axis('scaled')
        plt.xlim(-1.1,1.1)
        plt.ylim(-1.1,1.1)
    
    plt.tight_layout()
    fig = plt.gcf()
    fig.savefig(os.path.join(pdDir,'Rotated_Subspace.png'))
    #plt.show()
      
    np.save(os.path.join(pdDir, 'PD%s_CM_Rot.npy' % PD), CM_info)
    # In the above output file, the `CM_info` rows are each viable 2D subspace, with columns: 
    ##   1. 2D subspace 1d-index (zero-indexing)
    ##   2. 2D subspace optimal theta
    ##   3. 2D subspace coefficient of correlation (R^2) from parabola fit
    ##   4. 2D subspace 1st eigenvector (one-indexing)
    ##   5. 2D subspace 2nd eigenvector (one-indexing)

if __name__ == '__main__':
    path1 = os.path.splitext(sys.argv[0])[0]
    path2, tail = os.path.split(path1)
    op(path2, sys.argv[1])