import os, sys
import shutil
import numpy as np
import matplotlib
from matplotlib import rc
from matplotlib.figure import Figure
import matplotlib.pyplot as plt
from pylab import imshow, show
from matplotlib.ticker import MaxNLocator
from matplotlib import cm
import mrcfile
import imageio
import json
from scipy.spatial import distance
from scipy import spatial
import alphashape
from descartes import PolygonPatch
from shapely.geometry import Polygon
from shapely.ops import split
from shapely.geometry import LineString, asLineString
import heapq
import pickle
import imageio.core.util
def silence_imageio_warning(*args, **kwargs):
    pass
imageio.core.util._precision_warn = silence_imageio_warning
pyDir = os.path.dirname(os.path.abspath(__file__)) #python file location
parDir = os.path.abspath(os.path.join(pyDir, os.pardir))
rotDir = os.path.join(parDir, '2_Manifold_Rotations')
sys.path.append(rotDir)
import Generate_Nd_Rot
import ConicFit
import AbsValFit

if 1: #render with LaTeX font for figures
    rc('text', usetex=True)
    rc('font', family='serif')

# =============================================================================
# Least squares fit each 2D subspace per CM, project data points onto fit...
# ...stratify data points into bins (occupancy map), and save outputs:
# =============================================================================
# SETUP: First, make sure all user parameters are correct for your dataset...
#   ...below in the initial sections (e.g., 'PCA' for embedding type...
#   ...and correct input file names via directories for image stacks)
# RUNNING: To run a series of PDs at once: edit 'totalPDs' parameter below...
#   ...for the total number of PDs requested. As well, the number of CMs...
#   ...to process can also be altered at the top of the subsequent loop.
#   Finally, start batch processing via `python Spline_Binning.py`
# =============================================================================
# Author:    E. Seitz @ Columbia University - Frank Lab - 2020-2021
# Contact:   evan.e.seitz@gmail.com
# =============================================================================


# =============================================================================
# User parameters:
# =============================================================================
groundTruth = True #use GT indices for visualizations; see '0_Data_Inputs/GroundTruth_Indices'
viewCM = 2 #{1,2, etc.}; if using ground-truth, CM reference frame to use for color map indices
PCA = False #specify if manifolds from PCA or DM folder {if False, DM is True}
CTF = True #if CTF protocols previously used in DM
totalPDs = 126
totalCMs = 2 #total number of CMs to consider via leading results in previous algorithm
Bins = 20 #number of bins for each CM (i.e., energy landscape bins); **even numbers only**
printFigs = True #save figures of outputs throughout framework to file
R2_thresh = 0.55 #R^2 fit-score threshold... if above: Conic Fit is used; if below, Absolute Value is used. 

if groundTruth is True:
    CM1_idx = np.load(os.path.join(parDir, '0_Data_Inputs/GroundTruth_Indices/CM1_Indices.npy'), allow_pickle=True) #view in reference frame of CM1
    CM2_idx = np.load(os.path.join(parDir, '0_Data_Inputs/GroundTruth_Indices/CM2_Indices.npy'), allow_pickle=True) #view in reference frame of CM2
    if viewCM == 1:
        CM_idx = CM1_idx
    elif viewCM == 2:
        CM_idx = CM2_idx #(etc.)

# =============================================================================
# Main loop:
# =============================================================================
for PD in range(1,totalPDs+1):
    PD = "{0:0=3d}".format(PD)
    print('')
    print('PD:',PD)

    # =========================================================================
    # Import PD manifold:
    # =========================================================================
    #maniPath = os.path.join(parDir, '1_Embedding/PCA/Data_Manifolds_126/PD%s_SNRpt1_tau5_vec.npy' % PD)
    maniPath = os.path.join(parDir, '1_Embedding/DM/Data_Manifolds/PD_%s_vec.npy' % PD)
    print('Manifold Shape:', np.shape(np.load(maniPath)))
    U0 = np.load(maniPath) #eigenvectors
    
    # =========================================================================
    # Import PD image stack:
    # =========================================================================
    stackDir = os.path.join(parDir, '0_Data_Inputs')
    if CTF is True:
        stackPath = os.path.join(stackDir, 'CTF5k15k_SNRpt1_ELS_2D/Hsp2D_5k15k_PD_%s_filtered.mrcs' % PD) #note use of '_filtered' keyword
    elif CTF is False:
        stackPath = os.path.join(stackDir, 'CTF5k15k_SNRpt1_ELS_2D/Hsp2D_5k15k_PD_%s.mrcs' % PD)
    init_stack = mrcfile.mmap(stackPath)
    SS, box, box = init_stack.data.shape #'SS': total number of images; 'box': image dimension
    
    # =========================================================================
    # Import PD rotation info:
    # =========================================================================
    rotMatrices = os.path.join(parDir, '2_Manifold_Rotations/Data_Rotations/PD%s/PD%s_RotMatrices.npy' % (PD,PD))
    rotEigenfunctions = os.path.join(parDir, '2_Manifold_Rotations/Data_Rotations/PD%s/PD%s_RotEigenfunctions.npy' % (PD,PD))
    rotParameters = os.path.join(parDir, '2_Manifold_Rotations/Data_Rotations/PD%s/PD%s_RotParameters.npy' % (PD,PD))
    rotMatrix = np.load(rotMatrices)
    rotEigs = np.load(rotEigenfunctions)
    rotParams = np.load(rotParameters)
    dim = rotParams[0]
        
    v1_list = []
    v2_list = []
    for i in range(len(rotEigs)-1):
        if rotEigs[i] != 0:
            v1_list.append(i+1)
            v2_list.append(int(rotEigs[i]+1))

    # =========================================================================
    # Normalize manifold range and initiate subspace
    # =========================================================================
    def normalize(_d, to_sum=False, copy=True): #normalize all eigenfunctions
        # d is an (n x dimension) array
        d = _d if not copy else np.copy(_d)
        d -= np.min(d, axis=0)
        d /= (np.sum(d, axis=0) if to_sum else np.ptp(d, axis=0)) #normalize btw [0,1]
        #d = (2.*(d - np.min(d))/(np.ptp(d)))-1 #scale btw [-1,1]
        output_low, output_high = -1, 1
        d = ((d-np.min(d)) / (np.ptp(d))) * (output_high - output_low) + output_low
        return d
    
    U = normalize(U0) #rescale manifold between -1 and +1 (all axes)
    if PCA is True:
        U_init = U[:,0:dim]
    else: #if DM, don't use steady-state eigenvector (v_0)
        U_init = U[:,1:dim+1]
        
    # =========================================================================
    # Create directory for outputs:
    # =========================================================================
    pdDir = os.path.join(pyDir, 'bins/PD%s' % PD) #ZULU
    if not os.path.exists(pdDir):
        os.makedirs(pdDir)
                
    # Propagate output directory with template lists (to be filled in after examining subsequent outputs):
    if 1:
        file1 = open(os.path.join(pdDir,'CMs.txt'), 'w')
        file1.write("['1', '2', 'X', 'X', 'X', 'X']")
        file1.close() 
        file2 = open(os.path.join(pdDir,'Senses.txt'), 'w')
        file2.write("['F', 'R', 'X', 'X', 'X', 'X']")
        file2.close()
    
    # =========================================================================
    # Visualization parameters
    # =========================================================================
    s = 20
    lw = .5
    # For colormaps (if 'groundTruth' is True, above):
    cmap = 'nipy_spectral' #'gist_rainbow'sa
    Enum = np.arange(1, SS+1) 
    
    # =========================================================================
    # MAIN LOOP: fit each manifold, project points onto fits, bin, and save
    # =========================================================================    
    for CM in range(0,totalCMs): #[0,1]
        print('CM:', (CM+1))
        figIdx = 1
        # Create directory for each CM:
        outDir = os.path.join(pdDir, 'CM%s' % (CM+1))
        if os.path.exists(outDir):
            shutil.rmtree(outDir)
        if not os.path.exists(outDir):
            os.makedirs(outDir)
        # Load in angles for rotation operators:
        try:
            thetas = rotMatrix[CM]
        except:
            print('Loading error: rotation angles for CM not available.')
            continue
                
        v1 = v1_list[CM]-1 #2D subspace's 1st eigenvector
        v2 = v2_list[CM]-1 #2D subspace's 2nd eigenvector
                
        # =====================================================================
        # Use previously-defined optimal theta to rotate high-dim manifold into place
        # =====================================================================
        R = Generate_Nd_Rot.genNdRotations(dim, thetas)
        U_rot_Nd = np.matmul(R, U_init.T)
        U_rot_Nd = U_rot_Nd.T
        
        if groundTruth:
            plt.scatter(U_rot_Nd[:,v1], U_rot_Nd[:,v2], s=s, c=Enum, cmap=cmap, linewidth=lw, edgecolor='k', zorder=0)
            plt.axis('scaled')
            plt.xlim(-1.1,1.1)
            plt.ylim(-1.1,1.1)
            fig = plt.gcf()
            fig.savefig(os.path.join(outDir,'Fig_%s_Initial.png' % (figIdx)), dpi=200)
            figIdx += 1
            #plt.show()
            plt.clf()
                    
        # =====================================================================
        # Temporarily remove outliers for better fits:
        # =====================================================================
        tree = spatial.KDTree(U_rot_Nd[:,[v1,v2]])
        final_tree = tree.query_ball_tree(tree, r=.05, p=2.0, eps=0) #r: max radius
        inliers = []
        for i in final_tree:
            if len(i) > 10: #to bypass, set to 0
                for j in i:
                    inliers.append(j)
        inliers = np.unique(inliers)
        U_rot_Nd_in = U_rot_Nd[inliers]

        if 0:
            # =================================================================
            # 2D Parabolic Fit and Optimal In-Plane Rotation:
            # Method uses parabolic restraint (see paper)
            # Note: corresponding code still needs to be reviewed.
            # =================================================================
            cF, cF0, Theta, R2 = ConicFit.fit1(U_rot_Nd_in, v1, v2)
            # Define orientation of conic (facing up or down):
            if cF0[4] > 0: #upward-facing
                face = 'up'
            else: #downward-facing
                face = 'down'
        else:
            # =================================================================
            # General Conic Fit and Optimal In-Plane Rotation:
            # Note: no parabolic restraints; observed to be more robust...
            # ...and essential if CTF present.
            # =================================================================
            cF, cF0, Theta, R2 = ConicFit.fit2(U_rot_Nd_in, v1, v2)
            # Define orientation of conic (facing up or down):
            if cF0[4] < 0: #upward-facing
                face = 'up'
            else: #downward-facing
                face = 'down'
                
        # For reference in subsequent scripts:
        np.savetxt(os.path.join(outDir,'PD%s_CM%s_R2.txt' % (PD,CM+1)), np.array([R2]), fmt='%.3f')

        # =====================================================================
        # Plot non-rotated conic and fit versus rotated:
        # =====================================================================
        xy_range = np.linspace(-1.1, 1.1, 1000)
        XX, YY = np.meshgrid(xy_range, xy_range)
        plt.subplot(1,2,1) #non-rotated parabola
        Conic = cF[0]*XX**2 + cF[1]*XX*YY + cF[2]*YY**2 + cF[3]*XX + cF[4]*YY + cF[5]
        cs = plt.contour(XX, YY, Conic, [0], colors='C3', zorder=1)
        plt.scatter(U_rot_Nd[:,v1], U_rot_Nd[:,v2], s=1, c='gray', linewidth=lw, edgecolor='lightgray', zorder=-1)
        plt.scatter(U_rot_Nd_in[:,v1], U_rot_Nd_in[:,v2], s=s, c='white', linewidth=lw, edgecolor='lightgray', zorder=0)
        plt.axvline(0, c='gray', linewidth=.5)
        plt.axhline(0, c='gray', linewidth=.5)
        plt.axis('scaled')
        plt.xlim(-1.1, 1.1)
        plt.ylim(-1.1, 1.1)
        plt.subplot(1,2,2) #rotated parabola
        # Apply 2D in-plane rotation:
        R = np.array([[np.cos(Theta), np.sin(Theta)],[-1.*np.sin(Theta), np.cos(Theta)]])
        U_rot_2d = np.matmul(R, U_rot_Nd[:,[v1,v2]].T)
        U_rot_2d = U_rot_2d.T
        # Compute vertex form:
        qCoefs = [cF0[0]/(-1.*cF0[4]), cF0[3]/(-1.*cF0[4]), cF0[5]/(-1.*cF0[4])]
        h = -1.*qCoefs[1]/(2.*qCoefs[0])
        k = qCoefs[0]*h**2 + qCoefs[1]*h+qCoefs[2]
        plt.plot([h, h], [-1.1, 1.1], c='C0') #line of symmetry
        plt.plot([h, h], [-1.1, 1.1], c='C0') #line of symmetry
        #plt.scatter(h, k, c='C0')
        disc = cF0[1]**2 - 4.*cF0[0]*cF0[2] #discriminant
        plt.title(r'$\theta$=%.2f$^{\circ}$, R$^{2}$=%.2f, D=%.2f' % ((Theta*180/np.pi), R2, disc), fontsize=10) #np.arccos(cosTh)*180/np.pi) #degrees
        Conic0 = cF0[0]*XX**2 + cF0[1]*XX*YY + cF0[2]*YY**2 + cF0[3]*XX + cF0[4]*YY + cF0[5]
        cs0 = plt.contour(XX, YY, Conic0, [0], colors='C0', zorder=1)

        # Translate implicit conic equation into explicit form:
        cs0_pts = cs0.collections[0].get_paths()[0]
        cs0_v = cs0_pts.vertices
        cs0_x = cs0_v[:,0]
        cs0_y = cs0_v[:,1]
        # Split manifold into two halves using 'line of symmetry':
        split0_LHS_idx = []
        split0_RHS_idx = []
        for i in range(len(U_rot_2d[:,0])):
            px = U_rot_2d[i,0]
            py = U_rot_2d[i,1]
            d = (px-h)*((1.1)-(-1.1)) - (py-(1.1))*(h-h)
            if d <= 0:
                split0_LHS_idx.append(i)
            else:
                split0_RHS_idx.append(i)        
        # Split fit into two halves using 'line of symmetry:'
        fit0_LHS_x = []
        fit0_LHS_y = []
        fit0_RHS_x = []
        fit0_RHS_y = []
        for i in range(len(cs0_x)):
            px = cs0_x[i]
            py = cs0_y[i]
            d = (px-h)*((1.1)-(-1.1)) - (py-(1.1))*(h-h)
            if d <= 0:
                fit0_LHS_x.append(px)
                fit0_LHS_y.append(py)
            else:
                fit0_RHS_x.append(px)
                fit0_RHS_y.append(py)       
        plt.scatter(U_rot_2d[[split0_LHS_idx],0], U_rot_2d[[split0_LHS_idx],1], s=s, c='white', linewidth=lw, edgecolor='lightgray', zorder=0)
        plt.scatter(U_rot_2d[[split0_RHS_idx],0], U_rot_2d[[split0_RHS_idx],1], s=s, c='gray', linewidth=lw, edgecolor='lightgray', zorder=0)
        plt.axvline(0, c='gray', linewidth=.5)
        plt.axhline(0, c='gray', linewidth=.5)
        plt.axis('scaled')
        plt.xlim(-1.1,1.1)
        plt.ylim(-1.1,1.1)
        fig = plt.gcf()
        fig.savefig(os.path.join(outDir,'Fig_%s_ConicFit.png' % (figIdx)), dpi=200)
        figIdx += 1
        #plt.show()
        plt.clf()
                                                 
        '''if R2 < .15: #may need to tune as needed for specific datasets
            print('Fitting Error: fit score insufficient.')
            continue'''
        
        # =====================================================================
        # Arccos transformation:        
        # =====================================================================
        U_arc = np.copy(U_rot_2d) #arccos of rotated manifold
        U_arc = np.minimum(1, U_arc) #proper domain to avoid arccos errors
        U_arc = np.maximum(-1, U_arc) #proper domain to avoid arccos errors
        U_arc_2d = np.arccos(U_arc)
           
        # =====================================================================
        # Cut manifold into two halves and handle each half separately (for better overall fit):  
        # =====================================================================
        finalStackBin = np.zeros(shape=(Bins,box,box)) #binned image stacks (along fitted curve)
        finalCountBin = np.zeros(shape=(Bins)) #initiate for final 1D occupancy map
        finalImgIdxs = np.empty(Bins, dtype=object) #initiate for final image indices in each bin
        for i in range(Bins):
            finalImgIdxs[i] = []
           
        # Transform previous fits via arccos:
        fit00_LHS_x = []
        fit00_LHS_y = []
        fit00_RHS_x = []
        fit00_RHS_y = []
        for i in fit0_LHS_x:
            i = np.minimum(1, i)
            i = np.maximum(-1, i)
            j = np.arccos(i)
            fit00_LHS_x.append(j)
        for i in fit0_LHS_y:
            i = np.minimum(1, i)
            i = np.maximum(-1, i)
            j = np.arccos(i)
            fit00_LHS_y.append(j)
        for i in fit0_RHS_x:
            i = np.minimum(1, i)
            i = np.maximum(-1, i)
            j = np.arccos(i)
            fit00_RHS_x.append(j)
        for i in fit0_RHS_y:
            i = np.minimum(1, i)
            i = np.maximum(-1, i)
            j = np.arccos(i)
            fit00_RHS_y.append(j)
        # Arccos inverts orientation of graph:
        if face == 'up':
            face = 'down'
        elif face == 'down':
            face = 'up'
            
        # Rearrange order of half-manifolds as needed:
        if np.mean(U_arc_2d[[split0_LHS_idx],0]) > np.mean(U_arc_2d[[split0_RHS_idx],0]):
            split_LHS_idx = split0_RHS_idx
            split_RHS_idx = split0_LHS_idx
            fit1_LHS_x = fit00_RHS_x
            fit1_LHS_y = fit00_RHS_y
            fit1_RHS_x = fit00_LHS_x
            fit1_RHS_y = fit00_LHS_y
        else:
            split_LHS_idx = split0_LHS_idx
            split_RHS_idx = split0_RHS_idx
            fit1_LHS_x = fit00_LHS_x
            fit1_LHS_y = fit00_LHS_y
            fit1_RHS_x = fit00_RHS_x
            fit1_RHS_y = fit00_RHS_y
            
        if printFigs:
            plt.subplot(1,2,1)
            plt.scatter(U_arc_2d[[split_LHS_idx],0], U_arc_2d[[split_LHS_idx],1], s=s, c='white', linewidth=lw, edgecolor='k')
            plt.plot(fit1_LHS_x, fit1_LHS_y, c='r', linewidth=2, zorder=1)
            plt.xlabel(r'$\Phi_{%s}$' % (int(v1)+1), labelpad=5, fontsize=14)
            plt.ylabel(r'$\Phi_{%s}$' % (int(v2)+1), labelpad=7.5, fontsize=14)
            plt.title(len(split_LHS_idx))
            plt.axis('scaled')
            plt.ylim(0,np.pi)
            plt.subplot(1,2,2)
            plt.scatter(U_arc_2d[[split_RHS_idx],0], U_arc_2d[[split_RHS_idx],1], s=s, c='white', linewidth=lw, edgecolor='k')
            plt.plot(fit1_RHS_x, fit1_RHS_y, c='r', linewidth=2, zorder=1)
            plt.xlabel(r'$\Phi_{%s}$' % (int(v1)+1), labelpad=5, fontsize=14)
            #plt.ylabel(r'$\Phi_{%s}$' % (int(v2)+1), labelpad=7.5, fontsize=14)
            plt.axis('scaled')
            plt.ylim(0,np.pi)
            plt.title(len(split_RHS_idx))
            fig = plt.gcf()
            fig.savefig(os.path.join(outDir,'Fig_%s_Arccos.png' % figIdx), dpi=200)
            figIdx += 1
            #plt.show()
            plt.clf()
            
        if R2 < R2_thresh: #alternative fit required if suboptimal least-squares conic fit detected; R^2 threshold may need to be adjusted
            print('Suboptimal conic fit detected. Initiating alternative fit...')
            fit1_LHS_x, fit1_LHS_y, fit1_RHS_x, fit1_RHS_y, split_LHS_idx, split_RHS_idx, face = AbsValFit.op(U_arc_2d, groundTruth, CM_idx, v1, v2, figIdx, outDir)
            figIdx += 1

        for side in [0,1]:
            if side == 0:
                side_text = 'L'
                split_idx = split_LHS_idx
                bins = np.arange(0,Bins/2,1)
                fit1_x = fit1_LHS_x
                fit1_y = fit1_LHS_y
                center_fitX = np.amax(fit1_x)
            else:
                side_text = 'R'
                split_idx = split_RHS_idx
                bins = np.arange((Bins/2)+1,(Bins+1),1)
                fit1_x = fit1_RHS_x
                fit1_y = fit1_RHS_y
                center_fitX = np.amin(fit1_x)
            ss = len(split_idx) #new image count
            U_split = U_arc_2d[split_idx,:]

            # =================================================================
            # Temporarily remove outliers for better spline fit and Alpha Shape:
            # =================================================================
            tree = spatial.KDTree(U_split[:,[0,1]])
            final_tree = tree.query_ball_tree(tree, r=.06, p=2.0, eps=0) #r: max radius
            inliers = []
            outliers = []
            for i in final_tree:
                if len(i) > 10: #15; may need to adjust based on dataset noise
                    for j in i:
                        inliers.append(j)
                else:
                    for j in i:
                        outliers.append(j)
            # Undo outlier-assignments near vertex for subsequent, less-severe pruning:
            for out in outliers:
                if face == 'down':
                    if U_split[out,1] > 1.9:
                        inliers.append(out)
                elif face == 'up':
                    if U_split[out,1] < 1.1:
                        inliers.append(out)
                    
            inliers = np.unique(inliers)
            UX0 = [U_split[i,0] for i in inliers]
            UY0 = [U_split[i,1] for i in inliers]
            UXY0 = np.vstack((UX0, UY0))

            tree2 = spatial.KDTree(UXY0.T)
            final_tree = tree.query_ball_tree(tree2, r=.06, p=2.0, eps=0) #r: max radius
            inliers = []
            for i in final_tree:
                if len(i) > 10:
                    for j in i:
                        inliers.append(j)
            inliers = np.unique(inliers)
            UX_in = [UX0[i] for i in inliers]
            UY_in = [UY0[i] for i in inliers]
            pts = np.vstack((UX_in, UY_in))  

            # =================================================================
            # Find optimal "Alpha Shape" polygon for manifold:         
            # =================================================================
            points_in = np.array((UX_in, UY_in)).T #use version with outliers already pruned
            points_all = U_arc_2d[split_idx,:] #non-pruned, used as reference in figures only
            print('Optimizing alpha parameter... (%s/2)' % (side+1))
            Alphas = []
            for alpha in np.linspace(0.,50,50):
                alpha_shape = alphashape.alphashape(points_in, alpha)
                try:
                    if alpha_shape.geom_type == 'Polygon':
                        Alphas.append(alpha)
                    else: #i.e., geom_type == 'MultiPolygon'
                        print('Alpha converged: %.2f' % alpha)
                        break
                except:
                    print('Alpha converged: %.2f' % alpha)
                    break
            
            if 0: #last polygon generated in sequence before emergence of Multipolygon
                Alpha = Alphas[-1] 
            else: #wind back for coarser polygon
                windBack = int(len(Alphas)*.4) #may need to be adjusted; .6
                Alpha = Alphas[-(windBack)] 
            alpha_shape = alphashape.alphashape(points_in, Alpha)
            polygonMain = Polygon(alpha_shape)
            polyAreaTotal = polygonMain.area
            
            # Find intersection of spline fit with Alpha Shape polygon boundary:
            lineSpline = asLineString(np.vstack((fit1_x,fit1_y)).T)   
            boundary_both = lineSpline.intersection(polygonMain.exterior)
            
            try:
                # =============================================================
                # Note: if error emerges during this step, check if 
                # ...'len(boundary_both) > 2', which can occur in cases of
                # ...polygons with tightly-packed sections, resulting in >2
                # ...intersections of the alpha shape with the spline.
                # If this is the case, use a larger 'windBack' parameter above.
                # =============================================================
                boundary1 = boundary_both[0]
                boundary2 = boundary_both[1]
                if side == 0:
                    if np.array(boundary1)[0] < np.array(boundary2)[0]:
                        base_alphaX, base_alphaY = boundary1.xy #outer base-edge of Alpha Shape polygon
                    else:
                        base_alphaX, base_alphaY = boundary2.xy
                elif side == 1:
                    if np.array(boundary1)[0] > np.array(boundary2)[0]:
                        base_alphaX, base_alphaY = boundary1.xy
                    else:
                        base_alphaX, base_alphaY = boundary2.xy
            except: #Alpha Shape polygon only intersects spline once (at boundary) 
                base_alphaX, base_alphaY = boundary_both.xy
            
            base_alphaY = np.array(base_alphaY)[0]
            if CTF is True:
                base_alphaY0 = base_alphaY #save copy
                base_alphaY = (np.arccos(0) + base_alphaY)/2. #needed if 'CTF' is True
                if R2 < R2_thresh: #alternative fit required if suboptimal least-squares conic fit detected; see previous use above as well
                    if face == 'down':
                        base_alphaY = -np.pi
                    elif face == 'up':
                        base_alphaY = np.pi*2
            
            # Given Alpha shape boundaries, crop previous fits:         
            fit2_x = []
            fit2_y = []
            # Low R2 corresponds to more globular point-cloud; requires horizontal cutoff conditions at boundaries:
            if R2 < R2_thresh:
                x_idx = 0
                for x in fit1_x:
                    if side == 0:
                        if x >= base_alphaX:
                            fit2_x.append(x)
                            fit2_y.append(fit1_y[x_idx])
                    elif side == 1:
                        if x <= base_alphaX:
                            fit2_x.append(x)
                            fit2_y.append(fit1_y[x_idx])
                    x_idx += 1
            # High R2 corresponds to highly parabolic point-cloud; requires vertical cutoff conditions at boundaries:
            else: 
                y_idx = 0
                for y in fit1_y:
                    if face == 'up':
                        if y <= base_alphaY0:
                            fit2_x.append(fit1_x[y_idx])
                            fit2_y.append(y)
                    elif face == 'down':
                        if y >= base_alphaY0:
                            fit2_x.append(fit1_x[y_idx])
                            fit2_y.append(y)
                    y_idx += 1
                
            # Re-order x-values and y-values in sequence:
            if fit2_x[0] > fit2_x[-1]:
                fit2_x = fit2_x[::-1]
                fit2_y = fit2_y[::-1]            

            def find_nearest_val(array, value):
                n = [abs(i-value) for i in array]
                idx = n.index(min(n))
                return idx
                
            lineSplineFinal = asLineString(np.vstack((fit2_x,fit2_y)).T)
            if R2 >= R2_thresh:           
                if side == 0:
                    if face == 'up':
                        lineRayHorz = LineString([(center_fitX, base_alphaY), (center_fitX-3, base_alphaY-1)])
                        lineRayVert = LineString([(center_fitX, base_alphaY), (center_fitX, base_alphaY-3)])
                    elif face == 'down':
                        lineRayHorz = LineString([(center_fitX, base_alphaY), (center_fitX-3, base_alphaY+1)])
                        lineRayVert = LineString([(center_fitX, base_alphaY), (center_fitX, base_alphaY+3)])
                elif side == 1:
                    if face == 'up':
                        lineRayHorz = LineString([(center_fitX, base_alphaY), (center_fitX+3, base_alphaY-1)])
                        lineRayVert = LineString([(center_fitX, base_alphaY), (center_fitX, base_alphaY-3)])
                    elif face == 'down':
                        lineRayHorz = LineString([(center_fitX, base_alphaY), (center_fitX+3, base_alphaY+1)])
                        lineRayVert = LineString([(center_fitX, base_alphaY), (center_fitX, base_alphaY+3)])
    
                # =================================================================
                # Optional: Dilate the spline such that it takes up a % of the area...
                # ...of the alpha-shape, and then merge that dilated spline polygon...
                # ...with the alpha-shape polygon. This step acts as a safeguard for...
                # ...alpha-shapes that tend to cut into the point-cloud irregularly...
                # ...or point-clouds that have sections that are sparse or missing.
                # =================================================================
                dilatedAreas = []
                dilations = []
                dilatedPolys = []
                for b in np.linspace(0,0.5,30):
                    dilations.append(b)
                    dilated = lineSplineFinal.buffer(b)
                    try:
                        polySplit = split(dilated, lineRayHorz) #subdivide polygon with an intersecting line
                        polySegs = [] #for possibility of more than two segments generated during split
                        for poly in polySplit: #required in the case that very small segments are also produced along with the largest two
                            polySplitSolo = Polygon(poly)
                            polySegs.append(polySplitSolo.area)
                        polyIdxs = heapq.nlargest(2, range(len(polySegs)), key=polySegs.__getitem__) #keep polygon with largest area
                        dilated_cut1 = Polygon(polySplit[polyIdxs[0]])
    
                        polySplit = split(dilated_cut1, lineRayVert) #subdivide polygon with an intersecting line
                        polySegs = [] #for possibility of more than two segments generated during split
                        for poly in polySplit: #required in the case that very small segments are also produced along with the largest two
                            polySplitSolo = Polygon(poly)
                            polySegs.append(polySplitSolo.area)
                        polyIdxs = heapq.nlargest(2, range(len(polySegs)), key=polySegs.__getitem__) #keep polygon with largest area
                        dilated_cut2 = Polygon(polySplit[polyIdxs[0]])
                        
                        dilatedPolys.append(dilated_cut2)
                        dilatedAreas.append(dilated_cut2.area)
                    except:
                        dilatedAreas.append(99)
                        dilatedPolys.append(99)  
                dilation_idx = find_nearest_val(dilatedAreas, (polygonMain.area)*.5)
                dilated = dilatedPolys[dilation_idx]
                dilatedMerged = polygonMain.union(dilated)
            else:
                dilatedMerged = polygonMain
                
            # Plot Alpha Shape polygon with cropped fit overlaid:
            if printFigs: 
                plt.title('Alpha: %.2f, Area: %.2f' % (Alpha, polyAreaTotal))
                plt.gca().add_patch(PolygonPatch(dilatedMerged, facecolor='#99ccff', edgecolor='#6699cc', alpha=0.8))
                if 0:
                    LRx, LRy = lineRayHorz.xy
                    plt.plot(LRx, LRy, color='#999999', zorder=2)
                    LRx, LRy = lineRayVert.xy
                    plt.plot(LRx, LRy, color='#999999', zorder=2)
                #plt.gca().add_patch(PolygonPatch(alpha_shape, alpha=0.2))
                plt.scatter(*zip(*points_in), s=2, zorder=1)
                plt.scatter(*zip(*points_all), s=1, c='lightgray', zorder=-1)
                plt.plot(fit2_x, fit2_y, c='r', linewidth=2, zorder=1, alpha=.5)
                if CTF is True:
                    plt.scatter(center_fitX, base_alphaY, c='k', s=s)
                    plt.axhline(y=base_alphaY0, linewidth=1, c='k', zorder=1)
                elif CTF is False:
                    plt.scatter(center_fitX, base_alphaY, c='k', s=s)
                    plt.axhline(y=base_alphaY, linewidth=1, c='k', zorder=1)
                plt.axvline(x=center_fitX, linewidth=1, c='k', zorder=1)
                plt.xlabel(r'$\Phi_{%s}$' % (int(v1)+1), labelpad=5, fontsize=14)
                plt.ylabel(r'$\Phi_{%s}$' % (int(v2)+1), labelpad=7.5, fontsize=14)
                plt.axis('scaled')
                plt.ylim(0,np.pi)
                fig = plt.gcf()
                fig.savefig(os.path.join(outDir,'Fig_%s_Alpha_%s.png' % (figIdx, side_text)), dpi=200)
                figIdx += 1
                #plt.show()
                plt.clf()
                
            # Merged polygon becomes final polygon for subsequent steps:
            polygonMain = dilatedMerged
            polyAreaTotal = polygonMain.area
            
            # =================================================================
            # Calculate area ratios using rotating vector (ray):             
            # =================================================================
            polyAreaRatios1 = [] #percentage of each subdivided polygon to the whole
            polyAreaRatios2 = [] #both ratios are considered, to correct for error mentioned below
            lineCrosses = []
            angles = np.linspace(-45,89.99,300)
            angIdx = 0
            for ang in angles:
                if side == 0:
                    x0 = -10 #vector must be long enough as to completely intersect polygon
                    if face == 'down':
                        Ang = -1*ang*(np.pi/180.)
                    elif face == 'up':
                        Ang = ang*(np.pi/180.)
                elif side == 1:
                    x0 = 10 #vector must be long enough as to completely intersect polygon
                    if face == 'down':
                        Ang = ang*(np.pi/180.)
                    elif face == 'up':
                        Ang = -1*ang*(np.pi/180.)
                y0 = 0
                xR = x0*np.cos(Ang) - y0*np.sin(Ang)
                yR = x0*np.sin(Ang) + y0*np.cos(Ang)
                if CTF is True:
                    lineRay = LineString([(center_fitX, base_alphaY), (xR + center_fitX, yR + base_alphaY)]) #translate and rotate line
                else:
                    lineRay = LineString([(center_fitX, base_alphaY), (xR + center_fitX, yR + base_alphaY)]) #translate and rotate line
                try:
                    polySplit = split(polygonMain, lineRay) #subdivide polygon with an intersecting line
                    polySegs = [] #for possibility of more than two segments generated during split
                    for poly in polySplit: #required in the case that very small segments are also produced along with the largest two
                        polySplitSolo = Polygon(poly)
                        polySegs.append(polySplitSolo.area)
                    polyIdxs = heapq.nlargest(2, range(len(polySegs)), key=polySegs.__getitem__) #keep polygons with two largest areas
                    polySplit1 = Polygon(polySplit[polyIdxs[0]])
                    polySplit2 = Polygon(polySplit[polyIdxs[1]])
                    polyAreaRatios1.append(polySplit1.area / polyAreaTotal)
                    polyAreaRatios2.append(polySplit2.area / polyAreaTotal)
                    linesCross = lineSpline.intersection(lineRay)
                    try:
                        if len(linesCross) > 1: #i.e., if type = 'Multipoint' (can happen due to contiguous lines in LineString)
                            linesCross = linesCross[0]
                    except:
                        linesCross = linesCross
                    lineCrosses.append(linesCross)
                    if 0: #sanity check: view rotations and corresponding subdivisions; may need to decrease vector length above for better visualization
                        plt.clf()
                        if CTF is True:
                            plt.scatter(center_fitX, base_alphaY, c='magenta', zorder=3)
                        elif CTF is False:
                            plt.scatter(center_fitX, base_alphaY, c='magenta', zorder=3)
                        plt.gca().add_patch(PolygonPatch(polySplit1, alpha=0.2, color='red', label=('Area: %.4f' % polySplit1.area)))
                        plt.gca().add_patch(PolygonPatch(polySplit2, alpha=0.2, color='blue', label=('Area: %.4f' % polySplit2.area)))
                        LRx, LRy = lineRay.xy
                        plt.plot(LRx, LRy, color='#999999', zorder=2)
                        LSx, LSy = lineSpline.xy
                        plt.plot(LSx, LSy, c='k', zorder=1)
                        PCx, PCy = linesCross.xy
                        plt.scatter(PCx, PCy, c='cyan', zorder=3)
                        plt.legend(loc='best')
                        plt.title('Angle: %.2f, Ratio: %.3f, %.3f' % (ang, polyAreaRatios1[-1], polyAreaRatios2[-1]))
                        plt.axis('scaled')
                        fig = plt.gcf()
                        fig.savefig(os.path.join(outDir,'Ang_%s.png' % angIdx), dpi=200)
                        #plt.show()
                        plt.clf()
                except: #skip and append dummy value
                    polyAreaRatios1.append(99)
                    polyAreaRatios2.append(99)
                    lineCrosses.append(99)
                angIdx += 1
            # =================================================================
            # Note for above: the assignment of which segmented polygon "is which" sometimes...
            # flips, such that the ratio is non-monotonic. The below code adjusts for this...
            # by always using the 'polyAreaRatio' that creates a monotonically increasing sequence.
            # =================================================================
            polyAreaRatios = [] #corrected list
            for p in range(len(polyAreaRatios1)):
                if p == 0:
                    if polyAreaRatios1[p] < polyAreaRatios2[p]:
                        polyAreaRatios.append(polyAreaRatios1[p])
                    else:
                        polyAreaRatios.append(polyAreaRatios2[p])
                else:
                    if (polyAreaRatios1[p] > polyAreaRatios[-1]) and (polyAreaRatios2[p] > polyAreaRatios[-1]):
                        if polyAreaRatios1[p] < polyAreaRatios2[p]:
                            polyAreaRatios.append(polyAreaRatios1[p])
                        else:
                            polyAreaRatios.append(polyAreaRatios2[p])
                    elif (polyAreaRatios1[p] > polyAreaRatios[-1]) and (polyAreaRatios2[p] < polyAreaRatios[-1]):
                        polyAreaRatios.append(polyAreaRatios1[p])
                    else:
                        polyAreaRatios.append(polyAreaRatios2[p])
                        
            if 0: #sanity check: plot of area ratios should be monotonically increasing
                for i in polyAreaRatios:
                    if i < 99: #dummy value used above
                        plt.plot(polyAreaRatios)        
                plt.ylim(0,1)
                for i in range(1,11):
                    plt.axhline(y=i/10.)
                plt.show()
            
            binEdgesCoords = [] #list of bin edges
            binEdgesIdxs = [] #closest indices for each bin edge on fitted line
            if 0: #uniform area distribution for all bins
                for r in range(1,int((Bins/2))):
                    ratio = r/(Bins/2.)
                    ratio_idx = find_nearest_val(polyAreaRatios, ratio)
                    binEdgesCoords.append(lineCrosses[ratio_idx])
            else: #slightly less area for bin closest to center due to increased outlier trend; experimental!
                #eps = .0015
                eps = (.905/((Bins/2)-1) - (1/(Bins/2)))
                ratio = (1/(Bins/2.)) + eps
                ratioCurr = ratio
                for r in range(1,int((Bins/2))):
                    #print('ratio:', r, ratioCurr)
                    ratio_idx = find_nearest_val(polyAreaRatios, ratioCurr)
                    binEdgesCoords.append(lineCrosses[ratio_idx])
                    ratioCurr += ratio
                                                                
            if side == 0:
                binEdgesIdxs.append(int(0))
            elif side == 1:
                binEdgesIdxs.append(int(len(fit2_x)-1))
                                    
            for b in binEdgesCoords:
                binCoordX = b.xy[0][0]
                binCoordY = b.xy[1][0]
                temp_geodesics = [] 
                for pt in range(len(fit2_x)): #for every point on fit line
                    fitCoordX = fit2_x[pt]
                    fitCoordY = fit2_y[pt]
                    temp_geodesics.append(np.sqrt((fitCoordX - binCoordX)**2 + (fitCoordY - binCoordY)**2))
                val, idx = min((val, idx) for (idx, val) in enumerate(temp_geodesics)) #val = min distance; idx = its index                    
                
                if side == 0:
                    if idx <= binEdgesIdxs[-1]:
                        binEdgesIdxs.append(int(binEdgesIdxs[-1])+1)
                    else:
                        binEdgesIdxs.append(int(idx))
                elif side == 1:
                    if idx >= binEdgesIdxs[-1]:
                        binEdgesIdxs.append(int(binEdgesIdxs[-1])-1)
                    else:
                        binEdgesIdxs.append(int(idx))

            if side == 0:
                idx_max = max(range(len(fit2_x)), key=fit2_x.__getitem__)                        
                binEdgesIdxs.append(idx_max)
            elif side == 1:
                idx_max = min(range(len(fit2_x)), key=fit2_x.__getitem__)                        
                binEdgesIdxs.append(idx_max)
                
            #print(binEdgesIdxs)
                                                                
            if printFigs: #view final subdivision placements                        
                plt.gca().add_patch(PolygonPatch(polygonMain, alpha=0.2, label=('Area: %.4f' % polySplit2.area)))
                plt.plot(fit2_x, fit2_y, c='r', linewidth=1.5, zorder=1)
                #plt.scatter(center_fitX, base_alphaY, c='k', s=s, zorder=1)
                for pt in binEdgesIdxs:
                    plt.scatter(fit2_x[pt], fit2_y[pt], c='r', s=s, zorder=10)
                plt.scatter(*zip(*points_in), s=1, c='lightgray', zorder=-1)
                plt.xlabel(r'$\Phi_{%s}$' % (int(v1)+1), labelpad=5, fontsize=14)
                plt.ylabel(r'$\Phi_{%s}$' % (int(v2)+1), labelpad=7.5, fontsize=14)
                plt.axis('scaled')
                fig = plt.gcf()
                fig.savefig(os.path.join(outDir,'Fig_%s_BinEdges_%s.png' % (figIdx, side_text)), dpi=200)
                figIdx += 1
                #plt.show()
                plt.clf()
            
            # =================================================================
            # Function to reorder images along rays:
            # =================================================================                  
            def reorder2(mani, fitList, split_idx, base_x, base_y, side, face):
                mani_split = mani[split_idx,:]
                finalIdx = np.ones(shape=(np.shape(mani_split)[0],4)) #image index vs index of pt on fitted line its closest to
                xlist,ylist = list(zip(*fitList)) #x,y coords of each fit-point (vector) along curve fit
                ptB = np.array([[base_x],[base_y]]) #central-base point of parabola (from fit)
                t_size = 50 #number of interpolated points
                img_idx = 0
                for img in split_idx:#range(np.shape(mani)[0]): #for every image (point in manifold)
                    ptsT = np.empty(shape=(2,t_size)) #interpolated points (straight-line between image-point and base-point)
                    imgCoordX = mani[:,0][img]
                    imgCoordY = mani[:,1][img]
                    ptU = np.array([[imgCoordX],[imgCoordY]]) #image-point from point-cloud
                    t_idx = 0
                    for t in np.linspace(-1,1,t_size): #interpolation points
                        ptT = (1-t)*ptU + t*ptB
                        ptsT[0,t_idx] = ptT[0]
                        ptsT[1,t_idx] = ptT[1]
                        t_idx += 1
                    t_line = LineString([(base_x, base_y), (ptsT[0][0], ptsT[1][0])]) #simplified ray using only start and end point
                    fit_line = asLineString(np.vstack((xlist,ylist)).T)
                    t_cross = t_line.intersection(fit_line) #intersection of fit line with image ray
                    
                    if 0: #view scatter plots as sanity-check
                        if img%100==0:
                            Fx, Fy = fit_line.xy
                            plt.plot(Fx, Fy, c='cyan', linewidth=2, zorder=3)
                            Tx, Ty = t_line.xy
                            plt.plot(Tx, Ty, c='orange', linewidth=2, zorder=4)
                            plt.scatter(mani[:,0], mani[:,1], s=s, c='white', linewidth=lw, edgecolor='k', zorder=0)
                            plt.scatter(imgCoordX, imgCoordY, s=(s*.25), c='magenta', zorder=1)
                            #plt.scatter(ptsT[0,:], ptsT[1,:], s=.5, c='k', zorder=1) #brute force method (see below)
                            plt.scatter(base_x, base_y, c='green', s=30)
                            plt.plot(xlist, ylist, c='r', linewidth=2,zorder=2)
                            plt.show()
                        
                    if not t_cross.is_empty:
                        t_cross_x, t_cross_y = t_cross.xy
                    temp_geodesics = [] #list of all distances from the interpolated-points to points on the fitted curve
                    for pt in range(len(xlist)): #for every finite point on fit line
                        fitCoordX = xlist[pt]
                        fitCoordY = ylist[pt]
                        if t_cross.is_empty: #brute force, if ever necessary
                            temp_geodesics.append(distance.cdist([(fitCoordX, fitCoordY)], ptsT.T).min())
                        else: #optimized, works in majority of cases
                            temp_geodesics.append(np.sqrt((fitCoordX - np.array(t_cross_x))**2 + (fitCoordY - np.array(t_cross_y))**2))
                    val, idx = min((val, idx) for (idx, val) in enumerate(temp_geodesics)) #val = min distance; idx = its index
                    finalIdx[img_idx,0] = int(img) #image index
                    finalIdx[img_idx,1] = int(idx) #keep standard ray fit index
                    finalIdx[img_idx,2] = mani[:,0][img] #1st PC-coordinate of image
                    finalIdx[img_idx,3] = mani[:,1][img] #2nd PC-coodinate of image
                    # Rare event; ensure points that are distant outliers above/below base of parabola are correctly labelled:
                    '''if face == 'up':
                        if mani[:,1][img] > (base_y + 0.1):
                            if side == 0:
                                finalIdx[img_idx,1] = int(0)
                            elif side == 1:
                                finalIdx[img_idx,1] = int(len(xlist)-1)
                    elif face == 'down':
                        if mani[:,1][img] < (base_y - 0.1):
                            if side == 0:
                                finalIdx[img_idx,1] = int(0)
                            elif side == 1:
                                finalIdx[img_idx,1] = int(len(xlist)-1)'''             
                    img_idx += 1
                return finalIdx[np.argsort(finalIdx[:,1])] #sort array s.t. images placed in order along fitted curve
                              
            # =================================================================
            # Order and bin images along spline:        
            # =================================================================
            print('Performing ray projections (%s/2)...' % (side+1))
            fitList3 = zip(fit2_x, fit2_y) #each x-fit and y-fit coord, ordered along the curve
            if CTF is True:
                imageIdxSort3 = reorder2(U_arc_2d, fitList3, split_idx, center_fitX, base_alphaY, side, face) #project and order image-pts onto spline via rays
            elif CTF is False:
                imageIdxSort3 = reorder2(U_arc_2d, fitList3, split_idx, center_fitX, base_alphaY, side, face) #project and order image-pts onto spline via rays
            # Note for above: column 1 = image index; column 2 = fit index (sorted from low to high)

            if side == 1:
                binEdgesIdxs = binEdgesIdxs[::-1]
            slices = []
            for i in range(len(binEdgesIdxs)-1):
                slices.append([])
                slices[i] = range(binEdgesIdxs[i],binEdgesIdxs[i+1])
                                            
            for i in imageIdxSort3:
                imgOrder = int(i[0])
                for j in range(len(bins)):
                    sliceFirst = slices[j][0]
                    sliceLast = slices[j][-1]+1
                    if j == 0: #first bin
                        if sliceFirst <= int(i[1]) < sliceLast:
                            if side == 0:
                                finalStackBin[j] += init_stack.data[imgOrder]
                                finalCountBin[j] += 1
                                finalImgIdxs[j].append(imgOrder)
                            else:
                                finalStackBin[(j+int(Bins/2))] += init_stack.data[imgOrder]
                                finalCountBin[(j+int(Bins/2))] += 1
                                finalImgIdxs[(j+int(Bins/2))].append(imgOrder)
                    elif j == int(len(bins)-1): #last bin
                        if sliceFirst <= int(i[1]) <= sliceLast:
                            if side == 0:
                                finalStackBin[j] += init_stack.data[imgOrder]
                                finalCountBin[j] += 1
                                finalImgIdxs[j].append(imgOrder)
                            else:
                                finalStackBin[(j+int(Bins/2))] += init_stack.data[imgOrder]
                                finalCountBin[(j+int(Bins/2))] += 1
                                finalImgIdxs[(j+int(Bins/2))].append(imgOrder)
                    else: #all other bins
                        if sliceFirst <= int(i[1]) < sliceLast:
                            if side == 0:
                                finalStackBin[j] += init_stack.data[imgOrder]
                                finalCountBin[j] += 1
                                finalImgIdxs[j].append(imgOrder)
                            else:
                                finalStackBin[(j+int(Bins/2))] += init_stack.data[imgOrder]
                                finalCountBin[(j+int(Bins/2))] += 1
                                finalImgIdxs[(j+int(Bins/2))].append(imgOrder)
                        
            if 0: #plot final frames (bins) in sequence
                for b in bins:
                    plt.imshow(finalStackBin[b], cmap='gray')
                    plt.title(finalCountBin[b]) #occupancy of each bin
                    plt.show()
    
            # =================================================================
            # Save final bin information to file:
            # =================================================================                                   
            stackOut = os.path.join(outDir,'PD%s_CM%s_stack.mrcs' % (PD,CM+1))
            if os.path.exists(stackOut):
                os.remove(stackOut)
            final_bins = mrcfile.new_mmap(stackOut, shape=(Bins,box,box), mrc_mode=2, overwrite=True) #mrc_mode 2: float32
            for b in range(Bins):
                final_bins.data[b] = finalStackBin[b]
                if 0:
                    plt.imshow(final_bins.data[b], cmap='gray')
                    plt.show()
                
            for b in range(0,Bins):
                f = open(os.path.join(outDir,'PD%s_CM%s_Bin%02d.txt' % (PD,CM+1,(b+1))), 'w')
                json.dump(finalImgIdxs[b], f)
                f.close()
                
            final_bins.close()
            
            if side == 0: #for final figure only
                binEdgesIdxs_LHS = binEdgesIdxs
                fit2_x_LHS = fit2_x
                fit2_y_LHS = fit2_y
            else:
                binEdgesIdxs_RHS = binEdgesIdxs
                fit2_x_RHS = fit2_x
                fit2_y_RHS = fit2_y
                        
        np.savetxt(os.path.join(outDir,'PD%s_CM%s_Hist.txt' % (PD,CM+1)), finalCountBin, fmt='%i')
              
        if printFigs: #plot final binning via scatter plot on manifold
            color=iter(cm.tab20(np.linspace(0,1,Bins)))
            for b in range(Bins):
                c=next(color)
                plt.scatter(U_arc_2d[:,0][finalImgIdxs[b]], U_arc_2d[:,1][finalImgIdxs[b]], color=c, s=int(s/4))
            #plt.scatter(U_arc_2D[:,0], U_arc_2D[:,1], c='lime', s=int(s/4), zorder=-1)
            plt.axvline(center_fitX, c='k', linewidth=1, alpha=.5)
            plt.plot(fit2_x_LHS, fit2_y_LHS, c='k', linewidth=2, zorder=1)
            plt.plot(fit2_x_RHS, fit2_y_RHS, c='k', linewidth=2, zorder=1)
            if 0:
                for pt in binEdgesIdxs_LHS:
                    plt.scatter(fit2_x_LHS[pt], fit2_y_LHS[pt], c='k', s=s, zorder=1)
                for pt in binEdgesIdxs_RHS:
                    plt.scatter(fit2_x_RHS[pt], fit2_y_RHS[pt], c='k', s=s, zorder=1)
            plt.xlabel(r'$\Phi_{%s}$' % (int(v1)+1), labelpad=5)
            plt.ylabel(r'$\Phi_{%s}$' % (int(v2)+1), labelpad=7.5)
            plt.title('%s, %s:' % (len(split_LHS_idx), len(split_RHS_idx)))
            plt.axis('scaled')
            fig = plt.gcf()
            fig.savefig(os.path.join(outDir,'Fig_%s_OccScatter.png' % figIdx), dpi=200)
            figIdx += 1
            #plt.show()
            plt.clf()
        
        # =====================================================================
        # Save bar plot of occupancies:
        # =====================================================================
        plt.bar(range(1,Bins+1),finalCountBin)
        plt.title('Occupancy Map', fontsize=14)
        plt.xlabel(r'CM$_{%s}$ State' % (int(CM+1)), fontsize=14, labelpad=7.5)
        plt.ylabel('Occupancy', fontsize=14, labelpad=7.5)
        plt.xlim(0.25, Bins+.75)
        plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))
        plt.gca().yaxis.set_major_locator(MaxNLocator(integer=True))
        plt.gca().xaxis.set_major_locator(plt.MultipleLocator(1))
        #plt.axhline(y=SS/Bins, color='r', linestyle='-')
        fig = plt.gcf()
        fig.savefig(os.path.join(outDir,'Fig_%s_OccBar.png' % figIdx), dpi=200)
        figIdx += 1
        #plt.show()
        plt.clf()
                    
        # =====================================================================
        # Save current CM's sequence of states as gif:
        # =====================================================================
        #finalStackBin_uint8 = ((finalStackBin - finalStackBin.min()) * (1/(finalStackBin.max() - finalStackBin.min()) * 255)).astype('uint8')
        imageio.mimsave(os.path.join(outDir,'Fig_%s_2Dmovie.gif' % figIdx), finalStackBin)#_uint8) #Python3 uint8 conversion creates "blinking" gifs; just silence warning.
        figIdx += 1
        
        # =====================================================================
        # Check ground truth positions of states on parabola and sinusoid:
        # =====================================================================                
        if groundTruth is True: #check ground-truth bins    
            for F in [1,2]:
                if F == 1:
                    CM_idx = CM1_idx
                else:
                    CM_idx = CM2_idx
                plt.subplot(1,2,1)                          
                color=iter(cm.tab20(np.linspace(1, 0, np.shape(CM_idx)[0])))
                for b in range(np.shape(CM_idx)[0]):
                    c=next(color)
                    plt.scatter(U_arc_2d[:,0][CM_idx[b]], U_arc_2d[:,1][CM_idx[b]], color=c, s=s, edgecolor='k', linewidths=.1, zorder=1) #parabola
                plt.title('Ground Truth Bins')
                plt.axis('scaled')
                
                plt.subplot(1,2,2)                          
                color=iter(cm.tab20(np.linspace(0, 1, np.shape(CM_idx)[0])))
                for b in range(np.shape(CM_idx)[0]):
                    c=next(color)
                    plt.scatter(U_arc_2d[:,0][finalImgIdxs[b]], U_arc_2d[:,1][finalImgIdxs[b]], color=c, s=s, edgecolor='k', linewidths=.1, zorder=1) #parabola
                plt.title('Output Bins')
                plt.axis('scaled')
                fig = plt.gcf()
                fig.savefig(os.path.join(outDir,'Fig_%s_GroundTruth_%s.png' % (figIdx,F)), dpi=200)
                figIdx += 1
                #plt.show()
                plt.clf()
            
    init_stack.close()