# README
## Manifold Rotations folder

This folder contains scripts to find the optimal angle required to counter-rotate each PD manifold into its standard position (see our paper for a detailed discussion of this technique).

To setup for this procedure, first, make sure the correct number of PDs are defined (corresponding to your previous inputs) in the `1_RotHist_Batch.sh` script. Next, follow the instructions displayed at the top of the `Rotation_Histograms.py` script, and finally, initiate a batch computation via `sh 1_RotHist_Batch.sh` (making sure the proper Aniconda environment is first activated).

All outputs will be saved per PD to the `Data_Rotations` folder in this same directory. The `.npy` file in each PD output folder will be used in subsequent steps (with its contents described at the bottom of the `Rotation_Histograms.py` script). As well, you can check the accuracy of these rotation assignments via each PD's exported `Rotated_Subspace.png`, which displays the 2D subspaces corresponding to each conformational motion (CM, in decreasing order based on probability), as well as their appearance in the counter-rotated view. You should find that the leading (two) entries are parabolas that are properly aligned with the plane of their corresponding 2D subspaces. As well, if using the data we've provided, the colormap along these parabolas can be used to distinguish between corresponding CMs (as detailed in our paper).
