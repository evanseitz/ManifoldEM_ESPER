# README
## DM folder

This folder contains scripts to peform Diffusion Maps (DM) on the input files that were previously generated in the `0_Data_Inputs` folder. 

To setup for this procedure, a distance matrix must be generated for each previously created PD image stack. See the instructions for batch processing distances in `Distances.py`. Once these have been generated, next see the instructions in `DiffusionMaps.py` to generate corresponding manifolds.

As an important note, you will need to first perform a trial run of the `DiffusionMaps.py` script to view the corresponding Gaussian bandwidth plot (`plot of Ferguson method`) to best estimate the `eps` value for your dataset (see our paper). Once this has been obtained, change the `eps = 1e10` line to match the outcome of your analysis, and rerun the script a second (final) time. This value of `eps` should be consistent across all input PD image stacks generated with equivalent SNR.

All outputs will be stored in the `Data_Manifolds` folder. As an example of outputs, the eigenvalues and eigenvectors for our first example PD (generated with `SNR = infinite` and `tau = 1`) have been pre-generated in that folder.

On can next elect to run the `DM_Viewer.py` script to analyze these output files in the form of eigenvalue spectrum and manifold subspaces. This script provides a number of informative and adjustable plots to help view this complex information.
