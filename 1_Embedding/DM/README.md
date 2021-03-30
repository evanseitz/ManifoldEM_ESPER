# README
## DM folder

This folder contains scripts to peform Diffusion Maps (DM) on the input files that were previously generated in the `0_Data_Inputs` folder. 

To setup for this procedure, a distance matrix must be generated for each previously created PD image stack. See the instructions for batch processing distances in `Distances.py`. Once these have been generated, next see the instructions in `DiffusionMaps.py` to generate corresponding manifolds. All outputs will be stored in the `Data_Distances` and `Data_Manifolds` folders.

One can next elect to run the `DM_Viewer.py` script to analyze these output files in the form of eigenvalue spectrum and manifold subspaces. This script provides a number of informative and adjustable plots to help view this complex information.
