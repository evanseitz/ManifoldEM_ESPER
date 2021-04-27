# README
## Compile Bins folder

This folder contains scripts to stitch together all CM bins from each PD into a series of alignment files and stacks. Each generated alignment file is intended to correspond to a state, which can be accomplished for one or more degrees of freedom. The scripts are set up for (1) `_1D` which outputs a sequence of volumes independently for each degree of freedom; and (2) `_2D` which outputs an array of volumes via pairwise combination of two different degrees of freedom.

To setup for this procedure in 1D, please first read the description in the header of the `1_Compile_Bins_1D.py` script, and alter all user parameters there as you deem fit. To keep in mind, there are also several potentially important parameters scattered throughout this framework which may need to be adjusted depending on your data. Please see comments for more information. Once you are ready, a batch computation can be initiated via `python 1_Compile_Bins.py`. As well, a similar setup is required for the 2D procedure.

All outputs will be saved per bin to the `S2_bins` folder in this same directory. After all computations are complete for every PD and bin, you will need to run RELION to reconstruct the volumes for 1D (via `sh 1_Volumes_Bins_1D.sh`) or 2D (via `sh 2_Volumes_Bins_2D.sh`). For evaluating more than two CMs, this code will need to be altered. As well, note that the `--ctf` flag in each RELION command will need to be manually removed if input data was created without CTF.