# README
## PCA folder

This folder contains scripts to peform Principal Component Analysis (PCA) on the input files that were previously generated in the `0_Data_Inputs` folder. 

To setup for this procedure, first, make sure the correct number of PDs are defined (corresponding to your inputs) in the `1_PCA_Sequence.sh` script. Next, in the `PCA.py` script, alter the `dataPath` variable to match your input data (no change is needed if running our default settings in previous `0_Data_Inputs` scripts using the provided datasets).

Once these changes are accounted for, navigate to this folder in your command line interface (CLI), and, after activating the proper environment (see this repository's main `README` file), type `sh 1_PCA_Batch.sh` to initiate a batch of embeddings. All outputs will be stored in the `Data_Manifolds` folder. As an example of outputs, the eigenvalues and eigenvectors for the example five PDs (generated with `SNR = 0.1` and `tau = 5`) have been pre-generated in that folder. As a note, the `Data_Manifolds_126` has been provided solely for use in subsequent steps, and contains the 126 manifolds obtained from the 126 PD great circle described in our paper.

On can next elect to run the `PCA_Viewer.py` script to analyze these output files in the form of eigenvalue spectrum and manifold subspaces. This script provides a number of informative and adjustable plots to help view this complex information.
