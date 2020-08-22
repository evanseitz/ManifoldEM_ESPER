# README: 
## Data Inputs folder

This folder contains files corresponding to the input synthetic datasets, which must be created upstream for use in this repository. General instructions for creating synthetic data are included in our paper (see Supplementary Materials), with detailed code also provided in our external repository (https://github.com/evanseitz/cryoEM_synthetic_continua). As an example for your immediate use, we have provided a subset of our own synthetic data in these folders.

The `1_AtomicCoords_2D` folder contains the initial atomic coordinate files (PDBs) spanning 400 states with 2 degrees of freedom (`2D`). These are available for comparing final outputs with ground truth, and will not be used for any other purpose in this repository other than that validation.

The `2_PDs_2D` folder contains 400 images (corresponding to the aforementioned 400 states) for each of 5 projection directions (PDs). These are the same 5 PD datasets that are examined in the `Analysis: State Space 2` section of our paper. These images are pristine (SNR = infinity), and can be investigated immediately using code in other sections of this repository.

To emulate experimental conditions, however, these pristine images (in the `2_PDs_2D` folder) can be duplicated a fixed number of times (via a user defined `tau` parameter), with each duplicate image given unique additive Gaussian noise (via calculation of appropriate SNR, as also chosen by the user). First, alter the user parameters near the beginning of the `Generate_SNRtau.py` script. As an example of `tau` usage, `tau = 5` will result in 5 duplicates per state, and thus 5 x 400 = 2000 images per new PD stack. Next, the script `3_SNRtau_Batch.sh` must be altered to correspond to the number of PDs to process in sequence. After doing so, computations can be initiated via `sh 3_SNRtau_Batch.sh` (with the proper Aniconda environment activated beforehand) to generate new image stacks for each PD in the `3_PDs_2D_SNR_tau` folder.

As a note, due to data limitations, the `3_PDs_2D_SNR_tau` folder has no pre-generated data ready for use, and thus must be created by the user after downloading this repository. Once you have readied your data in this folder, proceed next to `1_Embedding`.
