# README: 
## Data Inputs folder

This folder contains files corresponding to the input synthetic datasets, which must be created upstream for use in this repository. General instructions for creating synthetic data are included in the SI of our paper, with detailed code also provided in our external repository (https://github.com/evanseitz/cryoEM_synthetic_continua). As an example, we have provided a subset of our own synthetic data in these folders.

The `1_AtomicCoords_2D` folder contains the initial atomic coordinate files (PDBs) spanning 400 states with 2 degrees of freedom (`2D`). These are available for comparing final outputs with ground truth, and will not be used for any other purpose in this repository other than that validation.

The `2_PDs_2D` folder contains 400 images (corresponding to the aforementioned 400 states) for each of 5 projection directions (PDs). These are the same 5 PD datasets that are examined in the `Analysis: State Space 2` section of our paper. These images are pristine (SNR = infinity), and can be investigated immediately using code in other sections of this repository.

To emulate experimental conditions, these pristine images can be duplicated a fixed number of times (via the `tau` parameter), with each duplicate image given unique additive Gaussian noise (via calculation of appropriate SNR, as chosen by the user). The script `3_Generate_SNR_tau.py` can be run on the PDs in `2_PDs_2D` to generate new image stacks in the `3_PDs_2D_SNR_tau` folder (with `SNR` and `tau` values alterable within that script). As an example of `tau` usage, `tau = 5` will result in 5 duplicates per state, and thus 5 x 400 = 2000 images per new PD stack. Finally, as a note, due to data limitations, the `3_PDs_2D_SNR_tau` folder has no pre-generated data ready for use, and thus must be created by the user after downloading this repository.

Once you have readied your data in this folder, proceed next to `1_Embedding`.
