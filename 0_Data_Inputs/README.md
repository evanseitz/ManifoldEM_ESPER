# README: 
## Data Inputs folder

This folder contains files corresponding to the input synthetic datasets, which must be created upstream for use in this repository. General instructions for creating synthetic data are included in our paper (see Supplementary Materials), with detailed code also provided in our [external repository](https://github.com/evanseitz/cryoEM_synthetic_continua), including installation instructions for several external packages required throughout this workflow. As an example for your immediate use, we have provided a subset of our own synthetic data in these folders.

The `2_PDs_2D` folder contains 400 images (corresponding to 400 states with two degrees of freedom) for each of 5 projection directions (PDs). These are the same 5 PD datasets that are examined in the initial analysis section of our paper. These images are pristine (SNR = infinity), and can be investigated immediately using code in other sections of this repository. To note, the contents of all image stacks can be easily viewed using RELION; e.g., `relion_display --i PD_001.mrcs`.

To emulate experimental conditions, however, these pristine images can be duplicated a fixed number of times (via a user defined `tau` parameter), with each duplicate image given unique additive Gaussian noise (via calculation of appropriate SNR, as also chosen by the user). This functions can be performed via code in the `Pristine_AddTauSNR` folder. As an example of `tau` usage, `tau = 5` will result in 5 duplicates per state, and thus 5 x 400 = 2000 images per new PD stack. Additionally, the folder `Pristine_AddCtfSNR` can be used to modify images with experimentally-relevant CTF.

Once you have readied your data in this folder by means of these sample PDs or a more elaborate construction, proceed next to `1_Embedding`.
