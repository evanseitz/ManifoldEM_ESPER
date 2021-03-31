# README
## Cryo-EM Heuristic Analysis

This repository contains the software implementation for our paper **Heuristic Analysis of Manifolds from Simulated Cryo-EM Ensemble Data** (Seitz, Schwander, Acosta-Reyes, Maji, Liao, Frank). It contains tools to apply the discussed methods to quasi-continuum models. This work was developed in the Frank research group at Columbia University in collaboration with Peter Schwander at the University of Wisconsin-Milwaukee (UWM).

The algorithms presented here in their current form are developed for analyzing synthetic data. Custom synthetic datasets can be generated as described in our supplementary materials. Additional information can also be found in our [previous paper](https://www.biorxiv.org/content/10.1101/864116v1): **Simulation of Cryo-EM Ensembles from Atomic Models of Molecules Exhibiting Continuous Conformations** (Seitz, Acosta-Reyes, Schwander, Frank); along with detailed code in the corresponding [repository](https://github.com/evanseitz/cryoEM_synthetic_continua).

Please note that additional alterations to this code will be required to make this workflow fully accessible to experimentally-obtained data. To note, much of the code necessary for processing and organizing such data into projection directions is already available in the first half of the ManifoldEM suite. To note, our workflow presented here branches off from the current ManifoldEM framework permanently after manifolds are created via Diffusion Maps and immediately before NLSA is performed.

## Instructions:

### Environment:
First, install [Anaconda](https://docs.anaconda.com/anaconda/install). Additionally, LaTex can be installed, e.g. via [TeX Live](https://tug.org/texlive)... if not, syntax for figure generation in these scripts will need to be individually altered. Next, with Anaconda sourced, create a new Anaconda environment:

`condo create -n Manifold python=3`

Next, activate this environment via `condo activate Manifold`, and install the following packages:

- `pip install numpy`
- `pip install matplotlib`
- `pip install scipy`
- `pip3 install -U scikit-learn scipy matplotlib`
- `pip install mrcfile`
- `pip install imageio`
- `conda config --add channels conda-forge`
- `conda config --set channel_priority strict`
- `conda install alphashape`
- `conda install -c conda-forge descartes`
- `pip install qiskit` #installs sympy
- `pip install latex` #if texlive installed above

Once these packages are installed within the Anaconda environment, the environment must be initiated each time before running these scripts via the command: `conda activate Manifold`

When you are done using the environment, always exit via: `conda deactivate`

### Additional Software:
In addition to the Anaconda environment detailed above, the following packages may also prove useful; some of which are required for final steps in this framework.
- Chimera
- PyMOL
- Phenix
- EMAN2
- RELION

### Usage:
Detailed instructions and comments for all procedures are provided in the code. Within this repository, we have also supplied a sample of pristine data that can be experimented with in the `0_Data_Inputs` folder. This example data can be further altered with additive Gaussian noise and introduction of noisy-duplicates via scripts in the `Pristine_AddNoiseTau` folder, as well as CTF via scripts in the `Pristine_AddCtfSNR` folder. A collection of manifolds are also provided in the `PCA/Data_Manifolds_126` folder so that users can jump straight in to experimenting with several of our downstream algorithms. These manifolds have been generated using 126 PDs with two degrees of freedom, with images uniformly duplicated and modified with experimentally-relevant SNR. 

## Attribution:
If this code is useful in your work, please cite: 
DOI

### License:
Copyright 2018-2021 Evan Seitz

The software, code sample and their documentation made available on this website could include technical or other mistakes, inaccuracies or typographical errors. We may make changes to the software or documentation made available on its web site at any time without prior notice. We assume no responsibility for errors or omissions in the software or documentation available from its web site. For further details, please see the LICENSE file.
