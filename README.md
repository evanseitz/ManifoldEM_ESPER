# README
## Cryo-EM Synthethic Analysis

This repository contains the software implementation for our paper **Heuristic Analysis of COnformational Reconstruction Techniques Using Simulated Cryo-EM Ensembles** (Seitz, Acosta-Reyes, ..., Frank): https://www.biorxiv.org/???. It contains tools to apply the discussed methods to continuum models.

### Instructions:

### Required Software:
- Python
  - numpy, pylab, matplotlib, mrcfile, csv, itertools
- Chimera
- PyMol
- Phenix
- EMAN2
- RELION

### Environment:
First, install Anaconda. Navigate to your project directory via the command line interface and install the environment corresponding to your operating system via:

`conda create --name synth --file env_linux_64.txt`

`conda create --name synth --file env_mac_64.txt`

Once the Anaconda environment is installed, it must be initiated each time before running (the majority of) these scripts via the command: `conda activate synth`

When you are done using the environment, always exit via: `conda deactivate`

### Attribution:
Please cite ...

DOI


### License:
Copyright 2018-2020 Evan Seitz

For further details, please see the LICENSE file.
