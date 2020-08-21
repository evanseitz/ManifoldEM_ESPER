# README
## Cryo-EM Synthethic Analysis

This repository contains the software implementation for our paper **Heuristic Analysis of Manifolds from Simulated Cryo-EM Ensemble Data** (Seitz, Schwander, Maji, Acosta-Reyes, Liao, Frank): https://www.biorxiv.org/?TBD?. It contains tools to apply the discussed methods to quasi-continuum models.

These algorithms use synthetic data generated as described in the SI. As well, a detailed description is also provided in our previous paper **Simulation of Cryo-EM Ensembles from Atomic Models of Molecules Exhibiting Continuous Conformations** (Seitz, Acosta-Reyes, Schwander, Frank), along with published repository:
- Paper: https://www.biorxiv.org/content/10.1101/864116v1
- Repository: https://github.com/evanseitz/cryoEM_synthetic_continua.

### Instructions:

### Required Software:
- Python
  - numpy, pylab, matplotlib, mrcfile, csv, itertools, sklearn
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
