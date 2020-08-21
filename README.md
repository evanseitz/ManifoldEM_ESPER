# README
## Cryo-EM Heuristic Analysis

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

Detailed directions for installing these libraries are available at https://github.com/evanseitz/cryoEM_synthetic_continua.

### Environment:
Users will first need to install Anaconda and then create an environment with the above Python requirements. Once the Anaconda environment is setup, remember to initiate it each time before running the scripts in this repository. As well, when you are done using the environment, always exit via: `conda deactivate`

### Attribution:
Please cite ...

DOI


### License:
Copyright 2018-2020 Evan Seitz

For further details, please see the LICENSE file.
