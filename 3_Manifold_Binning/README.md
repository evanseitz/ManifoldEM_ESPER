# README
## Manifold Binning folder

This folder contains scripts to stratify CM-subspaces from each PD into a sequence of bins.

To setup for this procedure, please read the description in the header of the `Manifold_Binning.py` script, and alter all user parameters there as you deem fit. To keep in mind, there are also several potentially important parameters scattered throughout this framework which may need to be adjusted depending on your data. Please see comments for more information. Once you are ready, a batch computation can be initiated via `python Manifold_Binning.py`.

All outputs will be saved per PD to the `bins` folder in this same directory. After all computations are complete for every PD, you will need to peruse this outputs folder and make several key decisions. Your decisions can be recorded in the `CMs.txt` and `Senses.txt` files within each PD subdirectory (e.g., `bins/PD001/Senses.txt`). You will need to watch the 2D movie in each `CM1` or `CM2` (etc.) subfolder and determine to which conformational motion it belongs. This is an arbitrary assignment so long as your label is consistently assigned across all PDs. Likewise, the sense - or direction - of that motion must also be defined (forward or reverse), and held consistent across all PDs.
