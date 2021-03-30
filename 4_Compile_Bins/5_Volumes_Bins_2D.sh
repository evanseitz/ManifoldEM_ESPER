# Copyright (c) Columbia University Evan Seitz 2020

# If `4_Generate_Align_2D.py` has been run, initiate this code via...
# ...`sh 5_Volumes_Bins_2D.sh` to create 3D volumes for each stack/STAR
# Note: the choice of --maxres can be adjusted below as needed for your data

cd S2_bins_GC3_v5 #may need to change based on user directory name
cd CM1_CM2
scriptDir="$( cd "$(dirname "$0")" ; pwd -P )"

for filename in *.star
do
    echo "input: $filename"
    echo "output: ${scriptDir}/${filename%.*}.mrc"
    relion_reconstruct --i "${filename}" --o "${scriptDir}/${filename%.*}.mrc" --maxres 3
done
