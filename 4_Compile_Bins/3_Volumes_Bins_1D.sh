# Copyright (c) Columbia University Evan Seitz 2020

# If `2_Generate_Align_1D.py` has been run, initiate this code via...
# ...`sh 3_Volumes_Bins_1D.sh` to create 3D volumes for each stack/STAR
# Note: each `relion_reconstruct` below call can also be given the parameter `maxres`
# e.g.: relion_reconstruct --i "${filename}" --o "${scriptDir}/${filename%.*}.mrc" --maxres 3

cd S2_bins
cd CM1
scriptDir="$( cd "$(dirname "$0")" ; pwd -P )"

for filename in *.star
do
    echo "input: $filename"
    echo "output: ${scriptDir}/${filename%.*}.mrc"
    relion_reconstruct --i "${filename}" --o "${scriptDir}/${filename%.*}.mrc"
done

cd ..
cd CM2
scriptDir="$( cd "$(dirname "$0")" ; pwd -P )"

for filename in *.star
do
    echo "input: $filename"
    echo "output: ${scriptDir}/${filename%.*}.mrc"
    relion_reconstruct --i "${filename}" --o "${scriptDir}/${filename%.*}.mrc"
done
