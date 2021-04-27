# Copyright (c) Columbia University Evan Seitz 2020

# If `1_Generate_Align_1D.py` has been run, initiate this code via...
# ...`sh 1_Volumes_Bins_1D.sh` to create 3D volumes for each stack/STAR
# Note: the choice of --maxres can be adjusted below as needed for your data;
# ...as can --ctf (remove this flag if no CTF was used during data construction)

cd S2_bins
cd CM1
scriptDir="$( cd "$(dirname "$0")" ; pwd -P )"

for filename in *.star
do
    echo "input: $filename"
    echo "output: ${scriptDir}/${filename%.*}.mrc"
    relion_reconstruct --i "${filename}" --o "${scriptDir}/${filename%.*}.mrc" --ctf #--maxres 3
done

cd ..
cd CM2
scriptDir="$( cd "$(dirname "$0")" ; pwd -P )"

for filename in *.star
do
    echo "input: $filename"
    echo "output: ${scriptDir}/${filename%.*}.mrc"
    relion_reconstruct --i "${filename}" --o "${scriptDir}/${filename%.*}.mrc" --ctf #--maxres 3
done
