# Copyright (c) Columbia University Evan Seitz 2020

# If `2_Compile_Bins_2D.py` has been run, initiate this code via...
# ...`sh 2_Volumes_Bins_2D.sh` to create 3D volumes for each stack/STAR
# Note: the choice of --maxres can be adjusted below as needed for your data;
# ...as can --ctf (remove this flag if no CTF was used during data construction)

cd S2_bins #may need to change based on user directory name
cd CM1_CM2
scriptDir="$( cd "$(dirname "$0")" ; pwd -P )"

for filename in *.star
do
    echo "input: $filename"
    echo "output: ${scriptDir}/${filename%.*}.mrc"
    relion_reconstruct --i "${filename}" --o "${scriptDir}/${filename%.*}.mrc" --ctf #--maxres 3
done
