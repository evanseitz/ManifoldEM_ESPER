#!/bin/sh

pyDir="${PWD}"
rotPath="$pyDir/Rotation_Histograms.py"

for i in {1..126}; #change to total number of PDs (e.g., {1..126} for 126 PDs)
do 
	foo=$(printf "%03d" $i)
	echo "${foo}"
	python $rotPath "${foo}"
done
