#!/bin/sh

pyDir="${PWD}"
dmPath="$pyDir/DiffusionMaps.py"

for i in {33..55}; #change to total number of PDs (e.g., {1..126} for 126 PDs)
do 
	foo=$(printf "%03d" $i)
	echo "${foo}"
	python $dmPath "${foo}"
done
