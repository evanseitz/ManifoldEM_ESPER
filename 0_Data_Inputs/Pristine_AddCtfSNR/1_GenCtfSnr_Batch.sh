#!/bin/bash

pyDir="${PWD}"
scriptPath="$pyDir/GenStackAlign_CtfSnr.py"

for i in $(seq 1 5); #change to total number of PDs (e.g., {1..126} for 126 PDs)
do 
	foo=$(printf "%03d" $i)
	echo "${foo}"
	python $scriptPath "${foo}"
done
