#!/bin/sh

pyDir="${PWD}"
snrPath="$pyDir/Generate_SNRtau.py"

for i in {1..5}; #change to total number of PDs (e.g., {1..126} for 126 PDs)
do 
	foo=$(printf "%03d" $i)
	echo "${foo}"
	python $snrPath "${foo}"
done