#!/bin/bash
set -e
set -v
#cp ../dataset/*/train/*.off data_train/raw
#cp ../dataset/*/test/*.off data_test/raw

declare -a arr=("airplane" "person" "bed" "bottle")
#declare -a arr=("xbox bed")
for i in "${arr[@]}"
do
cp -n ../dataset/$i/train/*.off data_train/raw
cp -n ../dataset/$i/test/*.off data_test/raw
done
