#!/bin/bash
echo "Enter dataset name"
read dataset
if [ $dataset == "KITTI" ]; then
    SEQS=( "00" "01" "02" "03" "04" "05" "06" "07" "08" "09" "10" )
else
    SEQS=( "00" "01" "02" "03" "04" )
fi
`mkdir -p data/$dataset/pose`
`mkdir -p data/$dataset/relative_pose`
`mkdir -p data/$dataset/calib`
`mkdir -p data/$dataset/scan`
for SEQ in ${SEQS[@]}; do
    `mkdir -p data/$dataset/scan/$SEQ`
done
`mkdir -p data/$dataset/preprocessed_data`
for SEQ in ${SEQS[@]}; do
    `mkdir -p data/$dataset/preprocessed_data/$SEQ`
done
`mkdir checkpoints`
`mkdir result`
`mkdir runs`