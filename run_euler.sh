#!/bin/bash

#SBATCH -n 1
#SBATCH -N 1
#SBATCH --mem-per-cpu=10G
#SBATCH --gpus-per-node=8
#SBATCH --time=0-08:00:00

#SBATCH --job-name=letr118
#SBATCH --output=letr118.out
#SBATHC --error=letr118.err

echo "Allocating 1 tasks on 1 nodes with 8 GPUs per Node"

rm -rf exp/res50_stage1/

conda activate deepl

bash script/train/a0_train_stage1_res50.sh  res50_stage1

echo "Done rendering"

exit 0
