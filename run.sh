#!/bin/bash

#SBATCH -n 8
#SBATCH -N 8
#SBATCH --mem-per-cpu=20G
#SBATCH --gpus-per-node=8
#SBATCH --time=0-08:00:00

#SBATCH --job-name=letr888_20G
#SBATCH --output=letr888_20G.out
#SBATHC --error=letr888_20G.err

echo "Allocating 8 tasks on 8 nodes with 8 GPUs per Node"

rm -rf exp/res50_stage1/

conda activate deepl

bash script/train/a0_train_stage1_res50.sh  res50_stage1

echo "Done rendering"

exit 0
