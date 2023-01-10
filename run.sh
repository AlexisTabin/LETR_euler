#!/bin/bash

#SBATCH -n 8
#SBATCH -N 8
#SBATCH --mem-per-cpu=40G
#SBATCH --gpus-per-node=8
#SBATCH --time=0-03:59:59

#SBATCH --job-name=letr
#SBATCH --output=letr.out
#SBATHC --error=letr.err

echo "Allocating 8 tasks on 8 nodes with 8 GPUs per Node"

rm -rf exp/res50_stage1/

bash script/train/a0_train_stage1_res50.sh  res50_stage1
echo "Done rendering"

exit 0
