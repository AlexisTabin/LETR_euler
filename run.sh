#!/bin/bash

#SBATCH -n 1
#SBATCH -N 1
#SBATCH --mem-per-cpu=40G

#SBATCH --gpus-per-node=1
#SBATCH --time=0-00:10:00

#SBATCH --job-name=letr
#SBATCH --output=letr_off.out
#SBATCH --error=letr_off.err
#SBATCH --mail-user=$USER@ethz.ch

echo "Allocating 8 tasks on 8 nodes with 8 GPUs per Node"

rm -rf exp/res50_stage1/

bash script/train/a0_train_stage1_res50.sh  res50_stage1
echo "Done rendering"

exit 0
