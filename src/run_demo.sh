#!/bin/bash

#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --mem-per-cpu=20G

#SBATCH --gpus-per-node=1
#SBATCH --time=0-3:59:59

#SBATCH --job-name=demo
#SBATCH --output=demo.out


echo "Running demo"


EXP_PATH=/cluster/scratch/atabin/LETR_euler/exp
echo "EXP_PATH: $EXP_PATH"
echo "Copying checkpoints to $EXP_PATH/checkpoints"
cp $EXP_PATH/res50_stage1_415_24h/checkpoints/checkpoint.pth $EXP_PATH/checkpoints/checkpoint_res50_s1.pth
cp $EXP_PATH/res50_stage2_617_24h/checkpoints/checkpoint.pth $EXP_PATH/checkpoints/checkpoint_res50_s2.pth
cp $EXP_PATH/res101_stage1_718_24h/checkpoints/checkpoint.pth $EXP_PATH/checkpoints/checkpoint_res101_s1.pth
cp $EXP_PATH/res101_stage2_415/checkpoints/checkpoint.pth $EXP_PATH/checkpoints/checkpoint_res101_s2.pth

python3 demo.py

echo "Done running demo"

exit 0
