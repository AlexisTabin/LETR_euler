#!/bin/bash

#SBATCH --ntasks=32
#SBATCH --nodes=4
#SBATCH --mem-per-cpu=40G

#SBATCH --gpus-per-node=8
#SBATCH --time=0-23:59:59

#SBATCH --job-name=24h_r101_32_48_a1
#SBATCH --output=24h_r101_32_48_a1.out
#SBATCH --error=24h_r101_32_48_a1.err

#SBATCH --mail-type=ALL    # notify on state change: BEGIN, END, FAIL or ALL
#SBATCH --mail-user=atabin@student.ethz.ch  # who to send email notification for job stats changes

echo "Allocating 32 tasks on 4 nodes with 8 GPUs per Node"
echo "Running A1"

rm -rf exp/res101_stage1_32_48_24h/

bash script/train/a1_train_stage1_res101.sh  res101_stage1_32_48_24h
echo "Done rendering"

exit 0
