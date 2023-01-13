#!/bin/bash

#SBATCH --ntasks=4
#SBATCH --nodes=1
#SBATCH --mem-per-cpu=40G

#SBATCH --gpus-per-node=5
#SBATCH --time=0-23:59:59

#SBATCH --job-name=r101_415_a3_24h
#SBATCH --output=r101_415_a3_24h.out
#SBATCH --error=r101_415_a3_24h.err

#SBATCH --mail-type=ALL    # notify on state change: BEGIN, END, FAIL or ALL
#SBATCH --mail-user=atabin@student.ethz.ch  # who to send email notification for job stats changes

echo "Allocating 12 tasks on 2 nodes with 7 GPUs per Node for 24h"
echo "Running A3"

rm -rf exp/res101_stage2_415/
bash script/train/a3_train_stage2_res101.sh  res101_stage2_415
echo "Done rendering"

exit 0
