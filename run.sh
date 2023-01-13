#!/bin/bash

#SBATCH --ntasks=7
#SBATCH --nodes=1
#SBATCH --mem-per-cpu=40G

#SBATCH --gpus-per-node=8
#SBATCH --time=0-23:59:59

#SBATCH --job-name=24h_r101_718_a1
#SBATCH --output=24h_r101_718_a1.out
#SBATCH --error=24h_r101_718_a1.err

#SBATCH --mail-type=ALL    # notify on state change: BEGIN, END, FAIL or ALL
#SBATCH --mail-user=atabin@student.ethz.ch  # who to send email notification for job stats changes

echo "Allocating 7 tasks on 1 nodes with 8 GPUs per Node"
echo "Running A1"

rm -rf exp/res101_stage1_718_24h/

bash script/train/a1_train_stage1_res101.sh  res101_stage1_718_24h
echo "Done rendering"

exit 0
