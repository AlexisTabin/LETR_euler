#!/bin/bash

#SBATCH --ntasks=2
#SBATCH --nodes=1
#SBATCH --mem-per-cpu=40G

#SBATCH --gpus-per-node=3
#SBATCH --time=0-3:59:59

#SBATCH --job-name=3h_r101_213_s1
#SBATCH --output=4h_r101_213_s1.out
#SBATCH --error=4h_r101_213_s1.err

#SBATCH --mail-type=ALL    # notify on state change: BEGIN, END, FAIL or ALL
#SBATCH --mail-user=atabin@student.ethz.ch  # who to send email notification for job stats changes

echo "Allocating 2 tasks on 1 nodes with 3 GPUs per Node"
echo "Running A1"

rm -rf exp/res101_stage1_213/

bash script/train/a1_train_stage1_res101.sh  res101_stage1_213
echo "Done rendering"

exit 0
