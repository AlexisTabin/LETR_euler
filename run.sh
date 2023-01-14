#!/bin/bash

#SBATCH --ntasks=5
#SBATCH --nodes=1
#SBATCH --mem-per-cpu=40G

#SBATCH --gpus-per-node=6
#SBATCH --time=0-23:59:59

#SBATCH --job-name=a3
#SBATCH --output=a3.out

#SBATCH --mail-type=ALL    # notify on state change: BEGIN, END, FAIL or ALL
#SBATCH --mail-user=atabin@student.ethz.ch  # who to send email notification for job stats changes

echo "Allocating 12 tasks on 2 nodes with 7 GPUs per Node for 24h"
echo "Running A3"

cp exp/res101_stage1_718_24h/checkpoints/checkpoint.pth exp/res101_stage1/checkpoints/checkpoint.pth

rm -rf exp/res101_stage2_516_from_a1_with_170_epochs/
bash script/train/a3_train_stage2_res101.sh  res101_stage2_516_from_a1_with_170_epochs
echo "Done rendering"

exit 0
