#!/bin/bash

#SBATCH --ntasks=8
#SBATCH --nodes=2
#SBATCH --mem-per-cpu=40G

#SBATCH --gpus-per-node=8
#SBATCH --time=0-0:59:59

#SBATCH --job-name=1h_r50_828_s2
#SBATCH --output=1h_r50_828_s2.out
#SBATCH --error=1h_r50_828_s2.err

#SBATCH --mail-type=ALL    # notify on state change: BEGIN, END, FAIL or ALL
#SBATCH --mail-user=atabin@student.ethz.ch  # who to send email notification for job stats changes

echo "Allocating 8 tasks on 2 nodes with 5 GPUs per Node for only 1h"
echo "Running A2"

rm -rf exp/res50_stage2_825_1h/
bash script/train/a2_train_stage2_res50.sh  res50_stage2_825_1h
echo "Done rendering"

exit 0
