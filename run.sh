#!/bin/bash

#SBATCH --ntasks=2
#SBATCH --nodes=1
#SBATCH --mem-per-cpu=40G

#SBATCH --gpus-per-node=4
#SBATCH --time=0-3:59:59

#SBATCH --job-name=4h_r50_214_s3
#SBATCH --output=4h_r50_214_s3.out
#SBATCH --error=4h_r50_214_s3.err

#SBATCH --mail-type=ALL    # notify on state change: BEGIN, END, FAIL or ALL
#SBATCH --mail-user=atabin@student.ethz.ch  # who to send email notification for job stats changes

echo "Allocating 2 tasks on 1 nodes with 4 GPUs per Node for 4h"
echo "Running A3"

rm -rf exp/res50_stage3_214_4h/
bash script/train/a4_train_stage3_res50.sh  res50_stage3_214_4h
echo "Done rendering"

exit 0
