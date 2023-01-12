#!/bin/bash

#SBATCH --ntasks=2
#SBATCH --nodes=1
#SBATCH --mem-per-cpu=40G

#SBATCH --gpus-per-node=4
#SBATCH --time=0-11:59:59

#SBATCH --job-name=4h_r50_214_s3
#SBATCH --output=4h_r50_214_s3.out
#SBATCH --error=4h_r50_214_s3.err

#SBATCH --mail-type=ALL    # notify on state change: BEGIN, END, FAIL or ALL
#SBATCH --mail-user=atabin@student.ethz.ch  # who to send email notification for job stats changes

echo "Allocating 2 tasks on 1 nodes with 4 GPUs per Node for 12h from resume"
echo "Running A4"

rm -rf exp/res50_stage3_214_12h_resumed/
bash script/train/a4_train_stage2_focal_res50.sh  res50_stage3_214_12h_resumed
echo "Done rendering"

exit 0
