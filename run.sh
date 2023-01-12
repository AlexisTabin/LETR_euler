#!/bin/bash

#SBATCH --ntasks=3
#SBATCH --nodes=1
#SBATCH --mem-per-cpu=40G

#SBATCH --gpus-per-node=4
#SBATCH --time=0-3:59:59

#SBATCH --job-name=3h_r50_314_s2_resume
#SBATCH --output=4h_r50_314_s2_resume.out
#SBATCH --error=4h_r50_314_s2_resume.err

#SBATCH --mail-type=ALL    # notify on state change: BEGIN, END, FAIL or ALL
#SBATCH --mail-user=atabin@student.ethz.ch  # who to send email notification for job stats changes

echo "Allocating 3 tasks on 1 nodes with 4 GPUs per Node for 4h from resume"
echo "Running A2 from resume"

rm -rf exp/res50_stage2_314_4h_resumed/
bash script/train/a2_train_stage2_res50.sh  res50_stage2_314_4h_resumed
echo "Done rendering"

exit 0
