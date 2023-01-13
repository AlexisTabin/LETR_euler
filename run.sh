#!/bin/bash

#SBATCH --ntasks=6
#SBATCH --nodes=1
#SBATCH --mem-per-cpu=40G

#SBATCH --gpus-per-node=7
#SBATCH --time=0-23:59:59

#SBATCH --job-name=24h_r50_617_a2
#SBATCH --output=24h_r50_617_a2.out
#SBATCH --error=24h_r50_617_a2.err

#SBATCH --mail-type=ALL    # notify on state change: BEGIN, END, FAIL or ALL
#SBATCH --mail-user=atabin@student.ethz.ch  # who to send email notification for job stats changes

echo "Allocating 6 tasks on 1 nodes with 7 GPUs per Node for 24h from resume"
echo "Running A2 from resume"

rm -rf exp/res50_stage2_617_24h/
bash script/train/a2_train_stage2_res50.sh  res50_stage2_617_24h
echo "Done rendering"

exit 0
