#!/bin/bash

#SBATCH --ntasks=8
#SBATCH --nodes=2
#SBATCH --mem-per-cpu=40G

#SBATCH --gpus-per-node=8
#SBATCH --time=0-3:59:59

#SBATCH --job-name=r50_828_s2
#SBATCH --output=r50_828_s2.out
#SBATCH --error=r50_828_s2.err

#SBATCH --mail-type=ALL    # notify on state change: BEGIN, END, FAIL or ALL
#SBATCH --mail-user=atabin@student.ethz.ch  # who to send email notification for job stats changes

echo "Allocating 8 tasks on 2 nodes with 8 GPUs per Node"
echo "Running A2"

rm -rf exp/res50_stage2_828/
bash script/train/a2_train_stage2_res50.sh  res50_stage2_828
echo "Done rendering"

exit 0
