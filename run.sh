#!/bin/bash

#SBATCH --ntasks=8
#SBATCH --nodes=1
#SBATCH --mem-per-cpu=40G

#SBATCH --gpus-per-node=8
#SBATCH --time=0-3:59:59

#SBATCH --job-name=resnet101_818
#SBATCH --output=resnet101_818.out
#SBATCH --error=resnet101_818.err

#SBATCH --mail-type=ALL    # notify on state change: BEGIN, END, FAIL or ALL
#SBATCH --mail-user=atabin@student.ethz.ch  # who to send email notification for job stats changes

echo "Allocating 8 tasks on 1 nodes with 8 GPUs per Node"
echo "Running A1"

rm -rf exp/res101_stage2_818/

bash script/train/a2_train_stage2_res50.sh  res50_stage2_818
echo "Done rendering"

exit 0
