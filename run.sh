#!/bin/bash

#SBATCH --ntasks=5
#SBATCH --nodes=1
#SBATCH --mem-per-cpu=60G

#SBATCH --gpus-per-node=rtx_3090:6
#SBATCH --time=0-23:59:59

#SBATCH --job-name=a2
#SBATCH --output=a2.out

#SBATCH --mail-type=ALL    # notify on state change: BEGIN, END, FAIL or ALL
#SBATCH --mail-user=atabin@student.ethz.ch  # who to send email notification for job stats changes

echo "Allocating 5 tasks on 1 nodes with 6 GPUs per Node for 24h from a0 with 200 epochs"
echo "Running A2"

rm -rf exp/res50_stage2_from_a0_with_200_epochs/
bash script/train/a2_train_stage2_res50.sh  res50_stage2_from_a0_with_200_epochs
echo "Done rendering"

exit 0
