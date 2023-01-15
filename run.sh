#!/bin/bash

#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=3
#SBATCH --mem-per-cpu=3300m

#SBATCH --gpus-per-node=2
#SBATCH --gres=gpumem:20g
#SBATCH --time=0-23:59:59


#SBATCH --job-name=a5
#SBATCH --output=a5.out

#SBATCH --mail-type=ALL    # notify on state change: BEGIN, END, FAIL or ALL
#SBATCH --mail-user=atabin@student.ethz.ch  # who to send email notification for job stats changes

echo "Running A5"

rm -rf exp/a5/
bash script/train/a5_train_stage2_focal_res101.sh  a5
echo "Done rendering"

exit 0
