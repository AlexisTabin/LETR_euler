#!/bin/bash

#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=3
#SBATCH --mem-per-cpu=3300m

#SBATCH --gpus-per-node=2
#SBATCH --gres=gpumem:20g
#SBATCH --time=0-23:59:59


#SBATCH --job-name=a4
#SBATCH --output=a4.out

#SBATCH --mail-type=ALL    # notify on state change: BEGIN, END, FAIL or ALL
#SBATCH --mail-user=atabin@student.ethz.ch  # who to send email notification for job stats changes

echo "Running A4"

rm -rf exp/a4/
bash script/train/a4_train_stage2_focal_res50.sh  a4
echo "Done rendering"

exit 0
