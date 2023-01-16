#!/bin/bash

#SBATCH --ntasks=10
#SBATCH --nodes=2
#SBATCH --cpus-per-task=3
#SBATCH --mem-per-cpu=3300m

#SBATCH --gpus-per-node=5
#SBATCH --gres=gpumem:20g
#SBATCH --time=0-23:59:59


#SBATCH --job-name=a3
#SBATCH --output=a3.out

#SBATCH --mail-type=ALL    # notify on state change: BEGIN, END, FAIL or ALL
#SBATCH --mail-user=atabin@student.ethz.ch  # who to send email notification for job stats changes

echo "Running A3"

cp exp/res101_stage1_718_24h/checkpoints/checkpoint.pth exp/res101_stage1/checkpoints/checkpoint.pth

rm -rf exp/res101_stage2_516_from_a1_with_170_epochs/
bash script/train/a3_train_stage2_res101.sh  res101_stage2_516_from_a1_with_170_epochs
echo "Done rendering"

exit 0
