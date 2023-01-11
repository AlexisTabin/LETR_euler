#!/bin/bash

#SBATCH --ntasks=8
#SBATCH --nodes=4
#SBATCH --mem-per-cpu=40G

#SBATCH --gpus-per-node=8
#SBATCH --time=0-3:59:59

#SBATCH --job-name=letr_848
#SBATCH --output=letr_848.out
#SBATCH --error=letr_848.err

#SBATCH --mail-type=BEGIN    # notify on state change: BEGIN, END, FAIL or ALL
#SBATCH --mail-user=$USER   # who to send email notification for job stats changes

echo "Allocating 8 tasks on 4 nodes with 8 GPUs per Node"

rm -rf exp/res50_stage1_848/

wandb login

bash script/train/a0_train_stage1_res50.sh  res50_stage1_848
echo "Done rendering"

exit 0
