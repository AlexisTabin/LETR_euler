#!/bin/bash

<<<<<<< HEAD
#SBATCH --ntasks=3
#SBATCH --nodes=1
#SBATCH --mem-per-cpu=40G

#SBATCH --gpus-per-node=4
#SBATCH --time=0-3:59:59

#SBATCH --job-name=3h_r50_314_s2
#SBATCH --output=4h_r50_314_s2.out
#SBATCH --error=4h_r50_314_s2.err
=======
#SBATCH --ntasks=8
#SBATCH --nodes=8
#SBATCH --mem-per-cpu=40G

#SBATCH --gpus-per-node=8
#SBATCH --time=0-11:59:59

#SBATCH --job-name=resnet101_888
#SBATCH --output=resnet101_888.out
#SBATCH --error=resnet101_888.err
>>>>>>> ce67311 (running res101 and a2)

#SBATCH --mail-type=ALL    # notify on state change: BEGIN, END, FAIL or ALL
#SBATCH --mail-user=atabin@student.ethz.ch  # who to send email notification for job stats changes

echo "Allocating 8 tasks on 8 nodes with 8 GPUs per Node"
echo "Running A1"

rm -rf exp/res101_stage1_888/

bash script/train/a1_train_stage1_res101.sh  res101_stage1_888
echo "Done rendering"

exit 0
