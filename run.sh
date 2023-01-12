#!/bin/bash

#SBATCH --ntasks=4
#SBATCH --nodes=1
#SBATCH --mem-per-cpu=40G

#SBATCH --gpus-per-node=6
#SBATCH --time=0-3:59:59

#SBATCH --job-name=3h_r50_416_s2
#SBATCH --output=4h_r50_416_s2.out
#SBATCH --error=4h_r50_416_s2.err

#SBATCH --mail-type=ALL    # notify on state change: BEGIN, END, FAIL or ALL
#SBATCH --mail-user=atabin@student.ethz.ch  # who to send email notification for job stats changes

echo "Allocating 4 tasks on 1 nodes with 6 GPUs per Node for 4h"
echo "Running A2"

rm -rf exp/res50_stage2_416_4h/
bash script/train/a2_train_stage2_res50.sh  res50_stage2_416_4h
echo "Done rendering"

exit 0
