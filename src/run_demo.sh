#!/bin/bash

#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --mem-per-cpu=20G

#SBATCH --gpus-per-node=1
#SBATCH --time=0-3:59:59

#SBATCH --job-name=demo
#SBATCH --output=demo.out

#SBATCH --mail-type=BEGIN    
#SBATCH --mail-user=$USER   

echo "Running demo"

python3 demo.py

echo "Done running demo"

exit 0
