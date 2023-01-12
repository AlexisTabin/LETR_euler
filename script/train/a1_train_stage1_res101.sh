# Fail the script if there is any failure
set -e

if [[ $# -eq 0 ]] ; then
    echo 'Require Experiment Name'
    exit 1
fi

# The name of this experiment.
name=$1

# Save logs and models under snap/gqa; make backup.
output=exp/$name
if [ ! -d "$output"  ]; then
    echo "folder not exist"
    mkdir -p $output/src
    cp -r src/* $output/src/
    cp $0 $output/run.bash

    PYTHONPATH=$PYTHONPATH:./src python -m torch.distributed.launch \
    --master_port=$((1000 + RANDOM % 9999)) --nproc_per_node=2 --use_env  src/main.py --coco_path data/dataset_marco \
    --output_dir $output --backbone resnet101 --wandb_name $name \
    --batch_size 1 --epochs 500 --lr_drop 200 --num_queries 1000  --num_gpus 2   --layer1_num 3 | tee -a $output/history.txt \

else
    echo "folder already exist"
fi




