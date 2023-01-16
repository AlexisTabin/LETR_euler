source ~/.bashrc

conda activate deepl

python3 src/main.py --coco_path dataset_marco \
    --LETRpost --backbone resnet50 --layer1_frozen --resume exp/res50_stage2_214_4h/checkpoints/checkpoint.pth \
    --no_opt --batch_size 1 --epochs 1 --lr_drop 25 --num_gpus 0 --device cpu --wandb_name 'test_local_a4' --num_queries 1000 \
    --lr 1e-5  --label_loss_func focal_loss \
    --label_loss_params '{"gamma":2.0}'  --save_freq 1 

# scp -r atabin@euler.ethz.ch:/cluster/scratch/atabin/LETR_euler/exp/res50_stage2_214_4h/ ./exp/
