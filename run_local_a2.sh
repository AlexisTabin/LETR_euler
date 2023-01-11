source ~/.bashrc

conda activate deepl

python3 src/main.py --coco_path dataset_marco \
    --LETRpost --backbone resnet50 --layer1_frozen --frozen_weights exp/res50_stage1_848/checkpoints/checkpoint.pth --no_opt \
    --batch_size 1 --epochs 1 --lr_drop 120 --num_gpus 0 --device cpu