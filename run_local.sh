source ~/.bashrc

conda activate deepl

python3 src/main.py --coco_path dataset_marco --backbone resnet50 --epochs 1 --device cpu