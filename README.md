# LETR Adapted : Line Segment Detection Using Transformers without Edges used to detect lines and classify them into textural/structural lines in an end-to-end fashion

## Introduction 
This repository contains an adapted version of the official code and pretrained models for [Line Segment Detection Using Transformers without Edges](https://arxiv.org/abs/2101.01909).

## Results and Checkpoints


| Name | sAP10 | sAP15 | sF10 | sF15 | URL|
| --- | --- | --- | --- | --- |--- |
| Wireframe | 65.6 | 68.0 | 66.1 | 67.4 | [LETR-R101](https://vcl.ucsd.edu/letr/checkpoints/res101/res101_stage2_focal.zip) |
| YorkUrban | 29.6 | 32.0 | 40.5 | 42.1 | [LETR-R50](https://vcl.ucsd.edu/letr/checkpoints/res50/res50_stage2_focal.zip) |

## Reproducing Results

### Installation

#### Conda

```bash
conda env create -f environment.yml
conda activate deepl
```


### Step3: Data Preparation

### Step4: Train Script Examples
1. Train a coarse-model (a.k.a. stage1 model).
    ```bash
    # Usage: bash script/*/*.sh [exp name]
    bash script/train/a0_train_stage1_res50.sh  res50_stage1 # LETR-R50  
    bash script/train/a1_train_stage1_res101.sh res101_stage1 # LETR-R101 
    ```

2. Train a fine-model (a.k.a. stage2 model).
    ```bash
    # Usage: bash script/*/*.sh [exp name]
    bash script/train/a2_train_stage2_res50.sh  res50_stage2  # LETR-R50
    bash script/train/a3_train_stage2_res101.sh res101_stage2 # LETR-R101 
    ```

3. Fine-tune the fine-model with focal loss (a.k.a. stage2_focal model).
    ```bash
    # Usage: bash script/*/*.sh [exp name]
    bash script/train/a4_train_stage2_focal_res50.sh   res50_stage2_focal # LETR-R50
    bash script/train/a5_train_stage2_focal_res101.sh  res101_stage2_focal # LETR-R101 
    ```

### Acknowledgments

This code is based on the implementations of [**LETR: Line Segment Detection Using Transformers without Edges**](https://github.com/mlpc-ucsd/LETR). 
