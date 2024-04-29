import os
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import lightning as pl
import segmentation_models_pytorch as smp
from dataset import TrainDataset
from config import cfg

from pprint import pprint
from torch.utils.data import DataLoader

import cv2
import numpy as np
import os
from pylsd import lsd

def produce_lsd_mask_with_normals(normals):
    res = torch.where(normals < 1 , torch.ones_like(normals)*100, torch.ones_like(normals))
    return res

#PyTorch
gpu_nums = cfg.TRAIN.gpu
wd = cfg.TRAIN.wd #weight decay for adamW

class IoULoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(IoULoss, self).__init__()

    def forward(self, inputs, targets, smooth=1):
        
        #comment out if your model contains a sigmoid or equivalent activation layer
        inputs = F.sigmoid(inputs)       
        
        #flatten label and prediction tensors
        inputs = inputs.view(-1)
        targets = targets.view(-1)
        
        #intersection is equivalent to True Positive count
        #union is the mutually inclusive area of all labels & predictions 
        intersection = (inputs * targets).sum()
        total = (inputs + targets).sum()
        union = total - intersection 
        
        IoU = (intersection + smooth)/(union + smooth)
                
        return 1 - IoU


def WMSELoss(input, target, weight=2.0):
    mask = torch.abs(input) > 1.0
    sqr = (input - target) ** 2
    return torch.mean(sqr*(weight * mask + ~mask))

class StructurenessModel(pl.LightningModule):

    def __init__(self, arch, encoder_name, in_channels, out_classes, **kwargs):
        super().__init__()
        self.model = smp.create_model(
            arch, encoder_name=encoder_name, in_channels=in_channels, classes=out_classes, **kwargs
        )

        # preprocessing parameteres for image
        params = smp.encoders.get_preprocessing_params(encoder_name)
        self.register_buffer("std", torch.tensor(params["std"]).view(1, 3, 1, 1))
        self.register_buffer("mean", torch.tensor(params["mean"]).view(1, 3, 1, 1))


        #from paper: https://hal.science/hal-01925321/document
        #self.loss_fn = nn.L1Loss()
        self.loss_fn = torch.nn.BCELoss()
        #check if less
        # self.loss_fn = torch.nn.MSELoss()
        # self.loss_fn = WMSELoss

    def forward(self, image):
        # normalize image
        image = (image - self.mean) / self.std
        mask = self.model(image)
        return mask

    def shared_step(self, batch, stage):
        image = batch["image"]
        line_proximity_weights = produce_lsd_mask_with_normals(batch['mask'])
        # Shape of the image should be (batch_size, num_channels, height, width)
        assert image.ndim == 4

        # Check that image dimensions are divisible by 32, 
        # encoder and decoder connected by `skip connections` and usually encoder have 5 stages of 
        # downsampling by factor 2 (2 ^ 5 = 32);
        h, w = image.shape[2:]
        assert h % 32 == 0 and w % 32 == 0

        mask = batch["mask"]

        # Shape of the mask should be [batch_size, num_classes (1), height, width]
        assert mask.ndim == 4

        # Check that mask values in between -1 and 1
        assert mask.max() <= 1.0 and mask.min() >= -1.0

        logits_mask = self.forward(image)
        logits_mask = logits_mask # we don't need class dimension
        prob_mask = logits_mask.sigmoid()
        #loss = self.loss_fn(prob_mask, (mask+1)/2)
        loss = self.loss_fn(prob_mask*line_proximity_weights, mask.abs()*line_proximity_weights)

        # prob_mask = logits_mask.tanh()
        #prob_mask_weighted = prob_mask * 
        #loss = self.loss_fn(prob_mask, mask)

        return {
            "loss": loss,
        }

    def shared_epoch_end(self, outputs, stage):

        metrics = {
            f"{stage}_loss_epoch": torch.mean(torch.tensor([x["loss"] for x in outputs]))
        }
        
        self.log_dict(metrics, prog_bar=True)

    def training_step(self, batch, batch_idx):
        loss = self.shared_step(batch, "train") 
        self.log(f"train_loss", loss)
        return loss

    def training_epoch_end(self, outputs):
        return self.shared_epoch_end(outputs, "train")

    def validation_step(self, batch, batch_idx):
        loss = self.shared_step(batch, "valid") 
        self.log(f"valid_loss", loss)
        return loss

    def validation_epoch_end(self, outputs):
        return self.shared_epoch_end(outputs, "valid")

    def test_step(self, batch, batch_idx):
        image = batch["image"]

        assert image.ndim == 4

        h, w = image.shape[2:]
        assert h % 32 == 0 and w % 32 == 0

        mask = batch["mask"]

        assert mask.ndim == 4
        assert mask.max() <= 1.0 and mask.min() >= -1.0

        logits_mask = self.forward(image)

        prob_mask = logits_mask.sigmoid()
        loss = self.loss_fn(prob_mask, mask.abs())



        # Save predicted normal difference
        prob_mask = prob_mask.squeeze(1)
        for i, savedir in enumerate(batch["savedir"]):
            dir = os.path.join(*savedir.split('/')[:-1])
            if not os.path.exists(dir):
                os.makedirs(dir)
            mask = torch.flip(prob_mask[i], [1]) if batch['flipped'][i] else prob_mask[i]
            np.savez_compressed(savedir, normals_diff=(2*mask-1).cpu().numpy())

        return {
            "loss": loss.item(),
        }

    def test_epoch_end(self, outputs):
        return self.shared_epoch_end(outputs, "test")

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=0.0001, weight_decay=wd)


cfg.merge_from_file("config/params_hypersim.yaml")



train_dataset = TrainDataset(
    cfg.DATASET.root_dataset,
    cfg.DATASET.list_train,
    cfg.DATASET)
train_dataloader = torch.utils.data.DataLoader(
    train_dataset,
    batch_size=cfg.TRAIN.batch_size,  # we have modified data_parallel
    shuffle=True,  # we do not use this param
    num_workers=cfg.TRAIN.workers,
    drop_last=True,
    pin_memory=True)

valid_dataset = TrainDataset(
    cfg.DATASET.root_dataset,
    cfg.DATASET.list_val,
    cfg.DATASET)

valid_dataloader = torch.utils.data.DataLoader(
    valid_dataset,
    batch_size=cfg.TRAIN.batch_size,  # we have modified data_parallel
    shuffle=False,  # we do not use this param
    num_workers=cfg.TRAIN.workers,
    drop_last=False,
    pin_memory=True)

test_dataset = TrainDataset(
    cfg.DATASET.root_dataset,
    cfg.DATASET.list_test,
    cfg.DATASET)

test_dataloader = torch.utils.data.DataLoader(
    test_dataset,
    batch_size=cfg.TRAIN.batch_size,  # we have modified data_parallel
    shuffle=False,  # we do not use this param
    num_workers=cfg.TRAIN.workers,
    drop_last=False,
    pin_memory=True)


model = StructurenessModel(cfg.MODEL.arch_decoder, cfg.MODEL.arch_encoder, in_channels=3, out_classes=1)


trainer = pl.Trainer(
    accelerator='gpu',
    devices = gpu_nums, 
    max_epochs=40,
    accumulate_grad_batches=32,
    default_root_dir=cfg.MODEL.path
)

print("TRAINING")

trainer.fit(
    model, 
    train_dataloaders=train_dataloader, 
    val_dataloaders=valid_dataloader,
)
print("VALIDATION")

# run validation dataset
valid_metrics = trainer.validate(model, dataloaders=valid_dataloader, verbose=False)
pprint(valid_metrics)

print("TESTING")
# run test dataset
test_metrics = trainer.test(model, dataloaders=test_dataloader, verbose=False)
pprint(test_metrics)
