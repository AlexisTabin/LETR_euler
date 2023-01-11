import os
import json
import torch
from torchvision import transforms
import numpy as np
from PIL import Image
import pickle
import cv2

class BaseDataset(torch.utils.data.Dataset):
    def __init__(self, odgt, opt, **kwargs):
        self.random_flip = opt.random_flip
        # parse the input list
        self.parse_input_list(odgt, **kwargs)

    def parse_input_list(self, odgt, max_sample=-1, start_idx=-1, end_idx=-1):
        if isinstance(odgt, list):
            self.list_sample = odgt
        elif isinstance(odgt, str):
            self.list_sample = [json.loads(x.rstrip()) for x in open(odgt, 'r')]

        if max_sample > 0:
            self.list_sample = self.list_sample[0:max_sample]
        if start_idx >= 0 and end_idx >= 0:     # divide file list
            self.list_sample = self.list_sample[start_idx:end_idx]

        self.num_sample = len(self.list_sample)
        assert self.num_sample > 0
        print('# samples: {}'.format(self.num_sample))

class TrainDataset(BaseDataset):
    def __init__(self, root_dataset, odgt, opt, **kwargs):
        super(TrainDataset, self).__init__(odgt, opt, **kwargs)
        self.root_dataset = root_dataset

    def __getitem__(self, index):
        this_record = self.list_sample[index]
        # load image and label
        image_path = os.path.join(self.root_dataset, this_record['fpath_img'])
        lines = os.path.join(self.root_dataset, this_record['fpath_lines'])
        img = cv2.imread(image_path)
        pick_file = open(lines, 'rb')
        data = pickle.load(pick_file)
        #assert(img.size[0] == segm.size[0])
        #assert(img.size[1] == segm.size[1])

        paths = this_record['fpath_img'].split('/')
        paths[2] = "pred_classification"
        camnum = paths[3].split('_')[2]
        paths[3] = f"cam_{camnum}"

        output = dict()
        output['image'] = img
        output['lines'] = data        
        output['savedir'] = os.path.join(self.root_dataset, *paths)
        output['filename'] = this_record['fpath_img']
        return output

    def __len__(self):
        return self.num_sample