import os
import json
import torch
from torchvision import transforms
import numpy as np
from PIL import Image
import pickle
from skimage import draw
from config import cfg


def clamp(n, smallest, largest): return max(smallest, min(n, largest-1)) #clamp function 

def produce_line_weight(self, img, lines):
    res = torch.ones(img.shape[1], img.shape[2])
    weight = float(cfg.DATASET.weight)
    #print(res.shape)
    for segment in lines: #iterate over lines
        padding = cfg.DATASET.padding
        for pad in range(-padding, padding):
            #get the two points of the line
            x0_pad, y0_pad = clamp(segment[0]+pad, 0, img.shape[2]), clamp(segment[1]+pad, 0, img.shape[1])
            x1_pad, y1_pad = clamp(segment[2]+pad, 0, img.shape[2]), clamp(segment[3]+pad, 0, img.shape[1])
            rr,cc = draw.line(int(y0_pad), int(x0_pad), int(y1_pad), int(x1_pad))
            #set the line to 10
            res[rr,cc] = weight
    return res

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
        segm_path = os.path.join(self.root_dataset, this_record['fpath_segm'])
        lines_path = os.path.join(self.root_dataset, this_record['fpath_line'])
        img = Image.open(image_path).convert('RGB')
        segm = Image.fromarray(np.load(segm_path)['normals_diff'])

                
        file = open(lines_path, 'rb')
        lines = pickle.load(file)
        file.close()

        assert(segm.mode == "F")
        assert(img.size[0] == segm.size[0])
        assert(img.size[1] == segm.size[1])

        flipped = self.random_flip and np.random.choice([0, 1])
        if flipped:
            img = img.transpose(Image.FLIP_LEFT_RIGHT)
            segm = segm.transpose(Image.FLIP_LEFT_RIGHT)

        segm = np.array(segm)
        segm[segm > 1] = 1
        segm[segm < -1] = -1
        segm[np.isnan(segm)] = 1

        paths = this_record['fpath_img'].split('/')
        paths[2] = "pred_normals_diff"
        camnum = paths[3].split('_')[2]
        paths[3] = f"cam_{camnum}"

        output = dict()
        output['image'] = np.array(img).transpose((2, 0, 1)).astype('uint8')
        output['mask'] = np.expand_dims(segm, axis=0)# batch_segms.contiguous()  
        output['line_weights'] = produce_line_weight(self, output['mask'], lines['segments'])     
        output['savedir'] = os.path.join(self.root_dataset, *paths)
        output['flipped'] = flipped
        return output

    def __len__(self):
        return self.num_sample
