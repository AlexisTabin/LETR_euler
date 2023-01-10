import pickle
import cv2
import pytlsd
import numpy as np
import json
from config import cfg
import os
from dataset import TrainDataset

def save_and_process(lines):
# change the format from x,y,x,y to x,y,dx, dy
# order: top point > bottom point
#        if same y coordinate, right point > left point
    
    new_lines_pairs = []
    for line in lines: # [ #lines, 2, 2 ]
        x1 = line[0]    # x1
        y1 = line[1]    # y1
        x2 = line[2]    # x2
        y2 = line[3]    # y2
        if x1 < x2:
            new_lines_pairs.append( [x1, y1, x2-x1, y2-y1] ) 
        elif  x1 > x2:
            new_lines_pairs.append( [x2, y2, x1-x2, y1-y2] )
        else:
            if y1 < y2:
                new_lines_pairs.append( [x1, y1, x2-x1, y2-y1] )
            else:
                new_lines_pairs.append( [x2,y1, x1-x2, y1-y2] )
    return new_lines_pairs

cfg.merge_from_file("LETR_euler\datalist\config\params_hypersim.yaml")

train_dataset = TrainDataset(
cfg.DATASET.root_dataset,
cfg.DATASET.list_train,
cfg.DATASET)

val_dataset = TrainDataset(
    cfg.DATASET.root_dataset,
    cfg.DATASET.list_val,
    cfg.DATASET)

test_dataset = TrainDataset(
    cfg.DATASET.root_dataset,
    cfg.DATASET.list_test,
    cfg.DATASET)

image_id = 0
anno_id = 0

dataset = train_dataset
def handle(data, im, image_id, anno_id):
    anno["images"].append({"file_name": data['filename'], "height": im.shape[0], "width": im.shape[1], "id": image_id})
    #lines = np.array(data["lines"]).reshape(-1, 2, 2)
    #os.makedirs(os.path.join(tar_dir, batch), exist_ok=True)

    image_path = os.path.join(data["filename"])
    line_set = save_and_process(data['lines']["segments"])
    label_set = data['lines']["structure"]
    for line, label in zip(line_set, label_set):
        info = {}
        info["id"] = anno_id
        anno_id += 1
        info["image_id"] = image_id
        info["category_id"] = 0 if label == True else 2 # 0 for structural, 2 for textural
        info["line"] = np.array(line, dtype=np.float16).tolist()
        info["area"] = 1
        anno["annotations"].append(info)

    image_id += 1
    #print("Finishing", image_path)
    return anno_id

anno = {}
anno['images'] = []
anno["annotations"] = []
anno["categories"] = [{"supercategory":"structural", "id": "0", "name": "structural"}, {"supercategory":"textural", "id": "2", "name": "textural"}]

for i in range(4):
    entry = dataset[i]
    anno_id = handle(entry, entry['image'], image_id, anno_id)
    image_id += 1
''' 
os.makedirs(os.path.join("annotations"), exist_ok=True)
anno_path = os.path.join("annotations", f"lines_.json")
with open(anno_path, 'w') as outfile:
    json.dump(anno, outfile)
'''
# Writing to sample.json
with open("LETR_euler/data_normals/annotations/lines_train2017_test.json", "w") as outfile:
    # Serializing json
    json_object = json.dump(anno, outfile)

image_id = 0
anno_id = 0

dataset = val_dataset
def handle(data, im, image_id, anno_id):
    print(data['filename'])
    anno["images"].append({"file_name": data['filename'], "height": im.shape[0], "width": im.shape[1], "id": image_id})
    #lines = np.array(data["lines"]).reshape(-1, 2, 2)
    #os.makedirs(os.path.join(tar_dir, batch), exist_ok=True)

    image_path = os.path.join(data["filename"])
    line_set = save_and_process(data['lines']["segments"])
    label_set = data['lines']["structure"]
    for line, label in zip(line_set, label_set):
        info = {}
        info["id"] = anno_id
        anno_id += 1
        info["image_id"] = image_id
        info["category_id"] = 0 if label == True else 2 # 0 for structural, 2 for textural
        info["line"] = np.array(line, dtype=np.float16).tolist()
        info["area"] = 1
        anno["annotations"].append(info)

    image_id += 1
    #print("Finishing", image_path)
    return anno_id

anno = {}
anno['images'] = []
anno["annotations"] = []
anno["categories"] = [{"supercategory":"structural", "id": "0", "name": "structural"}, {"supercategory":"textural", "id": "2", "name": "textural"}]

for i in range(4):
    entry = dataset[i]
    anno_id = handle(entry, entry['image'], image_id, anno_id)
    image_id += 1
''' 
os.makedirs(os.path.join("annotations"), exist_ok=True)
anno_path = os.path.join("annotations", f"lines_.json")
with open(anno_path, 'w') as outfile:
    json.dump(anno, outfile)
'''
# Writing to sample.json
with open("LETR_euler/data_normals/annotations/lines_val2017_test.json", "w") as outfile:
    # Serializing json
    json_object = json.dump(anno, outfile)


