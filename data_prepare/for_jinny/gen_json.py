import os
import os.path as osp
import glob
import numpy as np
import cv2
import json

root_path = '/home/yangbai/project/dataset'
anno_name = 'train_dataset.json'
#anno_name = 'val_dataset.json'


def read_class(class_path):
    with open (class_path, 'r') as f:
        classes = f.read().strip().split()
    return classes

def read_labels(labels_path):
    with open(labels_path, 'r') as f:
        annos = f.readlines()
        
    return annos


def fun(class_name):
    res = ''
    for i in range(len(class_name)):
        if class_name[i] >= '0' and class_name[i] <= '9':
            continue
        else:
            res += class_name[i]
    return res


def main():
    class_path = osp.join(root_path, 'class.txt')
    #labels_path = osp.join(root_path, 'labels.txt')
    folder = osp.join(root_path, 'annotations')
    labels_path = osp.join(root_path, 'train_dataset.txt')
    #labels_path = osp.join(root_path, 'val_dataset.txt')
    json_name = osp.join(folder, 'train_80.json')
    #json_name = osp.join(folder, 'val_20.json')
    classes = read_class(class_path)
    annos = read_labels(labels_path)
    
    dataset = {}
    dataset['images'] = []
    dataset['annotations'] = []
    dataset['categories'] = [] 

    super_list = []
    for i, cls in enumerate(classes, 1):
        supercat = fun(cls)
        super_list.append(supercat) 
        dataset['categories'].append({'id': int(i), 
                                      'name': cls,
                                      'supercategory': supercat})

    
    for k, index in enumerate(annos):
        parts = index.strip().split()
        print (parts)
        flag = 0
        if parts[0].endswith('.jpg'):
            img_path = osp.join(root_path, 'images', parts[0])
        else:
            flag = 1
            img_name = parts[0] + ' ' + parts[1]
            img_path = osp.join(root_path, 'images', img_name)

        print (img_path)
        img = cv2.imread(img_path)
        height, width, _ = img.shape

        if flag:
            img_name = parts[0] + ' ' + parts[1]
        else:
            img_name = parts[0]
        dataset['images'].append({'file_name': img_name,
                                  'id': k,
                                  'width': width,
                                  'height': height})
        
        if len(parts) == 6:
            cls_id = parts[1]
            x1 = float(parts[2])
            y1 = float(parts[3])
            x2 = float(parts[4])
            y2 = float(parts[5])
        elif len(parts) == 7:
            cls_id = parts[2]
            x1 = float(parts[3])
            y1 = float(parts[4])
            x2 = float(parts[5])
            y2 = float(parts[6])

#        cls_id = parts[1]   
#        x1 = float(parts[2])
#        y1 = float(parts[3])
#        x2 = float(parts[4])
#        y2 = float(parts[5])
        width = max(0, x2 - x1)
        height = max(0, y2 - y1)
        dataset['annotations'].append({
                'area': width * height,
                'bbox': [x1, y1, width, height],
                'category_id': int(cls_id),
                'id': k,
                'image_id': k,
                'is_crowd': 0,
                'segmentation': [[x1, y1, x2, y1, x2, y2, x1, y2]]
            })
    
    if not osp.exists(folder):
        os.makedirs(folder)

    with open(json_name, 'w') as fin:
        json.dump(dataset, fin)

if __name__ == '__main__':
    main()
