import os
import os.path as osp
import numpy as np
import random

root_path = '/home/yangbai/project/dataset'
src_path = '/home/yangbai/project/dataset/labels.txt'

def read_labels(src_path):
    with open(src_path, 'r') as f:
        annos = f.readlines()
    return annos



def divide_dataset(train_set, val_set):
    ratio = 0.8
    annos = read_labels(src_path)
    total_cnt = len(annos)
    print (total_cnt)
    train_cnt = total_cnt * ratio 
    val_cnt = int(total_cnt * (1 - ratio)) 

    print (type(annos[0]))
    val_res = random.sample(annos, val_cnt)
    set_val_res = set(val_res)
    set_annos = set(annos)
    set_train_res = set_annos - set_val_res
    train_res = list(set_train_res)

    print ('The length of train dataset', len(train_res))
    print ('The length of val dataset', len(val_res))
    print ('Total', len(train_res) + len(val_res))
    
    with open(train_set, 'a') as fin:
        for item in train_res:
            fin.write('%s' % item)

    with open(val_set, 'a') as fin:
        for item in val_res:
            fin.write('%s' % item)



def main():
    train_set = osp.join(root_path, 'train_dataset.txt')
    val_set = osp.join(root_path, 'val_dataset.txt')
    divide_dataset(train_set, val_set)



if __name__ == '__main__':
    main()
