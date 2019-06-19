import os
import os.path as osp
import glob
import numpy as np



src_path = '/home/yangbai/project/dataset'


def read_classes(class_path):
    with open(class_path, 'r') as f:
        classes = f.read().strip().split()
    return classes 

def cal(xml_path, cls):
    xml_list = glob.glob(xml_path + '/*.xml')
    return len(xml_list)


def main():
    cnt = 0
    class_path = osp.join(src_path, 'class.txt')
    classes = read_classes(class_path)
    for cls in classes:
        print ('Processing %s ...' % (cls))
        xml_path = osp.join(src_path, 'data_training', cls)
        temp = cal(xml_path, cls)  
        print ('%s has %d xml files' %(cls, temp))
        cnt += temp
    print (cnt)



if __name__ == '__main__':
    main()
