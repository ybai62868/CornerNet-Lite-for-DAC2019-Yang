import os
import glob
import numpy as np


txt_path = '/home/yangbai/project/dataset/label_all/'
res_path = '/home/yangbai/project/dataset/labels.txt'

def main():
    txt_list = glob.glob(txt_path + '*.txt')
    for txt in txt_list:
        with open(txt, 'r') as f:
            lines = f.read().splitlines()
            for line in lines:
                with open(res_path, 'a') as fin:
                    fin.write(line + '\n')

if __name__ == '__main__':
    main()
