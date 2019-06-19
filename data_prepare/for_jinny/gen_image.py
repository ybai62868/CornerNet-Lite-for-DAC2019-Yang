import os
import os.path as osp
import glob
import shutil

root_path = '/home/yangbai/project/dataset'

def read_class(class_path):
    with open(class_path, 'r') as f:
        classes = f.read().strip().split()
    return classes

def gen_image(src_path, des_path, cls):
    img_list = glob.glob(src_path + '/*.jpg')
    

    for img in img_list:
        img_name = img.split(osp.sep)[-1].split('.')[0]
        img_name += '.jpg'
        img_path = osp.join(des_path, img_name) 
        #print (img_path)
        print ('Processing %s ...' % cls)
        shutil.copyfile(img, img_path)


def main():
    class_path = osp.join(root_path, 'class.txt')
    classes = read_class(class_path)
    for cls in classes:
        src_path = osp.join(root_path, 'data_training', cls)
        des_path = osp.join(root_path, 'images', cls)
        if not osp.exists(des_path):
            os.makedirs(des_path)
        gen_image(src_path, des_path, cls)

if __name__ == '__main__':
    main()
