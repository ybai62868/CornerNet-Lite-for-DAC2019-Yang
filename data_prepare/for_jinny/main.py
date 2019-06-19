import json
import os
import os.path as osp
import cv2
import glob
import xml.dom.minidom


root_path = '/home/yangbai/project/dataset'

phase = 'train'
#split = 8000 # boundary 




def fun(class_name):
    res = ''
    for i in range(len(class_name)):
        if class_name[i] >= '0' and class_name[i] <= '9':
            continue
        else:
            res += class_name[i]
    return res



def generate(src_path, des_path, cls, idx):
    cnt = 0
    frame_list = [frame.split(osp.sep)[-1].split('.')[0] for frame in glob.glob(src_path + '/*.xml')]
    for frame in frame_list:
        print ('Processing frame %06d of class "%s" ...' % (cnt, cls))

        # get source data
        image_src = osp.join(src_path, frame + '.jpg')
        label_src = osp.join(src_path, frame + '.xml')

        # parse xml file to get information of label
        tr = xml.dom.minidom.parse(label_src)
        width = float(tr.getElementsByTagName('width')[0].firstChild.data)
        height = float(tr.getElementsByTagName('height')[0].firstChild.data)
        xmin = tr.getElementsByTagName('xmin')
        if xmin:
            xmin = float(xmin[0].firstChild.data)
        else:
            continue
        ymin = tr.getElementsByTagName('ymin')
        if ymin:
            ymin = float(ymin[0].firstChild.data)
        else:
            continue
        xmax = tr.getElementsByTagName('xmax')
        if xmax:
            xmax = float(xmax[0].firstChild.data)
        else:
            continue
        ymax = tr.getElementsByTagName('ymax')
        if ymax:
            ymax = float(ymax[0].firstChild.data)
        else:
            continue

        assert xmin < xmax
        assert ymin < ymax
        split = image_src.split('/')
        image_path_2 = image_src.split('/')[-2] + '/' + image_src.split('/')[-1]
        frame_label_file = '%s_%06d'
        print (image_path_2, cls, xmin, ymin, xmax, ymax)
        frame_name = '%s.txt' % (cls)
        frame_name_path = osp.join(des_path, frame_name)

        cls_id = cls2dac[cls]

        with open(frame_name_path, 'a') as f:
            f.write('%s %s %f %f %f %f\n' % (image_path_2, cls_id, xmin, ymin, xmax, ymax))

        cnt += 1


# image_path, label, x_min, y_min, x_max, y_max 

# dataset has 5 keys, info, licenses, images, annotations, categories.
dataset = {}
dataset['images'] = []
dataset['categories'] = []
dataset['annotations'] = []

with open(osp.join(root_path, 'class.txt')) as f:
    print (' Successful reading the class.txt ...')
    classes = f.read().strip().split()




# categories:
super_list = []
for i, cls in enumerate(classes, 1):
    supercat = fun(cls)
    super_list.append(supercat)
    dataset['categories'].append({'id': int(i), 'name': cls, 'supercategory': supercat})
print (dataset['categories'])
print (len(set(super_list)))


cls2dac = {}
cls2dac = {dac_id: ind + 1 for ind, dac_id in enumerate(classes)}
print (cls2dac)

cls_lst = classes
data_path = osp.join(root_path, 'data_training')

for idx, cls in enumerate(cls_lst, 1):
    src_path = osp.join(data_path, cls)
    des_path = osp.join(root_path, 'label_all')
    generate(src_path, des_path, cls, idx)



# images:
indexes = [f for f in os.listdir(osp.join(root_path, 'images'))]
if phase == 'train':
    indexes = [line for i,line in enumerate(_indexes) if i <= split]
elif phase == 'val':
    indexes = [line for i, line in enumerate(_indexes) if i > split]



# annotations:
with open(osp.join(root_path, 'annos.txt')) as tr:
    annos = tr.readlines()

for k, index in enumerate(indexes):
    img = cv2.imread(osp.join(root_path, 'images/') + index)
    height, width, _ = img.shape

    dataset['images'].append({'file_name': index, 
                             'id': k,
                             'width': width,
                             'height': height})
    
    for ii, anno in enumerate(annos):
        parts = anno.strip().split()
        if parts[0] == index:
            cls_id = parts[1]
            x1 = float(parts[2])
            y1 = float(parts[3])
            x2 = float(parts[4])
            y2 = float(parts[5])

            width = max(0, x2 - x1)
            height = max(0, y2 - y1)
            dataset['annotations'].append({
                'area': width * height,
                'bbox': [x1, y1, width, height],
                'category_id': int(cls_id),
                'id': i,
                'image_id': k,
                'is_crowd': 0,
                'segmentation': [[x1, y1, x2, y1, x2, y2, x1, y2]]
            })



folder = os.path.join(root_path, 'annotations')
if not osp.exists(folder):
    os.makedirs(folder)

json_name = osp.join(root_path, 'annotations/{}.json'.format(phase))
with open(json_name, 'w') as f:
    json.dump(dataset, f)
