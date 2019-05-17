import os
import json
import numpy as np

from .detection import DETECTION
from ..paths import get_file_path


class DAC(DETECTION):
    def __init__(self, db_config, split=None, sys_config=None):
        assert split is None or sys_config is not None
        super(DAC, self).__init__(db_config)

        self._mean    = np.array([0.40789654, 0.44719302, 0.47026115], dtype=np.float32)
        self._std     = np.array([0.28863828, 0.27408164, 0.27809835], dtype=np.float32)
        self._eig_val = np.array([0.2141788, 0.01817699, 0.00341571], dtype=np.float32)
        self._eig_vec = np.array([
            [-0.58752847, -0.69563484, 0.41340352],
            [-0.5832747, 0.00994535, -0.81221408],
            [-0.56089297, 0.71832671, 0.41158938]
        ], dtype=np.float32)


        self._dac_cls_ids = list(np.array(list(range(95))) + 1)

        self._dac_cls_names = [
            'boat1', 'boat2', 'boat3', 'boat4', 'boat5', 'boat6',
            'boat7', 'boat8', 'building1', 'building2', 'building3',
            'car1', 'car2', 'car3', 'car4', 'car5', 'car6',
            'car8', 'car9', 'car10', 'car11', 'car12', 'car13', 'car14',
            'car15', 'car16', 'car17', 'car18', 'car19', 'car20', 'car21',
            'car22', 'car23', 'car24', 'drone1', 'drone2', 'drone3', 'drone4',
            'group2', 'group3', 'horseride1', 'paraglider1', 'person1', 'person2',
            'person3', 'person4', 'person5', 'person6', 'person7', 'person8',
            'person9', 'person10', 'person11', 'person12', 'person13', 'person14',
            'person15', 'person16', 'person17', 'person18', 'person19', 'person20',
            'person21', 'person22', 'person23', 'person24', 'person25', 'person26',
            'person27', 'person28', 'person29', 'riding1', 'riding2', 'riding3',
            'riding4', 'riding5', 'riding6', 'riding7', 'riding8', 'riding9', 'riding10',
            'riding11', 'riding12', 'riding13', 'riding14', 'riding15', 'riding16',
            'riding17', 'truck1', 'truck2', 'wakeboard1', 'wakeboard2', 'wakeboard3',
            'wakeboard4', 'whale1'
        ] # 95 class

        self._cls2dac  = {ind + 1: dac_id for ind, dac_id in enumerate(self._dac_cls_ids)}
        self._dac2cls  = {dac_id: cls_id for cls_id, dac_id in self._cls2dac.items()}

        self._dac2name = {cls_id: cls_name for cls_id, cls_name in zip(self._dac_cls_ids, self._dac_cls_names)}
        self._name2dac = {cls_name: cls_id for cls_name, cls_id in self._dac2name.items()}


        if split is not None:
            dac_dir = os.path.join(sys_config.data_dir, 'dac')

            self._split = {
                #'trainval': 'trainval2018',
                'trainval': 'train_80',
                'minival': 'val_new_20',
                'testdev': 'testdev2018'
            }[split]

            #self._data_dir = os.path.join(dac_dir, 'images', self._split)
            self._data_dir = os.path.join(dac_dir, 'images')
            #self._anno_file = os.path.join(dac_dir, 'annotations', 'instances_{}.json'.format(self._split))
            self._anno_file = os.path.join(dac_dir, 'annotations', '{}.json'.format(self._split))


            self._detections, self._eval_ids = self._load_dac_annos()
            self._image_ids = list(self._detections.keys())
            self._db_inds = np.arange(len(self._image_ids))

    def _load_dac_annos(self):
        from pycocotools.coco import COCO

        dac = COCO(self._anno_file)
        self._dac = dac

        class_ids = dac.getCatIds()
        image_ids = dac.getImgIds()

        eval_ids = {}
        detections = {}
        for image_id in image_ids:
            image = dac.loadImgs(image_id)[0]
            dets = []

            eval_ids[image['file_name']] = image_id
            for class_id in class_ids:
                annotation_ids = dac.getAnnIds(imgIds=image['id'], catIds=class_id)
                annotations = dac.loadAnns(annotation_ids)
                category = self._dac2cls[class_id]
                for annotation in annotations:
                    det = annotation['bbox'] + [category]
                    det[2] += det[0]
                    det[3] += det[1]
                    dets.append(det)
            
            file_name = image['file_name']
            if len(dets) == 0:
                detections[file_name] = np.zeros((0, 5), dtype=np.float32)
            else:
                detections[file_name] = np.array(dets, dtype=np.float32)
        return detections, eval_ids

    def image_path(self, ind):
        if self._data_dir is None:
            raise ValueError('Data directory is not set')
        
        db_ind = self._db_inds[ind]
        file_name = self._image_ids[db_ind]
        return os.path.join(self._data_dir, file_name)
    
    def detections(self, ind):
        db_ind = self._db_inds[ind]
        file_name = self._image_ids[db_ind]
        return self._detections[file_name].copy()

    def cls2name(self, cls):
        dac = self._cls2dac[cls]
        return self._dac2name[dac]
    
    def _to_float(self, x):
        return float('{:.2f}'.format(x))

    def convert_to_coco(self, all_bboxes):
        detections = []
        for image_id in all_bboxes:
            coco_id = self._eval_ids[image_id]
            for cls_ind in all_bboxes[image_id]:
                category_id = self._cls2dac[cls_ind]
                for bbox in all_bboxes[image_id][cls_ind]:
                    bbox[2] -= bbox[0]
                    bbox[3] -= bbox[1]
                    
                    score = bbox[4]
                    bbox = list(map(self._to_float, bbox[0:4]))

                    detection = {
                        'image_id': int(coco_id),
                        'category_id': int(category_id),
                        'bbox': bbox,
                        'score': float('{:.2f}'.format(score))
                    }

                    detections.append(detection)
        return detections



    def evaluate(self, result_json, cls_ids, image_ids):
        from pycocotools.cocoeval import COCOeval

        if self._split == 'testdev':
            return None
        
        dac = self._dac

        eval_ids = [self._eval_ids[image_id] for image_id in image_ids]
        cat_ids = [self._cls2dac[cls_id] for cls_id in cls_ids]

        dac_dets = dac.loadRes(result_json)
        dac_eval = COCOeval(dac, dac_dets, 'bbox')
        dac_eval.params.imgIds = eval_ids
        dac_eval.params.catIds = cat_ids
        dac_eval.params.catIds = cat_ids
        dac_eval.evaluate()
        dac_eval.accumulate()
        dac_eval.summarize()
        return dac_eval.stats[0], dac_eval.stats[12:]

    


        

        

        


