# https://github.com/rbgirshick/yacs
from yacs.config import CfgNode as CN

_C = CN()

# Dataset params
_C.DATASET = CN(new_allowed=True)
_C.DATASET.DATASET_NAME = 'Bdkk100K'
_C.DATASET.ROOT = 'dataset/bdd100k'
_C.DATASET.IMAGE_ROOT = 'images/bdd100k/images/100k'
_C.DATASET.LABEL_ROOT = 'labels/bdd100k_det_20_labels_trainval/bdd100k/labels/det_20'
_C.DATASET.SEMANTIC_SEGMENTATION_ROOT = ''
_C.DATASET.INSTANCE_SEGMENTATION_ROOT = ''
_C.DATASET.PANOPTIC_SEGMENTATION = ''
_C.DATASET.DRIVABLE_AREA_MASK = 'labels/bdd100k_drivable_labels_trainval/bdd100k/labels/drivable/masks'
_C.DATASET.LANE_ROOT = ''
_C.DATASET.TRAIN = 'train'
_C.DATASET.TEST = 'val'
_C.DATASET.IMAGE_FORMAT = 'jpg'

_C.DATASET.TASKS = ['detection', 'drivable area segmentation', 'Lane Segmentation', 'semantic segmentation',
                    'instance segmentation', 'panoptic segmentation']

_C.DATASET.DETECTION_CLASSES = ['pedestrian', 'rider', 'car', 'truck', 'bus', 'train', 'motorcycle', 'bicycle',
                                'traffic light', 'traffic sign']
_C.DATASET.SEGMENTATION_CLASSES = ['road', 'sidewalk', 'building', 'wall', 'fence', 'pole', 'traffic light', 'traffic '
                                                                                                             'sign',
                                   'vegetation', 'terrain', 'sky', 'person', 'rider', 'car', 'truck', 'bus', 'train',
                                   'motorcycle', 'bicycle']

_C.DATASET.PANOPTIC_CLASSES = ['unlabeled', 'dynamic', 'ego vehicle', 'ground', 'static', 'parking', 'rail track',
                               'road', 'sidewalk', 'bridge', 'building', 'fence', 'garage', 'guard rail', 'tunnel',
                               'wall', 'banner', 'billboard', 'lane divider', 'parking sign', 'pole', 'polegroup',
                               'street light', 'traffic cone', 'traffic device', 'traffic light', 'traffic sign',
                               'traffic sign frame', 'terrain', 'vegetation', 'sky', 'person', 'rider', 'bicycle',
                               'bus', 'car', 'caravan', 'motorcycle', 'trailer', 'train', 'truck']


# Detection params
_C.DETECTION = CN(new_allowed=True)
_C.DETECTION.MODELS = ['Faster_RCNN']
_C.DETECTION.BACKBONE = ['resnet50', 'resnet101', 'resnet152']


# Drivable Area Segmentation params
_C.DRIVABLE_AREA = CN(new_allowed=True)


cfg = _C
