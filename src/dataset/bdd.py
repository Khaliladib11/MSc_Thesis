import os
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image
import json
from tqdm import tqdm
from collections import deque

import torch
from torch.utils import data
import torchvision.transforms as transforms


class BDD(data.Dataset):
    """
    Dataset class for the BDD100K dataset to be used in this project

    """

    def __init__(self,
                 cfg,
                 stage,
                 task='detection',
                 detection_cls=['pedestrian', 'rider', 'motorcycle', 'car', 'bus', 'motorcycle', 'bicycle',
                                'traffic light'],
                 segmentation_cls=None,
                 panoptic_cls=None,
                 db_path=None,
                 relative_path='..',
                 image_size=(400, 400),
                 transform=None
                 ):
        """
        Constructor for BDD class
        :param cfg: yacs configuration that contains all the necessary information about the dataset and labels
        :param stage: to select the stage (train, test)
        :param detection_cls: list contains the objects we want to detect
        :param task: string contains the task we want to perform
        :param db_path: db path for pre created db
        :param relative_path: relative dataset path
        :param image_size: tuble that contains the image size (w, h)
        :param transform: torchvision.transforms as input
        """
        assert stage in ['train', 'test'], "stage must be : 'train' or 'test'"
        assert task in cfg.DATASET.TASKS, f"You have to choose form the following tasks: {cfg.DATASET.TASKS}."

        assert all(cls in cfg.DATASET.DETECTION_CLASSES for cls in
                   detection_cls), f"Please choose classes from the following: {cfg.DATASET.DETECTION_CLASSES} "

        self.root = Path(relative_path) / Path(cfg.DATASET.ROOT)
        self.images_root = self.root / Path(cfg.DATASET.IMAGE_ROOT)
        self.labels_root = self.root / Path(cfg.DATASET.LABEL_ROOT)
        self.drivable_root = self.root / Path(cfg.DATASET.DRIVABLE_AREA_MUSK)
        self.semantic_segmentation_root = self.root / Path(cfg.DATASET.SEMANTIC_SEGMENTATION_ROOT)
        self.instance_segmentation_root = self.root / Path(cfg.DATASET.INSTANCE_SEGMENTATION_ROOT)
        self.panoptic_root = self.root / Path(cfg.DATASET.PANOPTIC_SEGMENTATION)
        self.lane_root = self.root / Path(cfg.DATASET.LANE_ROOT)

        self.stage = stage
        self.task = task
        self.detection_cls = detection_cls
        self.imgs_size = image_size
        self.transform = transform

        self.images = list(self.images_root.glob('**/*.jpg'))

        if detection_cls:
            self.cls_to_idx, self.idx_to_cls = self.create_idx(self.detection_cls)
        elif segmentation_cls:
            pass
        elif panoptic_cls:
            pass

        if db_path:
            with open(db_path, 'r') as f:
                self.db = json.load(f)
        else:
            self.db = self.__create_db()

    def create_idx(self, cls_list):
        cls_to_idx = {}
        idx_to_cls = {}

        for idx in range(len(cls_list)):
            cls_to_idx[cls_list[idx]] = idx
            idx_to_cls[idx] = cls_list[idx]

        return cls_to_idx, idx_to_cls

    def __create_detection_db(self):
        detection_db = deque()
        labels_path = self.labels_root / Path('det_train.json' if self.stage == 'train' else 'det_val.json')
        with open(labels_path, 'r') as labels_file:
            labels = json.load(labels_file)

        for item in tqdm(labels):
            image_path = str(self.images_root / Path('train' if self.stage == 'train' else 'test') / Path(item['name']))

            classes = []
            bboxes = []
            if 'labels' in item.keys():
                objects = item['labels']

                for obj in objects:

                    if obj['category'] in self.detection_cls:
                        x1 = obj['box2d']['x1']
                        y1 = obj['box2d']['y1']
                        x2 = obj['box2d']['x2']
                        y2 = obj['box2d']['y2']

                        bbox = [x1, y1, x2 - x1, y2 - y1]  # bbox of form: (x, y, w, h) MSCOCO format
                        cls = self.cls_to_idx[obj['category']]

                        bboxes.append(bbox)
                        classes.append(cls)

                if len(classes) > 0:
                    detection_db.append({
                        'image_path': image_path,
                        'bboxes': bboxes,
                        'classes': classes
                    })

        return detection_db

    def __create_db(self):
        db = None
        if self.task == 'detection':
            db = self.__create_detection_db()

        return db

    def image_transform(self, img):
        if self.transform is None:
            t_ = transforms.Compose([
                transforms.resize(self.imgs_size),
                transforms.toTensor(),
                transforms.Normalize(mean=[0.407, 0.457, 0.485],
                                     std=[0.229, 0.224, 0.225])
            ])
            return t_(img)
        else:
            return self.transform(img)

    def get_image(self, idx, apply_transform=False):
        image = Image.open(self.db[idx]['image_path'])
        if apply_transform:
            image = self.image_transform(image)

        return image

    def __len__(self):
        return len(self.db)

    def __getitem__(self, idx):
        X = self.get_image(idx)
        if self.task == 'detection':
            labels = self.db[idx]
            y = {'labels': torch.tensor(labels['classes']), 'boxes': torch.tensor(labels['bboxes'], dtype=torch.float32)}

        return X, y
