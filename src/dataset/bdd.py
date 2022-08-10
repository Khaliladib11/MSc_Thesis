import os
import random
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image
import json
from tqdm import tqdm
from collections import deque
from sklearn.model_selection import train_test_split

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
                 obj_cls,
                 db_path=None,
                 relative_path='..',
                 image_size=400,
                 transform=None,
                 seed=356,
                 ):
        """
        Constructor for BDD class
        :param cfg: yacs configuration that contains all the necessary information about the dataset and labels
        :param stage: to select the stage (train, test)
        :param obj_cls: list contains the objects we want to detect
        :param db_path: db path for pre created db
        :param relative_path: relative dataset path
        :param image_size: tuple that contains the image size (w, h)
        :param transform: torchvision. Transforms as input
        """
        assert stage in ['train', 'val', 'test'], "stage must be : 'train' or 'test'"

        self.root = Path(relative_path) / Path(cfg.DATASET.ROOT)
        self.images_root = self.root / Path(cfg.DATASET.IMAGE_ROOT)
        self.labels_root = self.root / Path(cfg.DATASET.LABEL_ROOT)
        self.drivable_root = self.root / Path(cfg.DATASET.DRIVABLE_AREA_MASK)
        self.semantic_segmentation_root = self.root / Path(cfg.DATASET.SEMANTIC_SEGMENTATION_ROOT)
        self.instance_segmentation_root = self.root / Path(cfg.DATASET.INSTANCE_SEGMENTATION_ROOT)
        self.panoptic_root = self.root / Path(cfg.DATASET.PANOPTIC_SEGMENTATION)
        self.lane_root = self.root / Path(cfg.DATASET.LANE_ROOT)

        self.stage = stage
        self.obj_cls = obj_cls
        self.image_size = image_size
        self.transform = transform

        self.images = list(self.images_root.glob('**/*.jpg'))

        self.db = deque()
        self.cls_to_idx, self.idx_to_cls = self.create_idx()

        random.seed(seed)

    def create_idx(self):
        cls_to_idx = {}
        idx_to_cls = {}
        idx = 0

        for obj in self.obj_cls:
            if obj == 'traffic light':
                """
                cls_to_idx['tl_NA'] = idx
                idx_to_cls[idx] = 'tl_NA'
                idx += 1
                """

                cls_to_idx['tl_G'] = idx
                idx_to_cls[idx] = 'tl_G'
                idx += 1

                cls_to_idx['tl_R'] = idx
                idx_to_cls[idx] = 'tl_R'
                idx += 1

                cls_to_idx['tl_Y'] = idx
                idx_to_cls[idx] = 'tl_Y'
                idx += 1

            else:
                cls_to_idx[self.obj_cls[idx]] = idx
                idx_to_cls[idx] = self.obj_cls[idx]
                idx += 1

        return cls_to_idx, idx_to_cls

    def split_data(self, db, train_size=80):
        db = list(db)
        to_idx = (train_size*len(db))//100
        if self.stage == 'train':
            train_db = db[:to_idx]
            return deque(train_db)
        elif self.stage == 'val':
            val_db = db[to_idx:]
            return deque(val_db)

        else:
            return deque(db)


    @staticmethod
    def xyxy_to_xywh(x1, y1, x2, y2):
        x = x1
        y = y1
        w = x2 - x1
        h = y2 - y2
        return (x, y, w, h)

    @staticmethod
    def xywh_to_xyxy(x, y, w, h):
        x1 = x
        y1 = y
        x2 = w - x1
        y2 = h - y1
        return (x1, y1, x2, y2)

    def image_transform(self, img):
        if self.transform is None:
            t_ = transforms.Compose([
                transforms.Resize((self.image_size, self.image_size)),
                transforms.ToTensor(),
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

    def export_db(self, path):
        print(f"Exporting {self.stage}_db DB...")
        with open(os.path.join(path, f'{self.stage}_db.json'), "w") as outfile:
            json.dump(list(self.db), outfile)
        print(f"DB {self.stage}_db Exported.")

    def __create_db(self):
        raise NotImplementedError

    def display_image(self, idx):
        raise NotImplementedError

    def __len__(self):
        return len(self.db)

    def __getitem__(self, idx):
        raise NotImplementedError
