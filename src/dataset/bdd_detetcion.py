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
from torchvision.utils import draw_bounding_boxes

from .bdd import BDD


COLOR_MAP = ['blue', 'orange', 'green', 'red', 'purple', 'brown', 'pink', 'gray', 'olive', 'cyan']

class BDD_Detection(BDD):

    def __init__(self,
                 cfg,
                 stage,
                 obj_cls=['pedestrian', 'rider', 'motorcycle', 'car', 'bus', 'motorcycle', 'bicycle',
                          'traffic light'],
                 db_path=None,
                 relative_path='..',
                 image_size=(400, 400),
                 transform=None):
        super(BDD_Detection, self).__init__(cfg, stage, obj_cls, db_path, relative_path, image_size, transform)

        if db_path:
            with open(db_path, 'r') as f:
                self.db = json.load(f)
        else:
            self.db = self.__create_db()

    def __create_db(self, format='xyxy'):
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

                    if obj['category'] in self.obj_cls:
                        x1 = obj['box2d']['x1']
                        y1 = obj['box2d']['y1']
                        x2 = obj['box2d']['x2']
                        y2 = obj['box2d']['y2']

                        #bbox = [x1, y1, x2 - x1, y2 - y1]  # bbox of form: (x, y, w, h) MSCOCO format
                        bbox = [x1, y1, x2, y2]
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

    def display_image(self, idx, display_labels=True):
        image = self.get_image(idx, apply_transform=False)

        classes = self.db[idx]['classes']
        bboxes = self.db[idx]['bboxes']

        # plot the image
        fig, ax = plt.subplots()
        ax.imshow(image)
        for i in range(len(classes)):
            # print(classes[i], color_map[classes[i]])
            bbox = bboxes[i]
            # for bbox in bboxes:
            # to load to correspond color map
            rect = patches.Rectangle((bbox[0], bbox[1]), bbox[2] - bbox[0], bbox[3] - bbox[1],
                                     edgecolor=COLOR_MAP[classes[i]],
                                     facecolor="none", linewidth=2)
            ax.add_patch(rect)
            if display_labels:
                ax.text(bbox[0], bbox[1] - 20, self.idx_to_cls[classes[i]], bbox={'facecolor': COLOR_MAP[classes[i]]},
                        fontsize=10)

        plt.axis('off')
        plt.show()

    def __getitem__(self, idx):
        X = self.get_image(idx, apply_transform=True)
        y = {
            'labels': torch.tensor(self.db[idx]['classes']),
            'boxes': torch.tensor(self.db[idx]['bboxes'])
        }

        return X, y
