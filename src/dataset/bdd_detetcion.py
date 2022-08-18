# Import Libraries
import random
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image
import json
from tqdm import tqdm
from collections import deque

import torch

from .bdd import BDD

# Define color map to be used when displaying the images with bounding boxes
COLOR_MAP = ['blue', 'orange', 'green', 'red', 'purple', 'brown', 'pink', 'gray', 'olive', 'cyan', 'blue']


class BDDDetection(BDD):
    """
    BDDDetection class, specific class for the detection task on BDD100K dataset
    """

    def __init__(self,
                 cfg,
                 stage,
                 obj_cls=['pedestrian', 'rider', 'motorcycle', 'car', 'bus', 'motorcycle', 'bicycle',
                          'traffic light'],
                 db_path=None,
                 relative_path='..',
                 image_size=400,
                 transform=None):
        """
        Constructor for BDDDetection class
        :param cfg: yacs configuration that contains all the necessary information about the dataset and labels
        :param stage: to select the stage (train, val, test)
        :param obj_cls: list contains the objects we want to detect
        :param db_path:  db path for pre created db
        :param relative_path: relative dataset path
        :param image_size:  tuple that contains the image size (w, h)
        :param transform: torchvision. Transforms as input
        """
        super(BDDDetection, self).__init__(cfg, stage, obj_cls, db_path, relative_path, image_size, transform)

        # check if the classes are in the DETECTION_CLASSES
        assert all(cls in cfg.DATASET.DETECTION_CLASSES for cls in
                   obj_cls), f"Please choose classes from the following: {cfg.DATASET.DETECTION_CLASSES}"

        # load pre created db
        if db_path:
            with open(db_path, 'r') as f:
                self.db = json.load(f)
        else:
            # or create it and split the data
            _db = self.__create_db()
            self.db = self.split_data(_db)

    def __create_db(self, format='xyxy'):
        """
        private method to create the database for the class
        :param format: the format of bounding boxes (xyxy or xywh)
        :return: deque object that holds the database
        """
        detection_db = deque()
        # labels_path = self.labels_root / Path('det_train.json' if self.stage == 'train' else 'det_val.json')

        # load the labels from the json file
        labels_path = self.labels_root / Path('det_val.json' if self.stage == 'test' else 'det_train.json')
        with open(labels_path, 'r') as labels_file:
            labels = json.load(labels_file)

        random.shuffle(labels)

        # loop through the labels
        for item in tqdm(labels):
            # image_path = str(self.images_root / Path('train' if self.stage == 'train' else 'test') / Path(item['name']))
            image_path = str(self.images_root / Path('val' if self.stage == 'test' else 'train') / Path(item['name']))

            classes = []  # list of classes in one image
            bboxes = []  # list of bboxes in one image

            # if the annotation has 'labels' key in the dic
            if 'labels' in item.keys():
                objects = item['labels']  # hold the objects
                # print(objects)

                # loop through object in objects
                for obj in objects:

                    category = obj['category']  # hold the category of the object

                    # if the category is in the classes we want to predict
                    if category in self.obj_cls:
                        # if the category is traffic light load it with the color
                        if category == 'traffic light':
                            # print(obj['attributes'])
                            color = obj['attributes']['trafficLightColor']
                            if color != 'NA':
                                # print(color)
                                category = "tl_" + color

                                x1 = obj['box2d']['x1']
                                y1 = obj['box2d']['y1']
                                x2 = obj['box2d']['x2']
                                y2 = obj['box2d']['y2']

                                # bbox = [x1, y1, x2 - x1, y2 - y1]  # bbox of form: (x, y, w, h) MSCOCO format
                                if format == 'xyxy':
                                    bbox = [x1, y1, x2, y2]

                                cls = self.cls_to_idx[category]

                                bboxes.append(bbox)
                                classes.append(cls)

                        else:
                            x1 = obj['box2d']['x1']
                            y1 = obj['box2d']['y1']
                            x2 = obj['box2d']['x2']
                            y2 = obj['box2d']['y2']

                            # bbox = [x1, y1, x2 - x1, y2 - y1]  # bbox of form: (x, y, w, h) MSCOCO format
                            if format == 'xyxy':
                                bbox = [x1, y1, x2, y2]

                            cls = self.cls_to_idx[category]

                            bboxes.append(bbox)
                            classes.append(cls)

                # if we have one and more objects in the image append the image path, bboxes and classes to the db
                if len(classes) > 0:
                    detection_db.append({
                        'image_path': image_path,
                        'bboxes': bboxes,
                        'classes': classes
                    })
        # finally, return the db
        return detection_db

    # method to display the image with the bounding boxes
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
                plt.text(bbox[0], bbox[1], self.idx_to_cls[classes[i]], verticalalignment="top", color="white",
                         bbox={'facecolor': COLOR_MAP[classes[i]], 'pad': 0})

        plt.axis('off')
        plt.show()

    # collate function to be used with the dataloader, since the not all the images has the same number of objects
    def collate_fn(self, batch):
        return tuple(zip(*batch))

    def __getitem__(self, idx):
        """
        method to return the item based on the index
        :param idx: index of the image in db
        :return: img and targets
        """
        img = self.get_image(idx, apply_transform=True)
        targets = {
            'labels': torch.tensor(self.db[idx]['classes'], dtype=torch.int64),
            'boxes': torch.tensor(self.db[idx]['bboxes'], dtype=torch.float)
        }

        return img, targets
