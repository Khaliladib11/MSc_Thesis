# Import Libraries
import os
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import albumentations as A
import cv2
from PIL import Image
import json
from collections import deque
from tqdm import tqdm
from sklearn.model_selection import train_test_split

from .bdd_utils import to_mask, bbox_from_instance_mask, get_coloured_mask

import torch
import torchvision.transforms as transforms

from .bdd import BDD

# Define color map to be used when displaying the images with bounding boxes
COLOR_MAP = ['blue', 'orange', 'green', 'red', 'purple', 'brown', 'pink', 'gray', 'olive', 'cyan', 'blue']


class BDDInstanceSegmentationDrivable(BDD):
    """
    BDD class for Instance Segmentation task
    """

    def __init__(self,
                 cfg,
                 stage,
                 obj_cls=['__bgr__', 'person', 'car', 'rider', 'bicycle', 'motorcycle', 'truck', 'bus'],
                 relative_path='..',
                 image_size=640,
                 transform=None):
        """
        Constructor for BDDInstanceSegmentation class
        :param cfg: yacs configuration that contains all the necessary information about the dataset and labels
        :param stage: to select the stage (train, val, test)
        :param obj_cls: list contains the objects we want to detect
        :param relative_path: relative dataset path
        :param image_size:  tuple that contains the image size (w, h)
        :param transform: torchvision. Transforms as input
        """
        super(BDDInstanceSegmentationDrivable, self).__init__(cfg=cfg,
                                                      stage=stage,
                                                      obj_cls=obj_cls,
                                                      relative_path=relative_path,
                                                      image_size=image_size,
                                                      transform=transform)

        self.cls_to_idx, self.idx_to_cls = self.create_idx()
        self.image_root = os.path.join(self.root, cfg.DATASET.IMAGE_10K_ROOT, 'train')
        self.images = os.listdir(self.image_root)
        self.instance_masks_path = os.path.join(self.root, cfg.DATASET.INSTANCE_SEGMENTATION_ROOT, 'train')
        self.instance_masks = os.listdir(self.instance_masks_path)
        self.polygon_root = os.path.join(self.root, cfg.DATASET.INSTANCE_SEGMENTATION_POLYGON_ROOT,
                                         'ins_seg_train.json')

        self.polygon_drviable_root = os.path.join(self.root, cfg.DATASET.DRIVABLE_AREA_POLYGON_ROOT, 'drivable_train.json')
        self.drivable_masks_path = os.path.join(self.root, cfg.DATASET.DRIVABLE_AREA_MASK, 'train')
        self.drivable_masks = os.listdir(self.drivable_masks_path)

        self.available_images = self.intersection()
        _db = self.__create_db()
        self.db = self.split_data(_db)

    def intersection(self):
        masks = [mask.replace('.png', '.jpg') for mask in self.drivable_masks]
        lst3 = [value for value in self.images if value in masks]
        return lst3

    # method to split the data into train and val based on percentage
    def split_data(self, db, train_size=80):
        db = list(db)

        train_db, test_db = train_test_split(db, test_size=1 - (train_size / 100), random_state=42)

        val_db, test_db = train_test_split(test_db, test_size=0.5, random_state=42)

        if self.stage == 'train':
            return deque(train_db)
        elif self.stage == 'val':
            return deque(val_db)
        elif self.stage == 'test':
            return deque(test_db)

    def data_augmentation(self, image, masks, bboxes, labels):
        """
        method to apply image augmentation technics to reduce overfitting
        :param image: numpy array with shape of HxWx3 (RGB image)
        :param masks: list of masks, each mask must have the same W and H with the image (2D mask)
        :param bboxes: list of bounding boxes, each box must have (xmin, ymin, xmax, ymax)
        :param labels: idx of the labels
        :return: image, masks, bboxes
        """
        class_labels = [self.idx_to_cls[label] for label in labels]
        for idx, box in enumerate(bboxes):
            box.append(class_labels[idx])

        augmentation_transform = A.Compose([
            A.HorizontalFlip(p=0.5),  # Random Flip with 0.5 probability
            A.CropAndPad(px=100, p=0.5),  # crop and add padding with 0.5 probability
            A.PixelDropout(dropout_prob=0.01, p=0.5),  # pixel dropout with 0.5 probability
        ], bbox_params=A.BboxParams(format='pascal_voc', min_visibility=0.3))  # return bbox with xyxy format

        transformed = augmentation_transform(image=image, masks=masks, bboxes=bboxes)

        transformed_boxes = []
        transformed_labels = []
        for box in transformed['bboxes']:
            box = list(box)
            label = box.pop()
            transformed_boxes.append(box)
            transformed_labels.append(label)

        labels = [self.cls_to_idx[label] for label in transformed_labels]

        return transformed['image'], transformed['masks'], transformed_boxes, labels

    def __create_db(self):
        """
        method to create the db of the class
        :return: deque object contains the necessary information
        """
        masks_polygon, drivable_polygon = self.__load_annotations()
        db = deque()
        for polygon_key in tqdm(masks_polygon):
            polygon = masks_polygon[polygon_key]
            if polygon['name'] in self.available_images:
                filtered_labels = self.__filter_labels(polygon['labels'])
                if 'labels' in drivable_polygon[polygon['name']].keys():
                    drivable_labels = drivable_polygon[polygon_key]['labels']
                    for drivable_label in drivable_labels:
                        drivable_label['category'] = 'road'
                        filtered_labels.append(drivable_label)

                if len(filtered_labels):
                    db.append({
                        'image_path': os.path.join(self.image_root, polygon['name']),
                        'mask_path': os.path.join(self.instance_masks_path, polygon['name'].replace('.jpg', '.png')),
                        'drivable_path': os.path.join(self.drivable_masks_path, polygon['name'].replace('.jpg', '.png')),
                        'labels': filtered_labels
                    })
        return db

    def __load_annotations(self):
        """
        method to load the annotation from json
        :return: list of annotations
        """
        with open(self.polygon_root, 'r') as f:
            polygon_annotation = json.load(f)

        with open(self.polygon_drviable_root, 'r') as f:
            polygon_drivable_annotation = json.load(f)

        masks_annotations = dict()
        drivable_annotations = dict()

        for polygon in polygon_annotation:
            masks_annotations[polygon['name']] = polygon

        for drivable_polygon in polygon_drivable_annotation:
            drivable_polygon
            drivable_annotations[drivable_polygon['name']] = drivable_polygon

        return masks_annotations, drivable_annotations

    def __filter_labels(self, labels):
        """
        method to filter the labels according to the objects passed to the constructor
        :param labels: list of dictionaries for the objects in the image
        :return: list of filtered labels
        """
        filtered_labels = []
        for label in labels:
            if label['category'] in self.obj_cls:
                filtered_labels.append(label)

        return filtered_labels

    def image_transform(self, img):
        """
        image transform if the given one is None
        :param img: PIL image
        :return: image tensor with applied transform on it
        """
        if self.transform is None:
            t_ = transforms.Compose([
                transforms.ToTensor(),  # convert the image to tensor
                transforms.Normalize(mean=[0.407, 0.457, 0.485],
                                     std=[0.229, 0.224, 0.225])  # normalize the image using mean ans std
            ])
            return t_(img)
        else:
            return self.transform(img)

    def get_image(self, idx, apply_transform=False):
        """
        method to return the image
        :param idx: index of the mask in the db
        :param apply_transform: Boolean value, if we want to apply the transform or not
        :return: PIL image or Tensor type
        """
        image_path = self.db[idx]['image_path']
        image = Image.open(image_path).convert('RGB')
        if apply_transform:
            image = self.image_transform(image)

        return image

    def get_drivable_mask(self, idx):
        """
        method to get the mask for the drivable area
        :param idx:
        :return:
        """
        drivable_mask_path = self.db[idx]['drivable_path']
        mask = np.array(Image.open(drivable_mask_path))
        # mask = torch.tensor(mask, dtype=torch.uint8)
        return mask

    def get_mask(self, idx):
        """
        method to get the mask of the image
        :param idx: index of the mask in the db
        :return: np array
        """
        mask_path = self.db[idx]['mask_path']
        mask = np.array(Image.open(mask_path))
        # mask = torch.tensor(mask, dtype=torch.uint8)
        return mask

    def _get_labels(self, idx):
        image_annotation = self.db[idx]
        mask_shape = np.array(Image.open(image_annotation['mask_path'])).shape
        target = {}
        boxes = []
        masks = []
        labels = []
        for label in image_annotation['labels']:
            poly2d = label['poly2d'][0]['vertices']
            mask = to_mask(mask_shape, poly2d)
            box = bbox_from_instance_mask(mask)
            label = self.cls_to_idx[label['category']]
            if box is not None:
                if box[0] != box[2] and box[1] != box[3]:
                    masks.append(np.array(mask, dtype=np.uint8))
                    boxes.append(box)
                    labels.append(label)


        target['boxes'] = boxes
        target['labels'] = labels
        target['masks'] = masks

        return target

    # method to display the image, task specific, to be implemented in the children classes
    def display_image(self, image, masks, boxes, labels):
        if isinstance(image, Image.Image):
            image = np.array(image)
        for idx, mask in enumerate(masks):
            rgb_mask, color = get_coloured_mask(mask)
            image = cv2.addWeighted(image, 1, rgb_mask, 0.5, 0)
            bbox = [(int(boxes[idx][0]), int(boxes[idx][1])), (int(boxes[idx][2]), int(boxes[idx][3]))]
            cv2.rectangle(image, bbox[0], bbox[1], color=color, thickness=2)
            cv2.putText(image, self.idx_to_cls[labels[idx]], bbox[0], cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0),
                        thickness=2)

        plt.imshow(image)
        plt.axis('off')
        plt.show()

    # collate function to be used with the dataloader, since the not all the images has the same number of objects
    def collate_fn(self, batch):
        return tuple(zip(*batch))

    def __len__(self):
        return len(self.db)

    def __getitem__(self, idx):
        image = self.get_image(idx, False)
        target = self._get_labels(idx)

        image, masks, bboxes, labels = self.data_augmentation(np.array(image), target['masks'], target['boxes'],
                                                              target['labels'])

        target['boxes'] = torch.tensor(bboxes, dtype=torch.float32)

        target['labels'] = torch.tensor(labels, dtype=torch.int64)

        target['masks'] = torch.tensor(np.array(masks, dtype=np.uint8))

        image_id = torch.tensor([idx])
        area = (target['boxes'][:, 3] - target['boxes'][:, 1]) * (target['boxes'][:, 2] - target['boxes'][:, 0])
        # suppose all instances are not crowd
        iscrowd = torch.zeros((len(self.cls_to_idx),), dtype=torch.int64)

        target['image_id'] = image_id
        target['area'] = area
        target['iscrowd'] = iscrowd

        image = self.image_transform(image)

        return image, target
