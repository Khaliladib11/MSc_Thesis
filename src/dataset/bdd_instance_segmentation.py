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

from .bdd_utils import to_mask, bbox_from_instance_mask, get_coloured_mask

import torch
import torchvision.transforms as transforms

from .bdd import BDD

# Define color map to be used when displaying the images with bounding boxes
COLOR_MAP = ['blue', 'orange', 'green', 'red', 'purple', 'brown', 'pink', 'gray', 'olive', 'cyan', 'blue']

class BDDInstanceSegmentation(BDD):
    """
    BDD class for Instance Segmentation task
    """

    def __init__(self,
                 cfg,
                 stage,
                 obj_cls=['__bgr__', 'person', 'car', 'rider', 'bicycle', 'motorcycle', 'truck', 'bus'],
                 relative_path='..',
                 image_size=400,
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
        super(BDDInstanceSegmentation, self).__init__(cfg=cfg,
                                                      stage=stage,
                                                      obj_cls=obj_cls,
                                                      relative_path=relative_path,
                                                      image_size=image_size,
                                                      transform=transform)

        # check if the classes are in the DETECTION_CLASSES
        assert all(cls in cfg.DATASET.INSTANCE_CLASSES for cls in
                   obj_cls), f"Please choose classes from the following: {cfg.DATASET.INSTANCE_CLASSES}"

        if self.stage == 'train':
            self.images_root = self.root / Path(cfg.DATASET.IMAGE_10K_ROOT + '/train')  # images root
            self.instance_segmentation_root = self.root / Path(
                cfg.DATASET.INSTANCE_SEGMENTATION_ROOT + '/train')  # ins seg masks root
            self.polygon_root = self.root / Path(
                cfg.DATASET.INSTANCE_SEGMENTATION_POLYGON_ROOT + '/ins_seg_train.json')  # polygon root
        elif self.stage == 'test':
            self.images_root = self.root / Path(cfg.DATASET.IMAGE_10K_ROOT + '/val')  # images root
            self.instance_segmentation_root = self.root / Path(
                cfg.DATASET.INSTANCE_SEGMENTATION_ROOT + '/val')  # ins seg masks root
            self.polygon_root = self.root / Path(
                cfg.DATASET.INSTANCE_SEGMENTATION_POLYGON_ROOT + '/ins_seg_val.json')  # polygon root

        _db = self.__create_db()
        self.db = self.split_data(_db)

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
            A.HorizontalFlip(p=1),  # Random Flip with 0.5 probability
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
        polygon_annotation = deque(self.__load_annotations())
        db = deque()
        for polygon in tqdm(polygon_annotation):
            filtered_labels = self.__filter_labels(polygon['labels'])

            if len(filtered_labels):
                db.append({
                    'image_path': self.images_root / Path(polygon['name']),
                    'mask_path': self.instance_segmentation_root / Path(polygon['name'].replace('.jpg', '.png')),
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

        return polygon_annotation

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
        for mask in masks:
            rgb_mask = get_coloured_mask(mask)
            image = cv2.addWeighted(image, 1, rgb_mask, 0.5, 0)

        fig, ax = plt.subplots(figsize=(20,20))
        ax.imshow(image)
        for i, mask in enumerate(labels):
            bbox = boxes[i]
            rect = patches.Rectangle((bbox[0], bbox[1]), bbox[2] - bbox[0], bbox[3] - bbox[1],
                                     edgecolor=COLOR_MAP[labels[i]],
                                     facecolor="none", linewidth=2)
            plt.text(bbox[0], bbox[1], self.idx_to_cls[labels[i]], verticalalignment="top",
                     color=COLOR_MAP[labels[i]])

            ax.add_patch(rect)

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

        image, masks, bboxes, labels = self.data_augmentation(np.array(image), target['masks'], target['boxes'], target['labels'])

        target['boxes'] = torch.tensor(bboxes)

        target['labels'] = torch.tensor(labels, dtype=torch.int64)

        target['masks'] = torch.tensor(np.array(masks, dtype=np.uint8))

        image = self.image_transform(image)

        return image, target
