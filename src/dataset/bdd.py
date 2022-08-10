# Import Libraries
import os
import random
from pathlib import Path
from PIL import Image
import json
from collections import deque

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
        :param stage: to select the stage (train, val, test)
        :param obj_cls: list contains the objects we want to detect
        :param db_path: db path for pre created db
        :param relative_path: relative dataset path
        :param image_size: image size when resizing the image
        :param transform: torchvision. Transforms as input
        """

        # Check the stage
        assert stage in ['train', 'val', 'test'], "stage must be : 'train', 'val', 'test'"

        # Load the root of all tasks
        self.root = Path(relative_path) / Path(cfg.DATASET.ROOT)  # Parent root
        self.images_root = self.root / Path(cfg.DATASET.IMAGE_ROOT)  # images root
        self.labels_root = self.root / Path(cfg.DATASET.LABEL_ROOT)  # detection root
        self.drivable_root = self.root / Path(cfg.DATASET.DRIVABLE_AREA_MASK)  # drivable area masks root
        self.semantic_segmentation_root = self.root / Path(cfg.DATASET.SEMANTIC_SEGMENTATION_ROOT)  # sem seg masks root
        self.instance_segmentation_root = self.root / Path(cfg.DATASET.INSTANCE_SEGMENTATION_ROOT)  # ins seg masks root
        self.panoptic_root = self.root / Path(cfg.DATASET.PANOPTIC_SEGMENTATION)  # panoptic seg masks root
        self.lane_root = self.root / Path(cfg.DATASET.LANE_ROOT)  # lane masks root

        self.stage = stage
        self.obj_cls = obj_cls
        self.image_size = image_size
        self.transform = transform

        # load images
        self.images = list(self.images_root.glob('**/*.jpg'))

        self.db = deque()  # deque object to hold the info

        # class to index and index to class mapping
        self.cls_to_idx, self.idx_to_cls = self.create_idx()

        random.seed(seed)  # Fixed seeds

    # method to create two different dictionaries for mapping
    def create_idx(self):
        cls_to_idx = {}
        idx_to_cls = {}
        idx = 0

        for obj in self.obj_cls:

            # if obj is a traffic light, add the class with the color except the NA
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

    # method to split the data into train and val based on percentage
    def split_data(self, db, train_size=80):
        db = list(db)
        to_idx = (train_size * len(db)) // 100
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
        """
        static method to convert x,y,x,y format to x,y,w,h format
        :param x1: x1 position
        :param y1: y1 position
        :param x2: x2 position
        :param y2: y2 position
        :return: x, y, w, h
        """
        x = x1
        y = y1
        w = x2 - x1
        h = y2 - y2
        return (x, y, w, h)

    @staticmethod
    def xywh_to_xyxy(x, y, w, h):
        """
        static method to convert x,y,w,h format to x,y,x,y format
        :param x: x position
        :param y: y position
        :param w: width
        :param h: height
        :return: x1, y1, x2, y2
        """
        x1 = x
        y1 = y
        x2 = w - x1
        y2 = h - y1
        return (x1, y1, x2, y2)

    def image_transform(self, img):
        """
        image transform if the given one is None
        :param img: PIL image
        :return: image tensor with applied transform on it
        """
        if self.transform is None:
            t_ = transforms.Compose([
                transforms.Resize((self.image_size, self.image_size)),  # resize the image
                transforms.ToTensor(),  # convert the image to tensor
                transforms.Normalize(mean=[0.407, 0.457, 0.485],
                                     std=[0.229, 0.224, 0.225])  # normalize the image using mean ans std
            ])
            return t_(img)
        else:
            return self.transform(img)

    def get_image(self, idx, apply_transform=False):
        """
        method the get image from the list of images
        :param idx: index of the image in the list of images
        :param apply_transform: Boolean value to check if we want to apply transform or no
        :return: PIL image or Tensor
        """
        image = Image.open(self.db[idx]['image_path'])
        if apply_transform:
            image = self.image_transform(image)

        return image

    def export_db(self, path):
        """
        method to export the database
        :param path: path to where we want to export the database
        :return: None
        """
        print(f"Exporting {self.stage}_db DB...")
        with open(os.path.join(path, f'{self.stage}_db.json'), "w") as outfile:
            json.dump(list(self.db), outfile)
        print(f"DB {self.stage}_db Exported.")

    # method to create the database, task specific, to be implemented in the children classes
    def __create_db(self):
        raise NotImplementedError

    # method to display the image, task specific, to be implemented in the children classes
    def display_image(self, idx):
        raise NotImplementedError

    def __len__(self):
        """
        method to return the length of the database
        :return: length of the db
        """
        return len(self.db)

    # method to get img and target from the db, task specific, to be implemented in the children classes
    def __getitem__(self, idx):
        raise NotImplementedError
