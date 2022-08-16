# Import Libraries
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import json

import torch
import torchvision.transforms as transforms

from .bdd import BDD


class BDDInstanceSegmentation(BDD):
    """
    BDD class for Instance Segmentation task
    """

    def __init__(self,
                 cfg,
                 stage,
                 obj_cls=['__bgr__', 'person', 'car', 'rider', 'bicycle', 'motorcycle'],
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

        self.images_root = self.root / Path(cfg.DATASET.IMAGE_10K_ROOT)  # images root

        _db = self.__create_db()
        self.db = self.split_data(_db)

    def __create_db(self):
        """
        method to create the db of the class
        :return: list ot Pathlib objects of the masks
        """
        masks_path = self.instance_segmentation_root / Path('val' if self.stage == 'test' else 'train')
        return list(masks_path.glob('**/*.png'))

    def get_image(self, idx, apply_transform=False):
        """
        method to return the image
        :param idx: index of the mask in the db
        :param apply_transform: Boolean value, if we want to apply the transform or not
        :return: PIL image or Tensor type
        """
        image_name = str(self.db[idx]).split('\\')[-1].replace('.png', '.jpg')
        image_path = str(self.images_root / Path('val' if self.stage == 'test' else 'train') / image_name)
        image = Image.open(image_path)
        if apply_transform:
            image = self.image_transform(image)

        return image

    def _get_mask(self, idx):
        """
        method to get the mask of the image
        :param idx: index of the mask in the db
        :return: np array
        """
        mask = np.array(Image.open(str(self.db[idx])))
        # mask = torch.tensor(mask, dtype=torch.uint8)
        return mask

    # method to display the image, task specific, to be implemented in the children classes
    def display_image(self, idx):
        raise NotImplementedError

    def __len__(self):
        pass

    def __getitem__(self, idx):
        pass
