from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import albumentations as A
from albumentations.pytorch import ToTensorV2


import torch
from torch.utils import data
import torchvision.transforms as transforms

from .bdd import BDD

COLOR_MAP = ['blue', 'red']


class BDDDrivableSegmentation(BDD):
    """
    BDDDrivableSegmentation class, specific class for the drivable area segmentation task on BDD100K dataset
    """
    def __init__(self,
                 cfg,
                 stage,
                 obj_cls=['direct', 'alternative', 'background'],
                 relative_path='..',
                 image_size=400,
                 transform=None):
        """
        Constructor for BDDDrivableSegmentation class
        :param cfg: yacs configuration that contains all the necessary information about the dataset and labels
        :param stage: to select the stage (train, val, test)
        :param obj_cls: list contains the objects we want to detect
        :param relative_path: relative dataset path
        :param image_size:  tuple that contains the image size (w, h)
        :param transform: torchvision. Transforms as input
        """
        super(BDDDrivableSegmentation, self).__init__(cfg=cfg,
                                                      stage=stage,
                                                      obj_cls=obj_cls,
                                                      relative_path=relative_path,
                                                      image_size=image_size,
                                                      transform=transform)

        self.cls_to_idx, self.idx_to_cls = self.create_idx()

        _db = self.__create_db()
        self.db = self.split_data(_db)

    def data_augmentation(self):
        if self.stage == 'train':
            compose = A.Compose([
                A.Resize(height=self.image_size, width=self.image_size),
                A.Rotate(limit=35, p=0.5),
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.1),
                A.Normalize(
                    mean=[0.0, 0.0, 0.0],
                    std=[1.0, 1.0, 1.0]
                ),
                ToTensorV2()
            ])

        else:
            compose = A.Compose([
                A.Resize(height=self.image_size, width=self.image_size),
                A.Normalize(
                    mean=[0.407, 0.457, 0.485],
                    sstd=[0.229, 0.224, 0.225]
                ),
                ToTensorV2()
            ])

        return compose

    def __create_db(self):
        """
        method to create the db of the class
        :return: list ot Pathlib objects of the masks
        """
        masks_path = self.drivable_root / Path('val' if self.stage == 'test' else 'train')
        return list(masks_path.glob('**/*.png'))

    def _get_mask(self, idx):
        """
        method to get the mask of the image
        :param idx: index of the mask in the db
        :return: np array
        """
        mask = np.array(Image.open(str(self.db[idx])))
        # mask = torch.tensor(mask, dtype=torch.uint8)
        return mask

    def get_image(self, idx, apply_transform=False):
        """
        method to return the image
        :param idx: index of the mask in the db
        :param apply_transform: Boolean value, if we want to apply the transform or not
        :return: PIL image or Tensor type
        """
        image_name = str(self.db[idx]).split('/')[-1].replace('.png', '.jpg')
        image_path = str(self.images_root / Path('val' if self.stage == 'test' else 'train') / image_name)
        image = Image.open(image_path)
        if apply_transform:
            image = self.image_transform(image)

        return image

    def display_image(self, idx, mask=None, alpha=0.5):
        """
        method to display the image with the mask
        :param idx: index of the mask in the db
        :param mask: mask of the image
        :param alpha: degree of transparency
        :return: None
        """
        image = np.array(self.get_image(idx, False))
        plt.imshow(image)
        if mask is not None:
            plt.imshow(mask, alpha=alpha)
            plt.title('Image with Mask')

        plt.axis('off')
        plt.show()

    def __getitem__(self, idx):
        """
        method to return the item based on the index
        :param idx: index of the image in db
        :return: img and mask
        """

        label = dict()
        img = self.get_image(idx, apply_transform=False)
        mask = self._get_mask(idx)
        augmentation = self.transform(image=img, mask=mask)
        image = augmentation['image']
        mask = augmentation['mask']
        # mask = torch.tensor(self._get_mask(idx), dtype=torch.long)
        mask = mask.long()
        return img, mask
