import os
import random
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import json
from tqdm import tqdm
from collections import deque

import torch
from torch.utils import data
import torchvision.transforms as transforms

from .bdd import BDD

COLOR_MAP = ['blue', 'red']

class BDDDrivableSegmentation(BDD):

    def __init__(self,
                 cfg,
                 stage,
                 obj_cls=['direct', 'alternative', 'background'],
                 relative_path='..',
                 image_size=400,
                 transform=None):
        super(BDDDrivableSegmentation, self).__init__(cfg=cfg,
                                                      stage=stage,
                                                      obj_cls=obj_cls,
                                                      relative_path=relative_path,
                                                      image_size=image_size,
                                                      transform=transform)

        _db = self.__create_db()
        self.db = self.split_data(_db)

    def __create_db(self):
        masks_path = self.drivable_root / Path('val' if self.stage == 'test' else 'train')
        return list(masks_path.glob('**/*.png'))

    def _get_mask(self, idx):
        mask = np.array(Image.open(str(self.db[idx])))
        #mask = torch.tensor(mask, dtype=torch.uint8)
        return mask

    def get_image(self, idx, apply_transform=False):
        image_name = str(self.db[idx]).split('\\')[-1].replace('.png', '.jpg')
        image_path = str(self.images_root / Path('val' if self.stage == 'test' else 'train') / image_name)
        image = Image.open(image_path)
        if apply_transform:
            image = self.image_transform(image)

        return image

    def display_image(self, idx, mask=None, alpha=0.5):
        image = np.array(self.get_image(idx, False))
        plt.imshow(image)
        if mask is not None:
            plt.imshow(mask, alpha=alpha)
            plt.title('Image with Mask')

        plt.axis('off')
        plt.show()



    def __getitem__(self, idx):
        X = self.get_image(idx, apply_transform=True)
        y = self._get_mask(idx)

        return X, y
