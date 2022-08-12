# Import Libraries
import numpy as np
import torch
from torchvision.io import read_image
from torchvision.ops import masks_to_boxes
from torchvision.utils import draw_bounding_boxes


############################################################
#  Bounding Boxes
############################################################


def extract_bbox_from_mask(mask_path):
    """
    Function to extract bounding boxes from masks
    :param mask_path: str, path for the mask
    :return: Tensor contains the bounding boxes
    """
    mask = read_image(mask_path)  # read the mask as a tensor type
    obj_ids = torch.unique(mask)  # get the unique colors
    obj_ids = obj_ids[:-1]  # get rid of the last color which is 2 (in my case 2 is the background)

    masks = mask == obj_ids[:, None, None]  # split the color-encoded mask into a set of boolean masks.
    boxes = masks_to_boxes(masks)  # get the bounding boxes

    return boxes
