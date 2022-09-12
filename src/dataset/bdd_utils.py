# Import Libraries
import numpy as np
import random
import json
import matplotlib.pyplot as plt
from matplotlib.path import Path
import torch
from torchvision.io import read_image
from torchvision.ops import masks_to_boxes


###############################################################################################################
# DISPLAY
###############################################################################################################
def to_mask(mask_shape, poly2d):
    """
    function to convert 2D polygon to 2D mask
    :param mask_shape: the shape of the mask we want to return
    :param poly2d: a list of x and y coordinates
    :return: np.array object
    """
    nx, ny = mask_shape[1], mask_shape[0]
    # Create vertex coordinates for each grid cell...
    # (<0,0> is at the top left of the grid in this system)
    x, y = np.meshgrid(np.arange(nx), np.arange(ny))
    x, y = x.flatten(), y.flatten()

    points = np.vstack((x, y)).T

    path = Path(poly2d)
    grid = path.contains_points(points)
    grid = grid.reshape((ny, nx))

    return grid


def bbox_from_instance_mask(mask):
    """
    Function to extract bounding boxes from masks
    :param mask_path: str, path for the mask
    :return: Tensor contains the bounding boxes
    """
    mask = torch.tensor(mask, dtype=torch.uint8)  # read the mask as a tensor type
    obj_ids = torch.unique(mask)  # get the unique colors
    obj_ids = obj_ids[1:]  # get rid of the last color which is 2 (in my case 2 is the background)

    masks = mask == obj_ids[:, None, None]  # split the color-encoded mask into a set of boolean masks.
    boxes = masks_to_boxes(masks)  # get the bounding boxes
    if len(boxes):
        return boxes[0].tolist()
    else:
        return None


def get_coloured_mask(mask):
    """
    random_colour_masks
      parameters:
        - image - predicted masks
      method:
        - the masks of each predicted object is given random colour for visualization
    """
    colours = [[0, 255, 0], [0, 0, 255], [255, 0, 0], [0, 255, 255], [255, 255, 0], [255, 0, 255], [80, 70, 180],
               [250, 80, 190], [245, 145, 50], [70, 150, 250], [50, 190, 190]]
    r = np.zeros_like(mask).astype(np.uint8)
    g = np.zeros_like(mask).astype(np.uint8)
    b = np.zeros_like(mask).astype(np.uint8)
    color = colours[random.randrange(0, 10)]
    r[mask == 1], g[mask == 1], b[mask == 1] = color
    coloured_mask = np.stack([r, g, b], axis=2)
    return coloured_mask, color