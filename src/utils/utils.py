# Import Libraries
import numpy as np
import os
import shutil
from collections import deque
from tqdm import tqdm
from PIL import Image
import matplotlib.pyplot as plt
from matplotlib.path import Path
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


############################################################
#  Data Format
############################################################

# method to convert xyxy to xywh yolo format
def convert_pascal_to_yolo(xmin, ymin, xmax, ymax, img_w, img_h):
    dw = 1. / (img_w)
    dh = 1. / (img_h)
    x = (xmin + xmax) / 2.0 - 1
    y = (ymin + ymax) / 2.0 - 1
    w = xmax - xmin
    h = ymax - ymin
    x = round(x * dw, 4)
    w = round(w * dw, 4)
    y = round(y * dh, 4)
    h = round(h * dh, 4)
    return (x, y, w, h)


def create_yolo_annotation(bdd):
    """
    method to convert to create new db from the bdd.db object that have yolo format
    :param bdd: BDD class instance
    :return: yolo_deque object that has all the annotation in yolo format
    """
    yolo_deque = deque()
    # YOLO_Format = namedtuple("YOLO", "cls x y w h")
    print("Start converting.")
    for item in tqdm(bdd.db):
        image_path = item['image_path']
        image = Image.open(image_path)
        width, height = image.size
        boxes = item['bboxes']
        classes = item['classes']
        yolo_boxes = []
        for i, box in enumerate(boxes):
            cls = classes[i]
            x, y, w, h = convert_pascal_to_yolo(box[0], box[1], box[2], box[3], width, height)
            # yolo_format = YOLO_Format(cls, x, y, w, h)
            yolo_boxes.append([cls, x, y, w, h])

        yolo_deque.append({
            'image_name': item['image_path'].split('\\')[-1],
            'image_path': image_path,
            'width': width,
            'height': height,
            'yolo_boxes': yolo_boxes
        })
    print("Finish from converting")
    return yolo_deque


def move_files(yolo_deque, images_folder_destination, labels_folder_destination):
    """
    method to copy the files from the original source to a new destination
    :param yolo_deque: deque object, which is the output of the create_yolo_annotation method
    :param images_folder_destination: the path to the folder where we want to copy the images to
    :param labels_folder_destination: the path to the folder where we want to create the labels (txt files)
    :return: None
    """
    assert os.path.isdir(images_folder_destination), "Folder does not exist!"
    assert os.path.isdir(labels_folder_destination), "Folder does not exist!"

    print("Start copying files.")
    for item in tqdm(yolo_deque):
        # shutil.copy(item['image_path'], os.path.join(folder_destination, 'images', stage))
        # create_annotation_file(item, folder_destination, stage)
        shutil.copy(item['image_path'], images_folder_destination)
        create_annotation_file(item, labels_folder_destination)

    print(f"All images are in {images_folder_destination} and all labels are in {labels_folder_destination}.")


def create_annotation_file(yolo_item, labels_folder_destination):
    text_file_name = yolo_item['image_name'].replace('.jpg', '.txt').split('/')[-1]
    file_path = os.path.join(labels_folder_destination, text_file_name)
    file = open(file_path, 'a')
    for line in yolo_item['yolo_boxes']:
        file.write(" ".join(str(item) for item in line))
        file.write('\n')
    file.close()

############################################################
#  Masks
############################################################

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


def bbox_from_instance_mask(mask_path):
    """
    Function to extract bounding boxes from masks
    :param mask_path: str, path for the mask
    :return: Tensor contains the bounding boxes
    """
    mask = read_image(mask_path)  # read the mask as a tensor type
    obj_ids = torch.unique(mask)  # get the unique colors
    obj_ids = obj_ids[1:]  # get rid of the last color which is 2 (in my case 2 is the background)

    masks = mask == obj_ids[:, None, None]  # split the color-encoded mask into a set of boolean masks.
    boxes = masks_to_boxes(masks)  # get the bounding boxes

    return boxes
