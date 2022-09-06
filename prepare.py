import os
import yaml
import argparse
from src.config.defaults import cfg
from src.utils.utils import *
from src.dataset.bdd_detetcion import BDDDetection

if __name__ == '__main__':
    # Define the parser
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, default='', help='')
    parser.add_argument('--yolo_version', type=str, default='yolov5', choices=['yolov5', 'yolov7'], help='which '
                                                                                                         'version of '
                                                                                                         'yolo do you '
                                                                                                         'want to use')
    parser.add_argument('--dataset_path', type=str, help='The path to the dataset folder where you want to upload the '
                                                         'data')

    # Fetch the params from the parser
    args = parser.parse_args()
    version = args.yolo_version
    dataset_path = args.dataset_path

    with open(args.data, 'r') as f:
        data = yaml.safe_load(f)  # data from .yaml file

    obj_cls = data['classes']  # the classes we want to work one
    relative_path = data['relative_path']  # relative path to the dataset

    ######################################## Datasets ########################################
    bdd_train_params = {
        'cfg': cfg,
        'relative_path': relative_path,
        'stage': 'train',
        'obj_cls': obj_cls
    }

    bdd_train = BDDDetection(**bdd_train_params)

    bdd_val_params = {
        'cfg': cfg,
        'relative_path': relative_path,
        'stage': 'val',
        'obj_cls': obj_cls
    }

    bdd_val = BDDDetection(**bdd_val_params)

    bdd_test_params = {
        'cfg': cfg,
        'stage': 'test',
        'relative_path': relative_path,
        'obj_cls': obj_cls
    }

    bdd_test = BDDDetection(**bdd_test_params)
    print(50*'#')
    print(
        f"We have {len(bdd_train)} training images, {len(bdd_val)} validation images and {len(bdd_test)} test images.")

    print(50 * '#')
    ######################################## Prepare Data ########################################

    """
    Yolov5 [https://github.com/ultralytics/yolov5] takes the data in YOLO annotation format
    YOLO format is basically (x, y, w, h), like COCO format but normalized
    the annotation must be in txt file, each image has it's own annotation file
    The data should be in a folder and under this folder we should have two folders: images and labels
    in images we should have three different folders: train, test, val
    same thing for labels
    """
    if version == 'yolov5':
        """
        Dataset folder's structure for Yolov5 is as follows:
        ├─dataset
        │ ├─images
        │ │ ├─train
        │ │ ├─test
        │ │ ├─val
        │ ├─labels
        │ │ ├─train
        │ │ ├─test
        │ │ ├─val
        """
        # train data
        images_training_path = os.path.join(dataset_path, 'images', 'train')
        labels_training_path = os.path.join(dataset_path, 'labels', 'train')

        # val data
        images_val_path = os.path.join(dataset_path, 'images', 'val')
        labels_val_path = os.path.join(dataset_path, 'labels', 'val')

        # test data
        images_test_path = os.path.join(dataset_path, 'images', 'test')
        labels_test_path = os.path.join(dataset_path, 'labels', 'test')

    elif version == 'yolov7':
        """
        Dataset folder's structure for Yolov7 is as follows:
        ├─dataset
        │ ├─train
        │ │ ├─images
        │ │ ├─labels
        │ ├─test
        │ │ ├─images
        │ │ ├─labels
        │ ├─val
        │ │ ├─images
        │ │ ├─labels
        """
        # train data
        images_training_path = os.path.join(dataset_path, 'train', 'images')
        labels_training_path = os.path.join(dataset_path, 'train', 'labels')
        # val data
        images_val_path = os.path.join(dataset_path, 'val', 'images')
        labels_val_path = os.path.join(dataset_path, 'val', 'labels')
        # test data
        images_test_path = os.path.join(dataset_path, 'test', 'images')
        labels_test_path = os.path.join(dataset_path, 'test', 'labels')

    print(50 * '#')
    # create annotation for training
    yolo_train = create_yolo_annotation(bdd_train)
    # move the data to the specific path
    move_files(yolo_train, images_training_path, labels_training_path)
    print(50 * '#')

    # create annotation for validation
    yolo_val = create_yolo_annotation(bdd_val)
    # move the data to the specific path
    move_files(yolo_train, images_val_path, labels_val_path)
    print(50 * '#')
    # create annotation for test
    yolo_test = create_yolo_annotation(bdd_test)
    # move the data to the specific path
    move_files(yolo_test, images_test_path, labels_test_path)
    print(50 * '#')