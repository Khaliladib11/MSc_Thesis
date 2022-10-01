import sys
import yaml
import argparse
from util import *
from src.models.Detection.Faster_RCNN import Faster_RCNN
from src.models.Segmentation.MaskRCNN import Mask_RCNN
from src.dataset.bdd_detetcion import BDDDetection
from src.dataset.bdd_instance_segmentation_drivable import BDDInstanceSegmentationDrivable
from src.config.defaults import cfg
from src.utils.DataLoaders import get_loader
from pytorch_lightning import Trainer
import pandas as pd
from pprint import pprint
import warnings

warnings.filterwarnings("ignore")

if __name__ == '__main__':
    # Define the parser
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, default="./data/fasterrcnn.yaml", help='data.yaml path')
    parser.add_argument('--weights', type=str, default=None, help='train from checkpoint')
    parser.add_argument('--pred', type=str, default='', help='Path to the predictions folder')
    parser.add_argument('--gt', type=str, default='', help='Path to the ground truth folder')
    parser.add_argument('--model', type=str, default='fasterrcnn', choices=['fasterrcnn', 'deeplab', 'maskrcnn', 'yolov5', 'yolov7'],
                        help='the model and task you want to perform')
    parser.add_argument('--save-path', type=str, default='./mAP_results.csv',
                        help='Path and name of the file you want to export.')

    # Fetch the params from the parser
    args = parser.parse_args()

    weights = args.weights  # Check point to continue training
    save_path = args.save_path
    model_name = args.model  # the name of the model: fastercnn, maskrcnn, deeplab

    with open(args.data, 'r') as f:
        data = yaml.safe_load(f)  # data from .yaml file

    obj_cls = data['classes']  # the classes we want to work one
    relative_path = data['relative_path']  # relative path to the dataset

    if model_name == "fasterrcnn":
        ## Load Model
        try:
            model = Faster_RCNN.load_from_checkpoint(weights)
        except Exception as e:
            print("Could not load the model weights. Please make sure you're providing the correct model weights.")
            sys.exit()

        bdd_params = {
            'cfg': cfg,
            'stage': 'test',
            'relative_path': relative_path,
            'obj_cls': obj_cls,
        }

        bdd = BDDDetection(**bdd_params)

        dataloader_args = {
            'dataset': bdd,
            'batch_size': 1,
            'shuffle': False,
            'collate_fn': bdd.collate_fn,
            'pin_memory': True,
            'num_workers': 1
        }
        dataloader = get_loader(**dataloader_args)

        trainer = Trainer(accelerator='gpu', devices=1)
        pred = trainer.predict(model, dataloader)

        print("Start Computing mAP for all classes.")
        mAP = model.metric.compute()

    elif model_name == 'maskrcnn':
        try:
            model = Mask_RCNN.load_from_checkpoint(weights)
        except Exception as e:
            print("Could not load the model weights. Please make sure you're providing the correct model weights.")
            sys.exit()


        bdd_params = {
            'cfg': cfg,
            'stage': 'test',
            'relative_path': relative_path,
            'obj_cls': obj_cls,
        }

        bdd = BDDInstanceSegmentationDrivable(**bdd_params)

        dataloader_args = {
            'dataset': bdd,
            'batch_size': 1,
            'shuffle': False,
            'collate_fn': bdd.collate_fn,
            'pin_memory': True,
            'num_workers': 1
        }
        dataloader = get_loader(**dataloader_args)

        trainer = Trainer(accelerator='gpu', devices=1)
        pred = trainer.predict(model, dataloader)

        print("Start Computing mAP for all classes.")
        mAP = model.metric.compute()

    elif model_name.startswith('yolo'):
        """
        We evaluate Yolo models the same way
        """
        # get the path where the *.txt file are stored. Those files are generated from the model
        prediction_path = args.pred
        # path to the ground truth folder, where we have the *.txt file. Those files are genearted from the prepate.py file
        gt_path = args.gt

        # check paths
        assert os.path.exists(prediction_path), f"Predictions does not exists at {prediction_path}"
        assert os.path.exists(gt_path), f"Predictions does not exists at {gt_path}"

        # Evaluation
        mAP = yolo_evaluation(prediction_path, gt_path)

    # print
    pprint(mAP)
    export_map_json(mAP, json_file_name=f'mAP_{model_name}.json' , to_save_path=save_path)
