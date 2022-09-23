import yaml
import argparse
from utils import *
from src.models.Detection.Faster_RCNN import Faster_RCNN
from src.dataset.bdd_detetcion import BDDDetection
from src.config.defaults import cfg
from src.utils.DataLoaders import get_loader
from pytorch_lightning import Trainer
import pandas as pd
import warnings

warnings.filterwarnings("ignore")


if __name__ == '__main__':
    # Define the parser
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, default="./data/fasterrcnn.yaml", help='data.yaml path')
    parser.add_argument('--weights', type=str, default=None, help='train from checkpoint')
    parser.add_argument('--model', type=str, default='fasterrcnn', choices=['fasterrcnn', 'deeplab', 'maskrcnn'],
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
        model = Faster_RCNN.load_from_checkpoint(weights)

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
        print("#" * 100)
        print("Mean Average Precision")
        print(f"Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = {round(mAP['map'].item(), 4)}")
        print(
            f"Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = {round(mAP['map_50'].item(), 4)}")
        print(
            f"Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = {round(mAP['map_75'].item(), 4)}")

        print('\n')
        print("Start Computing mAP for Person class.")
        print("#" * 100)
        mAP_person = model.metric_person.compute()
        print("Mean Average Precision for Person class")
        print(
            f"Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = {round(mAP_person['map'].item(), 4)}")
        print(
            f"Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = {round(mAP_person['map_50'].item(), 4)}")
        print(
            f"Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = {round(mAP_person['map_75'].item(), 4)}")

        print('\n')
        print("Start Computing mAP for Car class.")
        print("#" * 100)
        mAP_car = model.metric_car.compute()
        print("Mean Average Precision for Car class")
        print(
            f"Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = {round(mAP_car['map'].item(), 4)}")
        print(
            f"Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = {round(mAP_car['map_50'].item(), 4)}")
        print(
            f"Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = {round(mAP_car['map_75'].item(), 4)}")

        print('\n')
        print("Start Computing mAP for traffic light class.")
        print("#" * 100)
        mAP_tl = model.metric_tl.compute()
        print("Mean Average Precision for traffic light class")
        print(
            f"Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = {round(mAP_tl['map'].item(), 4)}")
        print(
            f"Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = {round(mAP_tl['map_50'].item(), 4)}")
        print(
            f"Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = {round(mAP_tl['map_75'].item(), 4)}")

        print('\n')
        print("Start Computing mAP for traffic sign class.")
        print("#" * 100)
        mAP_ts = model.metric_ts.compute()
        print("Mean Average Precision for traffic sign class")
        print(
            f"Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = {round(mAP_ts['map'].item(), 4)}")
        print(
            f"Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = {round(mAP_ts['map_50'].item(), 4)}")
        print(
            f"Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = {round(mAP_ts['map_75'].item(), 4)}")

        mAPs = {
            'all': {
                'mAP': round(mAP['map'].item(), 4),
                'mAP50': round(mAP['map_50'].item(), 4),
                'mAP75': round(mAP['map_75'].item(), 4),
            },
            'person': {
                'mAP': round(mAP_person['map'].item(), 4),
                'mAP50': round(mAP_person['map_50'].item(), 4),
                'mAP75': round(mAP_person['map_75'].item(), 4),
            },
            'car': {
                'mAP': round(mAP_car['map'].item(), 4),
                'mAP50': round(mAP_car['map_50'].item(), 4),
                'mAP75': round(mAP_car['map_75'].item(), 4),
            },
            'traffic light': {
                'mAP': round(mAP_tl['map'].item(), 4),
                'mAP50': round(mAP_tl['map_50'].item(), 4),
                'mAP75': round(mAP_tl['map_75'].item(), 4),
            },
            'traffic sign': {
                'mAP': round(mAP_ts['map'].item(), 4),
                'mAP50': round(mAP_ts['map_50'].item(), 4),
                'mAP75': round(mAP_ts['map_75'].item(), 4),
            }
        }

        # save the result as csv file
        export_map(mAPs, save_path)
