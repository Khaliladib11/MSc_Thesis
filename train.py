import os
import yaml
import argparse

from src.models.Detection.Faster_RCNN import Faster_RCNN
from src.models.Segmentation.MaskRCNN import Mask_RCNN
from src.models.Segmentation.DeepLab import DeepLab
from src.dataset.bdd_detetcion import BDDDetection
from src.dataset.bdd_instance_segmentation import BDDInstanceSegmentation
from src.dataset.bdd_drivable_segmentation import BDDDrivableSegmentation
from src.config.defaults import cfg
from src.utils.DataLoaders import get_loader

import torch
from pytorch_lightning import Trainer
from pytorch_lightning.utilities.model_summary import ModelSummary
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.loggers import CSVLogger

if __name__ == '__main__':
    # Define the parser
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch-size', type=int, default=16, help='total batch size for all GPUs')
    parser.add_argument('--img-size', type=int, default=640, help='train, val image size (pixels)')
    parser.add_argument('--data', type=str, default="./data/fasterrcnn.yaml", help='data.yaml path')
    parser.add_argument('--weights', type=str, default=None, help='train from checkpoint')
    parser.add_argument('--backbone', type=str, default='resnet50', help='choose the backbone you want to use - default: resnet50')
    parser.add_argument('--version', type=str, default='v2', choices=['v1', 'v2'], help='Version of MaskRCNN')
    parser.add_argument('--lr', type=float, default=1e-5, help='learning rate')
    parser.add_argument('--total_epochs', type=int, default=100, help='total_epochs')
    parser.add_argument('--num_workers', type=int, default=4, help='num_workers')
    parser.add_argument('--pin_memory', type=bool, default=False, help='pin_memory')
    parser.add_argument('--logger_path', type=str, help='where you want to log your data')
    parser.add_argument('--name', type=str, default=str, help='name of the model you want to save')
    parser.add_argument('--model', type=str, default='fasterrcnn', choices=['fasterrcnn', 'deeplab', 'maskrcnn'],
                        help='the model and task you want to perform')

    # Fetch the params from the parser
    args = parser.parse_args()

    batch_size = args.batch_size  # Batch Size
    lr = args.lr  # Learning Rate
    weights = args.weights  # Check point to continue training
    backbone = args.backbone  # Check point to continue training
    version = args.version  # version of MaskRCNN you want to use
    img_size = args.img_size  # Image size
    total_epochs = args.total_epochs  # number of epochs
    num_workers = args.num_workers
    pin_memory = args.pin_memory
    model = args.model
    logger_path = args.logger_path
    name = args.name

    with open(args.data, 'r') as f:
        data = yaml.safe_load(f)  # data from .yaml file

    obj_cls = data['classes']  # the classes we want to work one
    relative_path = data['relative_path']  # relative path to the dataset

    ######################################## Datasets ########################################

    if model == 'fasterrcnn':
        # Training dataset
        bdd_train_params = {
            'cfg': cfg,
            'stage': 'train',
            'relative_path': relative_path,
            'obj_cls': obj_cls
        }

        bdd_train = BDDDetection(**bdd_train_params)

        # Validation dataset
        bdd_val_params = {
            'cfg': cfg,
            'stage': 'val',
            'relative_path': relative_path,
            'obj_cls': obj_cls
        }

        bdd_val = BDDDetection(**bdd_val_params)

    elif model == 'deeplab':
        # Training dataset
        bdd_train_params = {
            'cfg': cfg,
            'stage': 'train',
            'relative_path': relative_path,
            'obj_cls': obj_cls,
            'image_size': img_size
        }

        bdd_train = BDDDrivableSegmentation(**bdd_train_params)

        # Validation dataset
        bdd_val_params = {
            'cfg': cfg,
            'stage': 'val',
            'relative_path': relative_path,
            'obj_cls': obj_cls,
            'image_size': img_size
        }

        bdd_val = BDDDrivableSegmentation(**bdd_val_params)

    elif model == 'maskrcnn':
        # Training dataset
        bdd_train_params = {
            'cfg': cfg,
            'stage': 'train',
            'relative_path': relative_path,
            'obj_cls': obj_cls,
            'image_size': img_size
        }

        bdd_train = BDDInstanceSegmentation(**bdd_train_params)

        # Validation dataset
        bdd_val_params = {
            'cfg': cfg,
            'stage': 'val',
            'relative_path': relative_path,
            'obj_cls': obj_cls,
            'image_size': img_size
        }

        bdd_val = BDDInstanceSegmentation(**bdd_val_params)

    print(f"Training Images: {len(bdd_train)}. Validation Images: {len(bdd_val)}.")

    ######################################## DataLoaders ########################################

    # training dataloader
    train_dataloader_args = {
        'dataset': bdd_train,
        'batch_size': batch_size,
        'shuffle': True,
        'collate_fn': bdd_train.collate_fn,
        'pin_memory': pin_memory,
        'num_workers': num_workers
    }
    train_dataloader = get_loader(**train_dataloader_args)

    # val dataloader
    val_dataloader_args = {
        'dataset': bdd_val,
        'batch_size': batch_size,
        'shuffle': False,
        'collate_fn': bdd_train.collate_fn,
        'pin_memory': pin_memory,
        'num_workers': num_workers
    }
    val_dataloader = get_loader(**val_dataloader_args)

    ######################################## Model ########################################

    # check device
    device = torch.device('cpu')
    if torch.cuda.is_available():
        device = torch.device('cuda')

    print(f"We are using {device}")

    # check model
    if model == 'fasterrcnn':
        faster_rcnn_params = {
            'cfg': cfg,
            'num_classes': len(bdd_train.cls_to_idx),
            'backbone': backbone,
            'learning_rate': lr,
            'weight_decay': 1e-3,
            'pretrained': True,
            'pretrained_backbone': True,
        }
        model = Faster_RCNN(**faster_rcnn_params)


    elif model == 'deeplab':
        deeplab_params = {
            'cfg': cfg,
            'num_classes': len(bdd_train.cls_to_idx),
            'backbone': backbone,
            'learning_rate': lr,
            'weight_decay': 1e-3,
            'pretrained': True,
            'pretrained_backbone': True,
        }
        model = DeepLab(**deeplab_params)

    elif model == 'maskrcnn':
        mask_rcnn_params = {
            'cfg': cfg,
            'num_classes': len(bdd_train.cls_to_idx),
            'version': version,
            'learning_rate': lr,
            'weight_decay': 1e-3,
            'pretrained': True,
            'pretrained_backbone': True,
        }
        model = Mask_RCNN(**mask_rcnn_params)

    ModelSummary(model)  # print model summary

    ######################################## Training ########################################

    early_stop_params = {
        'monitor': "val_loss",
        'patience': 5,
        'verbose': False,
        'mode': "min"
    }

    early_stop_callback = EarlyStopping(**early_stop_params)  # Early Stopping to avoid overfitting

    checkpoint_params = {
        'monitor': "val_loss",
        'mode': 'min',
        'every_n_train_steps': 0,
        'every_n_epochs': 1,
        'dirpath': logger_path
    }

    checkpoint_callback = ModelCheckpoint(**checkpoint_params)  # Model check

    wandb_logger = WandbLogger(name=name, project='Master Thesis', log_model='all')
    csv_logger = CSVLogger(save_dir=logger_path, name=name)

    if weights is not None:
        training_params = {
            'resume_from_checkpoint': weights,
            'profiler': "simple",
            "logger": [wandb_logger, csv_logger],
            'accelerator': 'gpu',
            'devices': 1,
            'max_epochs': total_epochs,
            'callbacks': [early_stop_callback, checkpoint_callback],
        }
        fit_params = {
            'model': model,
            'train_dataloaders': train_dataloader,
            'val_dataloaders': val_dataloader,
            'ckpt_path': weights,
        }

    else:
        training_params = {
            'profiler': "simple",
            "logger": [csv_logger, wandb_logger],
            'accelerator': 'gpu',
            'devices': 1,
            'max_epochs': total_epochs,
            'callbacks': [early_stop_callback, checkpoint_callback],
        }
        fit_params = {
            'model': model,
            'train_dataloaders': train_dataloader,
            'val_dataloaders': val_dataloader,
        }

    trainer = Trainer(**training_params)
    trainer.fit(**fit_params)

    print(f"Model's best weights: {checkpoint_callback.best_model_path}")
