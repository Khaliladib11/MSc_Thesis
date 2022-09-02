import os
import yaml
import argparse

from src.models.Detection.Faster_RCNN import Faster_RCNN
from src.dataset.bdd_detetcion import BDDDetection
from src.config.defaults import cfg
from src.utils.DataLoaders import get_loader

import torch
from pytorch_lightning import Trainer
from pytorch_lightning.utilities.model_summary import ModelSummary
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.loggers import CSVLogger

if __name__ == '__main__':
    # Define the parser
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch-size', type=int, default=16, help='total batch size for all GPUs')
    parser.add_argument('--img-size', type=int, default=640, help='train, val image size (pixels)')
    parser.add_argument('--data', type=str, default="data.yaml", help='data.yaml path')
    parser.add_argument('--weights', type=str, default=None, help='train from checkpoint')
    parser.add_argument('--lr', type=float, default=1e-5, help='learning rate')
    parser.add_argument('--total_epochs', type=int, default=100, help='total_epochs')
    parser.add_argument('--num_workers', type=int, default=4, help='num_workers')
    parser.add_argument('--pin_memory', type=bool, default=False, help='pin_memory')
    parser.add_argument('--model', type=str, default='fasterrcnn', choices=['fasterrcnn', 'deeplab', 'maskrcnn'],
                        help='the model and task you want to perform')

    # Fetch the params from the parser
    args = parser.parse_args()

    batch_size = args.batch_size  # Batch Size
    lr = args.lr  # Learning Rate
    weights = args.weights  # Check point to continue training
    img_size = args.img_size  # Image size
    total_epochs = args.total_epochs  # number of epochs
    num_workers = args.num_workers
    pin_memory = args.pin_memory
    model = args.model

    with open(args.data, 'r') as f:
        data = yaml.safe_load(f)  # data from .yamel file

    obj_cls = data['classes']  # the classes we want to work one
    relative_path = data['relative_path']  # relative path to the dataset

    ######################################## Datasets ########################################

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
            'backbone': 'resnet50',
            'learning_rate': lr,
            'weight_decay': 1e-3,
            'pretrained': True,
            'pretrained_backbone': True,
            'checkpoint_path': None,
        }
        model = Faster_RCNN(**faster_rcnn_params)

        ModelSummary(model)

    elif model == 'deeplab':
        pass

    elif model == 'maskrcnn':
        pass

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
        'dirpath': '../checkpoints/detection/FasterRCNN/v1'
    }

    checkpoint_callback = ModelCheckpoint(**checkpoint_params)  # Model check

    logger = CSVLogger(save_dir="../logs/FasterRCNN/v3/logs", name="fasterrcnn_v3")

    if weights is not None:
        training_params = {
            'resume_from_checkpoint': '../checkpoints/detection/FasterRCNN/v1/epoch=4-step=277110.ckpt',
            'profiler': "simple",
            "logger": logger,
            'accelerator': 'gpu',
            'devices': 1,
            'max_epochs': total_epochs,
            'callbacks': [early_stop_callback, checkpoint_callback],
        }
        fit_params = {
            'model': model,
            'train_dataloaders': train_dataloader,
            'val_dataloaders': val_dataloader,
            'ckpt_path': '../checkpoints/detection/FasterRCNN/v1/epoch=4-step=277110.ckpt',
        }

    else:
        training_params = {
            'profiler': "simple",
            "logger": logger,
            'accelerator': 'gpu',
            'devices': 1,
            'max_epochs': total_epochs,
            'callbacks': [early_stop_callback, checkpoint_callback],
        }
        fit_params = {
            'model': model,
            'train_dataloaders': train_dataloader,
            'val_dataloaders': val_dataloader,
            'ckpt_path': '../checkpoints/detection/FasterRCNN/v1/epoch=4-step=277110.ckpt',
        }

    trainer = Trainer(**training_params)
    trainer.fit(**fit_params)

    print(f"Model's best weights: {checkpoint_callback.best_model_path}")
