import json
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision

from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor, fasterrcnn_resnet50_fpn, fasterrcnn_resnet50_fpn_v2
from torchvision.models.detection import FasterRCNN_ResNet50_FPN_V2_Weights
from torchvision.models import ResNet50_Weights
from torchvision.ops import box_iou
import pytorch_lightning as pl
from torchmetrics.detection.mean_ap import MeanAveragePrecision


class Faster_RCNN(pl.LightningModule):

    def __init__(self,
                 cfg,
                 num_classes,
                 backbone,
                 learning_rate,
                 weight_decay,
                 pretrained,
                 pretrained_backbone,
                 checkpoint_path,
                 train_dataset,
                 val_dataset,
                 batch_size,
                 num_workers
                 ):
        super(Faster_RCNN, self).__init__()

        assert 0 <= learning_rate <= 1, "Learning Rate must be between 0 and 1"
        assert backbone in cfg.DETECTION.BACKBONE, f"Please choose backbone from the following: {cfg.DETECTION.BACKBONE}"
        assert num_classes > 0, "Number of classes must be greater than 0."

        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.num_workers = num_workers
        self.batch_size = batch_size
        self.num_classes = num_classes
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        
        self.metric = MeanAveragePrecision(box_format="xyxy", class_metrics=True)
        
        self.save_hyperparameters()

        if backbone == 'resnet50':
            params_ = {
                'weights': FasterRCNN_ResNet50_FPN_V2_Weights.DEFAULT,
                'weights_backbone': ResNet50_Weights.DEFAULT
            }
            self.model = fasterrcnn_resnet50_fpn_v2(**params_)

            in_features = self.model.roi_heads.box_predictor.cls_score.in_features
            self.model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

        else:
            resnet_backbone = torchvision.models.detection.backbone_utils.resnet_fpn_backbone(
                backbone,
                pretrained_backbone
            )
            self.model = FasterRCNN(resnet_backbone, num_classes)

        if checkpoint_path:
            self.load_checkpoint(checkpoint_path)

    def forward(self, x, *args, **kwargs):
        self.model.eval()
        return self.model(x)

    def train_dataloader(self):
        val_dataloader_args = {
            'dataset': self.val_dataset,
            'batch_size': self.batch_size,
            'shuffle': True,
            'collate_fn': self.val_dataset.collate_fn,
            'num_workers': self.num_workers
        }
        return DataLoader(**val_dataloader_args)


    def val_dataloader(self):
        train_dataloader_args = {
            'dataset': self.train_dataset,
            'batch_size': self.batch_size,
            'shuffle': False,
            'collate_fn': self.train_dataset.collate_fn,
            'num_workers': self.num_workers
        }
        return DataLoader(**train_dataloader_args)


    def training_step(self, train_batch, batch_idx):
        images, targets = train_batch

        targets = [{k: v for k, v in t.items()} for t in targets]

        loss_dict = self.model(images, targets)
        train_loss = sum(loss for loss in loss_dict.values())

        # self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)

        return train_loss

    def training_epoch_end(self, training_step_outputs):
        epoch_losses = torch.tensor([batch_loss['loss'].item() for batch_loss in training_step_outputs])
        # epoch_losses = torch.stack(training_step_outputs)
        loss_mean = torch.mean(epoch_losses)
        self.log('training_loss', loss_mean)
        
    
        

    def validation_step(self, val_batch, batch_idx):
        images, targets = val_batch

        targets = [{k: v for k, v in t.items()} for t in targets]

        outputs = self.model(images)
        
        if outputs[0]["boxes"].shape[0] == 0:
            # no box detected, 0 IOU
            self.metric.update(torch.tensor(0.0), targets)
        else:
            self.metric.update(outputs, targets)


    def validation_epoch_end(self, validation_step_outputs):
        mAPs = self.metric.compute()
        self.log_dict({
            'mAP at 0.50:0.05:0.95': mAPs['map'],
            'mAP at 0.50': mAPs['map_50'],
            'mAP at 0.75': mAPs['map_75']
        })
        
        self.metric.reset()

    def configure_optimizers(self):
        optimizer_params = {
            'params': self.model.parameters(),
            'lr': self.learning_rate,
            'weight_decay': self.weight_decay,
        }
        return torch.optim.Adam(**optimizer_params)
