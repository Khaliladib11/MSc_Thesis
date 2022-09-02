import json
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision

from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor, fasterrcnn_resnet50_fpn, \
    fasterrcnn_resnet50_fpn_v2, fasterrcnn_mobilenet_v3_large_fpn
from torchvision.models.detection import FasterRCNN_ResNet50_FPN_V2_Weights, FasterRCNN_MobileNet_V3_Large_FPN_Weights
from torchvision.models import ResNet50_Weights, MobileNet_V3_Large_Weights
from torchvision.ops import box_iou
import pytorch_lightning as pl


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
                 ):
        super(Faster_RCNN, self).__init__()

        assert 0 <= learning_rate <= 1, "Learning Rate must be between 0 and 1"
        assert backbone in cfg.DETECTION.BACKBONE, f"Please choose backbone from the following: {cfg.DETECTION.BACKBONE}"
        assert num_classes > 0, "Number of classes must be greater than 0."

        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.num_classes = num_classes
        self.save_hyperparameters()

        if backbone == 'resnet50':
            if pretrained:
                params_ = {
                    'weights': FasterRCNN_ResNet50_FPN_V2_Weights.DEFAULT,
                    'weights_backbone': ResNet50_Weights.DEFAULT
                }
                self.model = fasterrcnn_resnet50_fpn_v2(**params_)

            elif pretrained_backbone:
                params_ = {
                    'weights_backbone': ResNet50_Weights.DEFAULT
                }
                self.model = fasterrcnn_resnet50_fpn_v2(**params_)
            else:
                self.model = fasterrcnn_resnet50_fpn_v2()

            in_features = self.model.roi_heads.box_predictor.cls_score.in_features
            self.model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

        elif backbone == 'MobileNetV3-Large':
            if pretrained:
                params_ = {
                    'weights': FasterRCNN_MobileNet_V3_Large_FPN_Weights.DEFAULT,
                    'weights_backbone': MobileNet_V3_Large_Weights.DEFAULT
                }
                self.model = fasterrcnn_mobilenet_v3_large_fpn(**params_)

            elif pretrained_backbone:
                params_ = {
                    'weights_backbone': MobileNet_V3_Large_Weights.DEFAULT
                }
                self.model = fasterrcnn_mobilenet_v3_large_fpn(**params_)
            else:
                self.model = fasterrcnn_mobilenet_v3_large_fpn()

            in_features = self.model.roi_heads.box_predictor.cls_score.in_features
            self.model.roi_heads.box_predictor = FastRCNNPredictor(in_features, self.num_classes)

        if checkpoint_path:
            self.load_checkpoint(checkpoint_path)

    def forward(self, x, *args, **kwargs):
        return self.model(x)

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

        return {'train_loss': loss_mean}

    def validation_step(self, val_batch, batch_idx):
        self.model.train()
        images, targets = val_batch

        targets = [{k: v for k, v in t.items()} for t in targets]

        outputs = self.model(images, targets)

        val_loss = sum(loss for loss in outputs.values())
        return val_loss

    def validation_epoch_end(self, validation_step_outputs):
        epoch_losses = torch.tensor([batch_loss.item() for batch_loss in validation_step_outputs])
        # epoch_losses = torch.stack(training_step_outputs)
        loss_mean = torch.mean(epoch_losses)
        self.log('val_loss', loss_mean)
        return {'val_loss': loss_mean}

    def predict_step(self, batch):
        self.model.eval()
        images, targets = batch
        outputs = self.model(images)
        return outputs

    def configure_optimizers(self):
        optimizer_params = {
            'params': self.model.parameters(),
            'lr': self.learning_rate,
            'weight_decay': self.weight_decay,
        }
        return torch.optim.Adam(**optimizer_params)
