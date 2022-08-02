import json
from pathlib import Path

import torch
import torch.nn as nn
import torchvision

from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor, fasterrcnn_resnet50_fpn
from torchvision.ops import box_iou
import pytorch_lightning as pl


class FasterRCNN(pl.LightningModule):

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
        super(FasterRCNN, self).__init__()

        assert 0 <= learning_rate <= 1, "Learning Rate must be between 0 and 1"
        assert backbone in cfg.DETECTION.BACKBONE, f"Please choose backbone from the following: {cfg.DETECTION.BACKBONE}"
        assert num_classes > 0, "Number of classes must be greater than 0."

        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.num_classes = num_classes

        if backbone == 'resnet50':
            self.model = fasterrcnn_resnet50_fpn(pretrained, pretrained_backbone)

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

    def forward(self, x):
        self.model.eval()
        return self.model(x)

    def load_checkpoint(self, checkpoint):
        checkpoint = torch.load(Path(checkpoint))
        self.model.load_state_dict(checkpoint['model'])

    def training_step(self, train_batch, batch_idx):
        images, targets = train_batch

        targets = [{k: v for k, v in t.items()} for t in targets]

        loss_dict = self.model(images, targets)
        loss = sum(loss for loss in loss_dict.values())
        return {'loss': loss, 'log': loss_dict}

    def validation_step(self, val_batch, batch_idx):
        pass

    def configure_optimizers(self):
        optimizer_params = {
            'params': self.model.parameters(),
            'lr': self.learning_rate,
            'weight_decay': self.weight_decay,
            'momentum': 0.9
        }
        return torch.optim.Adam(**optimizer_params)
