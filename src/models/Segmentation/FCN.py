import json
from pathlib import Path

import torch
import torch.nn as nn
import torchvision

from torchvision.models.segmentation import fcn_resnet50, fcn_resnet101
from torchvision.models.segmentation import FCN_ResNet50_Weights, FCN_ResNet101_Weights
from torchvision.models import ResNet50_Weights, ResNet101_Weights
from torchvision.models.segmentation.fcn import FCNHead
import pytorch_lightning as pl




class FCN(pl.LightningModule):

    def __init__(self,
                 cfg,
                 num_classes,
                 backbone,
                 learning_rate,
                 weight_decay,
                 pretrained_backbone,
                 checkpoint_path,
                 train_loader,
                 val_loader
                 ):
        super(FCN, self).__init__()

        assert 0 <= learning_rate <= 1, "Learning Rate must be between 0 and 1"
        assert backbone in cfg.DETECTION.BACKBONE, f"Please choose backbone from the following: {cfg.DRIVABLE_AREA.BACKBONE}"
        assert num_classes > 0, "Number of classes must be greater than 0."

        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.num_classes = num_classes
        self.train_loader = train_loader
        self.val_loader = val_loader

        if backbone == 'resnet50':
            if pretrained_backbone:
                model_params = {
                    'weights': None,
                    'weights_backbone': ResNet50_Weights.DEFAULT,
                    'num_classes': self.num_classes,
                }
                self.model = fcn_resnet50(**model_params)
            else:
                self.model = fcn_resnet50(num_classes=self.num_classes)

        elif backbone == 'resnet101':
            if pretrained_backbone:
                model_params = {
                    'weights': None,
                    'weights_backbone': ResNet101_Weights.DEFAULT,
                    'num_classes': self.num_classes,
                }
                self.model = fcn_resnet101(**model_params)

            else:
                self.model = fcn_resnet101(num_classes=self.num_classes)

        self.logits = nn.LogSoftmax(dim=1)
        self.loss_function = nn.NLLLoss()

    def forward(self, x):
        self.model.eval()
        return self.model(x)

    def train_dataloader(self):
        return self.train_loader

    def val_dataloader(self):
        return self.val_loader

    def training_step(self, train_batch, batch_idx):
        images, targets = train_batch
        #targets = [{k: v for k, v in t.items()} for t in targets]
        outputs = self.model(images)['out']
        train_loss = self.loss_function(self.logits(outputs), )
        return train_loss

    def training_epoch_end(self, training_step_outputs):
        epoch_losses = torch.stack(training_step_outputs)
        loss_mean = torch.mean(epoch_losses)
        self.log('training_loss', loss_mean)

    def validation_step(self, val_batch, batch_idx):
        pass

    def validation_epoch_end(self, validation_step_outputs):
        pass

    def configure_optimizers(self):
        optimizer_params = {
            'params': self.model.parameters(),
            'lr': self.learning_rate,
            'weight_decay': self.weight_decay,
        }
        return torch.optim.Adam(**optimizer_params)
