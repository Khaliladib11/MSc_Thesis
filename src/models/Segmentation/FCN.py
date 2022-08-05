import json
from pathlib import Path

import torch
import torch.nn as nn
import torchvision

from torchvision.models.segmentation import fcn_resnet50, fcn_resnet101
import pytorch_lightning as pl


class FCN(pl.LightningModule):

    def __init__(self,
                 cfg,
                 num_classes,
                 backbone,
                 learning_rate,
                 weight_decay,
                 pretrained,
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

        model_params = {
            'pretrained_backbone': pretrained_backbone,
            'pretrained': pretrained,
            'num_classes': self.num_classes,
            'aux_loss': True
        }

        if backbone == 'resnet50':
            self.model = fcn_resnet50(**model_params)

        elif backbone == 'resnet101':
            self.model = fcn_resnet101(**model_params)

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
        pass
