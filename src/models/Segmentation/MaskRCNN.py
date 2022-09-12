import torch
import torch.nn as nn

import torchvision
import pytorch_lightning as pl

from .segmentation_models import get_maskrcnn


class Mask_RCNN(pl.LightningModule):

    def __init__(self,
                 cfg,
                 num_classes: int,
                 version: str,
                 learning_rate: float,
                 weight_decay: float,
                 pretrained: bool,
                 pretrained_backbone: bool,
                 ):
        """
        Constructor for the Mask_RCNN class
        :param cfg: yacs configuration that contains all the necessary information about the available backbones
        :param num_classes: the number of classes we want to classify for each pixel (with background)
        :param version: the version of the mask rcnn implementation you want to use
        :param learning_rate: The learning rate for the network
        :param weight_decay: decay for the regularization
        :param pretrained: bool value for pretrained network
        :param pretrained_backbone: bool value for pretrained backbone
        """
        super(Mask_RCNN, self).__init__()

        assert 0 <= learning_rate <= 1, "Learning Rate must be between 0 and 1"
        assert version in cfg.INSTANCE_SEGMENTATION.VERSION, f"Please choose version from the following: {cfg.INSTANCE_SEGMENTATION.VERSION}"
        assert num_classes > 0, "Number of classes must be greater than 0."

        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.num_classes = num_classes
        self.save_hyperparameters()

        self.model = get_maskrcnn(version=version,
                                  pretrained=pretrained,
                                  pretrained_weights=pretrained_backbone,
                                  num_classes=self.num_classes)

    def forward(self, x):
        return self.model(x)

    def training_step(self, train_batch, batch_idx) -> float:
        images, targets = train_batch

        targets = [{k: v for k, v in t.items()} for t in targets]

        loss_dict = self.model(images, targets)
        losses = sum(loss for loss in loss_dict.values())

        return losses

    def training_epoch_end(self, training_step_outputs) -> dict:
        epoch_losses = torch.tensor([batch_loss['loss'].item() for batch_loss in training_step_outputs])
        # epoch_losses = torch.stack(training_step_outputs)
        loss_mean = torch.mean(epoch_losses)
        self.log('training_loss', loss_mean)

    def validation_step(self, val_batch, batch_idx) -> float:
        self.model.train()
        images, targets = val_batch
        targets = [{k: v for k, v in t.items()} for t in targets]
        loss_dict = self.model(images, targets)
        va_losses = sum(loss for loss in loss_dict.values())

    def validation_epoch_end(self, validation_step_outputs) -> dict:
        epoch_losses = torch.tensor([batch_loss.item() for batch_loss in validation_step_outputs])
        # epoch_losses = torch.stack(training_step_outputs)
        loss_mean = torch.mean(epoch_losses)
        self.log('val_loss', loss_mean)
        return {'val_loss': loss_mean}

    def predict_step(self, batch):
        pass

    def configure_optimizers(self):
        optimizer_params = {
            'params': self.model.parameters(),
            'lr': self.learning_rate,
            'weight_decay': self.weight_decay,
        }
        return torch.optim.Adam(**optimizer_params)

