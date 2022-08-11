# Import Libraries
import torch
import torch.nn as nn

import torchvision

from torchvision.models.segmentation import DeepLabV3
from torchvision.models.segmentation import deeplabv3_resnet50, deeplabv3_resnet101, deeplabv3_mobilenet_v3_large
from torchvision.models.segmentation import DeepLabV3_ResNet50_Weights, DeepLabV3_ResNet101_Weights, \
    DeepLabV3_MobileNet_V3_Large_Weights
from torchvision.models import ResNet50_Weights, ResNet101_Weights, MobileNet_V3_Large_Weights
from torchvision.models.segmentation.deeplabv3 import DeepLabHead
import pytorch_lightning as pl

from torch.utils.data import DataLoader


class DeepLab(pl.LightningModule):
    """
    DeepLab Class for semantic segmentation task
    """

    def __init__(self,
                 cfg,
                 num_classes: int,
                 backbone: str,
                 learning_rate: float,
                 weight_decay: float,
                 pretrained: bool,
                 pretrained_backbone: bool,
                 train_loader: DataLoader,
                 val_loader: DataLoader
                 ):
        """
        Constructor for the DeepLab class
        :param cfg: yacs configuration that contains all the necessary information about the available backbones
        :param num_classes: the number of classes we want to classify for each pixel (with background)
        :param backbone: the name of the backbone we want to use
        :param learning_rate: The learning rate for the network
        :param weight_decay: decay for the regularization
        :param pretrained: bool value for pretrained network
        :param pretrained_backbone: bool value for pretrained backbone
        :param train_loader: DataLoader for training
        :param val_loader: DataLoader for testing
        """
        super(DeepLab, self).__init__()

        assert 0 <= learning_rate <= 1, "Learning Rate must be between 0 and 1"
        assert backbone in cfg.DRIVABLE_AREA.DEEPLAB_BACKBONE, f"Please choose backbone from the following: {cfg.DRIVABLE_AREA.DEEPLAB_BACKBONE} "
        assert num_classes > 0, "Number of classes must be greater than 0."

        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.num_classes = num_classes
        self.backbone = backbone
        self.train_loader = train_loader
        self.val_loader = val_loader

        self.model = self.__get_model(pretrained, pretrained_backbone)

        self.criterion = nn.CrossEntropyLoss()

        self.imagenet_stats = [[0.485, 0.456, 0.406], [0.485, 0.456, 0.406]]

    def __get_model(self, pretrained: bool, pretrained_backbone: bool) -> DeepLabV3:

        if self.backbone == 'resnet50':
            model = self.__get_deeplab_resnet50(pretrained, pretrained_backbone)

        elif self.backbone == 'resnet101':
            model = self.__get_deeplab_resnet101(pretrained, pretrained_backbone)

        elif self.backbone == 'mobilenet':
            model = self.__get_deeplab_mobile_net(pretrained, pretrained_backbone)

        return model

    def __get_deeplab_resnet50(self, pretrained: bool, pretrained_backbone: bool) -> DeepLabV3:
        if pretrained:
            model_params = {
                'weights': DeepLabV3_ResNet101_Weights.DEFAULT
            }
            model = deeplabv3_resnet50(**model_params)
            if self.num_classes != 21:
                model.classifier = DeepLabHead(2048, self.num_classes)

        elif pretrained_backbone:
            model_params = {
                'weights_backbone': ResNet50_Weights.DEFAULT
            }
            model = deeplabv3_resnet50(**model_params)
            if self.num_classes != 21:
                model.classifier = DeepLabHead(2048, self.num_classes)

        else:
            model_params = {
                'num_classes': self.num_classes
            }
            model = deeplabv3_resnet50(**model_params)

        return model

    def __get_deeplab_resnet101(self, pretrained, pretrained_backbone) -> DeepLabV3:
        if pretrained:
            model_params = {
                'weights': DeepLabV3_ResNet101_Weights.DEFAULT
            }
            model = deeplabv3_resnet101(**model_params)
            if self.num_classes != 21:
                model.classifier = DeepLabHead(2048, self.num_classes)

        elif pretrained_backbone:
            model_params = {
                'weights_backbone': ResNet101_Weights.DEFAULT
            }
            model = deeplabv3_resnet101(**model_params)
            if self.num_classes != 21:
                model.classifier = DeepLabHead(2048, self.num_classes)

        else:
            model_params = {
                'num_classes': self.num_classes
            }
            model = deeplabv3_resnet101(**model_params)

        return model

    def __get_deeplab_mobile_net(self, pretrained, pretrained_backbone) -> DeepLabV3:
        if pretrained:
            model_params = {
                'weights': DeepLabV3_MobileNet_V3_Large_Weights.DEFAULT
            }
            model = deeplabv3_mobilenet_v3_large(**model_params)
            if self.num_classes != 21:
                model.classifier = DeepLabHead(2048, self.num_classes)

        elif pretrained_backbone:
            model_params = {
                'weights_backbone': MobileNet_V3_Large_Weights.DEFAULT
            }
            model = deeplabv3_mobilenet_v3_large(**model_params)
            if self.num_classes != 21:
                model.classifier = DeepLabHead(2048, self.num_classes)

        else:
            model_params = {
                'num_classes': self.num_classes
            }
            model = deeplabv3_mobilenet_v3_large(**model_params)

        return model

    def forward(self, x):
        self.model.eval()
        return self.model(x)

    def train_dataloader(self):
        return self.train_loader

    def val_dataloader(self):
        return self.val_loader

    def training_step(self, train_batch, batch_idx):
        images, targets = train_batch
        outputs = self.model(images)['out']
        train_loss = self.criterion(outputs, targets['mask'])
        return train_loss

    def training_epoch_end(self, training_step_outputs):
        train_loss_mean = torch.mean(torch.stack(training_step_outputs))
        self.log('training_loss', train_loss_mean.item())

    def validation_step(self, val_batch, batch_idx):
        images, targets = val_batch
        outputs = self.model(images)['out']
        val_loss = self.criterion(outputs, targets['mask'])
        return val_loss

    def validation_epoch_end(self, validation_step_outputs):
        val_loss_mean = torch.mean(torch.stack(validation_step_outputs))
        self.log('training_loss', val_loss_mean.item())

    def image_transform(self, image):
        reprocess = torchvision.transforms.Compose([torchvision.transforms.ToTensor(),
                                                    torchvision.transforms.Normalize(mean=self.imagenet_stats[0],
                                                                                     std=self.imagenet_stats[1])])

        return reprocess(image)

    def predict_step(self, image):
        input_tensor = self.image_transform(image).unsqueeze(0)

        # Make the predictions for labels across the image
        self.model.eval()
        with torch.no_grad():
            output = self.model(input_tensor)["out"][0]
            output = output.argmax(0)

        # Return the predictions

        return output.cpu().numpy()

    def configure_optimizers(self):
        optimizer_params = {
            'params': self.model.parameters(),
            'lr': self.learning_rate,
            'weight_decay': self.weight_decay,
        }
        return torch.optim.Adam(**optimizer_params)
