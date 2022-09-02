from torchvision.models.segmentation import DeepLabV3
from torchvision.models.segmentation import deeplabv3_resnet50, deeplabv3_resnet101, deeplabv3_mobilenet_v3_large
from torchvision.models.segmentation import DeepLabV3_ResNet50_Weights, DeepLabV3_ResNet101_Weights, \
    DeepLabV3_MobileNet_V3_Large_Weights
from torchvision.models import ResNet50_Weights, ResNet101_Weights, MobileNet_V3_Large_Weights
from torchvision.models.segmentation.deeplabv3 import DeepLabHead


def get_deeplab(num_classes: int = 21,
                backbone: str = 'resnet50',
                pretrained: bool = True,
                pretrained_backbone: bool = True) -> DeepLabV3:
    """
    function to return the Deeplabv3+ model [https://arxiv.org/abs/1706.05587]
    :param num_classes: the number of classes (plus background)
    :param backbone: the backbone we want to use
    :param pretrained: if we want to use pretrained weights or not
    :param pretrained_backbone: if we want to use pretrained backbone or not
    :return: DeepLabV3 instance
    """

    # Resnet50 as backbone
    if backbone == 'resnet50':
        # if pretrained weights
        if pretrained:
            model_params = {
                'weights': DeepLabV3_ResNet50_Weights.DEFAULT
            }
            model = deeplabv3_resnet50(**model_params)

        # if pretrained backbone
        elif pretrained_backbone:
            model_params = {
                'weights_backbone': ResNet50_Weights.DEFAULT
            }
            model = deeplabv3_resnet50(**model_params)
        # training from scratch
        else:
            model = deeplabv3_resnet50()

    # Resnet101 as backbone
    if backbone == 'resnet101':
        # if pretrained weights
        if pretrained:
            model_params = {
                'weights': DeepLabV3_ResNet101_Weights.DEFAULT
            }
            model = deeplabv3_resnet101(**model_params)

        # if pretrained backbone
        elif pretrained_backbone:
            model_params = {
                'weights_backbone': ResNet101_Weights.DEFAULT
            }
            model = deeplabv3_resnet101(**model_params)
        # training from scratch
        else:
            model = deeplabv3_resnet101()

    # MobileNet backbone
    if backbone == 'mobilenet':
        # if pretrained weights
        if pretrained:
            model_params = {
                'weights': DeepLabV3_MobileNet_V3_Large_Weights.DEFAULT
            }
            model = deeplabv3_mobilenet_v3_large(**model_params)

        # if pretrained backbone
        elif pretrained_backbone:
            model_params = {
                'weights_backbone': MobileNet_V3_Large_Weights.DEFAULT
            }
            model = deeplabv3_mobilenet_v3_large(**model_params)
        # training from scratch
        else:
            model = deeplabv3_mobilenet_v3_large()

    # if the number of classes different from 91 (the number of classes of MSCOCO)
    # in case we want to use pretrained weights
    if num_classes != 21:
        model.classifier = DeepLabHead(2048, num_classes)

    return model
