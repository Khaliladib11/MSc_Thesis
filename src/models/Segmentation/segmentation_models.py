from torchvision.models.segmentation import DeepLabV3
from torchvision.models.segmentation import deeplabv3_resnet50, deeplabv3_resnet101, deeplabv3_mobilenet_v3_large
from torchvision.models.segmentation import DeepLabV3_ResNet50_Weights, DeepLabV3_ResNet101_Weights, \
    DeepLabV3_MobileNet_V3_Large_Weights
from torchvision.models import ResNet50_Weights, ResNet101_Weights, MobileNet_V3_Large_Weights
from torchvision.models.segmentation.deeplabv3 import DeepLabHead


from torchvision.models.detection import MaskRCNN
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
from torchvision.models.detection import MaskRCNN
from torchvision.models.detection import maskrcnn_resnet50_fpn, maskrcnn_resnet50_fpn_v2
from torchvision.models.detection import MaskRCNN_ResNet50_FPN_Weights, MaskRCNN_ResNet50_FPN_V2_Weights
from torchvision.models.resnet import ResNet50_Weights


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


def get_maskrcnn(version: str = 'v2',
                 pretrained: bool = True,
                 pretrained_weights: bool = True,
                 num_classes: int = 91) -> MaskRCNN:
    """
    function to return the Mask RCNN model
    :param version: which version we want to use
    :param pretrained: boolean value, if we want to use pretrained weights on COCO dataset
    :param pretrained_weights: boolean value, if we want to use pretrained backbone weights
    :param num_classes: the number of classes we want to classify (Note: classes + background)
    :return:
    """

    assert version in ['v1', 'v2'], 'You have to choose which version you want to use:\n ' \
                                    'v1 is Mask-RCNN with Resnet50 backbone and FPN network \n' \
                                    'v2 is improved version of Mask-RCNN with vision transformer.'

    if version == 'v1':
        if pretrained:
            model = maskrcnn_resnet50_fpn(weights=MaskRCNN_ResNet50_FPN_Weights.DEFAULT)
        elif pretrained_weights:
            model = maskrcnn_resnet50_fpn(weights_backbone=ResNet50_Weights.DEFAULT)

        else:
            model = maskrcnn_resnet50_fpn()

        if num_classes != 91:
            # get the number of input features for the classifier
            in_features = model.roi_heads.box_predictor.cls_score.in_features
            # now get the number of input features for the mask classifier
            in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
            hidden_layer = 256
            # replace the pre-trained head with a new one
            model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
            model.roi_heads.mask_predictor = MaskRCNNPredictor(in_features_mask,
                                                               hidden_layer,
                                                               num_classes)

    elif version == 'v2':
        if pretrained:
            model = maskrcnn_resnet50_fpn_v2(weights=MaskRCNN_ResNet50_FPN_V2_Weights.DEFAULT)
        elif pretrained_weights:
            model = maskrcnn_resnet50_fpn_v2(weights_backbone=ResNet50_Weights.DEFAULT)

        else:
            model = maskrcnn_resnet50_fpn_v2()

        if num_classes != 91:
            # get the number of input features for the classifier
            in_features = model.roi_heads.box_predictor.cls_score.in_features
            # now get the number of input features for the mask classifier
            in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
            hidden_layer = 256
            # replace the pre-trained head with a new one
            model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
            model.roi_heads.mask_predictor = MaskRCNNPredictor(in_features_mask,
                                                               hidden_layer,
                                                               num_classes)

    return model
