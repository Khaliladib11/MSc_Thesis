# Import Libraries

import torch
import torchvision

from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor, fasterrcnn_resnet50_fpn, \
    fasterrcnn_resnet50_fpn_v2, fasterrcnn_mobilenet_v3_large_fpn
from torchvision.models.detection import FasterRCNN_ResNet50_FPN_V2_Weights, FasterRCNN_MobileNet_V3_Large_FPN_Weights
from torchvision.models import ResNet50_Weights, MobileNet_V3_Large_Weights


############################################################
#  Detection Models
############################################################

def get_faster_rcnn(backbone: str = 'resnet50',
                    pretrained: bool = True,
                    pretrained_backbone: bool = True,
                    num_classes: int = 91
                    ) -> FasterRCNN:
    """
    Function to return the Faster RCNN model
    :param backbone: string to select the backbone
    :param pretrained: bool params to specify if we will be using pretrained network
    :param pretrained_backbone: bool params to specify if we will be using pretrained backbone
    :param num_classes: the number of classes we want to detect (plus background)
    :return: Model
    """

    model = None

    if backbone == 'resnet50':
        if pretrained:
            params_ = {
                'weights': FasterRCNN_ResNet50_FPN_V2_Weights.DEFAULT,
                'weights_backbone': ResNet50_Weights.DEFAULT
            }
            model = fasterrcnn_resnet50_fpn_v2(**params_)

        elif pretrained_backbone:
            params_ = {
                'weights_backbone': ResNet50_Weights.DEFAULT
            }
            model = fasterrcnn_resnet50_fpn_v2(**params_)
        else:
            model = fasterrcnn_resnet50_fpn_v2()

        in_features = model.roi_heads.box_predictor.cls_score.in_features
        model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    elif backbone == 'MobileNetV3-Large':
        if pretrained:
            params_ = {
                'weights': FasterRCNN_MobileNet_V3_Large_FPN_Weights.DEFAULT,
                'weights_backbone': MobileNet_V3_Large_Weights.DEFAULT
            }
            model = fasterrcnn_mobilenet_v3_large_fpn(**params_)

        elif pretrained_backbone:
            params_ = {
                'weights_backbone': MobileNet_V3_Large_Weights.DEFAULT
            }
            model = fasterrcnn_mobilenet_v3_large_fpn(**params_)
        else:
            model = fasterrcnn_mobilenet_v3_large_fpn()

        in_features = model.roi_heads.box_predictor.cls_score.in_features
        model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    return model
