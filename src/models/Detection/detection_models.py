from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor, fasterrcnn_resnet50_fpn, \
    fasterrcnn_resnet50_fpn_v2, fasterrcnn_mobilenet_v3_large_fpn
from torchvision.models.detection import FasterRCNN_ResNet50_FPN_V2_Weights, FasterRCNN_MobileNet_V3_Large_FPN_Weights
from torchvision.models import ResNet50_Weights, MobileNet_V3_Large_Weights


def get_fasterrcnn(num_classes: int = 91,
                   backbone: str = 'resnet50',
                   pretrained: bool = True,
                   pretrained_backbone: bool = True) -> FasterRCNN:
    """
    function to return the Faster RCNN model [https://arxiv.org/abs/1506.01497]
    :param num_classes: the number of classes (plus background)
    :param backbone: the backbone we want to use
    :param pretrained: if we want to use pretrained weights or not
    :param pretrained_backbone: if we want to use pretrained backbone or not
    :return: FasterRCNN instance
    """

    # Resnet50 as backbone
    if backbone == 'resnet50':
        # if pretrained weights
        if pretrained:
            params_ = {
                'weights': FasterRCNN_ResNet50_FPN_V2_Weights.DEFAULT,
                'weights_backbone': ResNet50_Weights.DEFAULT
            }
            model = fasterrcnn_resnet50_fpn_v2(**params_)

        # if pretrained backbone
        elif pretrained_backbone:
            params_ = {
                'weights_backbone': ResNet50_Weights.DEFAULT
            }
            model = fasterrcnn_resnet50_fpn_v2(**params_)

        # training from scratch
        else:
            model = fasterrcnn_resnet50_fpn_v2()

    # MobileNet backbone
    elif backbone == 'MobileNetV3-Large':
        # if pretrained weights
        if pretrained:
            params_ = {
                'weights': FasterRCNN_MobileNet_V3_Large_FPN_Weights.DEFAULT,
                'weights_backbone': MobileNet_V3_Large_Weights.DEFAULT
            }
            model = fasterrcnn_mobilenet_v3_large_fpn(**params_)

        # if pretrained backbone
        elif pretrained_backbone:
            params_ = {
                'weights_backbone': MobileNet_V3_Large_Weights.DEFAULT
            }
            model = fasterrcnn_mobilenet_v3_large_fpn(**params_)

        # training from scratch
        else:
            model = fasterrcnn_mobilenet_v3_large_fpn()

    # if the number of classes different from 91 (the number of classes of MSCOCO)
    # in case we want to use pretrained weights
    if num_classes != 91:
        in_features = model.roi_heads.box_predictor.cls_score.in_features
        model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    return model
