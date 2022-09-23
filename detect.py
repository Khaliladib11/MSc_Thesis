from src.models.Detection.Faster_RCNN import Faster_RCNN
from src.models.Segmentation.MaskRCNN import Mask_RCNN
from utils import *
import warnings
import argparse
import yaml

warnings.filterwarnings("ignore")

COLOR_MAP = ['blue', 'orange', 'green', 'red', 'purple', 'brown', 'pink', 'gray', 'olive', 'cyan', 'blue']


if __name__ == '__main__':
    # Define the parser
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', type=str, help='the path of the image or video')
    parser.add_argument('--data', type=str, default="./data/fasterrcnn.yaml", help='data.yaml path')
    parser.add_argument('--weights', type=str, default=None, help='train from checkpoint')
    parser.add_argument('--confidence-score', type=float, default=0.5, help='confidence score used to predict')
    parser.add_argument('--model', type=str, default='fasterrcnn', choices=['fasterrcnn', 'deeplab', 'maskrcnn'],
                        help='the model and task you want to perform')
    parser.add_argument('--save-path', type=str, default="predicted_image.jpg",
                        help='Path to save the image with bounding boxes')

    # Fetch the params from the parser
    args = parser.parse_args()

    source = args.source
    confidence_score = args.confidence_score
    weights = args.weights  # Check point to continue training
    save_path = args.save_path
    model_name = args.model  # the name of the model: fastercnn, maskrcnn, deeplab

    with open(args.data, 'r') as f:
        data = yaml.safe_load(f)  # data from .yaml file

    obj_cls = data['classes']  # the classes we want to work one
    relative_path = data['relative_path']  # relative path to the dataset

    # create idx and cls dict
    idx_to_cls = create_cls_dic(obj_cls)

    # check source path
    assert os.path.exists(source), "This image doesn't exists"

    # Load model
    if model_name == 'fasterrcnn':
        model = Faster_RCNN.load_from_checkpoint(weights)  # Faster RCNN
    elif model_name == 'maskrcnn':
        model = Mask_RCNN.load_from_checkpoint(weights)  # Mask RCNN

    # get prediction
    output, fps = detection_predict(model=model, image=source, confidence_score=confidence_score)
    print(f"Frame per Second: {fps}")  # speed
    display(image=source, prediction=output, save_path=save_path, idx_to_cls=idx_to_cls)  # display result and save it
