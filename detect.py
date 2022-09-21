import os
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import yaml
import argparse
from src.models.Detection.Faster_RCNN import Faster_RCNN
import torch
from src.dataset.bdd_detetcion import BDDDetection
from src.config.defaults import cfg
from src.utils.DataLoaders import get_loader
from pytorch_lightning import Trainer
import pandas as pd
import warnings

warnings.filterwarnings("ignore")

COLOR_MAP = ['blue', 'orange', 'green', 'red', 'purple', 'brown', 'pink', 'gray', 'olive', 'cyan', 'blue']
IDX_TO_CLS = {
    0: '__bgr__',
    1: 'pedestrian',
    2: 'car',
    3: 'traffic light',
    4: 'traffic sign',
}


def display_prediction(prediction, save_path):
    boxes = prediction['boxes']
    scores = prediction['scores']
    labels = prediction['labels']
    image = Image.open(image_path)
    fig, ax = plt.subplots()
    ax.imshow(image)
    for idx, score in enumerate(scores):
        bbox = boxes[idx]
        rect = patches.Rectangle((bbox[0], bbox[1]), bbox[2] - bbox[0], bbox[3] - bbox[1],
                                 edgecolor=COLOR_MAP[labels[idx]],
                                 facecolor="none", linewidth=2)
        ax.add_patch(rect)
        plt.text(bbox[0], bbox[1], IDX_TO_CLS[labels[idx]], verticalalignment="top", color="white",
                 bbox={'facecolor': COLOR_MAP[labels[idx]], 'pad': 0})

    plt.axis('off')
    plt.savefig(save_path)
    plt.show()


if __name__ == '__main__':
    # Define the parser
    parser = argparse.ArgumentParser()
    parser.add_argument('--image-path', type=str, help='the path of the image')
    parser.add_argument('--data', type=str, default="./data/fasterrcnn.yaml", help='data.yaml path')
    parser.add_argument('--weights', type=str, default=None, help='train from checkpoint')
    parser.add_argument('--confidence-score', type=float, default=0.5, help='confidence score used to predict')
    parser.add_argument('--model', type=str, default='fasterrcnn', choices=['fasterrcnn', 'deeplab', 'maskrcnn'],
                        help='the model and task you want to perform')
    parser.add_argument('--save-path', type=str, default="predicted_image.jpg",
                        help='Path to save the image with bounding boxes')

    # Fetch the params from the parser
    args = parser.parse_args()

    image_path = args.image_path
    confidence_score = args.confidence_score
    weights = args.weights  # Check point to continue training
    save_path = args.save_path
    model_name = args.model  # the name of the model: fastercnn, maskrcnn, deeplab

    with open(args.data, 'r') as f:
        data = yaml.safe_load(f)  # data from .yaml file

    obj_cls = data['classes']  # the classes we want to work one
    relative_path = data['relative_path']  # relative path to the dataset

    assert os.path.exists(image_path), "This image doesn't exists"

    if model_name == 'fasterrcnn':
        model = Faster_RCNN.load_from_checkpoint(weights)
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print('Using device:', device)

        prediction = model.predict(image_path, confidence_score, device)
        display_prediction(prediction, save_path)
