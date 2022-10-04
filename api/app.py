import os, sys, io
from PIL import Image
import matplotlib.pyplot as plt
import torch

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath('__file__')))
sys.path.append(BASE_DIR)

import torch
from src.models.Detection.Faster_RCNN import Faster_RCNN
from flask import Flask, request, jsonify, abort

app = Flask(__name__)

COLOR_MAP = ['blue', 'orange', 'green', 'red', 'purple', 'brown', 'pink', 'gray', 'olive', 'cyan', 'blue']

IDX_TO_CLS = {
    0: 'pedestrian',
    1: 'car',
    2: 'traffic light',
    3: 'traffic sign',
}

weights = {
    'fasterrcnn': '../weights/Faster RCNN/fasterrcnn/epoch=5-step=191964.ckpt',
    'yolov5s': '../weights/Yolov5/yolov5_train/yolo5s_bdd/weights/best.pt',
    'yolov5l': '../weights/Yolov5/yolov5_train/yolo5l_bdd/weights/best.pt',
    'yolov5x': '../weights/Yolov5/yolov5_train/yolo5x_bdd/weights/best.pt',
    'yolov7': '',
    'yolov7x': ''
}

path_to_yolov5 = '../../Training/yolov5'
path_to_yolov7 = '../../Training/yolov7'


try:
    fasterrcnn_model = Faster_RCNN.load_from_checkpoint(weights['fasterrcnn'])
    yolov5s = torch.hub.load(path_to_yolov5, 'custom', path=weights['yolov5s'], source='local')
    yolov5l = torch.hub.load(path_to_yolov5, 'custom', path=weights['yolov5l'], source='local')
    yolov5x = torch.hub.load(path_to_yolov5, 'custom', path=weights['yolov5x'], source='local')
    yolov7 = torch.hub.load(path_to_yolov7, 'custom', path_or_model=weights['yolov7'], source='local', force_reload=True)
    yolov7x = torch.hub.load(path_to_yolov7, 'custom', path_or_model=weights['yolov7x'], source='local', force_reload=True)

except Exception as e:
    print("There is problem with loading models")
    sys.exit(1)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

available_models = ['yolov5s', 'yolov5l', 'yolov5x', 'yolov7', 'yolov7x', 'fasterrcnn']
acceptable_file_format = ['jpg', 'jpeg', 'png']


def get_prediction(img_bytes, score, model_name) -> tuple:
    """
    function to predict from image byte
    :param img_bytes: image as byte array
    :param score: the confidence score we want to use
    :param model_name: the model name we want to use
    :return: tuple of three arrays
    """
    boxes = []  # array of boxes
    scores = []  # array of scores
    labels = []  # array of labels

    # Faster RCNN model
    if model_name == 'fasterrcnn':
        prediction = fasterrcnn_model.predict(img_bytes, score, device)  # predict using Faster RCNN model
        boxes = prediction['boxes']  # get the bounding boxes
        scores = prediction['scores']  # get the scores
        labels = [IDX_TO_CLS[label - 1] for label in prediction['labels']]  # convert the idx to labels
        # return the arrays
        return boxes, scores, labels

    # yolov5s model
    elif model_name == 'yolov5s':
        image = Image.open(io.BytesIO(img_bytes))  # open the image as Image.Image type
        yolov5s.conf = score  # change the confidence score
        df = yolov5s(image).pandas().xyxy[0]  # inference


    # yolov5l model
    elif model_name == 'yolov5l':
        image = Image.open(io.BytesIO(img_bytes))
        yolov5l.conf = score
        df = yolov5l(image).pandas().xyxy[0]  # inference

    # yolov5l model
    elif model_name == 'yolov5x':
        image = Image.open(io.BytesIO(img_bytes))
        yolov5x.conf = score
        df = yolov5x(image).pandas().xyxy[0]  # inference

    elif model_name == 'yolov5':
        image = Image.open(io.BytesIO(img_bytes))
        yolov7.conf = score
        df = yolov7(image).pandas().xyxy[0]  # inference

    elif model_name == 'yolov7x':
        image = Image.open(io.BytesIO(img_bytes))
        yolov7x.conf = score
        df = yolov7x(image).pandas().xyxy[0]  # inference

    # loop through the dataframe to get the predictions
    # yolov5 models return the data as pandas dataframe
    for index, row in df.iterrows():
        boxes.append([row['xmin'], row['ymin'], row['xmax'], row['ymax']])
        scores.append(row['confidence'])
        labels.append(row['name'])

    return boxes, scores, labels


@app.get('/')
def welcome():
    return "Hello World from Flask"


# prediction REST API
@app.route('/api/predictions', methods=['POST'])
def predict():
    # check the type of the request
    if request.method == 'POST':
        file = request.files['file']  # get the file
        model_name = request.form.get('model')  # get the model name
        confidence = request.form.get('confidence')  # get the confidence score

        # validate the file
        if not file:
            return jsonify({"error": "the 'file' field is None. Please provide a file with the request"}), 400

        # validate the model name
        elif not model_name:
            return jsonify({"error": "the 'model' field is None. Please provide a model with the request"}), 400

        # validate the confidence
        elif not confidence:
            return jsonify(
                {"error": "the 'confidence' field is None. Please provide a confidence score with the request"}), 400

        # convert the confidence to float type
        try:
            confidence = float(confidence)
        except TypeError:
            return jsonify({"error": "please provide a float number as confidence score between 0.0 and 1.0"}), 400

        # check if the value of confidence is between 0 and 1
        if 0.0 > confidence or confidence > 1.0:
            return jsonify({"error": "please provide a confidence score between 0.0 and 1.0"}), 400

        # check if the name of the model is available in the available_models list
        if model_name not in available_models:
            return jsonify(
                {"error": f"{model_name} not supported. please choose a model from {available_models}."}), 400

        # check the type of the file: jpg, png, jpeg
        if file.content_type.split('/')[-1] not in acceptable_file_format:
            return jsonify(
                {"error": f"The file type most be image. We only accept {acceptable_file_format} types."}), 400

        img_bytes = file.read()  # read the file as bytes

        # get predictions
        if isinstance(img_bytes, (bytes, bytearray)):
            boxes, scores, labels = get_prediction(img_bytes, confidence, model_name)
            if len(boxes) == 0 or len(scores) == 0 or len(labels) == 0:
                return jsonify({"message": "no objects detected in this image. Try to change the confidence score"})

            elif len(boxes) != len(scores) or len(scores) != len(labels) or len(boxes) != len(labels):
                return jsonify({"msg": "Something Wrong"}), 500

            return jsonify({'labels': labels, 'bboxes': boxes, 'scores': scores}), 200
        else:
            return jsonify({"msg": "Please Upload an image"}), 400
    else:
        abort(400)


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
