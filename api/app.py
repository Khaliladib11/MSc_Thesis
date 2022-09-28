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
    'yolov5x': '../weights/Yolov5/yolov5_train/yolo5x_bdd/weights/best.pt'
}

path_to_yolo = '../../Training/yolov5'

fasterrcnn_model = Faster_RCNN.load_from_checkpoint(weights['fasterrcnn'])
yolov5s, yolov5l, yolov5x = None, None, None
#yolov5s = torch.hub.load(path_to_yolo, 'custom', path=weights['yolov5s'], source='local')
#yolov5l = torch.hub.load(path_to_yolo, 'custom', path=weights['yolov5l'], source='local')
#yolov5x = torch.hub.load(path_to_yolo, 'custom', path=weights['yolov5x'], source='local')

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

available_models = ['yolov5s', 'yolov5l', 'yolov5x', 'fasterrcnn']
acceptable_file_format = ['jpg', 'jpeg', 'png']


def get_prediction(img_bytes, score, model_name):
    boxes = []
    scores = []
    labels = []

    if model_name == 'fasterrcnn':
        prediction = fasterrcnn_model.predict(img_bytes, score, device)
        boxes = prediction['boxes']
        scores = prediction['scores']
        labels = [IDX_TO_CLS[label - 1] for label in prediction['labels']]
        return boxes, scores, labels

    elif model_name == 'yolov5s':
        image = Image.open(io.BytesIO(img_bytes))
        yolov5s.conf = score
        df = yolov5s(image).pandas().xyxy[0]  # inference

    elif model_name == 'yolov5l':
        image = Image.open(io.BytesIO(img_bytes))
        yolov5l.conf = score
        df = yolov5l(image).pandas().xyxy[0]  # inference

    elif model_name == 'yolov5x':
        image = Image.open(io.BytesIO(img_bytes))
        yolov5x.conf = score
        df = yolov5x(image).pandas().xyxy[0]  # inference

    for index, row in df.iterrows():
        boxes.append([row['xmin'], row['ymin'], row['xmax'], row['ymax']])
        scores.append(row['confidence'])
        labels.append(row['name'])

    return boxes, scores, labels


@app.get('/')
def welcome():
    return "Hello World from Flask"


@app.route('/api/predictions', methods=['POST'])
def predict():
    if request.method == 'POST':
        file = request.files['file']
        model_name = request.form.get('model')
        confidence = request.form.get('confidence')
        if not file:
            return jsonify({"error": "the 'file' field is None. Please provide a file with the request"}), 400
        elif not model_name:
            return jsonify({"error": "the 'model' field is None. Please provide a model with the request"}), 400
        elif not confidence:
            return jsonify(
                {"error": "the 'confidence' field is None. Please provide a confidence score with the request"}), 400

        try:
            confidence = float(confidence)
        except TypeError:
            return jsonify({"error": "please provide a float number as confidence score between 0.0 and 1.0"}), 400

        if 0.0 > confidence or confidence > 1.0:
            return jsonify({"error": "please provide a confidence score between 0.0 and 1.0"}), 400

        if model_name not in available_models:
            return jsonify(
                {"error": f"{model_name} not supported. please choose a model from {available_models}."}), 400

        if file.content_type.split('/')[-1] not in acceptable_file_format:
            return jsonify(
                {"error": f"The file type most be image. We only accept {acceptable_file_format} types."}), 400

        img_bytes = file.read()

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
