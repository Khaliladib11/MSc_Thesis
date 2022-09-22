import os, sys

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath('__file__')))
sys.path.append(BASE_DIR)

import torch
from src.models.Detection.Faster_RCNN import Faster_RCNN
from flask import Flask, request, jsonify, abort

app = Flask(__name__)

COLOR_MAP = ['blue', 'orange', 'green', 'red', 'purple', 'brown', 'pink', 'gray', 'olive', 'cyan', 'blue']

IDX_TO_CLS = {
    0: '__bgr__',
    1: 'pedestrian',
    2: 'car',
    3: 'traffic light',
    4: 'traffic sign',
}

weights = '../weights/Faster RCNN/fasterrcnn/epoch=5-step=191964.ckpt'
model = Faster_RCNN.load_from_checkpoint(weights)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
confidence_score = 0.5


def get_prediction(img_bytes, score):
    prediction = model.predict(img_bytes, score, device)
    boxes = prediction['boxes']
    scores = prediction['scores']
    labels = [IDX_TO_CLS[label] for label in prediction['labels']]

    return boxes, scores, labels


@app.get('/')
def welcome():
    return "Hello World from Flask"


@app.route('/predict/fasterrcnn/', methods=['POST'])
def predict():
    if request.method == 'POST':
        file = request.files['file']
        img_bytes = file.read()
        if isinstance(img_bytes, (bytes, bytearray)):
            boxes, scores, labels = get_prediction(img_bytes, confidence_score)
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
