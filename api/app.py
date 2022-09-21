import os, sys
import io

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath('__file__')))
sys.path.append(BASE_DIR)

import torch
from src.models.Detection.Faster_RCNN import Faster_RCNN
from flask import Flask, request, jsonify

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


@app.get('/')
def welcome():
    return "Hello World from Flask"


@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        file = request.files['file']
        img_bytes = file.read()
        if isinstance(img_bytes, (bytes, bytearray)):
            prediction = model.predict(img_bytes, confidence_score, device)
            boxes = prediction['boxes']
            scores = prediction['scores']
            labels = prediction['labels']
        return jsonify({'labels': labels, 'bboxes': boxes, 'scores': scores})


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
