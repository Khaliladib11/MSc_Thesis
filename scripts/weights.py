import gdown

ids = {
    'fasterrcnn_weights.pt': '1Rj1xVxvDOjzYGexVejIR7-dsUIA8gKX-',
    'maskrcnn_weights.pt': '1CEXLdTieO4u_UGy5baVceqPRO7tgeokQ',
    'yolov5s_weights.pt': '1xoQKsDEV2yeca6pr-lJdSDY2NNgwJsZb',
    'yolov5l_weights.pt': '1kyd3c57Hj5WUjdCUPWHHgad_-A-g6-VF',
    'yolov5x_weights.pt': '1L9lP8Rv15rro2T6aOsjlu1yLNCMZXd5p',
    'yolov7_weights.pt': '1Fc4bq6FAaseU2GSKqgIqZ42dQTbpRcJU',
    'yolov7x_weights.pt': '1HcuVK4o78wrkwh63-YqdiG-CkUz9NMM8',
}

for idx in ids:
    gdown.download(id=ids[idx], output=idx, quiet=False)
