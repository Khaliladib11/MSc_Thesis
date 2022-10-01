# Import Libraries
import os
import io
import random
import time
import sys
import json
from pprint import pprint
import numpy as np
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
from PIL import Image
import cv2
import torch
from torchvision import transforms
from torchmetrics.detection.mean_ap import MeanAveragePrecision
import warnings

warnings.filterwarnings("ignore")

# Colors list
color_map = [[0, 255, 0], [0, 0, 255], [255, 0, 0], [0, 255, 255], [255, 255, 0], [255, 0, 255], [80, 70, 180],
             [250, 80, 190], [245, 145, 50], [70, 150, 250], [50, 190, 190]]


def export_map_json(mAP, json_file_name='mAP.json' , to_save_path='.') -> None:
    """
    Function used to exported the mean average precision results to a json file
    :param mAP: dictionary contains the mAP results
    :param json_file_name: the name of the json file
    :param to_save_path: the path where you want to save the json file
    :return: return None
    """
    with open(os.path.join(to_save_path, json_file_name), "w") as outfile:
        json.dump(mAP, outfile)

    print(f"mAP exported to {os.path.join(to_save_path, json_file_name)}.")



def create_cls_dic(objs) -> dict:
    """
    Function to create dictionary to map from index to class
    :param objs: list of object you want to classify (including background)
    :return: dictionary
    """
    idx_to_cls = {}
    i = 0
    for obj in objs:
        idx_to_cls[i] = obj
        i += 1

    return idx_to_cls


def image_transform(image) -> torch.Tensor:
    """
    Function to apply image transform before feed it to the model
    :param image: image of Image.Image or numpy.array type
    :return: torch.Tensor type
    """
    t = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225], )
    ])
    return t(image)


def detection_predict(model, image, confidence_score=0.5, device=None) -> tuple:
    """
    method to predict the bounding boxes and classes using faster rcnn model
    :param model: the model you want to use (Faster RCNN, Mask RCNN)
    :param image: the image, it can be path to an image or array of bytes
    :param confidence_score: confidence score used to predict
    :param device: the device (gpu, cpu)
    :return: tuple contains dictionary of the prediction and float number
    """

    # check image type
    if isinstance(image, (bytes, bytearray)):
        image = Image.open(io.BytesIO(image))  # byte array

    elif isinstance(image, str):
        assert os.path.exists(image), "This image doesn't exists"
        try:
            image = Image.open(image)  # path to an image
        except IOError:
            print("An exception occurred, make sure you have selected an image type.")
            sys.exit()

    elif isinstance(image, Image.Image) or isinstance(image, np.ndarray):
        # if it is Image.Image or ndarray then pass
        pass
    else:
        # else raise an exception
        raise Exception("You must input: bytes, Image type, path for an image, or numpy array.")
        sys.exit()

    # check device
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    tensor_image = image_transform(np.array(image))  # convert image to tensor and apply some transform
    tensor_image = tensor_image.unsqueeze(0)  # add batch dimension
    model.eval()  # put model on evaluation mode
    tensor_image = tensor_image.to(device)  # move tensor to device
    model = model.to(device)  # move model to device
    start_time = time.time()  # to measure the speed of inference
    prediction = model(tensor_image)  # predict the output
    end_time = (time.time() - start_time)  # time in sec
    fps = 1 / end_time
    prediction = prediction[0]  # remove the batch dim

    # convert the predictions to lists
    boxes = prediction['boxes'].tolist()
    labels = prediction['labels'].tolist()
    scores = prediction['scores'].tolist()

    output = {
        'boxes': [],
        'labels': [],
        'scores': [],
    }

    predict_masks = False
    if 'masks' in prediction.keys():
        masks = (prediction['masks']>0.5).squeeze().detach().cpu().numpy()
        predict_masks = True
        output['masks'] = []

    # Filter the prediction based on the confidence score threshold
    for idx, score in enumerate(scores):
        # check score
        if score > confidence_score:
            output['boxes'].append(boxes[idx])
            output['scores'].append(scores[idx])
            output['labels'].append(labels[idx])
            if predict_masks:
                output['masks'].append(masks[idx])

    return output, fps


def get_coloured_mask(mask):
    """
    random_colour_masks
      parameters:
        - image - predicted masks
      method:
        - the masks of each predicted object is given random colour for visualization
    """
    colours = color_map
    r = np.zeros_like(mask).astype(np.uint8)
    g = np.zeros_like(mask).astype(np.uint8)
    b = np.zeros_like(mask).astype(np.uint8)
    r[mask == 1], g[mask == 1], b[mask == 1] = colours[random.randrange(0, 10)]
    coloured_mask = np.stack([r, g, b], axis=2)
    return coloured_mask


def display(image, prediction, save_path, idx_to_cls, rect_th=2, text_size=0.5, text_th=2):
    """
    method to display image with bounding boxes and masks
    :param image: Image or numpy array image
    :param prediction: the dictionary that contains the preidction
    :param save_path: the path where to save the image with bounding boxes
    :param idx_to_cls: mapping from idx to classes
    :param rect_th: thickness of bounding boxes
    :param text_size: size of text
    :param text_th: thickness of text
    :return:
    """
    if isinstance(image, Image.Image):
        image = cv2.imread(np.array(image))
    if isinstance(image, np.ndarray):
        image = cv2.imread(image)
    elif isinstance(image, str):
        assert os.path.exists(image), "This image doesn't exists"
        try:
            image = cv2.imread(image)
            # image = Image.open(image)
        except IOError:
            print("An exception occurred, make sure you have selected an image type.")
            sys.exit()
    else:
        raise Exception("Prediction must have: boxes, scores and labels.")
        sys.exit()

    if all(label in prediction.keys() for label in ['boxes', 'scores', 'labels']):
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        for idx, box in enumerate(prediction['boxes']):
            if 'masks' in prediction.keys():
                rgb_mask = get_coloured_mask(prediction["masks"][idx])
                image = cv2.addWeighted(image, 1, rgb_mask, 0.5, 0)
            pt1 = (int(box[0]), int(box[1]))
            pt2 = (int(box[2]), int(box[3]))
            cv2.rectangle(image, pt1, pt2, color=color_map[prediction['labels'][idx]], thickness=rect_th)
            cv2.putText(image, idx_to_cls[prediction['labels'][idx]], pt1, cv2.FONT_HERSHEY_SIMPLEX, text_size,
                        color_map[prediction['labels'][idx]],
                        thickness=text_th)

        plt.figure(figsize=(10, 10))
        plt.imshow(image)
        plt.xticks([])
        plt.yticks([])
        plt.savefig(save_path)
        plt.show()
        print(f"Image saved to {save_path}")

    else:
        raise Exception("You must input: Image type, path for an image, or numpy array.")
        sys.exit()


def inference_video(model, video_source, idx_to_cls, confidence_score=0.5, device=None, save_name='video_inference'):
    if isinstance(video_source, str):
        assert os.path.exists(video_source), f"Video does not exist in {video_source}"
    else:
        raise Exception("You must input video path")
        sys.exit()

    # check device
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = model.to(device)
    model.eval()

    cap = cv2.VideoCapture(video_source)
    if not cap.isOpened():
        raise Exception("Error while trying to read video. Please check path again")
        sys.exit()

    idx_to_cls
    cls_to_idx = {}
    for idx in idx_to_cls:
        cls_to_idx[idx_to_cls[idx]] = idx

    COLORS = np.random.uniform(0, 255, size=(len(idx_to_cls), 3))

    # define codec and create VideoWriter object
    out = cv2.VideoWriter(f"inference_outputs/videos/{save_name}.mp4",
                          cv2.VideoWriter_fourcc(*'mp4v'), 30)

    frame_count = 0  # to count total frames
    total_fps = 0  # to get the final frames per second

    # read until end of video
    while cap.isOpened():
        # capture each frame of the video
        ret, frame = cap.read()
        if ret:
            image = frame.copy()
            tensor_image = image_transform(image).to(device)
            tensor_image = tensor_image.unsqueeze(0)
            start_time = time.time()
            with torch.no_grad():
                # get predictions for the current frame
                outputs = model(tensor_image)
            end_time = time.time()
            # get the current fps
            fps = 1 / (end_time - start_time)
            # add `fps` to `total_fps`
            total_fps += fps
            # increment frame count
            frame_count += 1

            # load all detection to CPU for further operations
            outputs = [{k: v.to('cpu') for k, v in t.items()} for t in outputs]
            outputs = outputs[0]
            # carry further only if there are detected boxes
            if len(outputs['boxes']) != 0:
                boxes = outputs['boxes'].data.numpy()
                scores = outputs['scores'].data.numpy()
                # filter out boxes according to `detection_threshold`
                boxes = boxes[scores >= confidence_score].astype(np.int32)
                draw_boxes = boxes.copy()
                # get all the predicited class names
                pred_classes = [idx_to_cls[i] for i in outputs[0]['labels'].cpu().numpy()]
                # draw the bounding boxes and write the class name on top of it
                for j, box in enumerate(draw_boxes):
                    class_name = pred_classes[j]
                    color = COLORS[cls_to_idx[class_name]]
                    cv2.rectangle(frame,
                                  (int(box[0]), int(box[1])),
                                  (int(box[2]), int(box[3])),
                                  color, 2)
                    cv2.putText(frame, class_name,
                                (int(box[0]), int(box[1] - 5)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, color,
                                2, lineType=cv2.LINE_AA)
                cv2.putText(frame, f"{fps:.1f} FPS",
                            (15, 25),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0),
                            2, lineType=cv2.LINE_AA)

                cv2.imshow('image', frame)
                out.write(frame)
                # press `q` to exit
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

        else:
            break

    # release VideoCapture()
    cap.release()
    # close all frames and video windows
    cv2.destroyAllWindows()

    # calculate and print the average FPS
    avg_fps = total_fps / frame_count
    print(f"Average FPS: {avg_fps:.3f}")


# YOLO evaluation

def yolo_to_pascal(x, y, w, h, width=1280, height=720) -> tuple:
    """
    Function to convert YOLO format (xywh) to Pascal voc format (xyxy)
    :param x: float number, x center
    :param y: float number, y center
    :param w: float width of the bounding box
    :param h: float height of the bounding box
    :param width: float width of the image
    :param height: float width of the image
    :return: tuple contains xmin, ymin, xmax, ymax
    """
    xmax = int((x * width) + (w * width) / 2.0)
    xmin = int((x * width) - (w * width) / 2.0)
    ymax = int((y * height) + (h * height) / 2.0)
    ymin = int((y * height) - (h * height) / 2.0)
    return (xmin, ymin, xmax, ymax)


def prepare_gt(file_path) -> dict:
    """
    Function to prepare th ground truth annotation from the file to use them in evaluation
    :param file_path: the path for the ground truth annotations
    :return: dictionary contains the bounding boxes and labels
    """
    output = {
        "boxes": [],
        "labels": [],
    }
    with open(file_path, 'r') as f:
        for line in f:
            line = line.split()
            label = int(line[0])
            box = [float(b) for b in line[1:]]
            xmin, ymin, xmax, ymax = yolo_to_pascal(box[0], box[1], box[2], box[3])
            output['boxes'].append([xmin, ymin, xmax, ymax])
            output['labels'].append(label)

    output['boxes'] = torch.tensor(output['boxes'])
    output['labels'] = torch.tensor(output['labels'])

    return output


def prepare_prediction(file_path) -> dict:
    """
    Function to prepare the predicted bounding boxes to use them in evaluation
    :param file_path: the path for the prediction files
    :return: dictionary contains the bounding boxes, scores and labels
    """
    output = {
        "boxes": [],
        "scores": [],
        "labels": [],
    }
    with open(file_path, 'r') as f:
        for line in f:
            line = line.split()
            label = int(line[0])
            box = [float(b) for b in line[1:-1]]
            score = float(line[-1])
            xmin, ymin, xmax, ymax = yolo_to_pascal(box[0], box[1], box[2], box[3])
            output['boxes'].append([xmin, ymin, xmax, ymax])
            output['scores'].append(score)
            output['labels'].append(label)

    output['boxes'] = torch.tensor(output['boxes'])
    output['scores'] = torch.tensor(output['scores'])
    output['labels'] = torch.tensor(output['labels'])
    return output


def yolo_evaluation(prediction_path, gt_path) -> dict:
    """
    Function to evaluate the performance of the YOLO algorithm
    :param prediction_path: the path for the prediction folder that contains the prediction files
    :param gt_path: the path for the ground truth folder that contains the ground truth files
    :return: dictionary contains the mean average precision
    """
    assert os.path.exists(prediction_path), f"{prediction_path} folder not found"
    assert os.path.exists(gt_path), f"{gt_path} folder not found"

    pred_files = os.listdir(prediction_path)

    metric = MeanAveragePrecision(box_format='xyxy', class_metrics=True)

    for file in tqdm(pred_files):
        pre = prepare_prediction(os.path.join(prediction_path, file))
        gt = prepare_gt(os.path.join(gt_path, file))
        metric.update([pre], [gt])

    mAP = metric.compute()

    return mAP
