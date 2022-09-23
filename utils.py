import os
import io
import random
import time

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import cv2
import torch
from torchvision import transforms
import warnings

warnings.filterwarnings("ignore")


def create_cls_dic(objs) -> dict:
    idx_to_cls = {}
    i = 0
    for obj in objs:
        idx_to_cls[i] = obj
        i += 1

    return idx_to_cls


def image_transform(image):
    t = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225], )
    ])
    return t(image)

def detection_predict(model, image, confidence_score=0.5, device=None) -> dict:
    """
    method to predict the bounding boxes and classes using faster rcnn model
    :param model: the faster rcnn model
    :param image: the image, it can be path to an image or array of bytes
    :param confidence_score: confidence score used to predict
    :param device: the device (gpu, cpu)
    :return: dict()
    """
    if isinstance(image, (bytes, bytearray)):
        image = Image.open(io.BytesIO(image))
    elif isinstance(image, str):
        assert os.path.exists(image), "This image doesn't exists"
        try:
            image = Image.open(image)
            # do stuff
        except IOError:
            print("An exception occurred, make sure you have selected an image type.")
    elif isinstance(image, Image) or isinstance(image, np):
        pass
    else:
        raise Exception("You must input: bytes, Image type, path for an image, or numpy array.")

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
        masks = prediction['masks'].tolist()
        predict_masks = True
        output['masks'] = []

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
    colours = [[0, 255, 0], [0, 0, 255], [255, 0, 0], [0, 255, 255], [255, 255, 0], [255, 0, 255], [80, 70, 180],
               [250, 80, 190], [245, 145, 50], [70, 150, 250], [50, 190, 190]]
    r = np.zeros_like(mask).astype(np.uint8)
    g = np.zeros_like(mask).astype(np.uint8)
    b = np.zeros_like(mask).astype(np.uint8)
    r[mask == 1], g[mask == 1], b[mask == 1] = colours[random.randrange(0, 10)]
    coloured_mask = np.stack([r, g, b], axis=2)
    return coloured_mask


color_map = [[0, 255, 0], [0, 0, 255], [255, 0, 0], [0, 255, 255], [255, 255, 0], [255, 0, 255], [80, 70, 180],
             [250, 80, 190], [245, 145, 50], [70, 150, 250], [50, 190, 190]]


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
    else:
        raise Exception("Prediction must have: boxes, scores and labels.")

    if all(label in prediction.keys() for label in ['boxes', 'scores', 'labels']):
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        for idx, box in enumerate(prediction['boxes']):
            if 'masks' in prediction.keys():
                rgb_mask = get_coloured_mask(prediction["masks"][idx])
                img = cv2.addWeighted(img, 1, rgb_mask, 0.5, 0)
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

    else:
        raise Exception("You must input: Image type, path for an image, or numpy array.")


def inference_video(model, video_source, idx_to_cls, confidence_score=0.5, device=None, save_name='video_inference'):
    if isinstance(video_source, str):
        assert os.path.exists(video_source), f"Video does not exist in {video_source}"
    else:
        raise Exception("You must input video path")

    # check device
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = model.to(device)
    model.eval()

    cap = cv2.VideoCapture(video_source)
    if not cap.isOpened():
        raise Exception("Error while trying to read video. Please check path again")

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
