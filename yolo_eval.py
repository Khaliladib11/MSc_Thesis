import os
import argparse
from utils import yolo_evaluation
from pprint import pprint

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--pred', type=str, default='', help='Path to the predictions folder')
    parser.add_argument('--gt', type=str, default='', help='Path to the ground truth folder')

    # Fetch the params from the parser
    args = parser.parse_args()
    prediction_path = args.pred
    gt_path = args.gt

    assert os.path.exists(prediction_path), f"Predictions does not exists at {prediction_path}"
    assert os.path.exists(gt_path), f"Predictions does not exists at {gt_path}"

    mAP = yolo_evaluation(prediction_path, gt_path)

    pprint(mAP)
