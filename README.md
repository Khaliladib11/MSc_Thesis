# MSc in Artificial Intelligence Thesis

![teaser](./doc/images/BDD100K.png)
 
 This repository contains the code for my Master thesis. I will cover different tasks in computer vision field, like object detection, semantic segmentation, instance segmentation and panoptic segmentation. I will be using different SOTA models.
 
 The main goal of this work is to compare the single task vs multi task learning models, as well as compare multi task models to YOLOP model. YOLOP model is a model that performs Object Detection and Semantic segmentation at the time.
 
 ---
 
 ## Dataset
 Dataset used is BDD100K which is available [here](https://www.bdd100k.com/).
 
 ---
 
 ## Project Structure
```python
├─src
│ ├─config
│ │ ├─defaults.py # default configuration
│ ├─dataset
│ │ ├─bdd.py  # Superclass dataset，Parent class
│ │ ├─bdd_detetcion.py # Subclass for detection task
│ │ ├─bdd_drivable_segmentation.py # Subclass for drivabel area segmetation task
│ │ ├─bdd_panoptic.py # Subclass for Panopric segmetation task
│ │ ├─bdd_segmentation.py # Subclass for semantic segmentation segmetation task
│ ├─dbs
│ │ ├─test_db.json # pre-created test db
│ │ ├─train_db.json # pre-created train db
│ │ ├─val_db.json # pre-created val db
│ ├─models
│ │ ├─Detection
│ │ │ ├─Faster_RCNN.py # Faster RCNN class
│ │ ├─Segmentation
│ │ │ ├─FCN.py # FCN class
│ │ │ ├─DeepLab.py # DeepLabv3+ class
│ ├─utils
│ │ ├─DataLoaders.py # dataloader function
| | ├─utils.py # useful function
| | ├─Data Augmentation.ipynb # Notebook contains tutorial for data augmentation
│ ├─BDD Detection.ipynb # Notebook contains tutorial how to use detection class
│ ├─BDD Drivable Area.ipynb # Notebook contains tutorial how to use drivable class
│ ├─BDD Instance Segmentation.ipynb # Notebook contains tutorial how to use instance segmentation class
│ ├─YOLO Format.ipynb # Notebook contains tutorial how convert xyxy format to yolo format
├─doc
│ ├─images # some images
├─notebooks
│ ├─Faster RCNN Notebook.ipynb # Faster RCNN notebook
│ ├─FCN Notebook.ipynb # FCN notebook
├─dataset
│ ├─bdd100k
│ │ ├─images
│ │ │ ├─10K
│ │ │ ├─100K
│ │ ├─labels
│ │ │ ├─det_20
│ │ │ ├─drivable
│ │ │ ├─lane
│ │ │ ├─pan_seg
│ │ │ ├─ins_seg
│ │ │ ├─pan_seg
```

---
## Train
### Object Detection
1- To train using **Faster RCNN** please take a look at the [Faster RCNN](https://github.com/Khaliladib11/MSc_Thesis/blob/main/notebooks/Faster%20RCNN%20Notebook.ipynb).

2- To train using yolo you have to follow these steps:
 - Clone the yolov5 in outside the folder of this project
```
git clone https://github.com/ultralytics/yolov5
```
 - Install all dependencies (recommadation: Create new virtual envirement when working with yolov5):
   ```
   cd ../yolov5
   pip install -r requirements.txt
   ```
 - Prepare the annotation to the YOLO format. Check the [YOLO notebook](https://github.com/Khaliladib11/MSc_Thesis/blob/main/src/YOLO%20Format.ipynb) for more details.
 
 - train the model:
    ```
   !python train.py --img 640 --cfg yolov5m.yaml --batch 1 --epochs 30 --data dataset.yaml --weights yolov5m.pt --name yolo_bdd
   ```
   For more information follow this [Tutorials](https://github.com/ultralytics/yolov5/wiki/Train-Custom-Data).
---
