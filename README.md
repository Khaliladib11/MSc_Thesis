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
│ │ ├─DataLoaders.py # some useful functions
├─doc
│ ├─images # some images
├─notebooks
│ ├─Faster RCNN Notebook.ipynb # Faster RCNN notebook
│ ├─FCN Notebook.ipynb # FCN notebook
```
