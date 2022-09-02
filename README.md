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
## Requirements

Please install requirements using: 
```bash
pip install -R requirements.txt
```

---
## Train
### Object Detection
#### Faster RCNN
To train F**aster RCNN** model run the following command:
```bash
python train.py --model fasterrcnn  --data './data/fasterrcnn.yaml' --batch-size 1 --img-size 640 --total_epochs 20
```
Where `model` to specify that we want **Faster RCNN** model, `batch-size` for batch size, `img-size` for resizing the images and finally `--total_epochs` for the epochs.

#### Yolov5
To train using **Yolov5**:

1- Create new virtual env using `virtualenv` or `conda` then activate it.
2- Clone **Yolov5** repository outside this project:
```bash
cd ..
git clone https://github.com/ultralytics/yolov5
```
3- Install all dependencies (recommendation: Create new virtual environment when working with yolov5):
   ```bash
   cd yolov5
   pip install -r requirements.txt
   ```
4- Create folder for the dataset inside yolov5 folder:
```bash
mkdir dataset
cd dataset
mkdir images labels
mkdir train test val
cd ..
cd labels
mkdir train test val
```
5- Create `dataset.yaml` file and place it inside `yolov5/data` (you can copy the one in data folder).

6- Prepare the data to be aligned with the YOLO format:
```bash
cd ../MSc_Thesis
python prepare.py --yolo_version yolov5 --dataset_path '../yolov5/dataset'
```
5- train the model:
```bash
cd ../yolov5
python train.py --img 640 --batch 1 --epochs 30 --data './data/dataset_yolov5.yaml' --weights yolov5m.pt --name yolo_bdd
```
For more information follow this [Tutorials](https://github.com/ultralytics/yolov5/wiki/Train-Custom-Data).

#### Yolov7
Almost the same as [Yolov5](####Yolov5)
1- Create new virtual env using `virtualenv` or `conda` then activate it.
2- Clone **Yolov7** repository outside this project:
```bash
cd ..
git clone https://github.com/WongKinYiu/yolov7.git
```
3- Install all dependencies (recommendation: Create new virtual environment when working with yolov5):
   ```bash
   cd yolov7
   pip install -r requirements.txt
   ```
4- Create folder for the dataset inside **yolov7** folder:
```bash
mkdir dataset
cd dataset
mkdir train test val
cd train
mkdir images labels
cd ../test
mkdir images labels
cd ../val
mkdir images labels
```
5- Create `dataset.yaml` file and place it inside `yolov7/data` (you can copy the one in data folder).

6- Prepare the data to be aligned with the YOLO format:
```bash
cd ../MSc_Thesis
python prepare.py --yolo_version yolov7 --dataset_path '../yolov7/dataset'
```
5- train the model:
```bash
cd ../yolov7
python train.py --img 640 --batch 1 --epochs 30 --data './data/dataset_yolov7.yaml' --weights yolov5m.pt --name yolo_bdd
```
For more information follow this [Tutorials](https://blog.paperspace.com/yolov7/).


---
