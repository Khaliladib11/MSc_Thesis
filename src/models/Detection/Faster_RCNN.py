import json
from pathlib import Path
from PIL import Image

import torch
from torchvision import transforms
from .detection_models import get_fasterrcnn
import pytorch_lightning as pl
from torchmetrics.detection.mean_ap import MeanAveragePrecision


class Faster_RCNN(pl.LightningModule):

    def __init__(self,
                 cfg,
                 num_classes,
                 backbone,
                 learning_rate,
                 weight_decay,
                 pretrained,
                 pretrained_backbone,
                 ):
        super(Faster_RCNN, self).__init__()

        assert 0 <= learning_rate <= 1, "Learning Rate must be between 0 and 1"
        assert backbone in cfg.DETECTION.BACKBONE, f"Please choose backbone from the following: {cfg.DETECTION.BACKBONE}"
        assert num_classes > 0, "Number of classes must be greater than 0."

        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.num_classes = num_classes
        self.save_hyperparameters()

        self.model = get_fasterrcnn(self.num_classes, backbone, pretrained, pretrained_backbone)

        self.transform = transforms.Compose([
            transforms.Resize(640),
            transforms.ToTensor(),  # convert the image to tensor
            transforms.Normalize(mean=[0.407, 0.457, 0.485],
                                 std=[0.229, 0.224, 0.225])  # normalize the image using mean ans std
        ])
        self.metric = MeanAveragePrecision()
        self.metric_person = MeanAveragePrecision()
        self.metric_car = MeanAveragePrecision()
        self.metric_tl = MeanAveragePrecision()
        self.metric_ts = MeanAveragePrecision()

    def forward(self, x, *args, **kwargs):
        return self.model(x)

    def training_step(self, train_batch, batch_idx):
        images, targets = train_batch

        targets = [{k: v for k, v in t.items()} for t in targets]

        loss_dict = self.model(images, targets)
        train_loss = sum(loss for loss in loss_dict.values())

        # self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)

        return train_loss

    def training_epoch_end(self, training_step_outputs):
        epoch_losses = torch.tensor([batch_loss['loss'].item() for batch_loss in training_step_outputs])
        # epoch_losses = torch.stack(training_step_outputs)
        loss_mean = torch.mean(epoch_losses)
        self.log('training_loss', loss_mean)

    def validation_step(self, val_batch, batch_idx):
        self.model.train()
        images, targets = val_batch

        targets = [{k: v for k, v in t.items()} for t in targets]

        outputs = self.model(images, targets)

        val_loss = sum(loss for loss in outputs.values())
        return val_loss

    def validation_epoch_end(self, validation_step_outputs):
        epoch_losses = torch.tensor([batch_loss.item() for batch_loss in validation_step_outputs])
        # epoch_losses = torch.stack(training_step_outputs)
        loss_mean = torch.mean(epoch_losses)
        self.log('val_loss', loss_mean)

    def filter_gt(self, pr, idx_label):
        boxes = pr['boxes'].tolist()
        labels = pr['labels'].tolist()
        pred = [{
            'boxes': [],
            'labels': [],
        }]
        for idx, label in enumerate(labels):
            if label == idx_label:
                pred[0]['boxes'].append(boxes[idx])
                pred[0]['labels'].append(labels[idx])
        if len(pred[0]['boxes']) > 0:
            pred[0]['boxes'] = torch.Tensor(pred[0]['boxes'])
            pred[0]['labels'] = torch.Tensor(pred[0]['labels'])
        return pred

    def filter_prediction(self, pr, idx_label):
        boxes = pr['boxes'].tolist()
        labels = pr['labels'].tolist()
        scores = pr['scores'].tolist()
        pred = [{
            'boxes': [],
            'labels': [],
            'scores': []
        }]
        for idx, label in enumerate(labels):
            if label == idx_label:
                pred[0]['boxes'].append(boxes[idx])
                pred[0]['labels'].append(labels[idx])
                pred[0]['scores'].append(scores[idx])

        pred[0]['boxes'] = torch.Tensor(pred[0]['boxes'])
        pred[0]['labels'] = torch.Tensor(pred[0]['labels'])
        pred[0]['scores'] = torch.Tensor(pred[0]['scores'])
        return pred

    def predict_step(self, batch, batch_idx):
        self.model.eval()
        images, targets = batch
        targets = [{k: v for k, v in t.items()} for t in targets]
        outputs = self.model(images)
        self.metric.update(outputs, targets)
        # Person
        pred_person = self.filter_prediction(outputs[0], 1)
        gt_person = self.filter_gt(targets[0], 1)

        if len(gt_person[0]['boxes']) > 0:
            self.metric_person.update(pred_person, gt_person)

        # Cars
        pred_car = self.filter_prediction(outputs[0], 2)
        gt_car = self.filter_gt(targets[0], 2)

        if len(gt_car[0]['boxes']) > 0:
            self.metric_car.update(pred_car, gt_car)

        # tl
        pred_tl = self.filter_prediction(outputs[0], 3)
        gt_tl = self.filter_gt(targets[0], 3)

        if len(gt_tl[0]['boxes']) > 0:
            self.metric_tl.update(pred_tl, gt_tl)
        # ts
        pred_ts = self.filter_prediction(outputs[0], 4)
        gt_ts = self.filter_gt(targets[0], 4)

        if len(gt_ts[0]['boxes']) > 0:
            self.metric_ts.update(pred_ts, gt_ts)

        return outputs

    def predict(self, image_path, threshold, device):
        self.model.eval()
        image = Image.open(image_path)
        tensor_image = self.transform(image)
        tensor_image = tensor_image.unsqueeze(0)
        tensor_image = tensor_image.to(device)
        model = self.model.to(device)
        yhat = model(tensor_image)
        yhat = yhat[0]
        boxes = yhat['boxes'].tolist()
        labels = yhat['labels'].tolist()
        scores = yhat['scores'].tolist()
        prediction = {
            "boxes": [],
            "scores": [],
            "labels": [],
        }

        for idx, score in enumerate(scores):
            if score > threshold:
                prediction['boxes'].append(boxes[idx])
                prediction['scores'].append(scores[idx])
                prediction['labels'].append(labels[idx])

        return prediction

    def configure_optimizers(self):
        optimizer_params = {
            'params': self.model.parameters(),
            'lr': self.learning_rate,
            'weight_decay': self.weight_decay,
        }
        return torch.optim.Adam(**optimizer_params)
