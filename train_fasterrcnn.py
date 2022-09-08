import os
import yaml
import argparse
from src.config.defaults import cfg
from src.models.Detection.detection_models import get_fasterrcnn
from src.dataset.bdd_detetcion import BDDDetection
from src.utils.DataLoaders import get_loader

import wandb
import torch
import torch.optim as optim
import pandas as pd
from tqdm import tqdm


def train_one_epoch(loader, model, optimizer, device):
    model.train()
    training_loss = []
    loop = tqdm(loader)
    for batch in loop:
        images, targets = batch
        images = list(image.to(device) for image in images)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        optimizer.zero_grad()
        loss_dict = model(images, targets)
        losses = sum(loss for loss in loss_dict.values())
        loop.set_postfix(loss=losses.item())
        training_loss += losses.item()
        losses.backward()
        optimizer.step()

    training_loss /= len(loader)

    return training_loss


def val_one_epoch(loader, model, device):
    model.eval()
    val_loss = []
    loop = tqdm(loader)
    for batch in loop:
        images, targets = batch
        images = list(image.to(device) for image in images)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        loss_dict = model(images, targets)
        losses = sum(loss for loss in loss_dict.values())
        loop.set_postfix(loss=losses.item())
        val_loss += losses.item

    val_loss /= len(loader)

    return val_loss


def logger(training_loss, val_loss, epoch, path_to_file):
    log_string = f"Training Loss: {training_loss}, Val Loss: {val_loss}, Epoch: {epoch}"
    with open(path_to_file, 'a') as file:
        file.write(log_string)


def load_check_point(model, optimizer, path):
    checkpoint = torch.load(path)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    training_losses = checkpoint['training_losses']
    validation_losses = checkpoint['validation_losses']
    best_val_loss = checkpoint['best_val_loss']
    return model, optimizer, epoch, training_losses, validation_losses, best_val_loss


def save_checkpoint(epoch, model, optimizer, training_losses, validation_losses, best_val_loss, path_to_save):
    print("########################################")
    print("Saving model...")
    print("########################################")
    torch.save({
        'epoch': epoch,
        'training_losses': training_losses,
        'validation_losses': validation_losses,
        'best_val_loss': best_val_loss,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }, path_to_save)
    print("########################################")
    print("Model saved")
    print("########################################")


def export_losses(training_losses, validation_losses, path):
    data = {
        "Training Losses": training_losses,
        "validation Losses": validation_losses
    }

    df = pd.DataFrame(data=data)
    df.to_csv(path)


def train(model,
          train_loader,
          val_loader,
          optimizer,
          epochs,
          device,
          path_to_save='./runs/train',
          checkpoint=None
          ):
    print("################################################################################")
    print('################################ Start Training ################################')
    print("################################################################################")

    logger_path = os.path.join(path_to_save, 'results.txt')
    best_path = os.path.join(path_to_save, 'best.pt')
    last_path = os.path.join(path_to_save, 'last.pt')

    if checkpoint is not None:
        checkpoint_params = {
            'model': model,
            'optimizer': optimizer,
            'path': checkpoint
        }
        model, optimizer, last_epoch, training_losses, validation_losses, best_val_loss = load_check_point(
            **checkpoint_params)
    else:
        training_losses = []
        validation_losses = []
        best_val_loss = 1000
        last_epoch = 0

    wandb.watch(model, log="all", log_freq=10)

    model.to(device)

    for epoch in range(last_epoch, epochs):

        train_loss = train_one_epoch(train_loader, model, optimizer, device)
        val_loss = val_one_epoch(val_loader, model, device)

        wandb.log({
            'epoch': epoch,
            'train_loss': train_loss,
            'val_loss': val_loss
        })

        training_losses.append(train_loss)
        validation_losses.append(val_loss)

        if val_loss < best_val_loss:
            save_checkpoint(epoch, model, optimizer, train_loss, val_loss, best_val_loss, best_path)

        logger(train_loss, val_loss, epoch, logger_path)

        save_checkpoint(epoch, model, optimizer, training_losses, validation_losses, best_val_loss, last_path)

        print("################################################################################")
        print('################################# End Training #################################')
        print("################################################################################")

    return training_losses, validation_losses


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch-size', type=int, default=16, help='total batch size for all GPUs')
    parser.add_argument('--img-size', type=int, default=640, help='train, val image size (pixels)')
    parser.add_argument('--data', type=str, default="data/fasterrcnn.yaml", help='fasterrcnn.yaml path')
    parser.add_argument('--backbone', type=str, default='resnet50',
                        help='choose the backbone you want to use - default: resnet50')
    parser.add_argument('--weights', type=str, default=None, help='train from checkpoint')
    parser.add_argument('--lr', type=float, default=1e-5, help='learning rate')
    parser.add_argument('--epochs', type=int, default=100, help='total number of epochs')
    parser.add_argument('--logger_path', type=str, help='where you want to log your data')
    parser.add_argument('--checkpoint_path', type=str, default='./checkpoints',
                        help="Path where you want to checkpoint.")
    parser.add_argument('--workers', type=int, default=4, help='number of workers')
    parser.add_argument('--pin-memory', type=bool, default=False, help='use pin memory to accelerate Dataloader.')
    parser.add_argument('--name', type=str, default='version1', help='name of the model you want to save')
    parser.add_argument('--project', type=str, default='Master Thesis', help='name of the Project to save in wandb')

    # Fetch the params from the parser
    args = parser.parse_args()

    batch_size = args.batch_size  # Batch Size
    img_size = args.img_size  # Image size

    with open(args.data, 'r') as f:
        data = yaml.safe_load(f)  # data from .yaml file

    obj_cls = data['classes']  # the classes we want to work one
    relative_path = data['relative_path']  # relative path to the dataset


    backbone = args.backbone  # Check point to continue training
    weights = args.weights  # Check point to continue training
    lr = args.lr  # Learning Rate
    epochs = args.epochs  # number of epochs
    workers = args.workers  # number of workers
    pin_memory = args.pin_memory  # pin memory
    logger_path = args.logger_path  # where you want to save the logs
    checkpoint_path = args.checkpoint_path  # path to checkpoints
    name = args.name  # name of the projects (version)
    project = args.project  # name of the projects

    ######################################## Datasets ########################################

    # Training dataset
    bdd_train_params = {
        'cfg': cfg,
        'stage': 'train',
        'relative_path': relative_path,
        'obj_cls': obj_cls,
    }

    bdd_train = BDDDetection(**bdd_train_params)

    # Validation dataset
    bdd_val_params = {
        'cfg': cfg,
        'stage': 'val',
        'relative_path': relative_path,
        'obj_cls': obj_cls,
    }

    bdd_val = BDDDetection(**bdd_val_params)

    print(50 * '#')
    print(f"Training Images: {len(bdd_train)}. Validation Images: {len(bdd_val)}.")
    print(50 * '#')

    ######################################## DataLoaders ########################################

    train_dataloader_args = {
        'dataset': bdd_train,
        'batch_size': batch_size,
        'shuffle': True,
        'collate_fn': bdd_train.collate_fn,
        'pin_memory': pin_memory,
        'num_workers': workers
    }
    train_dataloader = get_loader(**train_dataloader_args)

    # val dataloader
    val_dataloader_args = {
        'dataset': bdd_val,
        'batch_size': batch_size,
        'shuffle': False,
        'collate_fn': bdd_train.collate_fn,
        'pin_memory': pin_memory,
        'num_workers': workers
    }
    val_dataloader = get_loader(**val_dataloader_args)

    ######################################## Model ########################################

    model = get_fasterrcnn(num_classes=len(bdd_train.idx_to_cls),
                           backbone=backbone,
                           pretrained=True,
                           pretrained_backbone=True)

    opt_pars = {'lr': lr, 'weight_decay': 1e-3}
    optimizer = optim.Adam(list(model.parameters()), **opt_pars)

    device = torch.device('cpu')
    if torch.cuda.is_available():
        device = torch.device('cuda')

    print(f"We are using {device}")

    train_params = {
        "training_loader": train_dataloader,
        "validation_loader": val_dataloader,
        "model": model,
        "optimizer": optimizer,
        "device": device,
        "total_epochs": epochs,
        "path_to_save": logger_path,
        "checkpoint": checkpoint_path
    }

    wandb.config = {
        "learning_rate": lr,
        "epochs": epochs,
        "batch_size": batch_size
    }

    training_losses, validation_losses = train(**train_params)

    export_losses(training_losses, validation_losses, os.path.join(logger_path, 'losses.csv'))