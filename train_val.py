import torch
import os
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader, random_split
from sklearn.model_selection import train_test_split
import numpy as np
import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2
from PIL import Image
import cv2
import torchvision.transforms as transforms
from tqdm import tqdm
import torch.nn as nn
import torch.optim as optim
from model import UNET, initialize_with_vgg16_weights, NestedUNet
from utils import load_checkpoint, save_checkpoint, get_loaders, get_balanced_loaders, check_accuracy,get_loaders_for_fold,get_kfold_loaders, set_seed
from dataset import preprocessImage
import random
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import KFold


# Hyperparameters
LEARNING_RATE = 1e-4
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 1
NUM_EPOCHS = 2
NUM_WORKERS = 12
IMAGE_HEIGHT = 512
IMAGE_WIDTH = 512
PIN_MEMORY = True
LOAD_MODEL = False
TRAIN_IMG_DIR = "/Users/sujitsilas/Desktop/UCLA/Spring 2024/BE M224B Advances in Imaging Informatics /BE224_Spring2024_Data/trainImages/trainImages/"
TRAIN_MASK_DIR = "/Users/sujitsilas/Desktop/UCLA/Spring 2024/BE M224B Advances in Imaging Informatics /BE224_Spring2024_Data/trainMasks/trainMasks/"
VAL_IMG_DIR = "/Users/sujitsilas/Desktop/UCLA/Spring 2024/BE M224B Advances in Imaging Informatics /BE224_Spring2024_Data/testImages/testImages"



def train_fn(loader, model, optimizer, loss_fn, scaler):
    loop = tqdm(loader, desc="Train")
    losses = []

    for batch_idx, (data, targets) in enumerate(loop):
        data = data.to(device=DEVICE)
        targets = targets.unsqueeze(1).float().to(device=DEVICE)

        # forward
        with torch.cuda.amp.autocast():
            predictions = model(data)
            loss = loss_fn(predictions, targets)

        # backward
        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        # update tqdm loop
        loop.set_postfix(loss=loss.item())
        losses.append(loss.item())

    return losses

def validate_fn(loader, model, loss_fn, device="cuda"):
    model.eval()
    loop = tqdm(loader, desc="Validation")
    losses = []

    with torch.no_grad():
        for batch_idx, (data, targets) in enumerate(loop):
            data = data.to(device)
            targets = targets.unsqueeze(1).float().to(device)

            predictions = model(data)
            loss = loss_fn(predictions, targets)
            losses.append(loss.item())

            # update tqdm loop
            loop.set_postfix(loss=loss.item())

    val_loss = sum(losses) / len(losses)
    model.train()
    return val_loss

"""""
def main():
    train_transform = A.Compose(
        [
            A.Resize(height=IMAGE_HEIGHT, width=IMAGE_WIDTH),
            A.Rotate(limit=35, p=1.0),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.1),
            A.Normalize(
                mean=[0.0],
                std=[1.0],
                max_pixel_value=255.0,
            ),
            ToTensorV2(),
        ],
    )

    val_transforms = A.Compose(
        [
            A.Resize(height=IMAGE_HEIGHT, width=IMAGE_WIDTH),
            A.Normalize(
                mean=[0.0],
                std=[1.0],
                max_pixel_value=255.0,
            ),
            ToTensorV2(),
        ],
    )

    model = UNET(in_channels=3, out_channels=1).to(DEVICE)
    
    if torch.cuda.device_count() > 1:  # Check if multiple GPUs are available
        print("Using", torch.cuda.device_count(), "GPUs!")
        model = nn.DataParallel(model)
    initialize_with_vgg16_weights(model)

    loss_fn = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=2, factor=0.5, verbose=True)

    train_loader, val_loader = get_balanced_loaders(
        train_dir=TRAIN_IMG_DIR,
        train_maskdir=TRAIN_MASK_DIR,
        val_transform=val_transforms,
        train_transform=train_transform,
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY
    )

    scaler = torch.cuda.amp.GradScaler()

    train_losses = []
    avg_losses = []
    best_composite_score = float('-inf')
    best_checkpoint = None
    val_dice_scores = []
    val_jaccard_indices = []
    val_losses = []

    for epoch in range(NUM_EPOCHS):
        print(f"Epoch {epoch + 1}/{NUM_EPOCHS}")
        losses = train_fn(train_loader, model, optimizer, loss_fn, scaler)
        train_losses.extend(losses)

        avg_loss = sum(losses) / len(losses)
        avg_losses.append(avg_loss)

        val_loss = validate_fn(val_loader, model, loss_fn, device=DEVICE)
        val_losses.append(val_loss)

        scheduler.step(val_loss)

        # Check accuracy on validation set
        metrics = check_accuracy(val_loader, model, device=DEVICE)
        val_dice_scores.append(metrics["dice_score"])
        val_jaccard_indices.append(metrics["jaccard_index"])

        composite_score = metrics["composite_score"]

        if composite_score > best_composite_score:
            best_composite_score = composite_score
            best_checkpoint = {
                "state_dict": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "composite_score": best_composite_score
            }
            print(f"New best composite score: {best_composite_score:.4f}")
            save_checkpoint(best_checkpoint)

    plt.figure(figsize=(10, 5))
    plt.plot(range(1, NUM_EPOCHS + 1), avg_losses, label='Training Loss')
    plt.plot(range(1, NUM_EPOCHS + 1), val_losses, label='Validation Loss')
    plt.plot(range(1, NUM_EPOCHS + 1), val_dice_scores, label='Validation Dice Score')
    plt.plot(range(1, NUM_EPOCHS + 1), val_jaccard_indices, label='Validation Jaccard Index')
    plt.xlabel('Epoch')
    plt.ylabel('Metric')
    plt.title('Training Loss and Validation Metrics')
    plt.legend()
    plt.savefig('training_loss_and_validation_metrics_plot.png')
    plt.show()

if __name__ == "__main__":
    main()
"""""




def main():
    set_seed()

    train_transform = A.Compose(
        [
            A.Resize(height=IMAGE_HEIGHT, width=IMAGE_WIDTH),
            A.Rotate(limit=35, p=1.0),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.1),
            A.Normalize(
                mean=[0.0],
                std=[1.0],
                max_pixel_value=255.0,
            ),
            ToTensorV2(),
        ],
    )

    val_transforms = A.Compose(
        [
            A.Resize(height=IMAGE_HEIGHT, width=IMAGE_WIDTH),
            A.Normalize(
                mean=[0.0],
                std=[1.0],
                max_pixel_value=255.0,
            ),
            ToTensorV2(),
        ],
    )

    #model = UNET(in_channels=1, out_channels=1).to(DEVICE)
    model = NestedUNet(num_classes=1, input_channels=1).to(DEVICE)

    if torch.cuda.device_count() > 1:
        print("Using", torch.cuda.device_count(), "GPUs!")
        model = nn.DataParallel(model)

    initialize_with_vgg16_weights(model)

    loss_fn = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=2, factor=0.5, verbose=True)

    dataset = preprocessImage(TRAIN_IMG_DIR, TRAIN_MASK_DIR, transform=None)
    splits = get_kfold_loaders(dataset, k=10)

    for fold, (train_indices, val_indices) in enumerate(splits):
        print(f"Fold {fold + 1}/{len(splits)}")
        
        train_loader, val_loader = get_loaders_for_fold(dataset, train_indices, val_indices, train_transform, val_transforms, BATCH_SIZE, NUM_WORKERS, PIN_MEMORY)

        scaler = torch.cuda.amp.GradScaler()

        train_losses = []
        avg_losses = []
        best_composite_score = float('-inf')
        best_loss = float('inf')
        best_checkpoint = None
        avg_val_losses = []
        val_dice_scores = []
        val_jaccard_indices = []
        val_losses = []
        metrics_list = []

        for epoch in range(NUM_EPOCHS):
            print(f"Epoch {epoch + 1}/{NUM_EPOCHS}")
            losses = train_fn(train_loader, model, optimizer, loss_fn, scaler)
            train_losses.extend(losses)

            avg_loss = sum(losses) / len(losses)
            avg_losses.append(avg_loss)

            val_loss = validate_fn(val_loader, model, loss_fn, device=DEVICE)
            val_losses.append(val_loss)

            avg_val_loss = sum(val_losses) / len(val_losses)
            avg_val_losses.append(avg_val_loss)

            scheduler.step(val_loss)

            metrics = check_accuracy(val_loader, model, device=DEVICE)
            val_dice_scores.append(metrics["dice_score"])
            val_jaccard_indices.append(metrics["jaccard_index"])

            metrics['epoch'] = epoch + 1
            metrics_list.append(metrics)

            composite_score = metrics["composite_score"]

            if avg_val_loss < best_loss:
                best_loss = avg_val_loss
                best_checkpoint = {
                    "state_dict": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                }
                print(f"New lowest loss: {best_loss:.4f}")

        save_checkpoint(best_checkpoint)

        metrics_df = pd.DataFrame(metrics_list)
        model_params = f"Fold_{fold + 1}_LR_{LEARNING_RATE}_BS_{BATCH_SIZE}_Epochs_{NUM_EPOCHS}_IMG_{IMAGE_HEIGHT}x{IMAGE_WIDTH}"
        metrics_df.to_csv(f"metrics_{model_params}.csv", index=False)

        plt.figure(figsize=(10, 5))
        plt.plot(range(1, NUM_EPOCHS + 1), avg_losses, label='Training Loss')
        plt.plot(range(1, NUM_EPOCHS + 1), val_losses, label='Validation Loss')
        plt.plot(range(1, NUM_EPOCHS + 1), val_dice_scores, label='Validation Dice Score')
        plt.plot(range(1, NUM_EPOCHS + 1), val_jaccard_indices, label='Validation Jaccard Index')
        plt.xlabel('Epoch')
        plt.ylabel('Metric')
        plt.title(f'Training Loss and Validation Metrics ({model_params})')
        plt.legend()
        plt.savefig(f'training_loss_and_validation_metrics_plot_{model_params}.png')
        plt.show()

if __name__ == "__main__":
    main()
