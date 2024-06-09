import torch
import os
import pandas as pd
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
from utils import load_checkpoint, save_checkpoint, check_accuracy, get_loaders_for_fold, get_kfold_loaders, set_seed
from dataset import preprocessImage
import random
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import StratifiedKFold
import monai


# Hyperparameters
LEARNING_RATE = 1e-3
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 10
NUM_EPOCHS = 100
NUM_WORKERS = 12
IMAGE_HEIGHT = 512
IMAGE_WIDTH = 512
PIN_MEMORY = True
LOAD_MODEL = False
TRAIN_IMG_DIR = "/u/project/pallard/sujit009/Segmentation/trainImages/trainImages"
TRAIN_MASK_DIR = "/u/project/pallard/sujit009/Segmentation/trainMasks/trainMasks"
VAL_IMG_DIR = "/u/project/pallard/sujit009/Segmentation/testImages/testImages"
MODEL_TYPE="UNET_ImgM"


# Train
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


# main 

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

    val_transform = A.Compose(
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

    dataset = preprocessImage(TRAIN_IMG_DIR, TRAIN_MASK_DIR, transform=None)
    splits = get_kfold_loaders(dataset, k=5)

    best_fold_loss = float('inf')
    best_fold_composite_score = float('-inf')
    best_loss_checkpoint = None
    best_composite_score_checkpoint = None
    best_loss_fold = None
    best_composite_score_fold = None

    for fold, (train_indices, val_indices) in enumerate(splits):
        print(f"Fold {fold + 1}/{len(splits)}")

        # Initialize the model, optimizer, and scheduler for each fold
        model = UNET(in_channels=1, out_channels=1).to(DEVICE)
       

        if torch.cuda.device_count() > 1:
            print("Using", torch.cuda.device_count(), "GPUs!")
            model = nn.DataParallel(model)

        initialize_with_vgg16_weights(model)

        optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE)
        #optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
        
        train_loader, val_loader = get_loaders_for_fold(dataset, train_indices, val_indices, train_transform, val_transform, BATCH_SIZE, NUM_WORKERS, PIN_MEMORY)

        # Define total number of iterations (epochs * steps_per_epoch)
        #total_iterations = NUM_EPOCHS * len(train_loader)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS)

        scaler = torch.cuda.amp.GradScaler()

        train_losses = []
        avg_losses = []
        fold_best_loss = float('inf')
        fold_best_composite_score = float('-inf')
        fold_best_loss_checkpoint = None
        fold_best_composite_score_checkpoint = None
        avg_val_losses = []
        val_dice_scores = []
        val_jaccard_indices = []
        val_precisions = []
        val_composite_scores = []
        val_losses = []
        metrics_list = []

        for epoch in range(NUM_EPOCHS):
            print(f"Epoch {epoch + 1}/{NUM_EPOCHS}")
            losses = train_fn(train_loader, model, optimizer, nn.BCEWithLogitsLoss(), scaler)
            train_losses.extend(losses)

            avg_loss = sum(losses) / len(losses)
            avg_losses.append(avg_loss)

            val_loss = validate_fn(val_loader, model, nn.BCEWithLogitsLoss(), device=DEVICE)
            val_losses.append(val_loss)

            avg_val_loss = sum(val_losses) / len(val_losses)
            avg_val_losses.append(avg_val_loss)

            metrics = check_accuracy(val_loader, model, device=DEVICE)
            val_dice_scores.append(metrics["dice_score"])
            val_jaccard_indices.append(metrics["jaccard_index"])
            val_precisions.append(metrics["precision"])
            val_composite_scores.append(metrics["composite_score"])

            metrics['epoch'] = epoch + 1
            metrics_list.append(metrics)

            # Step the scheduler at each iteration
            #scheduler.step()

            if avg_val_loss < fold_best_loss:
                fold_best_loss = avg_val_loss
                fold_best_loss_checkpoint = {
                    "state_dict": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                }
                print(f"New lowest loss for fold {fold + 1}: {fold_best_loss:.4f}")

            if metrics["composite_score"] > fold_best_composite_score:
                fold_best_composite_score = metrics["composite_score"]
                fold_best_composite_score_checkpoint = {
                    "state_dict": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                }
                print(f"New highest composite score for fold {fold + 1}: {fold_best_composite_score:.4f}")

        save_checkpoint(fold_best_loss_checkpoint, filename=f"fold_{fold + 1}_best_loss.pth")
        save_checkpoint(fold_best_composite_score_checkpoint, filename=f"fold_{fold + 1}_best_composite_score.pth")

        metrics_df = pd.DataFrame(metrics_list)
        model_params = f"{MODEL_TYPE}_Fold_{fold + 1}_LR_{LEARNING_RATE}_BS_{BATCH_SIZE}_Epochs_{NUM_EPOCHS}"
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

        if fold_best_loss < best_fold_loss:
            best_fold_loss = fold_best_loss
            best_loss_checkpoint = fold_best_loss_checkpoint
            best_loss_fold = fold + 1

        if fold_best_composite_score > best_fold_composite_score:
            best_fold_composite_score = fold_best_composite_score
            best_composite_score_checkpoint = fold_best_composite_score_checkpoint
            best_composite_score_fold = fold + 1

    save_checkpoint(best_loss_checkpoint, filename=f"best_model_loss_fold_{best_loss_fold}.pth")
    save_checkpoint(best_composite_score_checkpoint, filename=f"best_model_composite_score_fold_{best_composite_score_fold}.pth")
    print(f"Best model saved from fold {best_loss_fold} with lowest loss: {best_fold_loss:.4f}")
    print(f"Best model saved from fold {best_composite_score_fold} with highest composite score: {best_fold_composite_score:.4f}")

if __name__ == "__main__":
    main()



