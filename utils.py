import torch
import torchvision
import os
from dataset import preprocessImage, preprocessTestImage
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
import torch.nn.functional as F
import torch.nn as nn
import numpy as np
from tqdm import tqdm
import random
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import KFold
import pandas as pd
from tqdm import tqdm
from sklearn.model_selection import StratifiedKFold

# Checkpoint saver
def save_checkpoint(state, filename="my_checkpoint.pth.tar"):
    print("=> Saving checkpoint")
    torch.save(state, filename)

# Checkpoint loader
def load_checkpoint(checkpoint, model):
    if "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
        state_dict = {key.replace("module.", ""): value for key, value in state_dict.items()}
        model.load_state_dict(state_dict)
    else:

        model.load_state_dict(checkpoint)



# Setting random seeds for reproducibility
def set_seed(seed=42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if using multi-GPU
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# k-fold cross-validation with stratification
def get_kfold_loaders(dataset, k=5, seed=42):

    # Get targets to stratify split
    targets = [dataset[i][1].sum().item() > 0 for i in range(len(dataset))]
    targets = np.array(targets, dtype=int)

    skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=seed)
    splits = list(skf.split(np.zeros(len(targets)), targets))
    return splits

# Dataloader
def get_loaders_for_fold(dataset, train_indices, val_indices, train_transform, val_transform, batch_size, num_workers, pin_memory):
    train_dataset = Subset(dataset, train_indices)
    val_dataset = Subset(dataset, val_indices)

    train_dataset.dataset.transform = train_transform
    val_dataset.dataset.transform = val_transform

    train_loader = DataLoader(train_dataset, batch_size=batch_size, num_workers=num_workers, pin_memory=pin_memory, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, num_workers=num_workers, pin_memory=pin_memory, shuffle=False)

    return train_loader, val_loader

# Validation
def check_accuracy(loader, model, device="cuda"):
    num_correct = 0
    num_pixels = 0
    dice_score = 0
    jaccard_index = 0
    sensitivity = 0
    specificity = 0
    precision = 0
    accuracy = 0
    f1_score = 0
    alpha = 0.5
    composite_score = 0

    model.eval()

    loop = tqdm(loader, desc="Validation", leave=False)
    
    with torch.no_grad():
        for x, y in loop:
            x = x.to(device)
            y = y.to(device).unsqueeze(1)
            preds = torch.sigmoid(model(x))
            preds = (preds > 0.5).float()

            num_correct += (preds == y).sum().item()
            num_pixels += torch.numel(preds)

            tp = (preds * y).sum().item()
            fp = ((preds == 1) & (y == 0)).sum().item()
            fn = ((preds == 0) & (y == 1)).sum().item()
            tn = ((preds == 0) & (y == 0)).sum().item()

            dice = (2 * tp) / (2 * tp + fp + fn + 1e-8)
            sens = tp / (tp + fn + 1e-8)

            dice_score += dice
            jaccard_index += tp / (tp + fp + fn + 1e-8)
            sensitivity += sens
            specificity += tn / (tn + fp + 1e-8)
            precision += tp / (tp + fp + 1e-8)
            accuracy += (tp + tn) / (tp + tn + fp + fn + 1e-8)
            f1_score += (2 * tp) / (2 * tp + fp + fn + 1e-8)
            
            composite_score += alpha * dice + (1 - alpha) * sens

    metrics = {
        "accuracy": num_correct / num_pixels,
        "dice_score": dice_score / len(loader),
        "jaccard_index": jaccard_index / len(loader),
        "sensitivity": sensitivity / len(loader),
        "specificity": specificity / len(loader),
        "precision": precision / len(loader),
        "f1_score": f1_score / len(loader),
        "composite_score": composite_score / len(loader)
    }

    print(f"Got {num_correct}/{num_pixels} with acc {num_correct/num_pixels*100:.2f}%")
    print(f"Dice score: {metrics['dice_score']:.4f}")
    print(f"Jaccard Index: {metrics['jaccard_index']:.4f}")
    print(f"Sensitivity: {metrics['sensitivity']:.4f}")
    print(f"Specificity: {metrics['specificity']:.4f}")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Accuracy: {metrics['accuracy']:.4f}")
    print(f"F1 Score: {metrics['f1_score']:.4f}")
    print(f"Composite Score (alpha=0.5): {metrics['composite_score']:.4f}")
    
    model.train()
    
    return metrics



# Predictions
def save_predictions_as_imgs(loader, model, img_ids, folder, device="cuda"):
    model.eval()
    for idx, (x, _) in enumerate(loader):  
        x = x.to(device=device)
        img_id = img_ids[idx]  # Get the img_id corresponding to the current batch
        with torch.no_grad():
            preds = torch.sigmoid(model(x))
            preds = (preds > 0.5).float() # modify to get different results


        for i in range(len(preds)):
            img_id = img_ids[idx * loader.batch_size + i]  # Get the img_id corresponding to the current prediction
            torchvision.utils.save_image(preds[i], f"{folder}/pred_{img_id}.png")
