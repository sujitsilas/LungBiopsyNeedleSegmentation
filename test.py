import os
import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import torchvision
from PIL import Image
from model import UNET, NestedUNet
from dataset import preprocessTestImage
from utils import load_checkpoint, save_predictions_as_imgs
import albumentations as A
import albumentations.augmentations.functional as F
from albumentations.pytorch import ToTensorV2
import random

# Hyperparameters
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 10
NUM_WORKERS = 12
IMAGE_HEIGHT = 512
IMAGE_WIDTH = 512
PIN_MEMORY = True
random.seed(42)


def main():
    # Define paths
    checkpoint_path = "/Users/sujitsilas/Desktop/UCLA/Spring 2024/BE M224B Advances in Imaging Informatics /BE224_Spring2024_Data/best_model_composite_score_fold_4.pth"
    test_img_dir = "/Users/sujitsilas/Desktop/UCLA/Spring 2024/BE M224B Advances in Imaging Informatics /BE224_Spring2024_Data/testImages/testImages/"
    generated_masks_dir = "/Users/sujitsilas/Desktop/UCLA/Spring 2024/BE M224B Advances in Imaging Informatics /BE224_Spring2024_Data/generatedMasks"

    val_transforms = A.Compose([
        A.Resize(height=IMAGE_HEIGHT, width=IMAGE_WIDTH),
        A.Normalize(mean=[0.0], std=[1.0], max_pixel_value=255.0),
        ToTensorV2(),
    ])

    print()

    model = UNET(in_channels=1, out_channels=1)
    checkpoint = torch.load(checkpoint_path, map_location=torch.device('cpu'))
    load_checkpoint(checkpoint, model=model)

    # Generate masks for test images and save them
    test_ds = preprocessTestImage(image_dir=test_img_dir, transform=val_transforms)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY, shuffle=False)
    img_ids = [img_id for _, img_id in test_loader.dataset]
    #save_predictions_as_imgs(loader=test_loader, model=model, img_ids=img_ids, folder=generated_masks_dir, device=DEVICE)
    save_predictions_as_imgs(loader=test_loader, model=model, img_ids=img_ids, folder=generated_masks_dir, device=DEVICE)


if __name__ == "__main__":
    main()

