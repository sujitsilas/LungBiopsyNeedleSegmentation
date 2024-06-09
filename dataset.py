import os
import cv2
import numpy as np
from PIL import Image
from torch.utils.data import Dataset

# Image preprocessing class for training data

class preprocessImage(Dataset):
    def __init__(self, image_dir, mask_dir, transform=None):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.transform = transform
        self.images = os.listdir(image_dir)
        self.masks = {os.path.splitext(file)[0]: file for file in os.listdir(mask_dir)}

    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, index):
        img_filename = self.images[index]
        img_path = os.path.join(self.image_dir, img_filename)

        img_id = os.path.splitext(img_filename)[0]

        img_filename = self.images[index]
        img_path = os.path.join(self.image_dir, img_filename)
        mask_filename = img_filename.replace(".jpg", "_mask.png")
        mask_path = os.path.join(self.mask_dir, mask_filename)
        
        image = np.array(Image.open(img_path).convert("L")) 
        mask = np.array(Image.open(mask_path).convert("L"), dtype=np.float32)
        mask[mask == 255.0] = 1.0

        if self.transform:
            augmentations = self.transform(image=image, mask=mask)
            image = augmentations["image"]
            mask = augmentations["mask"]

        return image, mask




"""""

# Image preprocessing class for test set with thresholding techniques applied
class preprocessImage(Dataset):
    def __init__(self, image_dir, mask_dir, transform=None):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.transform = transform
        self.images = os.listdir(image_dir)
        self.masks = {os.path.splitext(file)[0]: file for file in os.listdir(mask_dir)}

    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, index):
        # Get image and mask file paths
        img_filename = self.images[index]
        img_path = os.path.join(self.image_dir, img_filename)
        mask_filename = self.masks.get(os.path.splitext(img_filename)[0], img_filename.replace(".jpg", "_mask.png"))
        mask_path = os.path.join(self.mask_dir, mask_filename)
        
        # Load the image in grayscale
        image = np.array(Image.open(img_path).convert("L"), dtype=np.uint8)

        # Apply Otsu's thresholding to create a binary mask from the image
        _, otsu_thresh_image = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # Multiply the original image with the Otsu thresholded image
        multiplied_image = cv2.bitwise_and(image, image, mask=otsu_thresh_image)

        # Load the semantic segmentation mask
        mask = np.array(Image.open(mask_path).convert("L"), dtype=np.uint8)
        mask = (mask == 255).astype(np.uint8)  # Convert to binary mask if needed

        # Create a three-channel image combining the original, multiplied, and thresholded images
        three_channel_image = np.stack([image, multiplied_image, otsu_thresh_image], axis=-1)


        # Apply transformations if any
        if self.transform:
            augmented = self.transform(image=three_channel_image, mask=mask)
            three_channel_image = augmented["image"]
            mask = augmented["mask"]

        return three_channel_image, mask


"""""




# Image preprocessing class for test set

class preprocessTestImage(Dataset):
    def __init__(self, image_dir, transform=None):
        self.image_dir = image_dir
        self.transform = transform
        self.images = os.listdir(image_dir)

    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, index):
        img_filename = self.images[index]
        img_path = os.path.join(self.image_dir, img_filename)

        img_id = os.path.splitext(img_filename)[0]

        image = np.array(Image.open(img_path).convert("L")) 

        if self.transform:
            image = self.transform(image=image)['image']

        return image, img_id