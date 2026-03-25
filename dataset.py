import os
import cv2
import torch
from torch.utils.data import Dataset

# Optional: for augmentations
import albumentations as A
from albumentations.pytorch import ToTensorV2

class SegmentationDataset(Dataset):
    def __init__(self, image_dir, mask_dir, transform=None):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.transform = transform
        self.images = os.listdir(image_dir)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_name = self.images[idx]
        img_path = os.path.join(self.image_dir, img_name)
        mask_path = os.path.join(self.mask_dir, img_name)

        image = cv2.imread(img_path)
        mask = cv2.imread(mask_path, 0)  # grayscale mask

        if image is None or mask is None:
            raise FileNotFoundError(f"Image or mask not found: {img_path}, {mask_path}")

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Apply transform if provided
        if self.transform:
            augmented = self.transform(image=image, mask=mask)
            image = augmented['image']
            mask = augmented['mask']
        else:
            # Default resizing & normalization if no transform
            image = cv2.resize(image, (256, 256))
            mask = cv2.resize(mask, (256, 256), interpolation=cv2.INTER_NEAREST)
            image = image / 255.0
            mask = mask / 255.0
            image = torch.tensor(image).permute(2, 0, 1).float()
            mask = torch.tensor(mask).unsqueeze(0).float()

        return image, mask