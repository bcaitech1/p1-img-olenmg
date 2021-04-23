import os
import glob

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

import cv2


class ImageDataset(Dataset):
    """Image dataset for multi-label classification (0-17)

    Args:
        img_list (list[str]): image paths of dataset
        label_list (list[int]): label list(0-17) corresponding to img_list
        transform (albumentations.augmentations.transforms): default transform
        aug_transform (albumentations.augmentations.transforms): use this transform only with augmented minor data
        aug_target (list[bool]): for identifying if the data is the target of 'aug_transform' (augmented minor data)
    """

    def __init__(self, img_list, label_list, transform=None, aug_transform=None, aug_target=None):
        super(ImageDataset, self).__init__()
        self.img_list = img_list
        self.label_list = label_list
        self.transform = transform
        self.aug_transform = aug_transform
        self.aug_target = aug_target

    def __len__(self):
        return len(self.img_list)
    
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        
        image = cv2.imread(self.img_list[idx])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        label = self.label_list[idx]

        # if the user doesn't use minor data augmentation, aug_target is None.
        if self.aug_target and self.aug_target[idx]:
            image = self.aug_transform(image=image)['image']
        elif self.transform:
            image = self.transform(image=image)['image']

        return image, label


class MultiLabelDataset(Dataset):
    """Image dataset for multi-category classification (0-2/0-1/0-2)

    Args:
        img_list (list[str]): image paths of dataset
        label_list (list[int]): label list(0-17) corresponding to img_list
        transform (albumentations.augmentations.transforms): default transform
        aug_transform (albumentations.augmentations.transforms): use this transform only with augmented minor data
        aug_target (list[bool]): for identifying if the data is the target of 'aug_transform' (augmented minor data)
    """

    def __init__(self, img_list, label_list, transform=None, aug_transform=None, aug_target=None):
        super(ImageDataset, self).__init__()
        self.img_list = img_list
        self.label_list = label_list
        self.transform = transform
        self.aug_transform = aug_transform
        self.aug_target = aug_target
    
    def __len__(self):
        return len(self.img_list)
    
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        
        image = cv2.imread(self.img_list[idx])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        label = self.label_list[idx]

        mask_label = label // 6
        gender_label = label % 6 // 3
        age_label = label % 3

        # if the user doesn't use minor data augmentation, aug_target is None.
        if self.aug_target and self.aug_target[idx]:
            image = self.aug_transform(image=image)['image']
        elif self.transform:
            image = self.transform(image=image)['image']

        return image, (mask_label, gender_label, age_label)


class TestDataset(Dataset):
    """Image dataset for test dataset

    Args:
        root_dir (str): image paths of test dataset
        transform (albumentations.augmentations.transforms): default transform
    """

    def __init__(self, root_dir, transform):
        super(TestDataset, self).__init__()
        self.transform = transform
        self.img_list = sorted(glob.glob(os.path.join(root_dir, '*.*')))

    def __len__(self):
        return len(self.img_list)
    
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        image = cv2.imread(self.img_list[idx])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        if self.transform:
            image = self.transform(image=image)['image']
            
        return image
        