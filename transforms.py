import albumentations as A
import albumentations.pytorch

# -- default augmentation for training
transform_train = A.Compose([
    A.CenterCrop(450, 250, p=1),
    A.pytorch.transforms.ToTensor(),
])

# -- default augmentation for validation(inference)
transform_test = A.Compose([
    A.CenterCrop(450, 250, p=1),
    A.pytorch.transforms.ToTensor(),
])

# -- augmentaion for 'minor-data-only' augmentation
aug_transform = A.Compose([
    A.CenterCrop(450, 250, p=1),
    A.RandomBrightnessContrast(brightness_limit=(0.1, 0.4),
                                            contrast_limit=(0.1, 0.4), p=0.7),
    A.HorizontalFlip(p=0.5),
    A.CLAHE(p=0.8),
    A.ShiftScaleRotate(rotate_limit=10, p=0.7),
    A.pytorch.transforms.ToTensor(),
])
