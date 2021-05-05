# P Stage 1 - Image Classification <!-- omit in toc -->

- [Requirements](#requirements)
  - [Dependencies](#dependencies)
  - [Install Requirements](#install-requirements)
- [Features](#features)
- [Folder Structure](#folder-structure)
- [Usage](#usage)
  - [Config file format](#config-file-format)
- [Model list](#model-list)
- [Loss list](#loss-list)
- [Dataset list](#dataset-list)
- [Transforms list](#transforms-list)
  - [aug_transform](#aug_transform)
- [Others(deprecated)](#othersdeprecated)

## Requirements  
### Dependencies
- Python >= 3.7.7
- PyTorch >= 1.6.0
- torchvision >= 0.7.0
- opencv-python == 4.5.1.48
- albumentations == 0.5.2
- numpy == 1.18.5
- pandas == 1.1.5
- scikit-learn == 0.24.1
- timm == 0.4.5
- tqdm == 4.59.0
  
### Install Requirements
- `pip install -r requirements.txt`
  
## Features
- `.json` config file support for convenient parameter tuning  

## Folder Structure
  ```
  code_T1033/
  │
  ├── train.py - training
  ├── inference.py - inference
  │
  ├── config/ - holds configurations for training (my submission)
  │   ├── config.json - config template
  │   ├── multihead_effb0_with_aug.json - model 1
  │   ├── multihead_resnext50_with_aug.json - model 2
  │   └── soft_voting_ensemble.json - ensemble above two
  │
  ├── model - models
  ├── loss.py - loss functions
  │
  ├── dataset.py - pytorch dataset
  ├── transforms.py - transform components for train/test
  │
  └── deprecated/ - meaningful attempt but not used(not refined)
      ├── efficientNet-v11(scheduler).ipynb
      ├── efficientNet-v12(without aug).ipynb
      └── hard_voting.ipynb
  ```

## Usage
Train  
`python train.py --config config.json`   
  
Inference  
`python inference.py --config config.json`
  
### Config file format  
```javascript
{
    "train": {
        "seed": 444,                // random seed
        "name": "base_model",       // name(used for saved file names)
        
        "epochs": 10,               // training epochs
        "dataset": "ImageDataset",  // select dataset
        "batch_size": 64,           // training batch size
        "model": "BaseModel",       // model
        "optimizer": "Adam",        // optimizer
        "lr": 3e-04,                // learning rate
        "val_ratio": 0.2,           // validation ratio (split ratio)
        "criterion": "focal",       // loss function
        "lr_decay_step": 5,         // decay step for learning scheduler

        "data_dir": "/opt/ml/input/data/train/",    // path of data
        "model_dir": "/opt/ml/weight"               // path to save model
    },

    "inference": {
        "batch_size": 63,           // inference batch size
        "model": "BaseModel",       // model 
        "inf_mode": "single",       // default: single/ ensemble: ensemble models
        "name": "base_model",       // file name for loading parameters

        "data_dir": "/opt/ml/input/data/eval/",     // path of data
        "model_dir": "/opt/ml/weight",              // path to load model parameters 
        "output_dir": "/opt/ml/submission"          // path to save submission csv file
    }
}
```

## Model list
- `BaseModel` - Pretrained EfficientNet B0 with single head classifier(18 classes)
- `Effb4` - Pretrained EfficientNet B4 with single head classifier(18 classes)
- `ResNeXt50` - Pretrained ResNeXt50_32x4d with single head classifier(18 classes)
- `ResNeXt101` - Pretrained ResNeXt101_32x8d with single head classifier(18 classes)
- `MultiHead_Effb0` - Pretrained EfficientNet B0 with multi head classifier(3/2/3 classes)
- `MultiHead_Effb4` - Pretrained EfficientNet B4 with multi head classifier(3/2/3 classes)
- `MultiHead_ResNeSXt50` - Pretrained ResNeXt50 with multi head classifier(3/2/3 classes)
  
## Loss list
- `FocalLoss` - Focal loss  
  <img src="https://render.githubusercontent.com/render/math?math=FL=-\alpha_{t}*(1-p_{t})^{\gamma}*\log(p_{t})">

- `MultiClassLoss` - Sum of `nn.CrossEntropyLoss` of three categories  
  <img src="https://render.githubusercontent.com/render/math?math=MCE=w_{mask}*CE(o_{mask})%2bw_{gender}*CE(o_{gender})%2bw_{age}*CE(o_{age})">

## Dataset list
- `ImageDataset` - Image dataset for multi-label classification (0-17)
- `MultiLabelDataset` - Image dataset for multi-category classification (0-2/0-1/0-2)
- `TestDataset` - Image dataset for test dataset

## Transforms list
- `transform_train` - default augmentation for training
- `transform_test`- default augmentation for validation(inference)
- `aug_transform` - augmentaion for 'minor-data-only' augmentation

### aug_transform
```python
aug_transform = A.Compose([
    A.CenterCrop(450, 250, p=1),
    A.RandomBrightnessContrast(brightness_limit=(0.1, 0.4),
                                            contrast_limit=(0.1, 0.4), p=0.7),
    A.HorizontalFlip(p=0.5),
    A.CLAHE(p=0.8),
    A.ShiftScaleRotate(rotate_limit=10, p=0.7),
    A.pytorch.transforms.ToTensor(),
])
```
Through many experiments, only augmentations that did not disturb prediction were selected.   
**This applies only to resampled minor data.**
  
## Others(deprecated)
The models(files) below were deprecated but they also showed quite good performance.  

- efficientNet-v11(scheduler) - highest performance with age classification
- efficientNet-v12(without aug) - second-highest performance with age classification
- hard_voting.ipynb - support hard voting with submission file   
  
These source code files are not refined yet.

**For more explanation, please refer to my Wrap-up report. :)**
