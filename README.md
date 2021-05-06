# P Stage 1 - Image Classification <!-- omit in toc -->

- [Background](#background)
- [Input/Output](#inputoutput)
- [Requirements](#requirements)
  - [Dependencies](#dependencies)
  - [Install Requirements](#install-requirements)
- [Features](#features)
- [Folder Structure](#folder-structure)
- [Usage](#usage)
  - [Config file format](#config-file-format)
- [Model](#model)
  - [List](#list)
  - [Model architecture](#model-architecture)
- [Loss list](#loss-list)
- [Dataset list](#dataset-list)
- [Transforms list](#transforms-list)
  - [aug_transform](#aug_transform)
- [Others(deprecated)](#othersdeprecated)

## Background
COVID-19의 확산으로 우리나라는 물론 전 세계 사람들은 경제적, 생산적인 활동에 많은 제약을 가지게 되었습니다. 우리나라는 COVID-19 확산 방지를 위해 사회적 거리 두기를 단계적으로 시행하는 등의 많은 노력을 하고 있습니다. 과거 높은 사망률을 가진 사스(SARS)나 에볼라(Ebola)와는 달리 COVID-19의 치사율은 오히려 비교적 낮은 편에 속합니다. 그럼에도 불구하고, 이렇게 오랜 기간 동안 우리를 괴롭히고 있는 근본적인 이유는 바로 COVID-19의 강력한 전염력 때문입니다.

감염자의 입, 호흡기로부터 나오는 비말, 침 등으로 인해 다른 사람에게 쉽게 전파가 될 수 있기 때문에 감염 확산 방지를 위해 무엇보다 중요한 것은 모든 사람이 마스크로 코와 입을 가려서 혹시 모를 감염자로부터의 전파 경로를 원천 차단하는 것입니다. 이를 위해 공공 장소에 있는 사람들은 반드시 마스크를 착용해야 할 필요가 있으며, 무엇 보다도 코와 입을 완전히 가릴 수 있도록 올바르게 착용하는 것이 중요합니다. 하지만 넓은 공공장소에서 모든 사람들의 올바른 마스크 착용 상태를 검사하기 위해서는 추가적인 인적자원이 필요할 것입니다.

따라서, 우리는 카메라로 비춰진 사람 얼굴 이미지 만으로 이 사람이 마스크를 쓰고 있는지, 쓰지 않았는지, 정확히 쓴 것이 맞는지 자동으로 가려낼 수 있는 시스템이 필요합니다. 이 시스템이 공공장소 입구에 갖춰져 있다면 적은 인적자원으로도 충분히 검사가 가능할 것입니다.

## Input/Output
Input: pictures of people
- with mask
- without mask
- wrong wearing (i.e., not cover nose)

Output: 18 classes  
![image](https://user-images.githubusercontent.com/61135159/117261159-cba15580-ae8a-11eb-8273-1032d45f62c5.png)

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

## Model
### List
- `BaseModel` - Pretrained EfficientNet B0 with single head classifier(18 classes)
- `Effb4` - Pretrained EfficientNet B4 with single head classifier(18 classes)
- `ResNeXt50` - Pretrained ResNeXt50_32x4d with single head classifier(18 classes)
- `ResNeXt101` - Pretrained ResNeXt101_32x8d with single head classifier(18 classes)
- `MultiHead_Effb0` - Pretrained EfficientNet B0 with multi head classifier(3/2/3 classes)
- `MultiHead_Effb4` - Pretrained EfficientNet B4 with multi head classifier(3/2/3 classes)
- `MultiHead_ResNeXt50` - Pretrained ResNeXt50 with multi head classifier(3/2/3 classes)

### Model architecture
In the end, I used three head with backbone model(EfficientNetb0, ResNeXt50) and ensembled them.  

``` python
# model example
class MultiHead_Model(nn.Module):
    def __init__(self, num_classes):
        super(MultiHead_Model, self).__init__()
        self.model = BACKBONE
        # i.e. self.model = timm.create_model('resnext50_32x4d', pretrained=True)

        n_features = self.model.num_features # output features of backbone model
        self.mask_classifier = timm.models.layers.ClassifierHead(n_features, 3)
        self.gender_classifier = timm.models.layers.ClassifierHead(n_features, 2)
        self.age_classifier = timm.models.layers.ClassifierHead(n_features, 3)

    def forward(self, x):
        features = self.model.forward_features()
        x = self.mask_classifier(features)
        y = self.gender_classifier(features)
        z = self.age_classifier(features)

        return x, y, z
```

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
  
These source code files are not refined.

**For more explanation, please refer to my [Wrap-up report](./wrapup.pdf). :)**