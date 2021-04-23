import argparse
import glob
import os
import random
import copy
import time
import json
from types import SimpleNamespace
from tqdm import tqdm
from importlib import import_module

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score

from loss import create_criterion
from dataset import ImageDataset
from transforms import *

EXCEPT_DATA = ['004432_male_Asian_43', '001498-1_male_Asian_23', 
                '006359_female_Asian_18', '006360_female_Asian_18', 
                '006361_female_Asian_18', '006362_female_Asian_18',
                '006363_female_Asian_18', '006364_female_Asian_18']

def _seed_everything(seed):
    """Fix the seed

    Args:
        seed (int): seed number

    Returns:
        None
    """

    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    # torch.cuda.manual_seed_all(seed) # -- when using multi-GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)


def _label2label(mask, gender, age):
    """Convert multi-categorized label to single label

    Args:
        mask (int): mask label (0-2)
        gender (int): gender label (0-1)
        age (int): age label(0-2)

    Returns:
        single label
    """
    return mask * 6 + gender * 3 + age


def _path2label(img_paths, mode=0):
    """Labeling on image path

    Args:
        img_paths (list[str]): image paths
        mode (int): 0 - return 0-17 (single label)
                    1 - return 0-2/0-1/0-2 (multi category)

    Returns:
        label list
    """

    labels = []
    for img_path in img_paths:
        # Extract necessary word
        path, file = os.path.split(img_path)
        gender, _, age = path.split('/')[7].split('_')[1:]
        mask = file[0]

        # Encoding
        age_logit = 0 if int(age) < 30 else \
                    1 if int(age) < 58 \
                    else 2
        gender_logit = 0 if gender == 'male' else 1 
        mask_logit = 0 if mask == 'm' else \
                     1 if mask == 'i' else \
                     2

        if mode == 0:
            labels.append(mask_logit * 6 + gender_logit * 3 + age_logit)
        else:
            labels.append([mask_logit, gender_logit, age_logit])
        
    return labels


def _augment_minor_data(df_original, target):
    """Resampling(augment) only minor data

    Args:
        df_original(pd.DataFrame): dataframe of train data(include image path, label)
        target (str): column name of label(criteria)
    Returns:
        label list
    """

    df = df_original.copy()
    max_val = df[target].value_counts().astype('int64').max()

    for key, value in df[target].value_counts().items():
        target_df = df[df[target] == key]

        # Resample only minor data
        if value < max_val:
            target_df = target_df.sample(frac=(max_val-value)/value, replace=True)

            # 'aug_target == True' means this is a resampled row.
            target_df['aug_target'] = True

            df = df.append(target_df, ignore_index=True)

    return df


def _split_dataset(data_dir, csv_path, val_ratio, seed=444, augment=False):
    """Split dataset based on age&gender

    Photos of the same person are not be included both train dataset and test dataset
    
    Args:
        data_dir (str): path name containing train data
        csv_path (str): path cotaining train csv file
        val_ratio (str): size of validation dataset
        seed (int): seed number
        augment (bool): if true, execute _augment_minor_data function
    Returns:
        train_aug (list[bool]): aug_target 
        train_path, test_path, train_label, test_label
    """

    df = pd.read_csv(csv_path)

    # -- Delete abnormal data & Drop unnecessary columns
    df = df[df['path'].apply(lambda x: x not in EXCEPT_DATA)]
    df.drop(['id', 'race'], axis=1, inplace=True)
    
    # -- Image name to full path, Label encoding, aug_target setting
    df['path'] = data_dir + 'images/' + df['path']
    df['age'] = df['age'].apply(lambda x: 0 if int(x) < 30 else \
                                          1 if int(x) < 58 else 2)
    df['gender'] = df['gender'].apply(lambda x: 0 if x == 'male' else 1)
    df['aug_target'] = pd.Series(np.zeros(len(df)).astype('bool'))

    # -- data split based on age&gender, don't care whether to wear a mask
    df['semi_label'] = df['gender'] * 3 + df['age']
    df_label = df['semi_label'].tolist()
    train_df, test_df, _, _ = train_test_split(
                                    df,
                                    df_label,
                                    test_size=val_ratio,
                                    random_state=seed, 
                                    shuffle=True, 
                                    stratify=df_label
                              )

    if augment: # minor-data-only augmentation
        train_df = _augment_minor_data(train_df, 'semi_label')

    # -- Load all paths of dataset (mask1, mask2, ..., incorrect_mask, normal)
    train_aug = []
    train_path = []
    for target, path in zip(train_df['aug_target'], train_df['path']):
        train_path.extend(glob.glob(path + '/*.*'))
        if target:
            train_aug.extend([True] * 7)
        else:
            train_aug.extend([False] * 7)

    test_path = []
    for path in test_df['path']:
        test_path.extend(glob.glob(path + '/*.*'))

    # -- Labeling by path name
    train_label = _path2label(train_path, mode=1)
    test_label = _path2label(test_path, mode=1)

    return train_aug, train_path, test_path, train_label, test_label


def train(data_dir, model_dir, args):
    """Model training
    
    Args:
        data_dir (str): path name containing train data
        model_dir (str): path name to save model parameters
        args (argparse): rguments
    Returns:
        None
    """

    # -- settings
    _seed_everything(args.seed)
    use_cuda = torch.cuda.is_available()
    device = torch.device('cuda' if use_cuda else 'cpu')
    
    # -- dataset(with split)
    train_aug, X_train, X_test, y_train, y_test = _split_dataset(
                                data_dir=data_dir,
                                csv_path=os.path.join(data_dir, 'train.csv'),
                                val_ratio=args.val_ratio,
                                seed=args.seed,
                                augment=True
                            )
    print(len(X_train), len(X_test))

    dataset_module = getattr(import_module("dataset"), args.dataset)
    train_dataset = dataset_module(
        img_list=X_train,
        label_list=y_train,
        transform=transform_train,
        aug_transform=aug_transform,
        aug_target=train_aug,
    )
    test_dataset = dataset_module(
        img_list=X_test,
        label_list=y_test,
        transform=transform_test,
        aug_transform=None,
        aug_target=None,
    )

    # -- dataloader
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        num_workers=2,
        shuffle=True,
        pin_memory=use_cuda,
        drop_last=True
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        num_workers=2,
        shuffle=False,
        pin_memory=use_cuda,
        drop_last=True
    )
    dataloaders = {'train': train_loader,
                   'test': test_loader}

    # -- model
    model_module = getattr(import_module("model"), args.model)
    model = model_module(num_classes=18).to(device)
    # model = torch.nn.DataParallel(model) # for multi-GPU

    # -- loss & metric
    criterion = None
    if args.criterion == 'focal':
        criterion = create_criterion(
            args.criterion,
            gamma=5,
        )
    else:
        criterion = create_criterion(
            args.criterion,
            weight=None,
        )

    opt_module = getattr(import_module("torch.optim"), args.optimizer)
    optimizer = opt_module(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.lr,
        weight_decay=0.0
    )
    scheduler = ReduceLROnPlateau(optimizer, factor=0.1, patience=args.lr_decay_step, verbose=True)

    # -- Save best model parameters based on valid f1 score
    best_model_weights = copy.deepcopy(model.state_dict())
    best_f1 = 0.0
    best_epoch = 0

    since = time.time()
    for epoch in range(args.epochs):
        print(f"[Epoch {epoch + 1}/{args.epochs}] START")

        for phase in ['train', 'test']:
            if phase == 'train':
                model.train()
            else:
                model.eval()

            running_loss = 0.0
            running_corrects = 0
            running_true = []
            running_pred = []

            for inputs, labels in tqdm(dataloaders[phase]):
                inputs = inputs.to(device)
                labels = tuple([label.to(device) for label in labels])
                gt_labels = labels[0] * 6 + labels[1] * 3 + labels[2]

                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    preds = torch.max(outputs[0], 1)[1] * 6 + torch.max(outputs[1], 1)[1] * 3 + torch.max(outputs[2], 1)[1]
                    loss = criterion(outputs, labels)
                    
                    running_true.extend(gt_labels.cpu().numpy())
                    running_pred.extend(preds.cpu().numpy())

                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == gt_labels.data)
                
            if scheduler and phase == 'train':
                scheduler.step(running_loss)

            epoch_loss = running_loss / len(dataloaders[phase].dataset)
            epoch_acc = 100. * running_corrects.double() / len(dataloaders[phase].dataset)
            epoch_f1 = f1_score(running_true, running_pred, average='macro')

            print(f"[Epoch {epoch + 1}/{args.epochs}] {phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f} F1: {epoch_f1:.4f}")

            if phase == 'test' and epoch_f1 > best_f1:
                best_epoch = epoch
                best_f1 = epoch_f1
                best_model_weights = copy.deepcopy(model.state_dict())

    time_elapsed = time.time() - since
    print(f"Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s")
    print(f"Best val F1: {best_f1:.4f}")
    print('\n')

    torch.save(model.state_dict(), f"{args.model_dir}/{args.model}_{args.name}.pth")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # Setting
    parser.add_argument('--seed', type=int, default=444, help='random seed (default: 444)')
    parser.add_argument('--name', default='exp', help='model save at {SM_MODEL_DIR}/{name}')

    # Relevant to model training
    parser.add_argument('--epochs', type=int, default=10, help='number of epochs to train (default: 10)')
    parser.add_argument('--dataset', type=str, default='ImageDataset', help='dataset type (default: ImageDataset)')
    parser.add_argument('--batch_size', type=int, default=16, help='input batch size for training (default: 16)')
    parser.add_argument('--model', type=str, default='BaseModel', help='model type (default: BaseModel)')
    parser.add_argument('--optimizer', type=str, default='Adam', help='optimizer type (default: Adam)')
    parser.add_argument('--lr', type=float, default=3e-4, help='learning rate (default: 3e-4)')
    parser.add_argument('--val_ratio', type=float, default=0.2, help='ratio for validaton (default: 0.2)')
    parser.add_argument('--criterion', type=str, default='multi', help='criterion type (default: multi)')
    parser.add_argument('--lr_decay_step', type=int, default=5, help='learning rate scheduler decay step (default: 5)')

    # Container environment
    parser.add_argument('--data_dir', type=str, default=os.environ.get('SM_CHANNEL_TRAIN', '/opt/ml/input/data/train/'))
    parser.add_argument('--model_dir', type=str, default=os.environ.get('SM_MODEL_DIR', '/opt/ml/weight'))

    # Config
    parser.add_argument('--config', type=str, default=None, help='config file name')

    args = parser.parse_args()

    if args.config:
        with open(f"./config/{args.config}", "r") as json_file:
            args = json.load(json_file, object_hook=lambda d: SimpleNamespace(**d)).train
            print(args)
    else:
        print(args)

    data_dir = args.data_dir
    model_dir = args.model_dir

    train(data_dir, model_dir, args)
