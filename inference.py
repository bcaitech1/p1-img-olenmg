import argparse
import os
import json
from types import SimpleNamespace
from importlib import import_module
from tqdm import tqdm

import numpy as np
import pandas as pd
import torch
import torch.nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from dataset import TestDataset
from transforms import transform_test

from efficientnet_pytorch import EfficientNet

def _load_model(model_dir, saved_model, num_classes, device, args):
    """Load model parameters on [model_dir/saved_model]

    Args:
        model_dir (str): path name containing model parameters
        saved_model (str): file name of desired parameter file
        num_classes (int): number of classes
        device (torch.device): device
        args (argparse): arguments
    Returns:
        model(with loaded parameters)
    """

    # -- if mode is 'single'
    if args.inf_mode == 'single':
        model_name = args.model
    # -- if mode is 'ensemble'
    else:
        model_name = 'MultiHead_Effb0' if 'Effb0' in saved_model \
            else 'MultiHead_ResNeXt50' if 'ResNeXt50' in saved_model \
            else 'MultiHead_Effb4'
    
    model_cls = getattr(import_module("model"), model_name)
    model = model_cls(num_classes=num_classes)

    model_path = os.path.join(model_dir, saved_model)
    print(model_path)
    
    model.load_state_dict(torch.load(model_path, map_location=device)) 
    return model


@torch.no_grad()
def predict(model, dataloader, device):
    """predict with single model

    Args:
        model (torch.nn.Module): model
        dataloader (toch.nn.utils.DataLoader): test data loader
        device (torch.device): device
    Returns:
        predictions (logit)
    """

    predictions = []

    model.eval()
    for images in tqdm(dataloader):
        with torch.no_grad():
            images = images.float().to(device)
            output = model(images)
            predicted = torch.max(output[0], 1)[1] * 6 + torch.max(output[1], 1)[1] * 3 + torch.max(output[2], 1)[1]
            predictions.extend(predicted.cpu().numpy().tolist())
                                
    return np.array(predictions)


@torch.no_grad()
def predict_ensemble(models, weights, dataloader, device, args):
    """predict with multiple models(Ensemble) based on soft voting

    Args:
        models (list[torch.nn.Module]): list of models to be ensemble
        weights (list[int]): voting weights
        dataloader (toch.nn.utils.DataLoader): test data loader
        device (torch.device): device
    Returns:
        predictions (logit)
    """
    
    predictions = []

    for model in models:
        model.eval()
    
    for images in tqdm(dataloader):
        with torch.no_grad():
            images = images.float().to(device)

            mask_prob = torch.zeros((args.batch_size, 3), device=device)
            gender_prob = torch.zeros((args.batch_size, 2), device=device)
            age_prob = torch.zeros((args.batch_size, 3), device=device)

            for weight, model in zip(weights, models):
                output = model(images)

                mask_prob += F.softmax(output[0], dim=1) * weight
                gender_prob += F.softmax(output[1], dim=1) * weight
                age_prob += F.softmax(output[2], dim=1) * weight

            predicted = torch.max(mask_prob, 1)[1] * 6 + torch.max(gender_prob, 1)[1] * 3 + torch.max(age_prob, 1)[1]
            predictions.extend(predicted.cpu().numpy().tolist())

    return np.array(predictions)


def inference(data_dir, model_dir, output_dir, submission_name, args):
    """Inference with single model

    Args:
        data_dir (str): path name containing test data
        model_dir (str): path name containing model parameters
        output_dir (str): path name to save submission file
        submission_name (str): desired name of submission file
        args (argparse): arguments
    Returns:
        None
    """

    # -- setting
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    img_root = os.path.join(data_dir, 'images')
    info_path = os.path.join(data_dir, 'info.csv')
    info = pd.read_csv(info_path).sort_values(by='ImageID')

    # -- dataset/dataloader
    test_dataset = TestDataset(root_dir=img_root, transform=transform_test)
    test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=1)

    # -- load lastest model parameters
    weight = []
    for f_name in os.listdir(f"{model_dir}/"):
        written_time = os.path.getctime(f"{model_dir}/{f_name}")
        if args.name in f_name:
            weight.append((f_name, written_time))

    model_path = sorted(weight, key=lambda x: x[1], reverse=True)[0][0]
    model = _load_model(model_dir, model_path, 18, device, args).to(device)

    # -- predict&save
    pred = predict(model, test_dataloader, device)

    info['ans'] = pred
    info.sort_index(inplace=True)
    
    info.to_csv(os.path.join(output_dir, submission_name), index=False)
    print('test inference is done.')


def inference_with_ensemble(data_dir, model_dir, output_dir, submission_name, args):
    """Inference with multiple models(Ensemble) based on soft voting

    Args:
        data_dir (str): path name containing test data
        model_dir (str): path name containing model parameters
        output_dir (str): path name to save submission file
        submission_name (str): desired name of submission file
        args (argparse): arguments
    Returns:
        None
    """

    # -- setting
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    img_root = os.path.join(data_dir, 'images')
    info_path = os.path.join(data_dir, 'info.csv')
    info = pd.read_csv(info_path).sort_values(by='ImageID')

    # -- dataset/dataloader
    test_dataset = TestDataset(root_dir=img_root, transform=transform_test)
    test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=1)

    # -- load models to be ensemble
    model_paths = ['MultiHead_Effb0_all_multihead_effb0_final_aug.pth',
                   'MultiHead_ResNeXt50_all_multihead_resnext50_final_aug.pth',]
    weights = [1.0, 1.0]
    models = []
    for path in model_paths:
        model = _load_model(model_dir, path, 3, device, args).to(device)
        models.append(model)

    print(f"-- Start Ensemble --")
    print(f"model list: {[os.path.split(model_path)[1] for model_path in model_paths]}\n")

    # -- predict&save
    pred = predict_ensemble(models, weights, test_dataloader, device, args)

    info['ans'] = pred
    info.sort_index(inplace=True)
    
    info.to_csv(os.path.join(output_dir, submission_name), index=False)
    print('test inference is done.')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # -- Data and model checkpoints directories
    parser.add_argument('--batch_size', type=int, default=63, help='input batch size for validing (default: 63)')
    parser.add_argument('--model', type=str, default='BaseModel', help='model type (default: BaseModel)')
    parser.add_argument('--inf_mode', type=str, default='single', help='single/ensemble (default: single)')
    parser.add_argument('--name', default='exp', help='model name')

    # -- Container environment
    parser.add_argument('--data_dir', type=str, default=os.environ.get('SM_CHANNEL_EVAL', '/opt/ml/input/data/eval'))
    parser.add_argument('--model_dir', type=str, default=os.environ.get('SM_CHANNEL_MODEL', '/opt/ml/weight'))
    parser.add_argument('--output_dir', type=str, default=os.environ.get('SM_OUTPUT_DATA_DIR', '/opt/ml/submission'))

    # Config
    parser.add_argument('--config', type=str, default=None, help='config file name')

    args = parser.parse_args()

    if args.config:
        with open(f"./config/{args.config}", "r") as json_file:
            args = json.load(json_file, object_hook=lambda d: SimpleNamespace(**d)).inference
            print(args)
    else:
        print(args)

    data_dir = args.data_dir # path name containing test data
    model_dir = args.model_dir # path name containing model parameters
    output_dir = args.output_dir # path name to save submission file
    submission_name = f"{args.name}_submission.csv" # name of submission file

    os.makedirs(output_dir, exist_ok=True)
    
    if args.inf_mode == 'single':
        inference(data_dir, model_dir, output_dir, submission_name, args)    
    else:
        inference_with_ensemble(data_dir, model_dir, output_dir, submission_name, args)