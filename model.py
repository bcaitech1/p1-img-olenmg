import torch
import torch.nn as nn
import torch.nn.functional as F

import timm
from efficientnet_pytorch import EfficientNet

class BaseModel(nn.Module):
    """
    Pretrained EfficientNet B0 with single head classifier(18 classes)
    """

    def __init__(self, num_classes):
        super(BaseModel, self).__init__()
        self.model = EfficientNet.from_pretrained('efficientnet-b0', num_classes=num_classes)

    def forward(self, x):
        x = self.model(x)
        return x


class Effb4(nn.Module):
    """
    Pretrained EfficientNet B4 with single head classifier(18 classes)
    """

    def __init__(self, num_classes):
        super(Effb4, self).__init__()
        self.model = EfficientNet.from_pretrained('efficientnet-b4', num_classes=num_classes)

    def forward(self, x):
        x = self.model(x)
        return x


class ResNeXt50(nn.Module):
    """
    Pretrained ResNeXt50_32x4d with single head classifier(18 classes)
    """

    def __init__(self, num_classes=3):
        super(ResNeXt50, self).__init__()
        self.model = timm.create_model('resnext50_32x4d', pretrained=True, num_classes=num_classes)

    def forward(self, x):
        x = self.model(x)
        return x


class ResNeXt101(nn.Module):
    """
    Pretrained ResNeXt101_32x8d with single head classifier(18 classes)
    """

    def __init__(self, num_classes=3):
        super(ResNeXt101, self).__init__()
        self.model = timm.create_model('resnext101_32x8d', pretrained=True, num_classes=num_classes)

    def forward(self, x):
        x = self.model(x)
        return x


class MultiHead_Effb0(nn.Module):
    """
    Pretrained EfficientNet B0 with multi head classifier(3/2/3 classes)
    """

    def __init__(self, num_classes):
        super(MultiHead_Effb0, self).__init__()
        self.model = timm.create_model('efficientnet_b0', pretrained=True)
        n_features = self.model.num_features
        self.mask_classifier = timm.models.layers.ClassifierHead(n_features, 3)
        self.gender_classifier = timm.models.layers.ClassifierHead(n_features, 2)
        self.age_classifier = timm.models.layers.ClassifierHead(n_features, 3)

    def forward(self, x):
        features = self.model.forward_features(x)
        x = self.mask_classifier(features)
        y = self.gender_classifier(features)
        z = self.age_classifier(features)

        return x, y, z


class MultiHead_Effb4(nn.Module):
    """
    Pretrained EfficientNet B4 with multi head classifier(3/2/3 classes)
    """

    def __init__(self, num_classes):
        super(MultiHead_Effb4, self).__init__()
        self.model = EfficientNet.from_pretrained('efficientnet-b4', include_top=False)
        n_features = 1792
        self.mask_classifier = timm.models.layers.ClassifierHead(n_features, 3)
        self.gender_classifier = timm.models.layers.ClassifierHead(n_features, 2)
        self.age_classifier = timm.models.layers.ClassifierHead(n_features, 3)

    def forward(self, x):
        features = self.model(x)
        x = self.mask_classifier(features)
        y = self.gender_classifier(features)
        z = self.age_classifier(features)

        return x, y, z


class MultiHead_ResNeXt50(nn.Module):
    """
    Pretrained ResNeXt50 with multi head classifier(3/2/3 classes)
    """

    def __init__(self, num_classes):
        super(MultiHead_ResNeXt50, self).__init__()
        self.model = timm.create_model('resnext50_32x4d', pretrained=True)
        n_features = self.model.num_features
        self.mask_classifier = timm.models.layers.ClassifierHead(n_features, 3)
        self.gender_classifier = timm.models.layers.ClassifierHead(n_features, 2)
        self.age_classifier = timm.models.layers.ClassifierHead(n_features, 3)

    def forward(self, x):
        features = self.model.forward_features()
        x = self.mask_classifier(features)
        y = self.gender_classifier(features)
        z = self.age_classifier(features)

        return x, y, z
