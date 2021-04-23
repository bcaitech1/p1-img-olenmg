import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F 

class FocalLoss(nn.Module):
    """Focal loss

    FL(p _t) = - \alpha _t * (1 - p _t)^{\gamma} * \log(p _t)

    Args:
        gamma (float): gamma
        weight (Tensor): alpha _t
        reduction (str): reduction to apply to the output
    
    Shape:
        - Input: (N, C) where N is batch size and C is the number of classes
        - Output: (N)
    """

    def __init__(self, gamma, weight=None, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.weight = weight
        self.reduction = reduction

    def forward(self, input_tensor, target_tensor):
        log_prob = F.log_softmax(input_tensor, dim=-1)
        prob = torch.exp(log_prob)
        return F.nll_loss(
            ((1 - prob) ** self.gamma) * log_prob,
            target_tensor,
            weight=self.weight,
            reduction = self.reduction
        )


class MultiClassLoss(nn.Module):
    """Cross entropy for multi-category classification

    MCE = w_{mask} * CE(o_{mask}) + w_{gender} * CE(o_{gender}) + w_{age} * CE(o_{age})

    Args:
        weight (Tensor): rescaling weight given to each class

    Shape:
        - Input: tuple((N, 3), (N, 2), (N, 3)) where N is batch size. Each element is output of corresponding category.
        - Output: (N)
    """

    def __init__(self, weight=None):
        super(MultiClassLoss, self).__init__()
        self.weight = weight

    def forward(self, output, target):
        mask_loss = nn.CrossEntropyLoss()(output[0], target[0])
        gender_loss = nn.CrossEntropyLoss()(output[1], target[1])
        age_loss = nn.CrossEntropyLoss()(output[2], target[2])
        return mask_loss + gender_loss + age_loss


_criterion_entrypoints = {
    'cross_entropy': nn.CrossEntropyLoss,
    'focal': FocalLoss,
    'multi': MultiClassLoss,
}


#### Functions for ease of module loading ####

def criterion_entrypoint(criterion_name):
    return _criterion_entrypoints[criterion_name]


def is_criterion(criterion_name):
    return criterion_name in _criterion_entrypoints


def create_criterion(criterion_name, **kwargs):
    if is_criterion(criterion_name):
        create_fn = criterion_entrypoint(criterion_name)
        criterion = create_fn(**kwargs)
    else:
        raise RuntimeError('Unknown loss (%s)' % criterion_name)
    return criterion

##############################################