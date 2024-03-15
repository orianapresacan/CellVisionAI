import torch.nn as nn
import torch.nn.functional as F
import torch
import numpy as np


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

global alpha
global gamma
alpha = 0.8
gamma = 2

class FocalLoss(nn.Module):
    def forward(self, inputs, targets):
        # No flattening needed for segmentation tasks
        BCE = F.binary_cross_entropy_with_logits(inputs, targets, reduction='mean')
        BCE_EXP = torch.exp(-BCE)
        focal_loss = alpha * (1 - BCE_EXP)**gamma * BCE
        return focal_loss

class DiceLoss(nn.Module):
    def forward(self, inputs, targets, smooth=1):
        inputs = torch.sigmoid(inputs)  # Apply sigmoid to get probabilities
        # Compute Dice coefficient
        intersection = (inputs * targets).sum(dim=[2, 3])
        dice = (2. * intersection + smooth) / (inputs.sum(dim=[2, 3]) + targets.sum(dim=[2, 3]) + smooth)
        return 1 - dice.mean()


def criterion(x, y):
    """ Combined dice and focal loss.
    ARGS:
        x: (torch.Tensor) the model output
        y: (torch.Tensor) the target
    RETURNS:
        (torch.Tensor) the combined loss

    """
    focal, dice = FocalLoss(), DiceLoss()
    y = y.to(DEVICE)
    x = x.to(DEVICE)
    return 20 * focal(x, y) + dice(x, y)


def binary_iou(y_true, y_pred):
    """
    Calculate Intersection over Union (IoU) for binary segmentation masks.
    
    Args:
        y_true (np.ndarray): Ground truth binary mask.
        y_pred (np.ndarray): Predicted binary mask.
    
    Returns:
        float: IoU score.
    """
    intersection = np.logical_and(y_true, y_pred).sum()
    union = np.logical_or(y_true, y_pred).sum()
    if union == 0:
        return 0
    else:
        iou = intersection / union
        return iou