import torch
import torch.nn.functional as F


def binary_dice(pred, target, smooth=1):
    """
    Computes the Dice Loss for binary segmentation.
    Args:
        pred: Tensor of predictions (batch_size, 1, H, W).
        target: Tensor of ground truth (batch_size, 1, H, W).
        smooth: Smoothing factor to avoid division by zero.
    Returns:
        Scalar Dice Loss.
    """
    # Apply sigmoid to convert logits to probabilities
    pred = torch.sigmoid(pred)

    # Calculate intersection and union
    intersection = (pred * target).sum(dim=(2, 3))
    union = pred.sum(dim=(2, 3)) + target.sum(dim=(2, 3))

    # Compute Dice Coefficient
    dice = (2.0 * intersection + smooth) / (union + smooth)

    # Return Dice Loss
    return 1 - dice.mean()


def binary_diceCE(
    pred: torch.Tensor, target: torch.Tensor, dice_weight: float, ce_weight: float
):
    """Computes the weighted sum of the Dice Loss and the Cross Entropy loss
    for a binary segmentation task.
    !!!Warning: The function assumes that the input is the predicted logits of the model.
    Both Dice loss and BCE loss apply a Sigmoid function internally.

    Parameters
    ----------
    pred : Tensor
        Tensor of predictions (batch_size, 1, H, W).
    target : Tensor
        Tensor of ground truth (batch_size, 1, H, W).
    dice_weight : float
        Weight to apply to the output of the dice loss
    ce_weight : float
        Weight to apply to the output of the BCE loss

    Returns
    -------
    Tensor
        The weighted loss value.
    """

    bce_loss = F.binary_cross_entropy_with_logits(pred, target)
    dice_loss = binary_dice(pred, target)

    diceCE_loss = dice_weight * dice_loss + ce_weight * bce_loss
    return diceCE_loss
