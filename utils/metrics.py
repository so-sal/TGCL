import torch
import numpy as np
def soft_dice_per_class(pred_probs, target_onehot, eps=1e-6):
    B, C = pred_probs.shape[:2]
    dice_list = []
    for c in range(C):
        intersection = (pred_probs[:, c] * target_onehot[:, c]).sum(dim=(1,2,3))
        denominator  = (pred_probs[:, c].sum(dim=(1,2,3)) +
                        target_onehot[:, c].sum(dim=(1,2,3)))
        dice = (2.0 * intersection + eps) / (denominator + eps)
        dice_list.append(dice.mean().item())
    return dice_list


def hard_dice_per_class(pred_probs, target_onehot, eps=1e-6):
    """
    Calculate Hard Dice Score for each class using argmax-based predictions
    Args:
        pred_probs: Model's predicted probabilities (softmax outputs) [B, C, D, H, W]
        target_onehot: Ground truth (one-hot encoded) [B, C, D, H, W]
        eps: Small value to avoid division by zero

    Returns:
        dice_list: List of Dice Scores for each class
    """
    B, C = pred_probs.shape[:2]  # Batch size and number of classes
    dice_list = []

    # Convert softmax probabilities to hard predictions using argmax
    pred_hard = pred_probs.argmax(dim=1)  # [B, D, H, W]
    target_hard = target_onehot.argmax(dim=1)  # [B, D, H, W]

    dice_list = []
    for case_idx in range(B):  # Iterate over each case
        case_dice = []
        for c in range(C):  # Iterate over each class
            # Intersection and union for the current class
            intersection = ((pred_hard[case_idx] == c) & (target_hard[case_idx] == c)).sum()  # Per batch
            denominator = ((pred_hard[case_idx] == c).sum() +
                        (target_hard[case_idx] == c).sum())

            # Dice Score calculation
            dice = (2.0 * intersection + eps) / (denominator + eps)

            if (target_hard[case_idx] == c).sum() > 0:
                case_dice.append(dice.item())
            else:
                case_dice.append(float('nan'))
        dice_list.append(case_dice)
    return np.nanmean(np.vstack(dice_list), axis=0)
