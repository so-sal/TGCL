import torch
from utils.metrics import soft_dice_per_class, hard_dice_per_class
from torch import nn

def train_step(model, optimizer, img, label, crop, criterion_segmentation=None, criterion_classification=None,
               label_classification=False, weight_segmentation=1.00, max_norm=1.0, mode='segmentation', DeepSupervision=False):
    """
    Perform a single training step for segmentation and classification.
    
    Args:
        model: The model being trained.
        optimizer: Optimizer for the model.
        criterion_segmentation: Loss function for segmentation.
        img: Input image tensor.
        label: Ground truth segmentation labels.
        crop: Cropped region positions for the images.
        args: Argument object with configuration like weight_segmentation.
        max_norm: Gradient clipping max norm.
        mode: Mode of training ('segmentation', 'classification', 'both').
    
    Returns:
        loss: Computed loss for this step.
        dice_vals: List of Dice scores for the current batch.
    """
    optimizer.zero_grad()

    # Forward pass
    prediction = model(img, crop_pos=crop)
    
    # Compute segmentation loss
    loss = 0
    if mode in ['segmentation', 'both']:
        loss += criterion_segmentation(
            prediction['segmentation'], label #.argmax(axis=1)
        ) * weight_segmentation
    if mode in ['classification', 'both']:
        loss += criterion_classification(
            prediction['classification'], 
            label_classification.argmax(dim=1)
        ) * (1-weight_segmentation)
    
    # Combined loss if needed (commented out as per your original code)
    # Compute Dice scores
    if DeepSupervision:
        dice_vals = hard_dice_per_class(torch.softmax(prediction['segmentation'][0], dim=1), label)
    else:
        dice_vals = hard_dice_per_class(torch.softmax(prediction['segmentation'], dim=1), label)

    # Backward pass and optimization step
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=max_norm)
    optimizer.step()

    return loss.item(), dice_vals

def validation_step(model, img, label, crop, criterion_segmentation=None, criterion_classification=None,
                    label_classification=False, weight_segmentation=1.00, mode='segmentation'):
    """
    Perform a single validation step for segmentation and classification.
    
    Args:
        model: The model being validated.
        img: Input image tensor.
        label: Ground truth segmentation labels.
        crop: Cropped region positions for the images.
        criterion_segmentation: Loss function for segmentation.
        criterion_classification: Loss function for classification.
        label_classification: Classification labels.
        weight_segmentation: Weight for segmentation loss.
        mode: Mode of validation ('segmentation', 'classification', 'both').
    
    Returns:
        loss: Computed loss for this step.
        dice_vals: List of Dice scores for the current batch.
    """
    with torch.no_grad():  # Disable gradient computation
        # Forward pass
        prediction = model(img, crop_pos=crop)
        
        # Compute segmentation loss
        loss = 0
        if mode in ['segmentation', 'both']:
            loss += criterion_segmentation(
                prediction['segmentation'], 
                label #.argmax(axis=1).unsqueeze(1)
            ) * weight_segmentation
        if mode in ['classification', 'both']:
            loss += criterion_classification(
                prediction['classification'], 
                label_classification.argmax(dim=1)
            ) * (1-weight_segmentation)
        
        # Compute Dice scores
        dice_vals = hard_dice_per_class(
            torch.softmax(prediction['segmentation'], dim=1), 
            label
        )

    return loss.item(), dice_vals

def safe_item(x, default=0.0):
    if x is None:
        return default
    if isinstance(x, torch.Tensor):
        return x.item() if x.numel() == 1 else x.cpu()
    return x


def train_step_TGCF(model, optimizer, MRI_img, TRUS_img, MRI_label, TRUS_label, MRI_crop, TRUS_crop,
               criterion_segmentation, weight_segmentation_TRUS=0.8, weight_segmentation=1.0, weight_contrastive=1.0,
               weight_classification=1.0, max_norm=1.0, weight_self_supervision=0.1, DeepSupervision=False, criterion_classification=None):
    """
    model, optimizer, MRI_img, TRUS_img, MRI_label, TRUS_label, MRI_crop, TRUS_crop, criterion_segmentation = TGCF_MODEL, optimizer, MRI_Input_subset, TRUS_Input_subset, MRI_Label_subset, TRUS_Label_subset, MRI_Crop_subset, TRUS_Crop_subset, criterion_segmentation_ds
    Single training step for TGCF_MODEL (MRI + TRUS segmentation with contrastive loss).

    Args:
        model: TGCF model returning a dict with keys 'MRI_segmentation', 'TRUS_segmentation', 'contrastive_loss', 'loss_MRI_cancer_intra', 'loss_TRUS_cancer_intra', 'loss_MRI_cancer_inter', 'loss_TRUS_cancer_inter'.
        optimizer: Optimizer.
        MRI_img, TRUS_img: Input tensors for MRI and TRUS.
        MRI_label, TRUS_label: Ground truth segmentation labels.
        crop: Crop position information.
        criterion_segmentation: Loss function for segmentation.
        weight_segmentation: Weight for segmentation loss.
        weight_contrastive: Weight for contrastive loss.
        max_norm: Gradient clipping max norm.
        DeepSupervision: If True, use first element of deep supervision outputs.

    Returns:
        total_loss: Scalar loss value.
        dice_MRI: Dice scores for MRI segmentation.
        dice_TRUS: Dice scores for TRUS segmentation.
    """
    optimizer.zero_grad()
    
    # Forward pass: model returns separate segmentation outputs and contrastive loss.
    prediction = model(MRI_img, TRUS_img, MRI_label, TRUS_label, MRI_crop, TRUS_crop)
    
    if DeepSupervision:
        seg_MRI = prediction['MRI_segmentation']
        seg_TRUS = prediction['TRUS_segmentation']
    else:
        seg_MRI = prediction['MRI_segmentation'][0]
        seg_TRUS = prediction['TRUS_segmentation'][0]
    
    # Compute segmentation losses for MRI and TRUS branches.
    loss_seg_MRI = criterion_segmentation(seg_MRI, MRI_label)
    loss_seg_TRUS = criterion_segmentation(seg_TRUS, TRUS_label)
    loss_seg = weight_segmentation * (loss_seg_MRI * (1-weight_segmentation_TRUS) + loss_seg_TRUS * weight_segmentation_TRUS)
    
    # Contrastive loss is assumed to be computed inside the model forward.
    if weight_contrastive > 0.0:
        loss_contrastive = weight_contrastive * prediction['contrastive_loss']
        total_loss = loss_seg + loss_contrastive
    else:
        loss_contrastive = 0.0
        total_loss = loss_seg
    
    loss_self_supervision_MRI = prediction['loss_MRI_cancer_intra'] * weight_self_supervision
    loss_self_supervision_TRUS = prediction['loss_TRUS_cancer_intra'] * weight_self_supervision
    total_loss += loss_self_supervision_MRI + loss_self_supervision_TRUS

    # classification loss
    MRI_label[:,2].sum(axis=(1,2,3)) > 0
    if criterion_classification is not None:
        target_MRI  = (MRI_label [:,2].sum(axis=(1,2,3)) > 0).long()
        target_TRUS = (TRUS_label[:,2].sum(axis=(1,2,3)) > 0).long()
        loss_cls_MRI  = criterion_classification(prediction['MRI_classification' ], target_MRI)
        loss_cls_TRUS = criterion_classification(prediction['TRUS_classification'], target_TRUS)
        total_loss += weight_classification * (loss_cls_MRI * (1-weight_segmentation_TRUS) + loss_cls_TRUS * weight_segmentation_TRUS)

    
    total_loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=max_norm)
    optimizer.step()
    
    # Compute Dice scores (assumes hard_dice_per_class is defined elsewhere).
    if DeepSupervision:
        dice_MRI = hard_dice_per_class(torch.softmax(seg_MRI[0], dim=1), MRI_label)
        dice_TRUS = hard_dice_per_class(torch.softmax(seg_TRUS[0], dim=1), TRUS_label)
    else:
        dice_MRI = hard_dice_per_class(torch.softmax(seg_MRI, dim=1), MRI_label)
        dice_TRUS = hard_dice_per_class(torch.softmax(seg_TRUS, dim=1), TRUS_label)
    
    return {
        'total_loss': safe_item(total_loss),
        'seg_loss': safe_item(loss_seg),
        'TGCF_loss': safe_item(loss_contrastive),
        'self_loss': {'MRI':safe_item(loss_self_supervision_MRI), 'TRUS':safe_item(loss_self_supervision_TRUS)},
        'dice_MRI': dice_MRI,
        'dice_TRUS': dice_TRUS
    }


def validation_step_TGCF(model, MRI_img, TRUS_img, MRI_label, TRUS_label, MRI_crop, TRUS_crop,
                         criterion_segmentation, weight_segmentation_TRUS=0.8, weight_segmentation=1.0, 
                         weight_contrastive=1.0, weight_self_supervision={'MRI':1, 'TRUS':1}):
    """
    model, optimizer, MRI_img, TRUS_img, MRI_label, TRUS_label, MRI_crop, TRUS_crop, criterion_segmentation = TGCF_MODEL, optimizer, MRI_Input_subset, TRUS_Input_subset, MRI_Label_subset, TRUS_Label_subset, MRI_Crop_subset, TRUS_Crop_subset, criterion_segmentation_ds
    MRI_img, TRUS_img, MRI_label, TRUS_label, MRI_crop, TRUS_crop = MRI_Input, TRUS_Input, MRI_Label, TRUS_Label, MRI_Crop, TRUS_Crop
    
    Perform a single validation step for TGCF_MODEL (MRI + TRUS segmentation with contrastive loss).

    Args:
        model: TGCF model returning a dict with keys 'MRI_segmentation', 'TRUS_segmentation', 'contrastive_loss'.
        MRI_img, TRUS_img: Input tensors for MRI and TRUS.
        MRI_label, TRUS_label: Ground truth segmentation labels.
        MRI_crop, TRUS_crop: Crop position information.
        criterion_segmentation: Loss function for segmentation.
        weight_segmentation: Weight for segmentation loss.
        weight_contrastive: Weight for contrastive loss.
    Returns:
        Dictionary containing loss values and Dice scores.
    """
    with torch.no_grad():  # Disable gradient computation
        prediction = model(MRI_img, TRUS_img, MRI_label, TRUS_label, MRI_crop, TRUS_crop)

        seg_MRI = prediction['MRI_segmentation']
        seg_TRUS = prediction['TRUS_segmentation']

        # Compute segmentation losses for MRI and TRUS
        loss_seg_MRI = criterion_segmentation(seg_MRI, MRI_label)
        loss_seg_TRUS = criterion_segmentation(seg_TRUS, TRUS_label)
        loss_seg = weight_segmentation * (loss_seg_MRI * (1 - weight_segmentation_TRUS) + 
                                          loss_seg_TRUS * weight_segmentation_TRUS)

        # Contrastive loss (computed inside model forward)
        loss_contrastive = weight_contrastive * prediction['contrastive_loss']
        total_loss = loss_seg + loss_contrastive

        loss_self_supervision_MRI = prediction['loss_MRI_cancer_intra'] * weight_self_supervision['MRI']
        loss_self_supervision_TRUS = prediction['loss_TRUS_cancer_intra'] * weight_self_supervision['TRUS']
        total_loss += loss_self_supervision_MRI + loss_self_supervision_TRUS


        # Compute Dice scores
        dice_MRI = hard_dice_per_class(torch.softmax(seg_MRI, dim=1), MRI_label)
        dice_TRUS = hard_dice_per_class(torch.softmax(seg_TRUS, dim=1), TRUS_label)

    return {
        'total_loss': safe_item(total_loss),
        'seg_loss': safe_item(loss_seg),
        'TGCF_loss': safe_item(loss_contrastive),
        'self_loss': {'MRI':safe_item(loss_self_supervision_MRI), 'TRUS':safe_item(loss_self_supervision_TRUS)},
        'dice_MRI': dice_MRI,
        'dice_TRUS': dice_TRUS,
        'cls_MRI': safe_item(prediction['MRI_classification']),
        'cls_TRUS': safe_item(prediction['TRUS_classification'])
    }

def total_variation_loss_3d(y_pred):
    """ 3D TV Loss to encourage smoothness in segmentation masks """
    tv_d = torch.abs(y_pred[:, :, 1:, :, :] - y_pred[:, :, :-1, :, :]).mean()  # Depth 방향
    tv_h = torch.abs(y_pred[:, :, :, 1:, :] - y_pred[:, :, :, :-1, :]).mean()  # Height 방향
    tv_w = torch.abs(y_pred[:, :, :, :, 1:] - y_pred[:, :, :, :, :-1]).mean()  # Width 방향
    return tv_d + tv_h + tv_w

class DeepSupervisionLoss(nn.Module):
    def __init__(self, criterion, weights=[1.0, 0.2, 0.1, 0.05], tv_loss_weight = 0.0):
        super().__init__()
        self.weights = weights
        self.criterion = criterion
        self.tv_loss_weight = tv_loss_weight
        
    def forward(self, outputs, target):
        if isinstance(outputs, tuple):
            loss_list = [
                self.criterion(output, target) * self.weights[idx]
                for idx, output in enumerate(outputs)
            ]
            if self.tv_loss_weight > 0.0:
                tv_loss = total_variation_loss_3d(outputs[0])
                loss_list.append(tv_loss * self.tv_loss_weight)
            return sum(loss_list)
        else:
            return self.criterion(outputs, target)