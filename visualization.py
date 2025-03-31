# Token-Guided Contrastive Fusion Registration-Free Ultrasound Segmentation with MRI Supervision
# Label-Guided Token Contrastive Learning for Registration-Free Prostate Segmentation in Ultrasound Images with MRI Supervision

import argparse
from architecture.vision_transformer3D import VisionTransformer3D, TGCF, PatchEmbed3D, Block, MemEffAttention
from utils.generator import US_Generator, getData, collate_prostate_CS, reclassify_tensor, get_generator
from utils.inference import visualize_slices, inference_sliding_window, visualize_token_contrastive_with_labels

from utils.train import train_step_TGCF, validation_step_TGCF, DeepSupervisionLoss
from torch.utils.data import DataLoader
from utils.metrics import soft_dice_per_class, hard_dice_per_class

from functools import partial
import torch
from torch import nn
import yaml
import json
import pandas as pd
import numpy as np
from monai.losses import DiceCELoss, TverskyLoss, DiceLoss, DiceFocalLoss
from lion_pytorch import Lion
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts

import os
import gc
from tqdm import tqdm
import torch.nn.functional as F
import matplotlib.pyplot as plt

def main(args):
    TRAIN_GENERATOR_MRI,  VALID_GENERATOR_MRI,  VALID_GENERATOR_MRI_small,  TEST__GENERATOR_MRI  = get_generator(args.config_MRI)
    TRAIN_GENERATOR_TRUS, VALID_GENERATOR_TRUS, VALID_GENERATOR_TRUS_small, TEST__GENERATOR_TRUS = get_generator(args.config_TRUS)
    for mri, trus in zip([os.path.basename(i).split('_')[0] for i in TEST__GENERATOR_MRI. imageFileName.tolist()],
                         [os.path.basename(i).split('_')[0] for i in TEST__GENERATOR_TRUS.imageFileName.tolist()]):
        if mri != trus:
            print(mri, trus)
            raise ValueError('MRI and TRUS data does not match')
    TRAIN_DATALOADER_MRI = DataLoader(TRAIN_GENERATOR_MRI, batch_size=1, shuffle=False, num_workers=1,
                                        collate_fn=lambda x: collate_prostate_CS(x, crop=True, crop_size=args.img_size, random_crop_ratio=0.00))
    TRAIN_DATALOADER_TRUS = DataLoader(TRAIN_GENERATOR_TRUS, batch_size=1, shuffle=False, num_workers=1,
                                       collate_fn=lambda x: collate_prostate_CS(x, crop=True, crop_size=args.img_size, random_crop_ratio=0.00))

    MODEL_MRI = VisionTransformer3D(
        img_size=args.img_size,
        patch_size=args.patch_size,
        embed_dim=args.embed_dim,
        depth=12,
        num_heads=6,
        mlp_ratio=4,
        block_fn=partial(Block, attn_class=MemEffAttention),
        num_register_tokens=4,
        in_chans=1,
        out_chans_seg=args.out_channels_seg,
        out_chans_cls=args.out_channels_cls,
        embed_layer=PatchEmbed3D,
    ).to(args.device)

    MODEL_TRUS = VisionTransformer3D(
        img_size=args.img_size,
        patch_size=args.patch_size,
        embed_dim=args.embed_dim,
        depth=12,
        num_heads=6,
        mlp_ratio=4,
        block_fn=partial(Block, attn_class=MemEffAttention),
        num_register_tokens=4,
        in_chans=1,
        out_chans_seg=args.out_channels_seg,
        out_chans_cls=args.out_channels_cls,
        embed_layer=PatchEmbed3D,
    ).to(args.device)
    
    TGCF_MODEL = TGCF(MODEL_MRI, MODEL_TRUS, weight_token_contrast=args.weight_token_contrast).to(args.device)


    if os.path.exists(args.pretrained_weights_TGCF):
        print("Load pretrained weights for TGCF")
        args.pretrained_weights_TGCF = 'TGCF_TRUS_T2/TGCF_p8_eb384_bs12_ep27_loss[0.3591_0.8398_0.3445_0.5568]MRUS[Tr_0.297_0.214]_[Vl_0.172_0.114].pth.pth'
        TGCF_MODEL.load_state_dict(torch.load(args.pretrained_weights_TGCF, map_location=args.device))
    else:
        if os.path.exists(args.pretrained_weights_TRUS):
            print("Load pretrained weights for TRUS")
            TGCF_MODEL.MODEL_TRUS.load_state_dict(torch.load(args.pretrained_weights_TRUS, map_location=args.device))
        if os.path.exists(args.pretrained_weights_MRI):
            print("Load pretrained weights for MRI")
            TGCF_MODEL.MODEL_MRI.load_state_dict(torch.load(args.pretrained_weights_MRI, map_location=args.device))
    
    optimizer = torch.optim.AdamW(TGCF_MODEL.parameters(), lr=args.learning_rate, weight_decay=1e-2)
    segmentation_weights = torch.tensor(args.segmentation_weights, device=args.device)
    criterion_segmentation = DiceFocalLoss(to_onehot_y=False, softmax=True, weight=segmentation_weights)
    criterion_segmentation_ds = DeepSupervisionLoss(criterion_segmentation, weights=[1.0, 0.2, 0.01, 0.05], tv_loss_weight = 0.1)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.7, patience=2, min_lr=1e-8)

    TGCF_MODEL.eval()
    TEST_MRI_ITER = iter(TRAIN_DATALOADER_MRI)
    TEST_TRUS_ITER = iter(TRAIN_DATALOADER_TRUS)

    agg, cnt = [], []
    for batch_count in range(len(TEST_MRI_ITER)):
        MRI_Input,  MRI_Label,  MRI_Crop = next(TEST_MRI_ITER)
        TRUS_Input, TRUS_Label, TRUS_Crop = next(TEST_TRUS_ITER)

        if MRI_Label[:,2].sum() == 0 or TRUS_Label[:,2].sum() == 0:
            continue

        
        MRI_Input, MRI_Label = MRI_Input.to(args.device), MRI_Label.to(args.device)
        TRUS_Input, TRUS_Label = TRUS_Input.to(args.device), TRUS_Label.to(args.device)

        mri_crop_idx = MRI_Label[:, 2].sum(axis=(1, 2, 3)).argmax()
        trus_crop_idx = TRUS_Label[:, 2].sum(axis=(1, 2, 3)).argmax()

        info = TGCF_MODEL.get_token_contrastive_info(MRI_Input[mri_crop_idx:mri_crop_idx+1].to(args.device), TRUS_Input[mri_crop_idx:mri_crop_idx+1].to(args.device), MRI_Label[mri_crop_idx:mri_crop_idx+1].to(args.device),
                                                    TRUS_Label[trus_crop_idx:trus_crop_idx+1].to(args.device), MRI_Crop[trus_crop_idx:trus_crop_idx+1], TRUS_Crop[trus_crop_idx:trus_crop_idx+1], temperature=0.1)
        visualize_token_contrastive_with_labels(
            f1_sample=info['f1'],
            f2_sample=info['f2'],
            labels1=info['gt_patch_class_mri'].cpu().numpy(),  # Use ground truth patch classes
            confs1=info['conf1'].cpu().numpy(),
            labels2=info['gt_patch_class_trus'].cpu().numpy(),
            confs2=info['conf2'].cpu().numpy(),
            logits=info['logits'],  # Pass the precomputed logits to avoid re-calculation
            temperature=0.1,
            filename=f'zzz{batch_count}.png'
        )



        # --- Example Usage (with 'info' variable) ---
        agg_sim_matrix, counts = aggregated_similarity_matrix(info['f1'], info['f2'],
                                                    torch.tensor(info['prob1']),
                                                    torch.tensor(info['prob2'])) # prob을 사용
        agg.append(agg_sim_matrix)
        cnt.append(counts)


        # --- Visualization (same as before) ---
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))

        # Aggregated Similarity
        im = axes[0].imshow(agg_sim_matrix, cmap='viridis')
        cbar = fig.colorbar(im, ax=axes[0])
        cbar.ax.tick_params(labelsize=10)
        axes[0].set_xticks(range(3))
        axes[0].set_yticks(range(3))
        axes[0].set_xticklabels(['BG', 'Normal', 'Cancer'])
        axes[0].set_yticklabels(['BG', 'Normal', 'Cancer'])
        axes[0].set_xlabel('TRUS')
        axes[0].set_ylabel('MRI')
        axes[0].set_title('Aggregated Token Similarity')

        # Counts
        im2 = axes[1].imshow(counts, cmap='viridis', norm=plt.matplotlib.colors.LogNorm())
        cbar2 = fig.colorbar(im2, ax=axes[1])
        cbar2.ax.tick_params(labelsize=10)
        axes[1].set_xticks(range(3))
        axes[1].set_yticks(range(3))
        axes[1].set_xticklabels(['BG', 'Normal', 'Cancer'])
        axes[1].set_yticklabels(['BG', 'Normal', 'Cancer'])
        axes[1].set_xlabel('TRUS')
        axes[1].set_ylabel('MRI')
        axes[1].set_title('Number of Token Pairs')

        plt.tight_layout()
        plt.show()

def aggregated_similarity_matrix(f1, f2, prob1, prob2, temperature=0.1):
    """
    Computes the aggregated similarity matrix using the CORRECT label criteria.
    FEATURES ARE ASSUMED TO BE ALREADY NORMALIZED.

    Args:
        f1: MRI features (N1, D) - PyTorch Tensor (ALREADY NORMALIZED)
        f2: TRUS features (N2, D) - PyTorch Tensor (ALREADY NORMALIZED)
        prob1: MRI probabilities (N1, 3) - PyTorch Tensor
        prob2: TRUS probabilities (N2, 3) - PyTorch Tensor
        temperature: Temperature for similarity calculation.

    Returns:
        A 3x3 numpy array representing the aggregated similarity matrix.
        A 3x3 numpy array representing the counts.
    """
    num_classes = 3
    agg_matrix = np.zeros((num_classes, num_classes))
    counts = np.zeros((num_classes, num_classes))

    # f1_norm = F.normalize(f1, dim=-1)  # REMOVED: Already normalized
    # f2_norm = F.normalize(f2, dim=-1)  # REMOVED: Already normalized
    # logits = torch.matmul(f1_norm, f2_norm.T) / temperature # Use pre-computed
    logits = torch.matmul(f1, f2.T) / temperature


    # --- CORRECT Label Assignment ---
    def get_labels(prob):
        labels = torch.zeros(prob.shape[0], dtype=torch.long)  # 0: BG
        # Normal: >= 80% normal AND 0% cancer
        normal_mask = (prob[:, 1] >= 0.8) & (prob[:, 2] == 0)
        labels[normal_mask] = 1
        # Cancer: >= 10% cancer
        cancer_mask = prob[:, 2] >= 0.1
        labels[cancer_mask] = 2
        return labels

    labels1 = get_labels(prob1)  # Use probabilities to get labels
    labels2 = get_labels(prob2)

    for i in range(num_classes):
        for j in range(num_classes):
            mask1 = labels1 == i
            mask2 = labels2 == j
            if torch.any(mask1) and torch.any(mask2):
                agg_matrix[i, j] = logits[mask1][:, mask2].mean().item()
                counts[i, j] = mask1.sum() * mask2.sum()

    return agg_matrix, counts

def round_values(obj, decimals=3):
    if isinstance(obj, dict):
        return {k: round_values(v, decimals) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [round_values(v, decimals) for v in obj]
    elif isinstance(obj, tuple):
        return tuple(round_values(v, decimals) for v in obj)
    elif isinstance(obj, np.ndarray):
        return np.round(obj, decimals)
    elif isinstance(obj, (int, float)):
        return round(obj, decimals)
    return obj

def test_visualize(GENERATOR, data_idx, MODEL, filename, batch=None):
    with torch.no_grad():
        if batch is None:
            batch = [GENERATOR[data_idx]]
            Img, Label, Crop = collate_prostate_CS(batch, crop=False)
            z, y, x = Label[0,2].sum(axis=(1,2)).argmax().item(), Label[0,2].sum(axis=(0,2)).argmax().item(), Label[0,2].sum(axis=(0,1)).argmax().item()
            Crop = [[z-48, y-48, x-48]]
            Img = Img  [:,    z-48:z+48, y-48:y+48, x-48:x+48].to(args.device)
            Label = Label[:, :, z-48:z+48, y-48:y+48, x-48:x+48].to(args.device)
            img_idx = 0
            slices = [46, 47, 48, 49, 50]
        else:
            Img, Label, Crop = batch
            img_idx = Label[:, 2].sum(axis=(1,2,3)).argmax()
            Img, Label, Crop = Img[img_idx:img_idx+1].to(args.device), Label[img_idx:img_idx+1].to(args.device), Crop[img_idx:img_idx+1]
            img_idx = 0
            slices = [Label[img_idx, 2].sum(axis=(1,2)).argmax()+i-2 for i in range(5)]
            if max(slices) >= 96: slices = [slice - (max(slices)-96 + 1) for slice in slices]
            if min(slices) < 0: slices = [slice - min(slices) for slice in slices]
        prediction = MODEL(Img, crop_pos=Crop)
        prediction = torch.softmax(prediction['segmentation'], dim=1)
        visualize_slices(Img[img_idx], Label[img_idx], prediction.detach().cpu().numpy()[0], slices, filename)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Script with various default arguments.")   
    parser.add_argument('--cuda_device', type=int, default=7, help='Specify CUDA visible devices')
    parser.add_argument('--config_MRI',  type=str, default='configs/config_T2.yaml', help='Path to config file')
    parser.add_argument('--config_TRUS', type=str, default='configs/config_TRUS.yaml', help='Path to config file')
    parser.add_argument('--MR_sequence', type=str, default='T2', help='MRI sequence')

    # Model structure / Data structure
    parser.add_argument('--img_size', type=int, default=96, help='Image size in format (D, H, W)')
    parser.add_argument('--patch_size', type=int, default=8, help='Image size in format (D, H, W)')
    parser.add_argument('--embed_dim', type=int, default=384, help='Embedding dimension') # 384, 768, 1152,
    parser.add_argument('--image_load_size', type=int, default=12, help='Batch size for training')
    parser.add_argument('--batch_size', type=int, default=12, help='Batch size for training')
    parser.add_argument('--num_workers', type=int, default=6, help='Number of workers for data loading')
    
    parser.add_argument('--learning_rate', type=float, default=1e-4, help='Learning rate for optimizer')
    parser.add_argument('--n_crops_per_volume', type=int, default=18, help='crop numbers')
    parser.add_argument('--random_crop_ratio',  type=float, default=0.85, help='crop numbers')
    
    
    parser.add_argument('--num_epochs',  type=int, default=100, help='Number of epochs for training')
    parser.add_argument('--num_full_validation', type=int, default=20, help='Number of epochs for training')
    parser.add_argument('--save_dir',  type=str, default='TGCF_TRUS_T2/', help='directory of model weights')
    
    parser.add_argument('--weight_classification', type=float, default=0.00, help='Weight on the classification loss')
    parser.add_argument('--weight_segmentation', type=float, default=1.00, help='Weight on the segmentation loss')
    parser.add_argument('--weight_contrastive', type=float, default=1.50, help='Weight on the contrastive loss')
    parser.add_argument('--weight_segmentation_TRUS', type=float, default=0.95, help='Weight on the segmentation loss between TRUS vs MRI')
    parser.add_argument('--weight_token_contrast', type=float, nargs=3, default=[0.01, 0.05, 1.0], help='contrast loss weights on [bg, normal gland, cancer]')
    parser.add_argument('--weight_self_supervision', type=float, default=0.1, help='classification channel')
        
    parser.add_argument('--out_channels_cls', type=int, default=2, help='classification channel')
    parser.add_argument('--out_channels_seg', type=int, default=3, help='segmentation channel')
    parser.add_argument('--segmentation_weights', type=float, nargs=3, default=[0.01, 0.05, 1.00], help='[background, gland, indolent cancer, csPCa]')
    parser.add_argument('--pretrained_weights_TGCF', type=str, default='', help='Path to pretrained weights')
    parser.add_argument('--pretrained_weights_MRI', type=str, default='model_weights_ViT/T2_weights_0.158.pth', help='Path to pretrained weights')
    parser.add_argument('--pretrained_weights_TRUS', type=str, default='model_weights_ViT/TRUS_weights_0.101_noContrast.pth', help='Path to pretrained weights')
    args = parser.parse_args()    

    # args.config_MRI = "configs/config_T2.yaml"
    # args.pretrained_weights_MRI = 'asd'
    # args.cuda_device = 7

    if torch.cuda.is_available(): # Check GPU available
        print("Number of GPU(s) available:", torch.cuda.device_count())
        args.device = torch.device(f'cuda:{args.cuda_device}' if args.cuda_device else 'cuda:0')
    else:
        print("CUDA is not available.")
        args.device = torch.device('cpu')

    with open(args.config_TRUS, 'r') as f:
        args.config_TRUS = yaml.safe_load(f)

    with open(args.config_MRI, 'r') as f:
        args.config_MRI = yaml.safe_load(f)
    
    main(args)
