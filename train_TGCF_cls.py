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
from sklearn.metrics import roc_auc_score

def main(args):
    TRAIN_GENERATOR_MRI,  VALID_GENERATOR_MRI,  VALID_GENERATOR_MRI_small,  TEST__GENERATOR_MRI  = get_generator(args.config_MRI)
    TRAIN_GENERATOR_TRUS, VALID_GENERATOR_TRUS, VALID_GENERATOR_TRUS_small, TEST__GENERATOR_TRUS = get_generator(args.config_TRUS)
    for mri, trus in zip([os.path.basename(i).split('_')[0] for i in TRAIN_GENERATOR_MRI. imageFileName.tolist()],
                         [os.path.basename(i).split('_')[0] for i in TRAIN_GENERATOR_TRUS.imageFileName.tolist()]):
        if mri != trus:
            print(mri, trus)
            raise ValueError('MRI and TRUS data does not match')
        
    TRAIN_DATALOADER_MRI  = DataLoader(TRAIN_GENERATOR_MRI , batch_size=args.image_load_size, shuffle=False, num_workers=args.num_workers,
                                        collate_fn=lambda x: collate_prostate_CS(x, crop=True, crop_size=args.img_size,
                                        n_crops_per_volume=args.n_crops_per_volume, random_crop_ratio=args.random_crop_ratio))
    TRAIN_DATALOADER_TRUS = DataLoader(TRAIN_GENERATOR_TRUS, batch_size=args.image_load_size, shuffle=False, num_workers=args.num_workers,
                                        collate_fn=lambda x: collate_prostate_CS(x, crop=True, crop_size=args.img_size,
                                        n_crops_per_volume=args.n_crops_per_volume, random_crop_ratio=args.random_crop_ratio))
    
    VALID_DATALOADER_MRI  = DataLoader(VALID_GENERATOR_MRI_small , batch_size=1, shuffle=False, num_workers=1,
                                       collate_fn=lambda x: collate_prostate_CS(x, crop=True, crop_size=args.img_size, random_crop_ratio=0.00))
    VALID_DATALOADER_TRUS = DataLoader(VALID_GENERATOR_TRUS_small, batch_size=1, shuffle=False, num_workers=1,
                                       collate_fn=lambda x: collate_prostate_CS(x, crop=True, crop_size=args.img_size, random_crop_ratio=0.00))

    VALID_DATALOADER_MRI_All = DataLoader(VALID_GENERATOR_MRI, batch_size=1, shuffle=False, num_workers=1,
                                        collate_fn=lambda x: collate_prostate_CS(x, crop=True, crop_size=args.img_size, random_crop_ratio=0.00))
    VALID_DATALOADER_TRUS_All = DataLoader(VALID_GENERATOR_TRUS, batch_size=1, shuffle=False, num_workers=1,
                                       collate_fn=lambda x: collate_prostate_CS(x, crop=True, crop_size=args.img_size, random_crop_ratio=0.00))
    
    TEST__DATALOADER_MRI  = DataLoader(TEST__GENERATOR_MRI , batch_size=1, shuffle=False, num_workers=1,
                                        collate_fn=lambda x: collate_prostate_CS(x, crop=True, crop_size=args.img_size, random_crop_ratio=0.00))
    TEST__DATALOADER_TRUS = DataLoader(TEST__GENERATOR_TRUS, batch_size=1, shuffle=False, num_workers=1,
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
        TGCF_MODEL.load_state_dict(torch.load(args.pretrained_weights_TGCF, map_location=args.device))
    else:
        if os.path.exists(args.pretrained_weights_TRUS):
            print("Load pretrained weights for TRUS")
            TGCF_MODEL.MODEL_TRUS.load_state_dict(torch.load(args.pretrained_weights_TRUS, map_location=args.device))
        if os.path.exists(args.pretrained_weights_MRI):
            print("Load pretrained weights for MRI")
            TGCF_MODEL.MODEL_MRI.load_state_dict(torch.load(args.pretrained_weights_MRI, map_location=args.device))
    
    #optimizer = Lion(MODEL.parameters(), lr=args.learning_rate, weight_decay=1e-2)
    optimizer = torch.optim.AdamW(TGCF_MODEL.parameters(), lr=args.learning_rate, weight_decay=1e-2)
    segmentation_weights = torch.tensor(args.segmentation_weights, device=args.device)
    criterion_classification = nn.CrossEntropyLoss()
    #criterion_segmentation = DiceCELoss(to_onehot_y=False, softmax=True, weight=segmentation_weights)
    #criterion_segmentation = DiceLoss(to_onehot_y=False, softmax=True, weight=segmentation_weights)
    criterion_segmentation = DiceFocalLoss(to_onehot_y=False, softmax=True, weight=segmentation_weights)
    criterion_segmentation_ds = DeepSupervisionLoss(criterion_segmentation, weights=[1.0, 0.2, 0.01, 0.05], tv_loss_weight = 0.1)
    #criterion_segmentation = DiceLoss(include_background=True, to_onehot_y=True, softmax=True, weight=segmentation_weights)
    #scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=8, T_mult=2, eta_min=1e-6)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.7, patience=2, min_lr=1e-8)

    vis_a = 0
    estimated_steps_per_epoch = len(TRAIN_DATALOADER_MRI) // 2
    for epoch in range(1, args.num_epochs, 1):
        ##########################################################    
        #################### TRAINING SESSION ####################
        TGCF_MODEL.train()
        TRAIN_MRI_ITER = iter(TRAIN_DATALOADER_MRI)
        TRAIN_TRUS_ITER = iter(TRAIN_DATALOADER_TRUS)
        train_loss_seg, train_loss_TGCF, train_loss_self, train_count,   = 0.0, 0.0, {'MRI':0, 'TRUS':0}, 0
        train_MRI_dice, train_TRUS_dice = [], []
    
        for batch_count in tqdm(range(estimated_steps_per_epoch), desc="Training"):
            TGCF_MODEL.train()
            MRI_Input,  MRI_Label,  MRI_Crop  = next(TRAIN_MRI_ITER )
            TRUS_Input, TRUS_Label, TRUS_Crop = next(TRAIN_TRUS_ITER)
            
            if MRI_Label[:,2].sum() == 0 or TRUS_Label[:,2].sum() == 0: continue
            for i in range(0, len(MRI_Input), args.batch_size):
                MRI_Input_subset  = MRI_Input[i:i+args.batch_size].to(args.device)
                TRUS_Input_subset = TRUS_Input[i:i+args.batch_size].to(args.device)
                MRI_Label_subset  = MRI_Label[i:i+args.batch_size].to(args.device)
                TRUS_Label_subset = TRUS_Label[i:i+args.batch_size].to(args.device)
                MRI_Crop_subset   = MRI_Crop[i:i+args.batch_size]
                TRUS_Crop_subset  = TRUS_Crop[i:i+args.batch_size]

                
                if MRI_Label_subset[:,2].sum() == 0: continue
                if TRUS_Label_subset[:,2].sum() == 0: continue
                train_log = train_step_TGCF(TGCF_MODEL, optimizer,
                                            MRI_Input_subset, TRUS_Input_subset, MRI_Label_subset,
                                            TRUS_Label_subset, MRI_Crop_subset, TRUS_Crop_subset,
                                            criterion_segmentation_ds,
                                            weight_segmentation_TRUS=args.weight_segmentation_TRUS,
                                            weight_segmentation=args.weight_segmentation,
                                            weight_contrastive=args.weight_contrastive,
                                            weight_classification=args.weight_classification,
                                            weight_self_supervision=args.weight_self_supervision,
                                            DeepSupervision=True, criterion_classification=criterion_classification)
                print("Epc{epoch}_{batch_count}/{estimated_steps_per_epoch}:", {k: round_values(v) for k, v in train_log.items()})
            
                train_loss_seg  += train_log['seg_loss' ]
                train_loss_TGCF += train_log['TGCF_loss']
                train_loss_self['MRI']  += train_log['self_loss']['MRI']
                train_loss_self['TRUS'] += train_log['self_loss']['TRUS']

                train_count += 1
                train_MRI_dice.append(train_log['dice_MRI'])
                train_TRUS_dice.append(train_log['dice_TRUS'])

        train_loss = (train_loss_seg + train_loss_TGCF) / train_count
        train_loss_seg = train_loss_seg / train_count
        train_loss_TGCF = train_loss_TGCF / train_count
        train_MRI_dice = np.nanmean(np.vstack(train_MRI_dice), axis=0)
        train_TRUS_dice = np.nanmean(np.vstack(train_TRUS_dice), axis=0)

        ##########################################################    
        ################### VALIDATION SESSION ###################
        TGCF_MODEL.eval()

        valid_loss_seg, valid_loss_TGCF, valid_count, valid_MRI_dice, valid_TRUS_dice = 0.0, 0.0, 0, [], []
        valid_MRI_cls, valid_TRUS_cls, label_MRI_cls, label_TRUS_cls = [], [], [], []
        VALID_MRI_ITER_All = iter(TEST__DATALOADER_MRI)
        VALID_TRUS_ITER_All = iter(TEST__DATALOADER_TRUS)
        VALID_MRI_ITER  = VALID_MRI_ITER_All
        VALID_TRUS_ITER = VALID_TRUS_ITER_All

        MRI_label, MRI_pred = [], []
        TRUS_label, TRUS_pred = [], []
        for batch_count in tqdm(range(len(VALID_MRI_ITER)), desc="Validation"):
            MRI_Input,  MRI_Label,  MRI_Crop = next(VALID_MRI_ITER)
            TRUS_Input, TRUS_Label, TRUS_Crop = next(VALID_TRUS_ITER)
            MRI_Input, MRI_Label = MRI_Input.to(args.device), MRI_Label.to(args.device)
            TRUS_Input, TRUS_Label = TRUS_Input.to(args.device), TRUS_Label.to(args.device)

            if TRUS_Label[:,2].sum() == 0: TRUS_idx = TRUS_Label[:,1].sum(axis=(1,2,3)).argmax()
            else: TRUS_idx = TRUS_Label[:,2].sum(axis=(1,2,3)).argmax()
            if MRI_Label[:,2].sum() == 0: MRI_idx = MRI_Label[:,2].sum(axis=(1,2,3)).argmax()
            else: MRI_idx = MRI_Label[:,2].sum(axis=(1,2,3)).argmax()
            
            MRI_Input, MRI_Label = MRI_Input[MRI_idx:MRI_idx+1], MRI_Label[MRI_idx:MRI_idx+1]
            TRUS_Input, TRUS_Label = TRUS_Input[TRUS_idx:TRUS_idx+1], TRUS_Label[TRUS_idx:TRUS_idx+1]
            MRI_Crop, TRUS_Crop = MRI_Crop[MRI_idx:MRI_idx+1], TRUS_Crop[TRUS_idx:TRUS_idx+1]

            with torch.no_grad():
                prediction = TGCF_MODEL(MRI_Input, TRUS_Input, MRI_Label, TRUS_Label, MRI_Crop, TRUS_Crop)
            MRI_label.append(MRI_Label[:,2].sum().item() > 0)
            MRI_pred.append(torch.softmax(prediction['MRI_classification'], dim=1)[:,1].max().item())
            TRUS_label.append(TRUS_Label[:,2].sum().item() > 0)
            TRUS_pred.append(torch.softmax(prediction['TRUS_classification'], dim=1)[:,1].max().item())

        MRI_roc = roc_auc_score(MRI_label, MRI_pred)
        TRUS_roc = roc_auc_score(TRUS_label, TRUS_pred)
        #MRI_roc, TRUS_roc

        AUROC = MRI_roc, TRUS_roc
        os.makedirs(args.save_dir, exist_ok=True)
        filename = os.path.join(args.save_dir, f'TGCF_p{args.patch_size}_eb{args.embed_dim}_bs{args.batch_size}_ep{epoch}_loss[{train_loss_seg:.4f}_{train_loss_TGCF:.4f}')
        filename_TGCF = filename + f'MRUS[Tr_{train_MRI_dice[2]:.3f}_{train_TRUS_dice[2]:.3f}]_AUROC[{AUROC[0]:.3f}_{AUROC[1]:.3f}].pth'

        print(f"{epoch} - AUROC:, {AUROC}")
        torch.save(TGCF_MODEL.state_dict(), f'{filename_TGCF}.pth')
        scheduler.step(1-TRUS_roc)


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
    parser.add_argument('--image_load_size', type=int, default=3, help='Batch size for training')
    parser.add_argument('--batch_size', type=int, default=6, help='Batch size for training')
    parser.add_argument('--num_workers', type=int, default=3, help='Number of workers for data loading')
    
    parser.add_argument('--learning_rate', type=float, default=1e-4, help='Learning rate for optimizer')
    parser.add_argument('--n_crops_per_volume', type=int, default=18, help='crop numbers')
    parser.add_argument('--random_crop_ratio',  type=float, default=0.85, help='crop numbers')
    
    parser.add_argument('--num_epochs',  type=int, default=100, help='Number of epochs for training')
    parser.add_argument('--num_full_validation', type=int, default=20, help='Number of epochs for training')
    parser.add_argument('--save_dir',  type=str, default='TGCF_TRUS_T2_cls/', help='directory of model weights')
    
    parser.add_argument('--weight_classification', type=float, default=1.00, help='Weight on the classification loss')
    parser.add_argument('--weight_segmentation', type=float, default=0.10, help='Weight on the segmentation loss')
    parser.add_argument('--weight_contrastive', type=float, default=0.10, help='Weight on the contrastive loss')
    parser.add_argument('--weight_segmentation_TRUS', type=float, default=0.95, help='Weight on the segmentation loss between TRUS vs MRI')
    parser.add_argument('--weight_token_contrast', type=float, nargs=3, default=[0.01, 0.05, 1.0], help='contrast loss weights on [bg, normal gland, cancer]')
    parser.add_argument('--weight_self_supervision', type=float, default=0.05, help='classification channel')
        
    parser.add_argument('--out_channels_cls', type=int, default=2, help='classification channel')
    parser.add_argument('--out_channels_seg', type=int, default=3, help='segmentation channel')
    parser.add_argument('--segmentation_weights', type=float, nargs=3, default=[0.01, 0.05, 1.00], help='[background, gland, indolent cancer, csPCa]')
    parser.add_argument('--pretrained_weights_TGCF', type=str, default='TGCF_TRUS_T2_cls_self_cont/TGCF_p8_eb384_bs12_ep15_loss[0.0362_0.0545_0.0345_0.0195]MRUS[Tr_0.292_0.192]_[Vl_0.156_0.101]_AUROC0.8618386157656696.pth.pth', help='Path to pretrained weights')
    parser.add_argument('--pretrained_weights_MRI', type=str, default='model_weights_ViT/T2_weights_0.158.pth', help='Path to pretrained weights')
    parser.add_argument('--pretrained_weights_TRUS', type=str, default='model_weights_ViT/TRUS_weights_0.101_noContrast.pth', help='Path to pretrained weights')
    args = parser.parse_args()    

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
