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
import SimpleITK as sitk

def main(args):
    TRAIN_GENERATOR_MRI,  VALID_GENERATOR_MRI,  VALID_GENERATOR_MRI_small,  TEST__GENERATOR_MRI  = get_generator(args.config_MRI)
    TRAIN_GENERATOR_TRUS, VALID_GENERATOR_TRUS, VALID_GENERATOR_TRUS_small, TEST__GENERATOR_TRUS = get_generator(args.config_TRUS)
    
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


    print("Load pretrained weights for TGCF")
    TGCF_MODEL.load_state_dict(torch.load(args.pretrained_weights_TGCF, map_location=args.device))
    segmentation_weights = torch.tensor(args.segmentation_weights, device=args.device)
    criterion_segmentation = DiceFocalLoss(to_onehot_y=False, softmax=True, weight=segmentation_weights)
    criterion_segmentation_ds = DeepSupervisionLoss(criterion_segmentation, weights=[1.0, 0.2, 0.01, 0.05], tv_loss_weight = 0.1)
    
    VALID_DATALOADER_MRI = DataLoader(VALID_GENERATOR_MRI, batch_size=1, shuffle=False, num_workers=1,
                                        collate_fn=lambda x: collate_prostate_CS(x, crop=True, crop_size=args.img_size, random_crop_ratio=0.00))
    VALID_DATALOADER_TRUS = DataLoader(VALID_GENERATOR_TRUS, batch_size=1, shuffle=False, num_workers=1,
                                       collate_fn=lambda x: collate_prostate_CS(x, crop=True, crop_size=args.img_size, random_crop_ratio=0.00))

    TEST__DATALOADER_MRI = DataLoader(TEST__GENERATOR_MRI, batch_size=1, shuffle=False, num_workers=1,
                                        collate_fn=lambda x: collate_prostate_CS(x, crop=True, crop_size=args.img_size, random_crop_ratio=0.00))
    TEST__DATALOADER_TRUS = DataLoader(TEST__GENERATOR_TRUS, batch_size=1, shuffle=False, num_workers=1,
                                       collate_fn=lambda x: collate_prostate_CS(x, crop=True, crop_size=args.img_size, random_crop_ratio=0.00))

    TGCF_MODEL.eval()
    stride = 10
    # for Dataset, MRI_Generator, TRUS_Generator in zip(['Validation', 'Test'],
    #                                         [VALID_GENERATOR_MRI, TEST__GENERATOR_MRI],
                                            # [VALID_GENERATOR_TRUS, TEST__GENERATOR_TRUS]):
    if args.phase == 'validation':
        Dataset = 'Validation'
        MRI_Generator = VALID_GENERATOR_MRI
        TRUS_Generator = VALID_GENERATOR_TRUS
    elif args.phase == 'test':
        Dataset = 'Test'
        MRI_Generator = TEST__GENERATOR_MRI
        TRUS_Generator = TEST__GENERATOR_TRUS

    for batch_count in tqdm(range(len(MRI_Generator))):
        MRI_batch = [MRI_Generator[batch_count]]
        TRUS_batch = [TRUS_Generator[batch_count]]
        MRI_Input, MRI_Label, MRI_Crop = collate_prostate_CS(MRI_batch, crop=False)
        TRUS_Input, TRUS_Label, TRUS_Crop = collate_prostate_CS(TRUS_batch, crop=False)
        
        if TRUS_Label[:,2].sum() == 0: continue

        MRI_Input, MRI_Label = MRI_Input.to(args.device), MRI_Label.to(args.device)
        TRUS_Input, TRUS_Label = TRUS_Input.to(args.device), TRUS_Label.to(args.device)

        pred = sliding_window_inference(TGCF_MODEL, MRI_Input, MRI_Label, TRUS_Input, TRUS_Label,
                                                    window_size=96, stride=40)
        pred_softmax = torch.softmax(pred[0], dim=0)
        pred_sitk = sitk.GetImageFromArray(pred_softmax[2].numpy())
        origin_sitk = sitk.ReadImage(TRUS_Generator.imageFileName[batch_count])

        pred_sitk.SetSpacing(origin_sitk.GetSpacing())
        pred_sitk.SetOrigin(origin_sitk.GetOrigin())
        pred_sitk.SetDirection(origin_sitk.GetDirection())
        filename = os.path.basename(TRUS_Generator.imageFileName[batch_count]).split('_trus')[0]
        sitk.WriteImage(pred_sitk, f'results/{Dataset}_{args.MR_sequence}/{filename}_pred.nii.gz')
        
        gc.collect()
        torch.cuda.empty_cache()

def sliding_window_inference(TGCF_MODEL, MRI_Input, MRI_Label, TRUS_Input, TRUS_Label, window_size=96, stride=10):
    D, H, W = MRI_Input.shape[1:]
    # Hardcoded from TGCF_Prediction shape: [27, 3, 96, 96, 96]
    channels = 3
    final_pred = torch.zeros((1, channels, D, H, W))
    count_map = torch.zeros((D, H, W))

    for z in range(0, D - window_size + 1, stride):
        for y in range(0, H - window_size + 1, stride):
            for x in range(0, W - window_size + 1, stride):
                crop_MRI = [[z, y, x]]
                crop_TRUS = [[z, y, x]]
                with torch.no_grad():
                    pred = TGCF_MODEL(
                        MRI_Input[:, z:z+window_size, y:y+window_size, x:x+window_size],
                        TRUS_Input[:, z:z+window_size, y:y+window_size, x:x+window_size],
                        MRI_Label[:, :, z:z+window_size, y:y+window_size, x:x+window_size],
                        TRUS_Label[:, :, z:z+window_size, y:y+window_size, x:x+window_size],
                        crop_MRI,
                        crop_TRUS)
                patch_pred = pred['TRUS_segmentation'].cpu().numpy()
                final_pred[:, :, z:z+window_size, y:y+window_size, x:x+window_size] += patch_pred
                count_map[z:z+window_size, y:y+window_size, x:x+window_size] += 1
    return final_pred / count_map.unsqueeze(0).unsqueeze(0)

def reassemble_segmentation(predictions, crop_coords, original_size=(256, 256, 256), crop_size=96, num_classes=3):
    output_volume = torch.zeros((num_classes, *original_size), dtype=predictions.dtype, device=predictions.device)
    counts = torch.zeros(original_size, dtype=torch.int, device=predictions.device)
    for i, (z, y, x) in enumerate(crop_coords):
        output_volume[:, z:z+crop_size, y:y+crop_size, x:x+crop_size] += predictions[i]
        counts[z:z+crop_size, y:y+crop_size, x:x+crop_size] += 1
    counts[counts == 0] = 1
    output_volume = output_volume / counts
    return output_volume

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
    parser.add_argument('--cuda_device', type=int, default=5, help='Specify CUDA visible devices')
    parser.add_argument('--config_MRI',  type=str, default='configs/config_ADC.yaml', help='Path to config file')
    parser.add_argument('--config_TRUS', type=str, default='configs/config_TRUS.yaml', help='Path to config file')
    parser.add_argument('--MR_sequence', type=str, default='ADC', help='MRI sequence')
    parser.add_argument('--phase', type=str, default='validation', help='MRI sequence')

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
