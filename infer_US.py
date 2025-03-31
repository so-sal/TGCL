import argparse
from architecture.vision_transformer3D import VisionTransformer3D, PatchEmbed3D, Block, MemEffAttention
from utils.generator import US_Generator, getData, collate_prostate_CS, reclassify_tensor
from utils.inference import visualize_max_cancer, visualize_small_patch, visualize_slices, inference_sliding_window, saveData

from utils.train import train_step, validation_step, DeepSupervisionLoss
from torch.utils.data import DataLoader
from utils.metrics import soft_dice_per_class, hard_dice_per_class

from functools import partial
import torch
from torch import nn
import yaml
import json
import pandas as pd
import numpy as np
from monai.losses import DiceCELoss, TverskyLoss, DiceLoss
from lion_pytorch import Lion
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts

import os
import gc
from tqdm import tqdm
import torch.nn.functional as F
import matplotlib.pyplot as plt

def main(args):
    Dataset = getData(args.config['paths']['Image_path'],
                      args.config['paths']['Gland_path'],
                      args.config['paths']['Label_path'],
                      args.config['Modality'], args.config['file_extensions'])

    with open(args.config['SplitValidation']['testset'], 'r') as file:
        SplitValidation_dict_test = json.load(file)
        SplitValidation_dict_test = pd.DataFrame(SplitValidation_dict_test['bx_test'])
    Test_Dataset = Dataset.loc[SplitValidation_dict_test['Anon_ID']]
    Test_Dataset.reset_index(drop=True, inplace=True)

    TEST__GENERATOR = US_Generator(
                    imageFileName=Test_Dataset['Image'],
                    glandFileName=Test_Dataset['Gland'],
                    cancerFileName=Test_Dataset['Cancer'],
                    modality=Test_Dataset['Modality'],
                    cancerTo2=True,
                    Augmentation=False)
    TEST__DATALOADER       = DataLoader(TEST__GENERATOR,       batch_size=1, shuffle=False, num_workers=1, collate_fn=lambda x: collate_prostate_CS(x, crop=False))

    MODEL = VisionTransformer3D(
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

    args.pretrained_weights = 'RAUS_clsseg_patch8_emb384_batch18_epc90_AllValid_loss0.3525_[0.994-0.574-0.116].pth'
    pretrained_weights_path = os.path.join(args.save_dir, args.pretrained_weights)
    MODEL.load_state_dict(torch.load(pretrained_weights_path, map_location=args.device))
    
    #optimizer = Lion(MODEL.parameters(), lr=args.learning_rate, weight_decay=1e-2)
    optimizer = torch.optim.AdamW(MODEL.parameters(), lr=args.learning_rate, weight_decay=1e-2)
    segmentation_weights = torch.tensor(args.segmentation_weights, device=args.device)
    criterion_classification = nn.CrossEntropyLoss()
    criterion_segmentation = DiceCELoss(to_onehot_y=False, softmax=True, weight=segmentation_weights)
    criterion_segmentation = DiceLoss(to_onehot_y=False, softmax=True, weight=segmentation_weights)
    criterion_segmentation_ds = DeepSupervisionLoss(criterion_segmentation, weights=[1.0, 0.2, 0.1, 0.05], tv_loss_weight = 0.10)
    #criterion_segmentation = DiceLoss(include_background=True, to_onehot_y=True, softmax=True, weight=segmentation_weights)
    #scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=8, T_mult=2, eta_min=1e-6)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.7, patience=2, min_lr=1e-8)        
        
    for idx in range(len(TEST__GENERATOR)):
        save_filename = os.path.join("pred_result", os.path.basename(TEST__GENERATOR.imageFileName[idx]).split("_trus")[0] + '_prob.nii.gz')
        if os.path.exists(save_filename): continue
            
        batch = TEST__GENERATOR[idx]
        if batch[2].sum() > 10: continue
        
        Img, Label, Crop = collate_prostate_CS([batch], crop=False)
        final_pred = inference_sliding_window(MODEL, Img, window_size=(96, 96, 96), stride=5, num_classes=3)
        cancer_prob = torch.softmax(final_pred, dim=1)[0,2]
        saveData(cancer_prob.cpu().detach(),TEST__GENERATOR.imageFileName[idx], save_filename)
        
        

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
    parser.add_argument('--cuda_device', type=int, default=1, help='Specify CUDA visible devices')
    parser.add_argument('--config_file', type=str, default='configs/config_TRUS.yaml', help='Path to config file')

    # Model structure / Data structure
    parser.add_argument('--img_size', type=int, default=96, help='Image size in format (D, H, W)')
    parser.add_argument('--patch_size', type=int, default=8, help='Image size in format (D, H, W)')
    parser.add_argument('--embed_dim', type=int, default=384, help='Embedding dimension') # 384, 768, 1152,
    parser.add_argument('--image_load_size', type=int, default=12, help='Batch size for training')
    parser.add_argument('--batch_size', type=int, default=24, help='Batch size for training')
    
    parser.add_argument('--learning_rate', type=float, default=1e-4, help='Learning rate for optimizer')
    parser.add_argument('--n_crops_per_volume', type=int, default=18, help='crop numbers')
    parser.add_argument('--random_crop_ratio', type=float, default=0.85, help='crop numbers')
    
    parser.add_argument('--num_workers', type=int, default=6, help='Number of workers for data loading')
    parser.add_argument('--num_epochs', type=int, default=100, help='Number of epochs for training')
    parser.add_argument('--num_full_validation', type=int, default=10, help='Number of epochs for training')
    parser.add_argument('--save_dir', type=str, default='model_weights', help='directory of model weights')
    parser.add_argument('--weight_segmentation', type=float, default=0.99, help='segmentation loss의 가중치')
        
    parser.add_argument('--out_channels_cls', type=int, default=2, help='classification channel')
    parser.add_argument('--out_channels_seg', type=int, default=3, help='segmentation channel')
    parser.add_argument('--segmentation_weights', type=float, nargs=3, default=[0.02, 0.10, 1.00], help='[background, gland, indolent cancer, csPCa]')
    parser.add_argument('--pretrained_weights', type=str, default=None, help='Path to pretrained weights')

    args = parser.parse_args()    
    if torch.cuda.is_available(): # Check GPU available
        print("Number of GPU(s) available:", torch.cuda.device_count())
        args.device = torch.device(f'cuda:{args.cuda_device}' if args.cuda_device else 'cuda:0')
    else:
        print("CUDA is not available.")
        args.device = torch.device('cpu')

    with open('configs/config_TRUS.yaml', 'r') as f:
        args.config = yaml.safe_load(f)
    
    main(args)

# nohup python train.py --cuda_device 1 --image_load_size 6 --patch_size 8 --batch_size 18 --embed_dim 384 --learning_rate 0.0001 --pretrained_weights RAUS_clsseg_patch8_emb384_epc80_AllValid_loss0.4019_[0.993-0.568-0.025].pth > log_p8_b24_384 &
# nohup python train.py --cuda_device 0 --image_load_size 6 --patch_size 8 --batch_size  8 --embed_dim 384 --learning_rate 0.0001 --pretrained_weights RAUS_clsseg_patch8_emb384_epc80_AllValid_loss0.4019_[0.993-0.568-0.025].pth > log_p8_b8_384 &
# nohup python train.py --cuda_device 3 --image_load_size 6 --patch_size 8 --batch_size  8 --embed_dim 768 --learning_rate 0.0001 > log_p8_b12_768 &
# nohup python train.py --cuda_device 4 --image_load_size 6 --patch_size 6 --batch_size 12 --embed_dim 384 --learning_rate 0.0001 > log_p6_b12_374 &


