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

    with open(args.config['SplitValidation']['internal_split'], 'r') as file:
        SplitValidation_dict = json.load(file)
    with open(args.config['SplitValidation']['testset'], 'r') as file:
        SplitValidation_dict_test = json.load(file)
        SplitValidation_dict_test = pd.DataFrame(SplitValidation_dict_test['bx_test'])
    
    TrainDataset = Dataset.loc[SplitValidation_dict[args.config['FOLD_IDX']]['train']]
    ValidDataset = Dataset.loc[SplitValidation_dict[args.config['FOLD_IDX']]['val']]
    Test_Dataset = Dataset.loc[SplitValidation_dict_test['Anon_ID']]

    TrainDataset.reset_index(drop=True, inplace=True)
    ValidDataset.reset_index(drop=True, inplace=True)
    Test_Dataset.reset_index(drop=True, inplace=True)

    # TrainDataset = TrainDataset.iloc[:48, :]
    TRAIN_GENERATOR = US_Generator(
                    imageFileName=TrainDataset['Image'],
                    glandFileName=TrainDataset['Gland'],
                    cancerFileName=TrainDataset['Cancer'],
                    modality=TrainDataset['Modality'],
                    cancerTo2=True,
                    Augmentation=True)

    VALID_GENERATOR = US_Generator(
                    imageFileName=ValidDataset['Image'],
                    glandFileName=ValidDataset['Gland'],
                    cancerFileName=ValidDataset['Cancer'],
                    modality=ValidDataset['Modality'],
                    cancerTo2=True,
                    Augmentation=False)

    VALID_GENERATOR_small = US_Generator(
                    imageFileName=ValidDataset['Image'][:len(ValidDataset)//10],
                    glandFileName=ValidDataset['Gland'][:len(ValidDataset)//10],
                    cancerFileName=ValidDataset['Cancer'][:len(ValidDataset)//10],
                    modality=ValidDataset['Modality'][:len(ValidDataset)//10],
                    cancerTo2=True,
                    Augmentation=False)

    TEST__GENERATOR = US_Generator(
                    imageFileName=Test_Dataset['Image'],
                    glandFileName=Test_Dataset['Gland'],
                    cancerFileName=Test_Dataset['Cancer'],
                    modality=Test_Dataset['Modality'],
                    cancerTo2=True,
                    Augmentation=False)
    
    TRAIN_DATALOADER       = DataLoader(TRAIN_GENERATOR, batch_size=args.image_load_size, shuffle=True, num_workers=args.num_workers,
                                        collate_fn=lambda x: collate_prostate_CS(x, crop=True, crop_size=args.img_size,
                                        n_crops_per_volume=args.n_crops_per_volume, random_crop_ratio=args.random_crop_ratio))
    VALID_DATALOADER_ALL   = DataLoader(VALID_GENERATOR,       batch_size=1, shuffle=False, num_workers=1, collate_fn=lambda x: collate_prostate_CS(x, crop=True, crop_size=args.img_size, random_crop_ratio=0.00))
    VALID_DATALOADER_SMALL = DataLoader(VALID_GENERATOR_small, batch_size=1, shuffle=False, num_workers=1, collate_fn=lambda x: collate_prostate_CS(x, crop=True, crop_size=args.img_size, random_crop_ratio=0.00))
    TEST__DATALOADER       = DataLoader(TEST__GENERATOR,       batch_size=1, shuffle=False, num_workers=1, collate_fn=lambda x: collate_prostate_CS(x, crop=False))

    # for idx in range(len(TRAIN_GENERATOR)):
    #     batch = [TRAIN_GENERATOR[idx]]

    # for Img, Label, Crop in tqdm(TRAIN_DATALOADER):
    #     print(Img.shape, Label.shape, len(Crop))

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

    if args.pretrained_weights is not None:
        args.pretrained_weights = 'RAUS_clsseg_patch8_emb384_batch18_epc90_AllValid_loss0.3525_[0.994-0.574-0.116].pth'
        pretrained_weights_path = os.path.join(args.save_dir, args.pretrained_weights)
        MODEL.load_state_dict(torch.load(pretrained_weights_path, map_location=args.device))
    
    optimizer = torch.optim.AdamW(MODEL.parameters(), lr=args.learning_rate, weight_decay=1e-2)
    segmentation_weights = torch.tensor(args.segmentation_weights, device=args.device)
    criterion_classification = nn.CrossEntropyLoss()
    criterion_segmentation = DiceCELoss(to_onehot_y=False, softmax=True, weight=segmentation_weights)
    criterion_segmentation = DiceLoss(to_onehot_y=False, softmax=True, weight=segmentation_weights)
    criterion_segmentation_ds = DeepSupervisionLoss(criterion_segmentation, weights=[1.0, 0.2, 0.1, 0.05], tv_loss_weight = 0.10)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.7, patience=2, min_lr=1e-8)
    
    estimated_steps_per_epoch = int(len(TRAIN_DATALOADER) / args.image_load_size * args.n_crops_per_volume)
    estimated_steps_per_epoch = estimated_steps_per_epoch // 5

    for epoch in range(1, args.num_epochs, 1):
        ########## TRAINING
        MODEL.train()
        train_loss, train_count, train_dice = 0.0, 0, []
        for batch_count, (Img, Label, Crop) in tqdm(enumerate(TRAIN_DATALOADER), total=estimated_steps_per_epoch, desc="Training"):
            if len(torch.unique(Label.argmax(dim=1))) != args.out_channels_seg: continue
            Img = Img.to(args.device)
            Label = Label.to(args.device)
            # Label[:, 2].sum(axis=(1,2,3)), Label[:, 2].sum(axis=(1,2,3)).argmax()
            # if there is no cancer in the samples in the batch, skip the batch
        
            MODEL.train()
            train_loss, train_count, train_dice = 0.0, 0, []
            for i in range(0, len(Img), args.batch_size):
                optimizer.zero_grad()
                Img_subset = Img[i:i+args.batch_size]
                Label_subset = Label[i:i+args.batch_size]
                Crop_subset = Crop[i:i+args.batch_size]
                
                if len(torch.unique(Label_subset.argmax(dim=1))) != args.out_channels_seg: continue
                
                loss, dice = train_step(MODEL, optimizer, Img_subset, Label_subset, Crop_subset,
                           criterion_segmentation=criterion_segmentation_ds, mode='segmentation', DeepSupervision=True)
                train_loss += loss
                train_count += 1
                train_dice.append(dice)
            MODEL.eval()
            train_loss = train_loss / train_count
            train_dice = np.nanmean(np.vstack(train_dice), axis=0)
        
            if train_count > estimated_steps_per_epoch:
               break
        train_loss = train_loss / train_count
        train_dice = np.nanmean(np.vstack(train_dice), axis=0)

        MODEL.eval()
        test_visualize(GENERATOR=TRAIN_GENERATOR, data_idx=3 , MODEL=MODEL, filename=f'visualization_log/RAUS_clsseg_patch{args.patch_size}_emb{args.embed_dim}_batch{args.batch_size}_epc{epoch}_train_[{train_loss:.3f}]_[{"-".join([f"{dice:.3f}"for dice in train_dice])}]_generator.png')
        test_visualize(GENERATOR=TRAIN_GENERATOR, data_idx=3 , MODEL=MODEL, filename=f'visualization_log/RAUS_clsseg_patch{args.patch_size}_emb{args.embed_dim}_batch{args.batch_size}_epc{epoch}_train_[{train_loss:.3f}]_[{"-".join([f"{dice:.3f}"for dice in train_dice])}]_batch.png', batch=[Img_subset, Label_subset, Crop_subset])
        print(f"Epoch {epoch}, Train Loss: {train_loss:.4f}, Train Dice: {train_dice}, LR: {optimizer.param_groups[0]['lr']}")

        # refresh memory and cache
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        gc.collect()
        

        ########## VALIDATION
        valid_loss, valid_count, valid_dice = 0.0, 0, []
        MODEL.eval()
         ## VALIDATION FOR SMALL DATASET
        if epoch % args.num_full_validation != 0:
            for batch_count, (Img, Label, Crop) in tqdm(enumerate(VALID_DATALOADER_SMALL), total=len(VALID_DATALOADER_SMALL), desc="Training"):
                Img = Img.to(args.device)
                Label = Label.to(args.device)
                if len(torch.unique(Label.argmax(dim=1))) != args.out_channels_seg: continue
                loss, dice = validation_step(MODEL, Img, Label, Crop, criterion_segmentation=criterion_segmentation,mode='segmentation')
                valid_loss += loss
                valid_count += 1
                valid_dice.append(dice)
            valid_loss = valid_loss / valid_count
            valid_dice = np.nanmean(np.vstack(valid_dice), axis=0)
            filename = f'{args.save_dir}/RAUS_clsseg_patch{args.patch_size}_emb{args.embed_dim}_batch{args.batch_size}_epc{epoch}_SMLValid_loss{valid_loss:.4f}_[{"-".join([f"{dice:.3f}"for dice in valid_dice])}]'
            test_visualize(GENERATOR=VALID_GENERATOR, data_idx=12, MODEL=MODEL, filename=f'{filename}.png')
            test_visualize(GENERATOR=VALID_GENERATOR, data_idx=12, MODEL=MODEL, filename=f'{filename}.png')
            test_visualize(GENERATOR=VALID_GENERATOR, data_idx=12, MODEL=MODEL, filename=f'{filename}.png')
            torch.save(MODEL.state_dict(), f'{filename}.pth')
            print(f"Epoch {epoch}, Valid Loss: {valid_loss:.4f}, Valid Dice: {valid_dice}, LR: {optimizer.param_groups[0]['lr']}")
        else: ## VALIDATION FOR ALL DATASET
            for batch_count, (Img, Label, Crop) in tqdm(enumerate(VALID_DATALOADER_ALL), total=len(VALID_DATALOADER_ALL), desc="Training"):
                Img = Img.to(args.device)
                Label = Label.to(args.device)
                if len(torch.unique(Label.argmax(dim=1))) != args.out_channels_seg: continue
                loss, dice = validation_step(MODEL, Img, Label, Crop, criterion_segmentation=criterion_segmentation,mode='segmentation')
                valid_loss += loss
                valid_count += 1
                valid_dice.append(dice)
            valid_loss = valid_loss / valid_count
            valid_dice = np.nanmean(np.vstack(valid_dice), axis=0)
            filename = f'{args.save_dir}/RAUS_clsseg_patch{args.patch_size}_emb{args.embed_dim}_batch{args.batch_size}_epc{epoch}_AllValid_loss{valid_loss:.4f}_[{"-".join([f"{dice:.3f}"for dice in valid_dice])}]'
            test_visualize(GENERATOR=VALID_GENERATOR, data_idx=12, MODEL=MODEL, filename=f'{filename}.png')
            torch.save(MODEL.state_dict(), f'{filename}.pth')
            print(f"Epoch {epoch}, Valid Loss: {valid_loss:.4f}, Valid Dice: {valid_dice}, LR: {optimizer.param_groups[0]['lr']}")
        scheduler.step(valid_loss)

    for idx in range(len(TEST__GENERATOR)):
        batch = TEST__GENERATOR[idx]
        if batch[2].sum() <= 10:
            continue
        Img, Label, Crop = collate_prostate_CS([batch], crop=False)
        
        final_pred = inference_sliding_window(MODEL, Img, window_size=(96, 96, 96), stride=5, num_classes=3)
        cancer_prob = torch.softmax(final_pred, dim=1)[0,2]

        save_filename = os.path.join("pred_result", os.path.basename(TEST__GENERATOR.imageFileName[idx]).split("_trus")[0] + '_prob.nii.gz')
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
