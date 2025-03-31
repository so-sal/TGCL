import glob
from icecream import ic
import os
import numpy as np
import nibabel as nib
import albumentations as A
import SimpleITK as sitk
import pandas as pd
import json

from torch.utils.data import Dataset, DataLoader
import torch
from torchvision import transforms
from monai.transforms import (
    Compose,
    RandAffined,
    RandGaussianNoised,
    RandGaussianSmoothd,
    RandAdjustContrastd
)

import monai.transforms as mt
from typing import List, Optional, Dict, Tuple

class US_Generator(Dataset):
    def __init__(self, 
                 imageFileName: pd.Series, 
                 glandFileName: pd.Series,
                 cancerFileName: Optional[pd.Series] = None,
                 modality: Optional[pd.Series] = None,
                 cancerTo2: bool = False, # True: 2 classes (normal + indolent vs csPCa), False: 3 classes (normal vs indolent vs csPCa)
                 Augmentation: bool = False,
                 filter_background_prob: Dict[str, float] = {'TRUS': 1.00, 'MRI': 1.00},
                 img_size: int = 256,
                 Image_Only: bool = False,
                 return_modal: bool = False,
                 preprocess: bool = False,
                 add_channel: bool = False,
                 spacing: tuple = (0.5, 0.5, 0.5),
                 resample_MRI: bool = True):
        
        self.imageFileName = imageFileName
        self.glandFileName = glandFileName
        self.cancerFileName = cancerFileName if isinstance(cancerFileName, pd.Series) else glandFileName
        self.modality = modality if isinstance(modality, pd.Series) else pd.Series(['TRUS'] * len(imageFileName))
        
        self.Image_Only = Image_Only
        self.return_modal = return_modal
        self.preprocess = preprocess
        self.add_channel = add_channel
        self.cancerTo2 = cancerTo2
        self.Augmentation = Augmentation
        self.filter_background_prob = filter_background_prob
        
        self.spacing = spacing
        self.resample_MRI = resample_MRI
        
        if self.Augmentation:
            self.transform3D = mt.Compose([
                mt.EnsureChannelFirstd(keys=['image', 'label1', 'label2'], channel_dim=0),
                mt.RandAffined(
                    keys=['image', 'label1', 'label2'],
                    prob=0.3,
                    rotate_range=(0.1, 0.1, 0.1),        # reduced rotations for ultrasound anatomy
                    scale_range=(0.03, 0.03, 0.03),        # slight scaling variations
                    translate_range=(2, 2, 2),             # smaller translations
                    mode=('bilinear', 'nearest', 'nearest'),
                    padding_mode='zeros'
                ),
                mt.RandGaussianNoised(
                    keys=['image'],
                    prob=0.2,                            # increased probability to mimic ultrasound speckle
                    std=0.03,                            # lower noise level
                ),
                mt.RandGaussianSmoothd(
                    keys=['image'],
                    prob=0.2,                            # increased smoothing probability
                    sigma_x=(0.1, 0.5),
                    sigma_y=(0.1, 0.5),
                    sigma_z=(0.1, 0.5),
                ),
                mt.RandAdjustContrastd(
                    keys=['image'],
                    prob=0.1,
                    gamma=(0.8, 1.2)                     # tighter gamma range for subtle contrast variations
                ),
                mt.EnsureTyped(keys=['image', 'label1', 'label2'])
            ])
        else:
            self.transform3D = mt.Compose([
                mt.EnsureChannelFirstd(keys=['image', 'label1', 'label2']),
                mt.Orientationd(keys=['image', 'label1', 'label2'], axcodes='RAS'),
                mt.EnsureTyped(keys=['image', 'label1', 'label2'])
            ])

    def resample_image(self, image, new_spacing=(0.5, 0.5, 0.5), interpolator=sitk.sitkLinear):
        orig_spacing = image.GetSpacing()
        orig_size = image.GetSize()
        new_size = [int(round(osz * ospc / nspc)) for osz, ospc, nspc in zip(orig_size, orig_spacing, new_spacing)]
        
        resampler = sitk.ResampleImageFilter()
        resampler.SetOutputSpacing(new_spacing)
        resampler.SetSize(new_size)
        resampler.SetOutputOrigin(image.GetOrigin())
        resampler.SetOutputDirection(image.GetDirection())
        resampler.SetInterpolator(interpolator)
        return resampler.Execute(image)
        
    def _loadImage(self, idx: int) -> Tuple[np.ndarray, ...]:
        # Load images while preserving metadata
        image = sitk.ReadImage(self.imageFileName[idx])
        gland = sitk.ReadImage(self.glandFileName[idx])
        cancer = sitk.ReadImage(self.cancerFileName[idx])

        spacing = image.GetSpacing()
        direction = image.GetDirection()
        origin = image.GetOrigin()

        if self.modality[idx] in ["MRI", "ADC", "T2", "DWI"] and self.resample_MRI == True:
            image  = self.resample_image(image)
            gland  = self.resample_image(gland, interpolator=sitk.sitkNearestNeighbor)
            cancer = self.resample_image(cancer, interpolator=sitk.sitkNearestNeighbor)
        
        if self.Image_Only:
            return sitk.GetArrayFromImage(image).astype(np.float32) / 255.0
        
        
        # Ensure consistent spatial properties
        for img in [gland, cancer]:
            img.SetSpacing(spacing)
            img.SetDirection(direction)
            img.SetOrigin(origin)
        
        # Convert to numpy with proper orientation
        image_array = sitk.GetArrayFromImage(image).astype(np.float32) / 255.0
        gland_array = sitk.GetArrayFromImage(gland).astype(np.float32)
        cancer_array = sitk.GetArrayFromImage(cancer).astype(np.float32)
        
        return image_array, gland_array, cancer_array

    def _cancerTo2(self, cancer: np.ndarray) -> np.ndarray:
        cancer[cancer == 1] = 0 # setting indolent cancer to normal
        cancer[cancer > 1] = 1
        return cancer

    def _cancerTo3(self, cancer: np.ndarray) -> np.ndarray:
        cancer[cancer > 2] = 2
        return cancer

    def normalize(self, image: np.ndarray, gland: Optional[np.ndarray] = None, modality: str = 'MRI') -> np.ndarray:
        if modality in ['MRI', 'ADC', 'T2', 'DWI']:
            params = [0.374586, 0.097537] if gland is None else [
                image[gland == 1].mean(),
                image[gland == 1].std()
            ]
        else:  # TRUS
            params = [0.217590, 0.137208] if gland is None else [
                image[gland == 1].mean(),
                image[gland == 1].std()
            ]
        return (image - params[0]) / params[1]

    def get3Channel(self, image: np.ndarray, gland: np.ndarray, cancer: np.ndarray,
                   filter_background: float = 1.0) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        if filter_background == 1.0:
            slices_idx = list(range(len(image)))
        else:
            prostate_present = np.any(gland == 1, axis=(1, 2))
            non_prostate_idx = np.where(~prostate_present)[0]
            num_select = int(len(non_prostate_idx) * filter_background)
            selected_idx = np.random.choice(non_prostate_idx, size=num_select, replace=False)
            slices_idx = sorted(np.where(prostate_present)[0].tolist() + selected_idx.tolist())

        images_3ch, glands_3ch, cancers_3ch = [], [], []
        for idx in slices_idx:
            if idx == 0: idx += 1
            elif idx == image.shape[0] - 1: idx -= 1
            
            images_3ch.append(image[idx-1:idx+2])
            glands_3ch.append(gland[idx])
            cancers_3ch.append(cancer[idx])

        return (
            np.stack(images_3ch),
            np.stack(glands_3ch),
            np.stack(cancers_3ch)
        )

    def __len__(self) -> int:
        return len(self.imageFileName)

    def __getitem__(self, idx: int) -> Tuple[np.ndarray, ...]:
        if self.Image_Only:
            image = self._loadImage(idx)
            data = {'image': image, 'label1': image, 'label2': image}
            transformed = self.transform3D(data)
            image = transformed['image'].squeeze(0).as_tensor()
            image = self.normalize(image, modality=self.modality[idx])
            
            if self.add_channel:
                image, _, _ = self.get3Channel(image, image, image, self.filter_background_prob)
            return (image, self.modality[idx]) if self.return_modal else image
        
        image, gland, cancer = self._loadImage(idx)
        if gland.max() == 255.: 
            gland /= 255.
            
        cancer = self._cancerTo2(cancer) if self.cancerTo2 else self._cancerTo3(cancer)
        
        if self.Augmentation:
            transformed = self.transform3D({
                'image': image[np.newaxis, ...],
                'label1': gland[np.newaxis, ...],
                'label2': cancer[np.newaxis, ...]
            })
            image = transformed['image'].squeeze(0).as_tensor()
            gland = transformed['label1'].squeeze(0).as_tensor()
            cancer = transformed['label2'].squeeze(0).as_tensor()

        image = self.normalize(image, gland, self.modality[idx])
        
        if self.add_channel:
            filter_prob = self.filter_background_prob[self.modality[idx]]
            image, gland, cancer = self.get3Channel(image, gland, cancer, filter_prob)
            
        if self.return_modal:
            return image, gland, cancer, self.modality[idx]
            
        return image, gland, cancer

def safe_to_tensor(data):
    """Converts MetaTensor or numpy array to torch.Tensor."""
    if isinstance(data, torch.Tensor):
        return data.float()  # Ensure float type
    return torch.from_numpy(data).float()

    
def stack_labels(cancerTo2, backg, glands, cancers):
    """ Function for label stacking """
    if cancerTo2:
        return torch.stack([
            backg,
            glands,
            (cancers >= 1).float(),
        ], dim=1)  # [B, 3, D, H, W]
    else:
        return torch.stack([
            backg,
            glands,
            (cancers == 1).float(),
            (cancers >= 2).float(),
        ], dim=1)  # [B, 4, D, H, W]

def get_crop(vol_img, vol_lbl, coords, crop_size, volume_shape):
    """
    Extract a crop from the given volume and label based on coordinates.
    Ensures the crop does not exceed image boundaries.
    """
    d_crop, h_crop, w_crop = crop_size
    D, H, W = volume_shape

    z_start = max(0, min(coords[0], D - d_crop))
    y_start = max(0, min(coords[1], H - h_crop))
    x_start = max(0, min(coords[2], W - w_crop))

    crop_img = vol_img[
        z_start:z_start + d_crop,
        y_start:y_start + h_crop,
        x_start:x_start + w_crop
    ]
    crop_lbl = vol_lbl[
        :,
        z_start:z_start + d_crop,
        y_start:y_start + h_crop,
        x_start:x_start + w_crop
    ]

    return crop_img, crop_lbl, (z_start, y_start, x_start)

def random_crop_3d_batch(images, labels, crop_size=(96, 96, 96), n_crops=4):
    """
    Randomly crops 3D volumes with priority for cancer regions.
    """
    B, D, H, W = images.shape
    C = labels.shape[1]
    d_crop, h_crop, w_crop = crop_size

    all_crops_img = []
    all_crops_lbl = []
    crop_coords = []

    for b_idx in range(B):
        vol_img = images[b_idx]
        vol_lbl = labels[b_idx]

        # Find cancer coordinates
        cancer_mask = (vol_lbl[2] > 0)
        cancer_coords = torch.nonzero(cancer_mask, as_tuple=False)

        # Determine number of crops for the cancer regions and random crops
        if len(cancer_coords) > 0:
            n_cancer_crops = n_crops // 2
            n_random_crops = n_crops - n_cancer_crops
        else:
            n_cancer_crops = 0
            n_random_crops = n_crops

        # Cancer focus crops
        for _ in range(n_cancer_crops):
            if len(cancer_coords) > 0:
                idx = torch.randint(0, len(cancer_coords), (1,)).item()
                z, y, x = cancer_coords[idx]

                # find the location of cancer pixel, and randomly crop around it (location-crop_size ~ location)
                z_start = max(0, min(z - torch.randint(0, d_crop, (1,)).item(), D - d_crop))
                y_start = max(0, min(y - torch.randint(0, h_crop, (1,)).item(), H - h_crop))
                x_start = max(0, min(x - torch.randint(0, w_crop, (1,)).item(), W - w_crop))

                crop_img, crop_lbl, coords = get_crop(
                    vol_img, vol_lbl, (z_start, y_start, x_start), crop_size, (D, H, W)
                )
                all_crops_img.append(crop_img.unsqueeze(0))
                all_crops_lbl.append(crop_lbl.unsqueeze(0))
                crop_coords.append(coords)

        # Random crops
        for _ in range(n_random_crops):
            z_start = torch.randint(0, D - d_crop + 1, (1,)).item()
            y_start = torch.randint(0, H - h_crop + 1, (1,)).item()
            x_start = torch.randint(0, W - w_crop + 1, (1,)).item()

            # randomly crop
            crop_img, crop_lbl, coords = get_crop(
                vol_img, vol_lbl, (z_start, y_start, x_start), crop_size, (D, H, W)
            )
            all_crops_img.append(crop_img.unsqueeze(0))
            all_crops_lbl.append(crop_lbl.unsqueeze(0))
            crop_coords.append(coords)


    all_crops_img = torch.cat(all_crops_img, dim=0)
    all_crops_lbl = torch.cat(all_crops_lbl, dim=0)

    return all_crops_img, all_crops_lbl, crop_coords

def fixed_stride_crop_3d_batch(images, labels, crop_size=(96, 96, 96), stride=80):
    """
    Extracts all 3D crops from volumes using fixed stride for all dimensions.
    """
    B, D, H, W = images.shape
    d_crop, h_crop, w_crop = crop_size

    all_crops_img = []
    all_crops_lbl = []
    crop_coords = []

    for b_idx in range(B):
        vol_img = images[b_idx]
        vol_lbl = labels[b_idx]

        for z_start in range(0, D - d_crop + 1, stride):
            for y_start in range(0, H - h_crop + 1, stride):
                for x_start in range(0, W - w_crop + 1, stride):
                    # Extract crop
                    crop_img, crop_lbl, coords = get_crop(
                        vol_img, vol_lbl, (z_start, y_start, x_start), crop_size, (D, H, W)
                    )
                    all_crops_img.append(crop_img.unsqueeze(0))
                    all_crops_lbl.append(crop_lbl.unsqueeze(0))
                    crop_coords.append(coords)

    all_crops_img = torch.cat(all_crops_img, dim=0)
    all_crops_lbl = torch.cat(all_crops_lbl, dim=0)

    return all_crops_img, all_crops_lbl, crop_coords

def pad_or_crop_volume(volume, target_depth):
    d, h, w = volume.shape
    if d < target_depth:
        pad_total = target_depth - d
        pad_before = pad_total // 2
        pad_after = pad_total - pad_before
        volume = torch.nn.functional.pad(volume, (0, 0, 0, 0, pad_before, pad_after))
    elif d > target_depth:
        crop_start = (d - target_depth) // 2
        volume = volume[crop_start:crop_start+target_depth, :, :]
    return volume

def collate_prostate_CS(batch, crop=True, crop_size=96, n_crops_per_volume=15, cancerTo2=True, random_crop_ratio=0.85, fixed_stride=80, axis_size=256):
    """
    Each batch item: (image3D [D,H,W], gland3D [D,H,W], cancer3D [D,H,W])
    - crop: boolean indicating whether to crop the images
    - crop_size: size of the crops to be extracted
    - n_crops_per_volume: number of crops to extract per volume
    - cancerTo2: boolean indicating whether to convert cancer labels to 3 classes (background, normal, cancer), if false then it will distinguish between indolent and csPCa
    - random_crop_ratio: ratio of random crops to fixed crops
        random_crop: randomly crop but focusing on cancer regions
        fixed_stride_crop: crop all the patches with fixed stride
    - fixed_stride: stride for fixed crops
    Returns:
      images: [B, D, H, W] or [B*n_crops, D_crop, H_crop, W_crop]
      labels: [B, 4, D, H, W] or [B*n_crops, 4, D_crop, H_crop, W_crop]
      coords: list of (batch_index, z_start, y_start, x_start) for each crop
    """

    # Convert each item to tensor
    imgs = [safe_to_tensor(item[0]) for item in batch]
    glands = [safe_to_tensor(item[1]) for item in batch]
    cancers = [safe_to_tensor(item[2]) for item in batch]

    axial_dims = [img.shape[0] for img in imgs]
    if len(set(axial_dims)) > 1 or axial_dims[0] != axis_size:
        target_depth = axis_size
        imgs = [pad_or_crop_volume(img, target_depth) for img in imgs]
        glands = [pad_or_crop_volume(g, target_depth) for g in glands]
        cancers = [pad_or_crop_volume(c, target_depth) for c in cancers]

    images = torch.stack(imgs)
    glands = torch.stack(glands)
    cancers = torch.stack(cancers)
    
    # Overwrite gland with 0 where cancer >= 1
    glands[cancers >= 1] = 0
    backg = (~((glands == 1) + (cancers == 1) + (cancers == 2))).float()
    labels = stack_labels(cancerTo2, backg, glands, cancers)

    if crop:
        # Determine the number of volumes for random crop and fixed crop
        B = images.shape[0]
        n_random = int(round(B * random_crop_ratio, 0))

        # Separate indices for random and fixed crop
        random_indices = torch.randperm(B)[:n_random]
        fixed_indices = torch.randperm(B)[n_random:]

        crops_img, crops_lbl, all_coords = [], [], []
        if len(random_indices) > 0:
            random_crops_img, random_crops_lbl, random_coords = random_crop_3d_batch(
                images[random_indices],
                labels[random_indices],
                crop_size=(crop_size, crop_size, crop_size),
                n_crops=n_crops_per_volume
            )
            crops_img.append(random_crops_img)
            crops_lbl.append(random_crops_lbl)
            all_coords.extend(random_coords)

        if len(fixed_indices) > 0:
            fixed_crops_img, fixed_crops_lbl, fixed_coords = fixed_stride_crop_3d_batch(
                images[fixed_indices],
                labels[fixed_indices],
                crop_size=(crop_size, crop_size, crop_size),
                stride=fixed_stride,  # Example stride (customizable)
            )
            crops_img.append(fixed_crops_img)
            crops_lbl.append(fixed_crops_lbl)
            all_coords.extend(fixed_coords)


        # Combine random and fixed crops
        all_crops_img = torch.cat(crops_img, dim=0)
        all_crops_lbl = torch.cat(crops_lbl, dim=0)
        

        # Shuffle the combined crops
        perm = torch.randperm(all_crops_img.size(0))
        all_crops_img = all_crops_img[perm]
        all_crops_lbl = all_crops_lbl[perm]
        all_coords = [all_coords[i] for i in perm.tolist()]
        return all_crops_img, all_crops_lbl, all_coords

    return images, labels, None



def getData(Image_path, Gland_path, Label_path, Modality, file_extensions):
    """
    This function collects image, gland, and label file paths for a given modality, organizing them into a DataFrame.
    It is designed to handle data where file names include a modality-specific identifier, such as '84216_001_trus.nii.gz'.
    
    Parameters:
    - Image_path: Path to the directory containing the image files.
    - Gland_path: Path to the directory containing the gland annotation files.
    - Label_path: Path to the directory containing the cancer label files.
    - Modality: A string representing the imaging modality (e.g., 'MRI', 'US') used in file naming.
    - file_extensions: A dictionary specifying the postfix to be added to each filename for images, glands, and labels.
    
    Returns:
    - A pandas DataFrame with columns for [image paths, gland paths, label paths, modality, and patient ID].
    """

    Image, Gland, Label = [], [], []
    Modal, patID = [], []

    img = [i.split(file_extensions['Image_name'])[0] for i in os.listdir(Image_path)] # patientID_modality.nii.gz
    gld = [i.split(file_extensions['Gland_name'])[0] for i in os.listdir(Gland_path)] # patientID_modality_prostate_label.nii.gz
    can = [i.split(file_extensions['Cancer_name'])[0] for i in os.listdir(Label_path)] # patientID_modality_roi_bxconfirmed_label.nii.gz
    
    intersection_filenames = set(img) & set(gld) & set(can)
    for filename in sorted(intersection_filenames):
        Image.append(Image_path + filename + file_extensions['Image_name'])
        Gland.append(Gland_path + filename + file_extensions['Gland_name'])
        Label.append(Label_path + filename + file_extensions['Cancer_name'])
        Modal.append(Modality)
        patID.append(filename)

    Dataset = pd.DataFrame({'Image':Image, 'Gland':Gland, 'Cancer':Label, 'Modality':Modal, 'PatientID':patID})
    Dataset.set_index('PatientID', inplace=True)
    return Dataset

def aggregate(df):
    return pd.concat(df.values(), axis=0).reset_index(drop=True)

def align_pred_with_ref(pred_tensor, ref_sitk):
    pred_np = pred_tensor.cpu().numpy()
    pred_sitk = sitk.GetImageFromArray(pred_np)
    pred_sitk.SetSpacing(ref_sitk.GetSpacing())
    pred_sitk.SetOrigin(ref_sitk.GetOrigin())
    pred_sitk.SetDirection(ref_sitk.GetDirection())
    return pred_sitk

def resize_pred_to_label(pred_sitk, label_sitk):
    # Ensure the orientation matches
    if pred_sitk.GetDirection() != label_sitk.GetDirection():
        pred_sitk = sitk.Resample(pred_sitk, label_sitk, sitk.Transform(), sitk.sitkNearestNeighbor)
    
    # Resample pred_sitk to match label_sitk
    resample = sitk.ResampleImageFilter()
    resample.SetReferenceImage(label_sitk)
    resample.SetInterpolator(sitk.sitkNearestNeighbor)
    resample.SetTransform(sitk.Transform())
    resample.SetDefaultPixelValue(0)
    resized_pred = resample.Execute(pred_sitk)
    return resized_pred

def input_verification_crop_or_pad(image,size,physical_size):
    """
    Calculate target size for cropping and/or padding input image
    Parameters:
    - image: image to be resized (sitk.Image or numpy.ndarray)
    - size: target size in voxels (z, y, x)
    - physical_size: target size in mm (z, y, x)
    Either size or physical_size must be provided.
    Returns:
    - shape of original image (in convention of SimpleITK (x, y, z) or numpy (z, y, x))
    - size of target image (in convention of SimpleITK (x, y, z) or numpy (z, y, x))
    """
    # input conversion and verification
    if physical_size is not None:
        # convert physical size to voxel size (only supported for SimpleITK)
        if not isinstance(image, sitk.Image):
            raise ValueError("Crop/padding by physical size is only supported for SimpleITK images.")
        # spacing_zyx = list(image.GetSpacing())[::-1]
        spacing_zyx = list(image.GetSpacing()) # (0.5,0.5,3.0)
        size_zyx = [length/spacing for length, spacing in zip(physical_size, spacing_zyx)]
        size_zyx = [int(np.round(x)) for x in size_zyx] # size after cropping

        if size is None:
            # use physical size
            size = size_zyx
        else:
            # verify size
            # print("size",size)
            # print("size_zyx",size_zyx)
            if size != size_zyx:
                raise ValueError(f"Size and physical size do not match. Size: {size}, physical size: "
                                 f"{physical_size}, spacing: {spacing_zyx}")
    if isinstance(image, sitk.Image):
        # determine shape and convert convention of (z, y, x) to (x, y, z) for SimpleITK
        shape = image.GetSize()
        # size = list(size)[::-1]
        size = list(size)
    else:
        # determine shape for numpy array
        assert isinstance(image, (np.ndarray, np.generic))
        shape = image.shape
        size = list(size) # size before cropping
    rank = len(size)
    assert rank <= len(shape) <= rank + 1, \
        f"Example size doesn't fit image size. Got shape={shape}, output size={size}"
    return shape, size

def crop_or_pad(image, size, physical_size, crop_only = False,center_of_mass = None):
    """
    Resize image by cropping and/or padding
    Parameters:
    - image: image to be resized (sitk.Image or numpy.ndarray)
    - size: target size in voxels (z, y, x)       -->(256,256,3.0)
    - physical_size: target size in mm (z, y, x)  -->(256*0.5,256*0.5,num_slice*3.0)
    Either size or physical_size must be provided.
    Returns:
    - resized image (same type as input)
    """
    # input conversion and verification
    #print('center_of_mass',center_of_mass)
    shape, size = input_verification_crop_or_pad(image, size, physical_size)

    # set identity operations for cropping and padding
    # print('----size------',size)
    rank = len(size)
    padding = [[0, 0] for _ in range(rank)]
    slicer = [slice(None) for _ in range(rank)]

    # for each dimension, determine process (cropping or padding)
    if center_of_mass is not None:
        desired_com = [int(x / 2) for x in size]
        displacement = [(x-y) for x,y in zip(desired_com,center_of_mass)]
        for i in range(rank):
            # if shape[i] < size[i]:
            if  int(center_of_mass[i] - np.floor((size[i]) / 2.))<0 or int(center_of_mass[i] + np.floor((size[i]) / 2.))>=256:
                if crop_only:
                    continue
                padding[i][0] = max((size[i] - shape[i]) // 2 + displacement[i],0)
                padding[i][1] = max(size[i] - shape[i] - padding[i][0],0)
                # padding[i][0] = np.maximum(np.floor((np.array(size[i]) - np.array(shape[i])) / 2).astype(int) - displacement[i], 0)
                # padding[i][1] = size[i] - padding[i][0] - shape[i]
            # else:
            idx_start = max(int(center_of_mass[i] - np.floor((size[i]) / 2.)),0)
            idx_end = min(int(center_of_mass[i] + np.floor((size[i]) / 2.)),shape[i])
            #print('idx_start',idx_start)
            #print('idx_end',idx_end)
            slicer[i] = slice(idx_start, idx_end)
    else:
        for i in range(rank):
            if shape[i] < size[i]:
                if crop_only:
                    continue
                # set padding settings
                padding[i][0] = (size[i] - shape[i]) // 2
                padding[i][1] = size[i] - shape[i] - padding[i][0]
            else:
                # create slicer object to crop image
                idx_start = int(np.floor((shape[i] - size[i]) / 2.))
                idx_end = idx_start + size[i]
                slicer[i] = slice(idx_start, idx_end)

    # print("slicer",slicer)

    # crop and/or pad image
    if isinstance(image, sitk.Image):
        #print('padding',padding)
        pad_filter = sitk.ConstantPadImageFilter()
        pad_filter.SetPadLowerBound([pad[0] for pad in padding])
        pad_filter.SetPadUpperBound([pad[1] for pad in padding])
        return pad_filter.Execute(image)[tuple(slicer)]
    else:
        return np.pad(image[tuple(slicer)], padding)


def main():
    ## Load data directly
    DataPath = '/home/sosal/Data/'
    Image_path = os.path.join(DataPath, 'TRUS/*')
    Gland_path = os.path.join(DataPath, 'TRUS_Prostate_Label/*')
    Label_path = os.path.join(DataPath, 'TRUS_ROI_Bxconfirmed_Label/*')
    
    Image_file = glob.glob(Image_path) # assume list of Image, Gland, Label is same!
    Gland_file = glob.glob(Gland_path)
    Label_file = glob.glob(Label_path)
    
    # Assume all of the modality of files is TRUS
    modality = ['TRUS' for i in range(len(Image_file))]

    Generator = US_Generator(
        Image_file, Gland_file, Label_file, modality, 
        cancerTo2=True, Augmentation=True, filter_background_prob=.1) #, transformUS, transformMRI)
    
    a, b, c = Generator.__getitem__(10)
    ic(a.shape, b.shape, c.shape)
    ic(a.max(), a.min(), a.mean())
    ic(b.max(), b.min())
    ic(c.max(), c.min())


def reclassify_tensor(x, out_channels_cls=2):
    # Calculate sum for each class: [batch_size, num_classes]
    class_sums = x.sum(dim=(2, 3, 4))  

    # Create empty tensor for new classification: [batch_size, 3]
    batch_size = x.shape[0]
    new_labels = torch.zeros((batch_size, out_channels_cls), dtype=torch.float32)

    for i in range(batch_size):
        # Priority: csPCa > indolent > normal

        if class_sums.shape[1] >= 4:
            if class_sums[i, 3] > 0:  # class 4 (index 3): csPCa
                new_labels[i, 2] = 2
        elif class_sums[i, 2] > 0:  # class 3 (index 2): indolent  
            new_labels[i, 1] = 1
        elif class_sums[i, 0] > 0 or class_sums[i, 1] > 0:  # class 1,2 (index 0,1): normal
            new_labels[i, 0] = 1
            
    return new_labels

def get_generator(config):
    Dataset = getData(config['paths']['Image_path'],
                      config['paths']['Gland_path'],
                      config['paths']['Label_path'],
                      config['Modality'], config['file_extensions'])

    with open(config['SplitValidation']['internal_split'], 'r') as file:
        SplitValidation_dict = json.load(file)
    with open(config['SplitValidation']['testset'], 'r') as file:
        SplitValidation_dict_test = json.load(file)
        SplitValidation_dict_test = pd.DataFrame(SplitValidation_dict_test['bx_test'])
    
    TrainDataset = Dataset.loc[SplitValidation_dict[config['FOLD_IDX']]['train']]
    ValidDataset = Dataset.loc[SplitValidation_dict[config['FOLD_IDX']]['val']]
    Test_Dataset = Dataset.loc[SplitValidation_dict_test['Anon_ID']]

    cancer_idx = [0,1,3,4,5,6,7,8,10,11,12,15,16,17,19,20,22,23,24,26,27,28,29,30,31,32,33,34]
    ValidDataset_small = ValidDataset.iloc[cancer_idx, :]


    TrainDataset.reset_index(drop=True, inplace=True)
    ValidDataset_small.reset_index(drop=True, inplace=True)
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
                    imageFileName=ValidDataset_small['Image'],
                    glandFileName=ValidDataset_small['Gland'],
                    cancerFileName=ValidDataset_small['Cancer'],
                    modality=ValidDataset_small['Modality'],
                    cancerTo2=True,
                    Augmentation=False)

    TEST__GENERATOR = US_Generator(
                    imageFileName=Test_Dataset['Image'],
                    glandFileName=Test_Dataset['Gland'],
                    cancerFileName=Test_Dataset['Cancer'],
                    modality=Test_Dataset['Modality'],
                    cancerTo2=True,
                    Augmentation=False)

    return TRAIN_GENERATOR, VALID_GENERATOR, VALID_GENERATOR_small, TEST__GENERATOR


if __name__ == "__main__":
    main()


