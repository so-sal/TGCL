# Label-Guided Token Contrastive Learning for Registration-Free Prostate Cancer Detection

This repository contains the official PyTorch implementation of our proposed method:

**"Label-Guided Token Contrastive Learning for Registration-Free Prostate Cancer Detection in Ultrasound Images with MRI Supervision."**

## Overview

We introduce a novel framework leveraging Vision Transformer-based **Token-Guided Contrastive Fusion (TGCF)** for robust prostate cancer segmentation and classification using ultrasound (US) imaging, supervised by MRI without the need for explicit image registration.

## Repository Structure

- `architecture`: Contains the 3D Vision Transformer-based models. and `TGCF` modules.
- `evaluation_prostate_cancer_detection_on_MRI/`: Evaluation scripts for prostate cancer detection performance.
- `utils/`: Helper functions for data loading, training, inference, visualization, and metrics.

### Main Scripts
```bash

### Cross-modal training
python train_TGCF_cls.py --cuda_device 0 \
                         --config_MRI configs/config_T2.yaml \
                         --config_TRUS configs/config_TRUS.yaml \
                         --num_epochs 100

### Independent modal training:
train_MRI.py
train_US.py

### Inference
python inference_US.py --cuda_device 0 \
                       --phase test \
                       --pretrained_weights_TGCF path_to_weights.pth

### Evaluation
lesion_level_eval_2class_allcases.py

### Visualization
visualization.py

