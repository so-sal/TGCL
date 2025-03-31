import numpy as np
import cv2
import torch
import umap
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

def minMax(data):
    return (data-data.min()) / (data.max()-data.min())

def minMax_3d(data): # axis_2
    data_new = data.copy()
    for idx in range(data.shape[2]):
        data_new[:, :, idx] = minMax(data_new[:,:,idx])
    return data_new

def getUmapHeatmap(Image, Label, MODEL, slice_idx):
    x = Image[slice_idx:slice_idx+1]
    batch_size = x.shape[0]
    mask_dim = (x.shape[2] / MODEL.patch_size, x.shape[3] / MODEL.patch_size)
    
    # DINO Part
    x = MODEL.backbone.forward_features(x.cuda())
    x = x['x_norm_patchtokens'].cpu().detach()
    x_2d = x.reshape(batch_size, int(mask_dim[0]),int(mask_dim[1]), MODEL.embedding_size)
    x_2d = x_2d.permute(0,3,1,2)
    x_2d = x_2d.numpy()
    resized_x = np.zeros((x_2d.shape[0], x_2d.shape[1], 256, 256))
    for i in range(x_2d.shape[0]):
        for j in range(x_2d.shape[1]):
            resized_x[i, j] = cv2.resize(x_2d[i, j], (256, 256), interpolation=cv2.INTER_LINEAR)
    x_2d = torch.tensor(resized_x)
    x_2d = x_2d.permute(0,2,3,1)
    x_2d = x_2d.reshape( x_2d.shape[0], 256*256, x_2d.shape[3])
    
    n_components = 3
    reducer = umap.UMAP(n_components=n_components)
    
    gland_region = np.where(Label[slice_idx].argmax(axis=0).reshape( 256* 256) >= 1)[0]
    embedding = reducer.fit_transform(x_2d[0, gland_region])  # 차원 축소
    heatmap = np.zeros( (256*256, 3) )
    for x_idx, heatmap_idx in enumerate(gland_region):
        heatmap[heatmap_idx,:] = embedding[x_idx,:]
    heatmap = heatmap.reshape(256, 256, 3)
    return heatmap

def Visualize_UMAP_pred(Images, Labels, preds, heatmap, slice_idx, filename, modality='TRUS', label_size=256):
    yellow = mcolors.LinearSegmentedColormap.from_list("color1", [(1, 1, 1, 0), (1, 1, 0, 1)], N=2)
    red = mcolors.LinearSegmentedColormap.from_list("color1", [(1, 1, 1, 0), (1, 0, 0, 1)], N=2)
    colors = [(0, 0, 0, 0), (1, 1, 0, 1), (1, 0, 0, 1)]  # RGBA for 0, 1, 2
    jet = plt.cm.get_cmap("jet")  # Get the original jet colormap
    colors = jet(np.arange(256))  # Extract those colors as an array
    colors[:, -1] = np.linspace(0, 1, 256)  # Modify the alpha channel based on the value
    custom_jet = mcolors.LinearSegmentedColormap.from_list("custom_jet", colors)

    Image = cv2.resize(Images[slice_idx, 1].numpy(), (label_size, label_size))
    fig, axs = plt.subplots(2, 4, figsize=(12, 6))
    titles = [[f"Original {modality}", f"Gland label", f"Cancer label", "Prediction"],
              ["UMAP comp1", "UMAP comp2", "UMAP comp3", "UMAP 1-3 RGB"]]
    
    for i in range(2):
        for j in range(4):
            axs[i,j].imshow(Image, cmap="gray")
            axs[i,j].set_title(titles[i][j])
            axs[i,j].axis('off')
            
    axs[0,1].imshow((Labels[slice_idx, 1] + Labels[slice_idx, 2]) > 0, alpha=0.5, cmap=yellow)
    axs[0,2].imshow(Labels[slice_idx, 2], alpha=0.5, cmap=red)
    axs[0,3].imshow(preds[slice_idx].argmax(axis=0), alpha=0.5, vmin=0, vmax=2, cmap=custom_jet)
    axs[1,0].imshow(cv2.resize(minMax(heatmap[:,:,0]), (label_size,label_size) ), alpha=0.7, cmap=custom_jet)
    axs[1,1].imshow(cv2.resize(minMax(heatmap[:,:,1]), (label_size,label_size) ), alpha=0.7, cmap=custom_jet)
    axs[1,2].imshow(cv2.resize(minMax(heatmap[:,:,2]), (label_size,label_size) ), alpha=0.7, cmap=custom_jet)
    axs[1,3].imshow(cv2.resize(minMax_3d(heatmap),     (label_size,label_size) ), alpha=0.5)
    plt.tight_layout()
    plt.savefig(filename)
