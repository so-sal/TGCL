from scipy import interpolate
from matplotlib.colors import LinearSegmentedColormap, ListedColormap
import matplotlib.patches as patches
import cv2
import matplotlib.pyplot as plt
import torch
import numpy as np
import SimpleITK as sitk

import numpy as np
import seaborn as sns
import torch.nn.functional as F
from sklearn.cluster import KMeans

# pred = inference_sliding_window(MODEL, Image[i:i+1].to(args.device), window_size=window_size, overlap=overlap)
def inference_sliding_window(model, image, window_size=(96, 96, 96), stride=20, num_classes=3):
    """
    Sliding window inference for 3D segmentation
    
    Args:
        model: trained segmentation model
        image: input image (1, d, h, w)
        window_size: Sliding window size (d, h, w)
        overlap: overlap ratio for stride (0~1)
    """
    model.eval()
    device = next(model.parameters()).device
    
    # calculate stride
    # 출력 텐서 초기화 (클래스별 확률맵)
    D, H, W = image.shape[1:]
    pred = torch.zeros((1, num_classes, D, H, W), device=device)
    count = torch.zeros((1, 1, D, H, W), device=device)
    
    with torch.no_grad():
        for d in range(0, D - window_size[0] + 1, stride):
            for h in range(0, H - window_size[1] + 1, stride):
                for w in range(0, W - window_size[2] + 1, stride):
                    # 윈도우 추출
                    window = image[:, 
                                 d:d + window_size[0],
                                 h:h + window_size[1],
                                 w:w + window_size[2]]
                    
                    # inference
                    window = window.to(device)
                    output = model(window, crop_pos=[[d, h, w]])['segmentation']
                    
                    # accumulating
                    pred[:, :,
                         d:d + window_size[0],
                         h:h + window_size[1],
                         w:w + window_size[2]] += output
                    
                    count[:, :,
                          d:d + window_size[0],
                          h:h + window_size[1],
                          w:w + window_size[2]] += 1
    
    # Calculate final prediction (averaging)
    final_pred = pred / count
    
    return final_pred

# visualize_max_cancer(TestDataset, ViTFeatureExtractor, 
def visualize_max_cancer(TestDataset, MODEL, filename, args, window_size=(96, 96, 96), stride=20, num_classes=3):
    Image, Label = TestDataset
    Pred, Dice = [], []
    slice_idxes = Label[:, 2:4].sum(axis=(1,3,4)).argmax(1)
    with torch.no_grad():
        for i in range(Image.shape[0]):
            pred = inference_sliding_window(MODEL, Image[i:i+1].to(args.device), window_size=window_size, stride=stride, num_classes=num_classes)
            pred = torch.softmax(pred, dim=1).cpu().detach()
            Pred.append(pred)
            call = (pred[:, 2:4].sum(axis=1)[0]>0.3).float().numpy()
            y = Label[i,2:4].sum(axis=0).numpy()
            Dice.append(2. * np.sum(y * call) / (np.sum(call) + np.sum(y)))
    Pred = torch.cat(Pred, dim=0)
    Pred = torch.stack([Pred[case_idx, :, slice_idx] for case_idx, slice_idx in enumerate(slice_idxes)])
    Image = torch.stack([Image[case_idx, slice_idx] for case_idx, slice_idx in enumerate(slice_idxes)])
    Label = torch.stack([Label[case_idx, :, slice_idx] for case_idx, slice_idx in enumerate(slice_idxes)])

    jet = plt.cm.jet
    colors = jet(np.arange(256))
    colors[:int(0.10*256), -1] = 0
    jet_transp = LinearSegmentedColormap.from_list("custom_jet", colors)

    colors = [(0,0,0,0), (1,1,0,1), (1,0,0,1)]  # 투명, 노랑
    label_cmap = ListedColormap(colors)

    yel_cmap = ListedColormap([(0,0,0,0), (1,1,0,1)])
    red_cmap = ListedColormap([(0,0,0,0), (1,0,0,1)])
    
    segmentation_maps = Label.argmax(axis=1).cpu().numpy()
    segmentation_maps[segmentation_maps > 2] = 2
    
    contour_image_gls, contour_image_ccs, contour_image_boths = [], [], []
    for idx in range(Label.shape[0]):
        segmentation_map = segmentation_maps[idx]
        mask_label1 = (segmentation_map == 1).astype(np.uint8)
        mask_label2 = (segmentation_map == 2).astype(np.uint8)
        mask_combined = np.logical_or(segmentation_map == 1, segmentation_map == 2).astype(np.uint8)
        contours_gl = smooth_contours(mask_label1)
        contours_cc = smooth_contours(mask_label2)
        contours_both = smooth_contours(mask_combined)
        contour_image_gl = np.zeros(segmentation_map.shape, dtype=np.uint8)
        contour_image_cc = np.zeros(segmentation_map.shape, dtype=np.uint8)
        contour_image_both = np.zeros(segmentation_map.shape, dtype=np.uint8)
        cv2.drawContours(contour_image_gl, contours_gl, -1, 1, 1)
        cv2.drawContours(contour_image_cc, contours_cc, -1, 1, 1)
        cv2.drawContours(contour_image_both, contours_both, -1, 1, 1)
        contour_image_gls.append(contour_image_gl)
        contour_image_ccs.append(contour_image_cc)
        contour_image_boths.append(contour_image_both)

    fig, axes = plt.subplots(2, len(Image), figsize=(len(Image)*3,6))
    for i in range(Image.shape[0]):
        axes[0,i].set_title(f"Dice: {Dice[i]:.3f}")
        for j in range(2):
            axes[j,i].imshow(Image[i], cmap="gray")
            axes[j,i].imshow(contour_image_boths[i], vmin=0, vmax=1, cmap=yel_cmap, alpha=1.0)
            axes[j,i].imshow(contour_image_ccs[i], vmin=0, vmax=1, cmap=red_cmap, alpha=1.0)
            axes[j,i].axis('off')
            # show dice score per column title

    for i in range(Image.shape[0]):
        axes[1,i].imshow( Pred[i, 2:4].sum(axis=0).cpu().numpy(), cmap=jet_transp, alpha=0.7)
    plt.tight_layout()
    plt.savefig(filename)

def visualize_small_patch(MODEL, Img, Label, Crop, slides, filename):
    Img_vis = Img[slides]
    Label_vis = Label[slides]
    Crop_vis = [Crop[slides[i]] for i in range(len(slides))]
    slice_idx = [Label_vis[i,2].sum(axis=(1,2)).argmax().item() for i in range(len(Label_vis))]
    
    fig, axes = plt.subplots( len(slides), 4, figsize=(12, 3*len(slides)))
    for i in range(len(slides)):
        for j in range(4):
            axes[i, j].imshow(Img_vis[i, slice_idx[i], ].cpu(), cmap='gray')
        
    for i in range(len(slides)):
        prediction_vis = MODEL(Img_vis[i:i+1], crop_pos=Crop_vis[i:i+1])
        prediction_vis = torch.softmax(prediction_vis['segmentation'], dim=1)

        axes[i,1].imshow(Label_vis[i, 1, slice_idx[i], ].cpu().numpy(), vmin=0, vmax=2, cmap='jet', alpha=0.5)
        axes[i,1].imshow(Label_vis[i, 2, slice_idx[i], ].cpu().numpy(), vmin=0, vmax=1, cmap='jet', alpha=0.5)
        axes[i,2].imshow(prediction_vis[0, 1:3].sum(axis=0)[slice_idx[i], ].cpu().detach().numpy(), vmin=0, vmax=2, cmap='jet', alpha=0.5)
        axes[i,3].imshow(prediction_vis[0, 2, slice_idx[i], ].cpu().detach().numpy(), vmin=0, vmax=1.0, cmap='jet', alpha=0.5)
    
    plt.savefig(filename)
    plt.close()
    
def visualize_slices(Img, Label, prediction, slice_indices, filename):
    """
    Visualize multiple slices along the z-axis for a single 3D volume (Img, Label, and Prediction).

    Args:
        Img: Input image, shape (96, 96, 96).
        Label: Ground truth label, shape (C, 96, 96, 96).
        prediction: Model prediction, shape (C, 96, 96, 96).
        slice_indices: List of z-axis slice indices to visualize.
        filename: Filename to save the visualization.
    """
    ncol = 5
    fig, axes = plt.subplots(len(slice_indices), ncol, figsize=(ncol*3, 3*len(slice_indices)))

    # Ensure axes is 2D (for single slice case)
    if len(slice_indices) == 1:
        axes = np.expand_dims(axes, axis=0)

    for i, z_idx in enumerate(slice_indices):
        # Image
        for j in range(ncol):
            axes[i, j].imshow(Img[z_idx].cpu().numpy(), cmap="gray")
            axes[i, j].set_title(f"Image (Slice {z_idx})")

        # Label: gland (Class 1) and cancer (Class 2)
        axes[i, 1].imshow(Label[1, z_idx].cpu().numpy(), vmin=0, vmax=2, cmap="jet", alpha=0.5)
        axes[i, 1].imshow(Label[2, z_idx].cpu().numpy(), vmin=0, vmax=1, cmap="jet", alpha=0.5)
        axes[i, 1].set_title(f"Label (Slice {z_idx})")

        # Prediction: gland + cancer
        axes[i, 2].imshow(prediction[1:3].sum(axis=0)[z_idx], vmin=0, vmax=2, cmap="jet", alpha=0.5)
        axes[i, 2].set_title(f"Prediction (Gland+Cancer, Slice {z_idx})")

        # Prediction: cancer (Class 2 only)
        axes[i, 3].set_title(f"Prediction (Cancer, Slice {z_idx})")
        axes[i, 3].imshow(prediction[2, z_idx], vmin=0, vmax=1.0, cmap="jet", alpha=0.5)

        axes[i, 4].set_title(f"Prediction (argmax Slice {z_idx})")
        axes[i, 4].imshow(prediction.argmax(axis=0)[z_idx], vmin=0, vmax=2.0, cmap="jet", alpha=0.5)

    # Save the figure
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()

def smooth_contours(mask, closing_kernel_size=5, epsilon_factor=0.01, min_contour_length=5):
    closed_mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((closing_kernel_size, closing_kernel_size), np.uint8))
    contours, _ = cv2.findContours(closed_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    smooth_contours = []
    for contour in contours:
        if len(contour) < min_contour_length:
            continue
        epsilon = epsilon_factor * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon, True)
        if len(approx) < 3:
            continue
        x, y = approx.squeeze().T
        if len(x) < 3:
            continue
        try:
            tck, _ = interpolate.splprep([x, y], s=0, per=True)
            smooth_contour = np.column_stack(interpolate.splev(np.linspace(0, 1, 1000), tck)).astype(np.int32)
            smooth_contours.append(smooth_contour)
        except:
            smooth_contours.append(approx)
    return smooth_contours

def saveData(pred, origin_t2, save_filename):
    origin_t2_sitk = sitk.ReadImage(origin_t2)
    pred_image = sitk.GetImageFromArray(pred)
    pred_image.SetSpacing(origin_t2_sitk.GetSpacing())
    pred_image.SetDirection(origin_t2_sitk.GetDirection())
    pred_image.SetOrigin(origin_t2_sitk.GetOrigin())
    sitk.WriteImage(pred_image, save_filename)

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.nn.functional as F
from sklearn.cluster import KMeans
from scipy.cluster.hierarchy import linkage, dendrogram, leaves_list

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.nn.functional as F
from scipy.cluster.hierarchy import linkage, leaves_list

def visualize_token_contrastive_with_labels(f1_sample, f2_sample, labels1, confs1, labels2, confs2,
                                            logits=None, temperature=0.1, sort_tokens=True, filename=None):
    import numpy as np
    import matplotlib.pyplot as plt
    import seaborn as sns
    from scipy.cluster.hierarchy import linkage, leaves_list
    import torch.nn.functional as F

    # Compute logits if not provided.
    if logits is None:
        f1_norm = F.normalize(f1_sample, dim=-1)
        f2_norm = F.normalize(f2_sample, dim=-1)
        logits = torch.matmul(f1_norm, f2_norm.T) / temperature

    logits_np = logits.detach().cpu().numpy()
    N = f1_sample.shape[0]

    # Normalize logits for better visual contrast.
    logits_np = (logits_np - logits_np.min()) / (logits_np.max() - logits_np.min())

    # Sort tokens using the original indices.
    labels1_order = np.argsort(labels1)
    labels2_order = np.argsort(labels2)

    labels1_sorted = labels1[labels1_order]
    labels2_sorted = labels2[labels2_order]
    confs1_sorted = confs1[labels1_order]
    confs2_sorted = confs2[labels2_order]

    # Reorder the similarity matrix accordingly.
    logits_np_sorted = logits_np[np.ix_(labels1_order, labels2_order)]

    # Optionally perform hierarchical clustering within each class for MRI tokens (y-axis).
    for class_idx in range(3):
        class_indices = np.where(labels1_sorted == class_idx)[0]
        if len(class_indices) > 1:
            Z = linkage(logits_np_sorted[class_indices, :], method='ward')
            sorted_class_indices = class_indices[leaves_list(Z)]
            logits_np_sorted[class_indices, :] = logits_np_sorted[sorted_class_indices, :]
            confs1_sorted[class_indices] = confs1_sorted[sorted_class_indices]

    # And for TRUS tokens (x-axis).
    for class_idx in range(3):
        class_indices = np.where(labels2_sorted == class_idx)[0]
        if len(class_indices) > 1:
            Z = linkage(logits_np_sorted[:, class_indices].T, method='ward')
            sorted_class_indices = class_indices[leaves_list(Z)]
            logits_np_sorted[:, class_indices] = logits_np_sorted[:, sorted_class_indices]
            confs2_sorted[class_indices] = confs2_sorted[sorted_class_indices]

    def get_rgba(label, conf):
        if label == 0:
            return (0, 0, 0, 1)          # Black for Background.
        elif label == 1:
            return (1, 1, 0, conf)         # Yellow for Normal.
        elif label == 2:
            alpha = 1.0 if conf >= 0.1 else 0.5
            return (1, 0, 0, alpha)        # Red for Cancer.
        else:
            return (0.5, 0.5, 0.5, 1.0)

    # Use MRI labels/confidences for y-axis, TRUS for x-axis.
    colors_y = [get_rgba(labels1_sorted[i], confs1_sorted[i]) for i in range(N)]
    colors_x = [get_rgba(labels2_sorted[i], confs2_sorted[i]) for i in range(N)]

    # Create the figure.
    fig = plt.figure(figsize=(12, 10))
    gs = fig.add_gridspec(nrows=2, ncols=3, width_ratios=[0.05, 1, 0.05],
                          height_ratios=[0.05, 1], wspace=0.02, hspace=0.02)

    # Top colorbar (TRUS tokens)
    ax_top = fig.add_subplot(gs[0, 1])
    ax_top.imshow(np.array([colors_x]), aspect='auto')
    ax_top.axis('off')

    # Left colorbar (MRI tokens)
    ax_left = fig.add_subplot(gs[1, 0])
    ax_left.imshow(np.array([[c] for c in colors_y]), aspect='auto')
    ax_left.axis('off')

    # Heatmap
    ax_heatmap = fig.add_subplot(gs[1, 1])
    hm = sns.heatmap(logits_np_sorted, cmap='jet', ax=ax_heatmap, cbar=False,
                     xticklabels=False, yticklabels=False)
    ax_heatmap.set_title("Hierarchical Token-Level Contrastive Similarity Matrix")
    ax_heatmap.set_xlabel("Tokens (TRUS)")
    ax_heatmap.set_ylabel("Tokens (MRI)")

    # Colorbar for heatmap.
    ax_cbar = fig.add_subplot(gs[1, 2])
    cbar = fig.colorbar(hm.get_children()[0], cax=ax_cbar)
    cbar.ax.tick_params(labelsize=10)
    ax_top.set_xlim(ax_heatmap.get_xlim())

    if filename is not None:
        plt.savefig(filename, bbox_inches='tight')
        plt.close(fig)
    else:
        plt.show()


# Example usage:
# Assume f_MRI[0] and f_TRUS[0] are (N, D) embeddings for one sample,
# and conf1, pred1, conf2, pred2 are obtained from torch.max(prob, dim=-1) for each modality.
# For demonstration, here's how to call the function:
# visualize_token_contrastive_with_labels(f1_sample=f_MRI[0],
#                                        f2_sample=f_TRUS[0],
#                                        labels1=pred1.cpu().numpy(),
#                                        confs1=conf1.cpu().numpy(),
#                                        labels2=pred2.cpu().numpy(),
#                                        confs2=conf2.cpu().numpy(),
#                                        temperature=0.1)
