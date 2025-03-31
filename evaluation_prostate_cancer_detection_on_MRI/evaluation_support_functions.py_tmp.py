# -*- coding: utf-8 -*-
"""
Created on Tue Oct  5 19:55:07 2021

@author: Indrani

lesion-level evaluation based on sextants
code used for labels paper eval
evaluate indolent, aggressive and cancer 


"""
from tqdm import tqdm
import torch
import numpy as np
import SimpleITK as sitk
import os
from sklearn import metrics
from scipy import interp
import matplotlib.pyplot as plt
import pandas as pd
from skimage.measure import label
from scipy.spatial.distance import dice as dice_dist
import csv
from scipy.stats import iqr
from skimage.morphology import dilation, disk, erosion, closing
from scipy import ndimage
import math
from sklearn.metrics import average_precision_score
from natsort import natsorted
from util_functions.train_functions import tensor_shuffle, getPatchTokens
from util_functions.Prostate_DataGenerator import US_MRI_Generator, collate_prostate_position, getData

#%%

def Evaluate_ROC_Dice_Fusion(MODELs, MODEL_Fusion, GENERATORs, collate_fn, args, config, Thresholds=[0.5],
                      cancer_only = False, percentagg = 0.01, modal=False, pos=True,
                      clinically_significant=False): #
    for key, MODEL in MODELs.items():
        MODEL.eval()
    MODEL_Fusion.eval()
    results = {Threshold: [] for Threshold in Thresholds}
    
    for sample_idx in tqdm(range(len(GENERATORs['T2']))):
        case = os.path.basename(GENERATORs['T2' ].imageFileName[sample_idx]).split(f"_{config['Modality'].lower()}")[0]
        Image_T2,  Label, Posit = collate_fn([GENERATORs['T2' ].__getitem__(sample_idx)])
        Image_ADC, Label, Posit = collate_fn([GENERATORs['ADC'].__getitem__(sample_idx)])
        Image_DWI, Label, Posit = collate_fn([GENERATORs['DWI'].__getitem__(sample_idx)])

        Image_T2 , Label, Posit = tensor_shuffle(Image_T2 , Label, args.device, pos = Posit, shuffle=False)
        Image_ADC, Label, Posit = tensor_shuffle(Image_ADC, Label, args.device, pos = Posit, shuffle=False)
        Image_DWI, Label, Posit = tensor_shuffle(Image_DWI, Label, args.device, pos = Posit, shuffle=False)

        with torch.no_grad():
            Tokens_T2  = getPatchTokens(MODELs['T2' ], Image_T2 , Posit, args).to(args.device)
            Tokens_ADC = getPatchTokens(MODELs['ADC'], Image_ADC, Posit, args).to(args.device)
            Tokens_DWI = getPatchTokens(MODELs['DWI'], Image_DWI, Posit, args).to(args.device)
            preds = MODEL_Fusion(Tokens_T2, Tokens_ADC, Tokens_DWI).cpu().detach()
            preds = torch.softmax(preds, dim=1).numpy()

        mask = sitk.ReadImage(GENERATORs['T2'].glandFileName[sample_idx])
        mask_np = sitk.GetArrayFromImage(mask)
        mask_np[mask_np>0] = 1
    
        label = sitk.ReadImage(GENERATORs['T2'].cancerFileName[sample_idx])
        label_np = sitk.GetArrayFromImage(label)
        #label_np[label_np>0] = 1
        if preds.shape[1] == 4:
            if clinically_significant:
                preds[:, 1] = preds[:, 1] + preds[:, 2]
                preds[:, 2] = preds[:, 3]
                preds = preds[:, :3]
                Label[:, 1] = Label[:, 1] + Label[:, 2]
                Label[:, 2] = Label[:, 3]
                Label = Label[:, :3]
            else:
                preds[:, 2] = preds[:, 2] + preds[:, 3]
                preds = preds[:, :3]
                Label[:, 2] = Label[:, 2] + Label[:, 3]
                Label = Label[:, :3]

        for Threshold in Thresholds:
            try:
                res = evaluate(case, mask, mask_np, label_np, preds, float(Threshold), percentagg, cancer_only)
                if res != None:
                    results[Threshold].append(res)
            except:
                pass
    return results

def Evaluate_ROC_Dice(MODEL, Generator, collate_fn, args, config, Thresholds=[0.5],
                      cancer_only = False, percentagg = 0.01, modal=False, pos=True,
                      clinically_significant=False): #
    MODEL.eval()
    results = {Threshold: [] for Threshold in Thresholds}
    with torch.no_grad():
        for sample_idx in tqdm(range(len(Generator))):
            case = os.path.basename(Generator.imageFileName[sample_idx]).split(f"_{config['Modality'].lower()}")[0]
            Posit, Modal = False, False
            if pos==False:
                image, mask_gland, mask_cancer = Generator.__getitem__(sample_idx)
                Image, Label = collate_fn([[image, mask_gland, mask_cancer]])
            elif modal == False:
                image, mask_gland, mask_cancer = Generator.__getitem__(sample_idx)
                Image, Label, Posit = collate_fn([[image, mask_gland, mask_cancer]])
            elif modal == True:
                image, mask_gland, mask_cancer, modality = Generator.__getitem__(sample_idx)
                Image, Label, Posit, Modal = collate_fn([[image, mask_gland, mask_cancer, modality]])
            
            preds = []
            for idx in range(0, len(Image), args.small_batchsize):
                if pos == False:
                    pred = MODEL(Image[idx:idx+args.small_batchsize].to(args.device)).cpu().detach()
                elif modal == False:
                    pred = MODEL(Image[idx:idx+args.small_batchsize].to(args.device),
                                Posit[idx:idx+args.small_batchsize].to(args.device).float()
                                ).cpu().detach()
                elif modal == True:
                    pred = MODEL(Image[idx:idx+args.small_batchsize].to(args.device),
                                Posit[idx:idx+args.small_batchsize].to(args.device).float(),
                                Modal[idx:idx+args.small_batchsize].to(args.device)
                                ).cpu().detach()
                pred = torch.softmax(pred, dim=1).numpy()
                preds.append(pred)
            pred = np.vstack(preds)
            
            mask = sitk.ReadImage(Generator.glandFileName[sample_idx])
            mask_np = sitk.GetArrayFromImage(mask)
            mask_np[mask_np>0] = 1
        
            label = sitk.ReadImage(Generator.cancerFileName[sample_idx])
            label_np = sitk.GetArrayFromImage(label)
            #label_np[label_np>0] = 1
            if pred.shape[1] == 4:
                if clinically_significant:
                    pred[:, 1] = pred[:, 1] + pred[:, 2]
                    pred[:, 2] = pred[:, 3]
                    pred = pred[:, :3]
                else:
                    pred[:, 2] = pred[:, 2] + pred[:, 3]
                    pred = pred[:, :3]

            for Threshold in Thresholds:
                try:
                    res = evaluate(case, mask, mask_np, label_np, pred, Threshold, percentagg, cancer_only)
                    if res != None:
                        results[Threshold].append(res)
                except:
                    pass
    return results


# def predFlip(MODEL, x, pos, target_flip, args):
#     x = torch.flip(x.to(args.device), target_flip)
#     pred = MODEL(x, pos.to(args.device).float()).cpu().detach()
#     target_flip_pred = [i for i in target_flip if i != 1]
#     pred = torch.flip(pred, target_flip_pred)
#     return pred

# def save_image(tensor, label = None, pred = None, filename="test1.png"):
#     np_img = minMax(tensor.permute(1,2,0).numpy())
#     plt.imshow( cv2.resize(np_img, (256, 256)), cmap="gray")
#     if label is not None:
#         y = label[1]
#         y[label[2] == 1] = 2
#         plt.imshow(y, alpha=0.3, vmin=0, vmax=2)
#     if pred is not None:
#         plt.imshow(pred.argmax(axis=0), alpha=0.3, vmin=0, vmax=2)
#     plt.savefig(filename)

# x = Image[20].unsqueeze(0)
# y = Label[20].unsqueeze(0)
# pos = Posit[20].unsqueeze(0)
# save_image( x[0], predFlip(MODEL, x, pos[20], [1], args).cpu(), filename="test_flip1.png" )
# save_image( x[0].cpu(), filename="test_noflip1.png" )

# torch.flip(x, [1, 2]).shape
# save_image( torch.flip(x, [1, 2])[0]

# target_flips = [[2], [3], [2,3], [1], [1,2], [1,3], [1,2,3]]
# for target_flip in target_flips:
#     pred = predFlip(MODEL, x, pos, target_flip, args)
#     pred = torch.softmax(pred, dim=1).numpy()
#     filename = '_'.join([str(i) for i in target_flip])
#     save_image( x[0], pred=pred[0], filename=f"test_flip_{filename}.png")
#     print(f"test_flip_{filename}.png")

# results = [ [] for i in range(12) ]
# with torch.no_grad():
#     for sample_idx in tqdm(range(len(Generator))):
#         case = os.path.basename(Generator.imageFileName[sample_idx]).split(f"_{config['Modality'].lower()}")[0]
#         image, mask_gland, mask_cancer = Generator.__getitem__(sample_idx)
#         Image, Label, Posit = collate_fn([[image, mask_gland, mask_cancer]])
        
#         pred_all = []
#         preds = []
#         for idx in range(0, len(Image), args.small_batchsize):
#             x = Image[idx:idx+args.small_batchsize].to(args.device)
#             pos = Posit[idx:idx+args.small_batchsize].to(args.device).float()
#             pred = MODEL(x, pos).cpu().detach()
#             pred = torch.softmax(pred, dim=1).numpy()
#             preds.append(pred)
#         pred = np.vstack(preds)
#         pred_all.append(pred)        
#         for target_flip in target_flips:
#             preds = []
#             for idx in range(0, len(Image), args.small_batchsize):
#                 x = Image[idx:idx+args.small_batchsize].to(args.device)
#                 pos = Posit[idx:idx+args.small_batchsize].to(args.device).float()
#                 pred = predFlip(MODEL, x, pos, target_flip, args)
#                 pred = torch.softmax(pred, dim=1).numpy()
#                 preds.append(pred)
#             pred = np.vstack(preds)
#             pred_all.append(pred)
        
#         mask = sitk.ReadImage(Generator.glandFileName[sample_idx])
#         mask_np = sitk.GetArrayFromImage(mask)
#         mask_np[mask_np>0] = 1
    
#         label_np = sitk.ReadImage(Generator.cancerFileName[sample_idx])
#         label_np = sitk.GetArrayFromImage(label_np)
#         #label_np[label_np>0] = 1


#         pred_final = [pred_all[0], pred_all[1], pred_all[2], pred_all[3], pred_all[4], pred_all[5], pred_all[6], pred_all[7],
#                      (pred_all[0] + pred_all[2]) / 2,
#                      (pred_all[0] + pred_all[2] + pred_all[4]) / 3,
#                      (pred_all[0] + pred_all[2] + pred_all[4] + pred_all[6] ) / 4,
#                      (pred_all[0] + pred_all[1] + pred_all[2] + pred_all[3] + pred_all[4] + pred_all[5] + pred_all[6] + pred_all[7]) / 7]
        
#         for i in range(len(pred_final)):
#             res = evaluate(case, mask, mask_np, label_np, pred_final[i], Threshold, percentagg, cancer_only)
#             results[i].append(res)
# [pd.DataFrame(results[i])['ROC AUC'].mean() for i in range( len(pred_final) )]



def agg_overlap_cs_lesion(label_np, agg_label_np, pred_agg_label):
    """
    finds overlap of predicted aggressive pixels and actual agg pixels in mixed cs lesions
    used to see how well the model differentiates agg and ind pixels within a mixed lesion
    """
    
    list_of_lesions = np.unique(label_np)[1:]
    num_lesions = len(list_of_lesions)
    all_agg_overlap = []
    # for each lesion (can be mixed with agg and ind components), find overlap between predicted aggressive and actual agg pixels
    for k in range(0, num_lesions):
        lesion = list_of_lesions[k]
        copy_lesion_label = np.copy(label_np)
        copy_lesion_label[label_np != lesion] = 0
        copy_lesion_label[copy_lesion_label > 0] = 1
        
        copy_agg_label = np.copy(agg_label_np)
        copy_agg_label[copy_lesion_label!=1] = 0 # everything other than this lesion not considered
        
        copy_pred_agg_label = np.copy(pred_agg_label)
        copy_pred_agg_label[copy_agg_label==0] = 0 # looks at agg pred within GT agg labels
        
        len_agg_labels = len(copy_agg_label[copy_agg_label>0])
        if len_agg_labels >0:
            len_predicted_agg_labels = len(copy_pred_agg_label[copy_pred_agg_label>0])
            agg_overlap = float(len_predicted_agg_labels/len_agg_labels)        # computes the overlap between predicted and agg GT pixels
            all_agg_overlap.append(agg_overlap)
    
    return all_agg_overlap

def ind_overlap_ind_lesion(label_np, ind_label_np, pred_ind_label):
    """
    finds overlap with predicted and actual indolent pixels 
    used to see how well the model differentiates agg and ind pixels within a mixed lesion
    """
    list_of_lesions = np.unique(label_np)[1:]
    num_lesions = len(list_of_lesions)
    all_ind_overlap = []
    # for each lesion, overlap between predicted indolent and actual indolent GT pixels
    for k in range(0, num_lesions):
        lesion = list_of_lesions[k]
        copy_lesion_label = np.copy(label_np)
        copy_lesion_label[label_np != lesion] = 0
        copy_lesion_label[copy_lesion_label > 0] = 1
        
        copy_ind_label = np.copy(ind_label_np)
        copy_ind_label[copy_lesion_label!=1] = 0
        
        copy_pred_ind_label = np.copy(pred_ind_label)
        copy_pred_ind_label[copy_ind_label==0] = 0 # looks at agg pred within GT agg labels
        
        len_ind_labels = len(copy_ind_label[copy_ind_label>0])
        len_predicted_ind_labels = len(copy_pred_ind_label[copy_pred_ind_label>0])
        ind_overlap = float(len_predicted_ind_labels/len_ind_labels)       # computes the overlap 
        all_ind_overlap.append(ind_overlap)
    
    return all_ind_overlap


def lesion_classifier(lesion_label_np, pred_label, pred_aggind_label, pred_prob, mask_np, cohort, spacing_im, percentagg, pred_type):
    """
    TP: true positive
    TN: true negative
    FP: false positive
    FN: false negative
    computes TP,TN, FP, FN and returns a list of true and predicted values, which will be used later for computing eval metrics
    
    ***inputs to function***
    -------------------------------------------------------------------------------------------------------------------------------------
    lesion_label_np: GT lesion label 
    pred_label: predicted label -- will have 0,1 values at each pixel, depending on analysis (cancer/clinically significant/agg), everything in positive class is 1
    pred_aggind_label: predicted agg/ind labels -- will have values 0, 1, 2 at each pixel, 0-- normal, 1-- ind, 2 -- agg
    pred_prob: predicted probability maps -- depending on analysis, choose cancer prob map or agg cancer prob map, etc
    mask_np: prostate gland segmentation GT
    cohort: RP/Bx 
    spacing_im: pixel spacing on MRI (0.29mm for media and labels paper)
    percentagg: %of agg pixels in a lesion for it to be considered clinically significant (cs)
    pred_type: cancer, cs or ind, depending on whether evaluating based on cancer vs. all, cs cancer vs. all, or ind vs all.
    --------------------------------------------------------------------------------------------------------------------------------------
    *** outputs from function****
    ---------------------------------------------------------------------------------------------------------------------------------------
    true, pred, predprob, percent_FP, len_correct_pred_grades, vol_sextant_FP, percent_agg_FP, FP_agg_vol
    true: vector of GT pos and GT neg, on a sextant-level, e.g. [1,0,0,0] if there is 1 cancer lesion, and 3 negative sextants
    pred: vector of predicted pos and neg, based on predicted label values, e.g. [1, 1, 0, 0] -- so lesion is detected, but also 1 FP sextant
    predprob: vector of predicted probabilities on the sextant level, e.g. [0.9, 0.7, 0.1, 0.03]
    percent_FP: if there is FP prediction in a sextant, what percentage of the sextant has the FP prediction
    len_correct_pred_grades: number of lesions correctly detected as cs or cancer or indolent, based on pred type
    vol_sextant_FP: if there are FP predictions, what are the volumes of those FP predictions?
    percent_agg_FP: if there are FP predictions, what percent is agg FP
    FP_agg_vol: what is the volume of FP agg predictions?
    
    -----------------------------------------------------------------------------------------------------------------------------------------
     
    """
    #print ("aggind label unique:", np.unique(pred_aggind_label))
   
    # find true positives and false negatives using findTPFN function. Uses GT lesions for deciding TP and FN
    num_lesions, len_TP, len_FN, len_correct_pred_grades, GT_pos, pred_pos, predprob_pos = findTPFN(mask_np, pred_label, pred_aggind_label, pred_prob, lesion_label_np,percentagg, pred_type)
    
    # find true negatives and false positives using findTNFP function. Uses sextant-based analysis for the prostate.
    GT_neg, pred_neg, predprob_neg, percent_FP, vol_sextant_FP, percent_agg_FP, FP_agg_vol = findTNFP(mask_np, pred_label, pred_aggind_label, pred_prob, lesion_label_np,spacing_im)

    # form a vector of true and predictions. 
    # true includes ground truth positives and ground truth negatives on a sextant-level, e.g., if there is 1 lesion, and 3 negative sextants, true = [1,0,0,0] 
    true = np.concatenate((GT_pos, GT_neg))
    
    # pred includes predicted positives and negatives on a sextant-level, e.g., if lesion is correctly detected, but ..
    #there is 1 more false positive out of the 3 negative sextants, pred = [1, 1, 0, 0]
    pred = np.concatenate((pred_pos, pred_neg))
    # preprob is a vector similar to pred, but with the probability values, instead of the predicted label values,..
    # e.g., for above e.g. pred, predprob = [0.9, 0.7, 0.1, 0.03]
    predprob = np.concatenate((predprob_pos, predprob_neg))
    
    return true, pred, predprob, percent_FP, len_correct_pred_grades, vol_sextant_FP, percent_agg_FP, FP_agg_vol

def findTPFN(mask_np, pred_np, pred_aggind_np, pred_prob, label_np, percentagg, pred_type):
    """
    find lesion-level TP and FN based on actual labels
    if 90th percentile of predictions is true, then lesion is detected
    -- this is in line with the Ron Summers paper (sumathipala paper) for true positives
    
    ***inputs to function***
    -------------------------------------------------------------------------------------------------------------------------------------
     
    mask_np: prostate gland segmentation GT
    pred_np: predicted label -- will have 0,1 values at each pixel, depending on analysis (cancer/clinically significant/agg), everything in positive class is 1
    pred_aggind_np: predicted agg/ind labels -- will have values 0, 1, 2 at each pixel, 0-- normal, 1-- ind, 2 -- agg
    pred_prob: predicted probability maps -- depending on analysis, choose cancer prob map or agg cancer prob map, etc
    label_np: GT lesion label
    percentagg: %of agg pixels in a lesion for it to be considered clinically significant (cs)
    pred_type: cancer, cs or ind, depending on whether evaluating based on cancer vs. all, cs cancer vs. all, or ind vs all.
    --------------------------------------------------------------------------------------------------------------------------------------
    *** outputs from function****
    ---------------------------------------------------------------------------------------------------------------------------------------
    
    num_lesions: number of lesions in the case
    detected_lesions: number of lesions detected by model
    missed_lesions: number of lesions missed by model
    correctly detected_grades: number of lesions correctly identified as agg/ind/cancer
    GT_pos: vector with number of lesions, e.g., if there are two lesions, GT_pos = [1,1]
    pred_pos: vector with pred labels, e.g., if model detects 1 lesion correctly out of two, pred_pos = [1,0]
    predprob_pos: vector with predicted probs, e.g., for above case, predprob_pos = [0.9, 0.02]
    
    -----------------------------------------------------------------------------------------------------------------------------------------
     
    
    """
    
    list_of_lesions = np.unique(label_np)[1:]
    num_lesions = len(list_of_lesions)
    indiv_lesion_np = np.zeros((num_lesions, np.shape(label_np)[0], np.shape(label_np)[1], np.shape(label_np)[2]),
                               dtype=int)

    # the 90th percentile probability is considered to see if lesion is detected or not
    pred_90percent = np.zeros((num_lesions), dtype=float)
    predprob_90percent = np.zeros((num_lesions), dtype=float)
    les_classifier = np.zeros((num_lesions), dtype=int)
    les_grade_classifier = np.zeros((num_lesions), dtype=int)
    for k in range(0, num_lesions):
        lesion = list_of_lesions[k]
        copy_lesion_label = np.copy(label_np)
        copy_lesion_label[label_np != lesion] = 0
        copy_lesion_label[copy_lesion_label > 0] = 1
        indiv_lesion_np[k, :, :, :] = copy_lesion_label
        
        # check if lesion is detected
        pred_les_pix = pred_np[indiv_lesion_np[k, :, :, :] > 0]
        pred_90percent[k] = np.percentile(pred_les_pix, 90)
        
        predprob_les_pix = pred_prob[indiv_lesion_np[k, :, :, :] > 0]
        predprob_90percent[k] = np.percentile(predprob_les_pix, 90)
        
        pred_positive_percent = len(pred_les_pix[pred_les_pix == 1])/len(pred_les_pix)
        #print ("pred_positive percent, and pred_90percent:", pred_positive_percent, pred_90percent)
        
        if pred_90percent[k] >= 1:
            les_classifier[k] = 1 #lesion detected
        
        if les_classifier[k] == 1: # if lesion is detected, check if it correctly predicts agg/indolent
            # check if lesion is detected correctly as per histologic grade 
            pred_les_aggind_pix = pred_aggind_np[indiv_lesion_np[k, :, :, :] > 0]
            #% of aggressive cancer of total predicted cancer
            pred_agg_precent = len(pred_les_aggind_pix[pred_les_aggind_pix == 2])/len(pred_les_pix) 
            # check to see if the detected lesion is also correctly classified as cancer/cs cancer/indolent   
            if pred_type == 'cs':
                 if pred_agg_precent>=percentagg:
                    les_grade_classifier[k] =1
            
            elif pred_type == 'ind':
                if pred_agg_precent<=percentagg:
                    les_grade_classifier[k] =1
                    
            elif pred_type == 'cancer': # dont care about agg/indolent, as long as it is detected
                les_grade_classifier[k] =1       
                
    detected_lesions = np.sum(les_classifier)  # true positive
    missed_lesions = num_lesions - detected_lesions  # false negative
    correctly_detected_grades = np.sum(les_grade_classifier)
    
    GT_pos = np.ones((num_lesions), dtype=float)
    pred_pos = pred_90percent
    predprob_pos = predprob_90percent
    return num_lesions, detected_lesions, missed_lesions, correctly_detected_grades, GT_pos, pred_pos, predprob_pos

def findTNFP(mask_np, pred_np, pred_aggind_np, pred_prob, label_np, spacing_im, percent_cancer=0.05):
    """
    function to compute TN and FP based on sextants
    sextant is TN is percent of cancer in that sextant <= percent_cancer
    ***inputs to function***
    -------------------------------------------------------------------------------------------------------------------------------------
     
    mask_np: prostate gland segmentation GT
    pred_np: predicted label -- will have 0,1 values at each pixel, depending on analysis (cancer/clinically significant/agg), everything in positive class is 1
    pred_aggind_np: predicted agg/ind labels -- will have values 0, 1, 2 at each pixel, 0-- normal, 1-- ind, 2 -- agg
    pred_prob: predicted probability maps -- depending on analysis, choose cancer prob map or agg cancer prob map, etc
    label_np: GT lesion label
    spacing_im: pixel spacing of input MRI
    percent_cancer: threshold percent of cancer GT pixels in a sextant for it to be considered GT neg.
    ----------------------------------------------------------------------------------------------------------------------------------------
    *** outputs from function****
    ---------------------------------------------------------------------------------------------------------------------------------------
    GT_neg, pred_neg, predprob_neg, percent_FP, FP_vol, percent_agg_FP, FP_agg_vol
    GT_neg: vector of zeros with length equal to number of negative sextants, e.g. GT_neg = [0,0,0,0]
    pred_neg: vector with pred labels, e.g., if model correctly predicts all except second sextant negative , pred_neg = [0,1,0,0]
    pred_prob_neg: vector with pred probabilities, e.g, predprob_neg = [0.01, 0.8, 0.1, 0.03]
    percent_FP: percent of FP cancer predictions in each sextant
    FP_vol: volume of FP predictions 
    percent_agg_FP: percentage of agg FP predictions in each sextant
    FP_agg_vol: volume of FP agg cancer predictions
    
    -----------------------------------------------------------------------------------------------------------------------------------------
    
    """
    
    GT_neg = []
    pred_neg = []
    predprob_neg = []
    percent_FP = []
    FP_vol = []
    percent_agg_FP = []
    FP_agg_vol = []
    
    # dividing the prostate into sextants 
    coor = np.nonzero(mask_np)
    prostate = np.unique(coor[0])
    prostate_regions = np.array_split(mask_np[prostate, :, :], 3, axis=0)
    label_regions = np.array_split(label_np[prostate, :, :], 3, axis=0)
    pred_regions = np.array_split(pred_np[prostate, :, :], 3, axis=0)
    pred_aggind_regions = np.array_split(pred_aggind_np[prostate, :, :], 3, axis=0)
    predprob_regions = np.array_split(pred_prob[prostate, :, :], 3, axis=0)

    for prostate_region, label_region, pred_region, pred_aggind_region, predprob_region in zip(prostate_regions, label_regions, pred_regions, pred_aggind_regions, predprob_regions):
        if prostate_region.size == 0:
            continue
        prostate_left_right = np.array_split(prostate_region, 2, axis=2)
        label_left_right = np.array_split(label_region, 2, axis=2)
        pred_left_right = np.array_split(pred_region, 2, axis=2)
        pred_aggind_left_right = np.array_split(pred_aggind_region, 2, axis=2)
        predprob_left_right = np.array_split(predprob_region, 2, axis=2)
        for prostate_half, label_half, pred_half, pred_aggind_half, predprob_half in zip(prostate_left_right, label_left_right, pred_left_right, pred_aggind_left_right, predprob_left_right):
            # checking if a sextant is GT neg based on percent of cancer GT pixels in that sextant, if below the percent_cancer threshold, then sextant is GT neg
            if (np.sum(np.logical_and(label_half > 0, prostate_half > 0)) / np.sum(prostate_half > 0)) <= percent_cancer: # percentage of cancer pixels
                GT_neg.append(0) 
                normal_tissue = np.logical_and(label_half == 0, prostate_half > 0)
                pred_neg.append(np.percentile(pred_half[normal_tissue], 90)) # 90th percentile of predicted labels in the sextant
                predprob_neg.append(np.percentile(predprob_half[normal_tissue], 90)) # 90th percentile of predicted probabilities in the sextant
                #print ("90th percentile", np.percentile(pred_half[normal_tissue], 90))
                #compute percent of FP
                if np.percentile(pred_half[normal_tissue], 90)>0:
                    #FP = len(pred_half[normal_tissue]>0)
                    #print ("len FP:", np.logical_and(pred_half > 0, normal_tissue == True))
                    #print (FP)
                    #print(len(normal_tissue[normal_tissue == True]))
                    # print ("unique values in pred_aggind_half:", np.unique(pred_aggind_half))
                    FP_sextant = np.logical_and(pred_half > 0, normal_tissue == True)
                    FP_agg_sextant = np.logical_and(pred_aggind_half >=2, normal_tissue == True)
                    FP = np.sum(FP_sextant)
                    FP_agg = np.sum(FP_agg_sextant)
                    # print ("false positive pixels", FP)
                    # print ("false positive agg pixels", FP_agg)
                    # print ("normal", np.sum(normal_tissue),len(normal_tissue[normal_tissue == True]) )
                    percent_FP.append(float(FP)/len(normal_tissue[normal_tissue == True]))
                    percent_agg_FP.append(float(FP_agg)/len(normal_tissue[normal_tissue == True]))
                    
                    #print (np.shape(pred_half))
                    """
                    plt.imshow(pred_half[1,:,:])
                    plt.show()
                    plt.imshow(normal_tissue[1,:,:])
                    plt.show()
                    """
                    #print ("spacing:", spacing_im)
                    FP_vol.append(FP*spacing_im[0]*spacing_im[1]*spacing_im[2])
                    FP_agg_vol.append(FP_agg*spacing_im[0]*spacing_im[1]*spacing_im[2])
                
                """
                if (np.sum(np.logical_and(pred_half > 0, prostate_half > 0)) / np.sum(prostate_half > 0)) <= percent_cancer: # percentage of cancer pixels
                    pred_neg.append(0)    
            
                """
    return GT_neg, pred_neg, predprob_neg, percent_FP, FP_vol, percent_agg_FP, FP_agg_vol


def findTNFP_bx(mask_np, pred_np, pred_prob, label_np):
    # for biopsy, we look at any % of cancer, not a threshold
    # this is because radiologist labels are often small, and for a single slice
    # so any sextant with any percentage of cancer is not considered GT neg
    # finds TN and FP for bx cohort
    
    #*********************IB comment: this function can be integrated with the findTPFN function above, empahsizing the difference that percent_cancer for BX cohort is o****
    """
    function to compute TN and FP based on sextants
    sextant is TN is percent of cancer in that sextant <= percent_cancer
    ***inputs to function***
    -------------------------------------------------------------------------------------------------------------------------------------
     
    mask_np: prostate gland segmentation GT
    pred_np: predicted label -- will have 0,1 values at each pixel, depending on analysis (cancer/clinically significant/agg), everything in positive class is 1
    pred_aggind_np: predicted agg/ind labels -- will have values 0, 1, 2 at each pixel, 0-- normal, 1-- ind, 2 -- agg
    pred_prob: predicted probability maps -- depending on analysis, choose cancer prob map or agg cancer prob map, etc
    label_np: GT lesion label
    ----------------------------------------------------------------------------------------------------------------------------------------
    *** outputs from function****
    ---------------------------------------------------------------------------------------------------------------------------------------
    
    GT_neg: vector of zeros with length equal to number of negative sextants, e.g. GT_neg = [0,0,0,0]
    pred_neg: vector with pred labels, e.g., if model correctly predicts all except second sextant negative , pred_neg = [0,1,0,0]
    pred_prob_neg: vector with pred probabilities, e.g, predprob_neg = [0.01, 0.8, 0.1, 0.03]
   
    -----------------------------------------------------------------------------------------------------------------------------------------
    
   
    """
    
    GT_neg = []
    pred_neg = []
    predprob_neg = []
    
    coor = np.nonzero(mask_np)
    prostate = np.unique(coor[0])
    prostate_regions = np.array_split(mask_np[prostate, :, :], 3, axis=0)
    label_regions = np.array_split(label_np[prostate, :, :], 3, axis=0)
    pred_regions = np.array_split(pred_np[prostate, :, :], 3, axis=0)
    predprob_regions = np.array_split(pred_prob[prostate, :, :], 3, axis=0)

    for prostate_region, label_region, pred_region, predprob_region in zip(prostate_regions, label_regions, pred_regions, predprob_regions):
        if prostate_region.size == 0:
            continue
        label_left_right = np.array_split(label_region, 2, axis=2)
        pred_left_right = np.array_split(pred_region, 2, axis=2)
        prostate_left_right = np.array_split(prostate_region, 2, axis=2)
        predprob_left_right = np.array_split(predprob_region, 2, axis=2)
        for prostate_half, label_half, pred_half, predprob_half in zip(prostate_left_right, label_left_right, pred_left_right, predprob_left_right):
            #print (np.sum(np.logical_and(label_half > 0, prostate_half > 0))/np.sum(prostate_half > 0))
              
            if np.sum(np.logical_and(label_half > 0, prostate_half > 0)) >0:
                GT_neg.append(0)
                normal_tissue = np.logical_and(label_half == 0, prostate_half > 0)
                pred_neg.append(np.percentile(pred_half[normal_tissue], 90))
                predprob_neg.append(np.percentile(predprob_half[normal_tissue], 90))
                """
                if (np.sum(np.logical_and(pred_half > 0, prostate_half > 0)) / np.sum(prostate_half > 0)) <= percent_cancer: # percentage of cancer pixels
                    pred_neg.append(0)    
            
                """
    return GT_neg, pred_neg, predprob_neg
    
def eval_per_patient(true, pred, pred_datatype, true_label_prostate, pred_label_prostate):
    
    """
    function to per-patient evaluation metrics
    ***inputs to function***
    -------------------------------------------------------------------------------------------------------------------------------------
    true, pred, pred_datatype, true_label_prostate, pred_label_prostate 
    true: vector of true values, e,g. for sextant-level eval, true = [1,0,0,0]
    pred: vector of pred values (can be probabilities or pred labels), if pred_probabilities are given, then pred = [1, 0,1, 0]
    pred_datatype: pred prob values for the prediction type needed, e.g. can be prob values of cancer/agg cancer/ind cancer. e.g., pred_datatype = [0.9, 0.1, 0.8, 0.02]
    true_label_prostate: label map within the prostate -- used for computing dice score
    pred_label_prostate: predicted label within the prostate -- used for computing dice score
    ------------------------------------------------------------------------------------------------------------------------------------------
    *** outputs from function****
    ---------------------------------------------------------------------------------------------------------------------------------------
    pat_roc_auc: per-patient ROC-AUC
    pat_pr_auc: per-patient precision recall AUC
    pat_sens: per-patient sensitivity
    pat_spec: per-patient specificity
    pat_prec: per-patient precision
    pat_dice: per-patient dice
    pat_npv: per-patient negative predictive values
    pat_F1score:per-patient F1 score
    pat_acc: per-patient accuracy
    total_positives: total number of GT positives
    total_negatives: totla number of GT negatives
   
    -----------------------------------------------------------------------------------------------------------------------------------------
    
    """
    
    
    # check to see if there are more than 1 GT class
    # important to check since per-patient metrics can only be computed if there are both pos and neg GT classes for each patient
    # 1 GT class can happen if (i) a patient has a very big lesion covering all sextants, hence no GT negative class, or ..
    # (ii) if all sextants are negative, i.e. no positive lesion 
    # no positive lesion can happen if the patient is (i)normal w/o cancer, or (ii) there is a lesion that gets thresholded with 250mm3 vol threshold, or (iii) for bx cohort, rad could not find lesion
    if len(np.unique(true)) > 1:
        #pat_auc = metrics.roc_auc_score(true, pred)
        pat_roc_auc = metrics.roc_auc_score(true, pred_datatype)
        precision, recall, _ = metrics.precision_recall_curve(true, pred_datatype)
        pat_pr_auc = metrics.auc(recall, precision)
        
    else:
        #print ("only one GT class")
        pat_roc_auc = None
        pat_pr_auc = None
    #true_positives = np.sum(true[pred >= thresh])
    true_positives = np.sum(true[pred == 1])
    total_positives = np.sum(true)
    if total_positives > 0:
        pat_sens = true_positives / total_positives
    else:
        pat_sens = None

    #true_negatives = np.sum(1 - true[pred < thresh])
    true_negatives = np.sum(1 - true[pred == 0])
    total_negatives = np.sum(1 - true)
    if total_negatives > 0:
        pat_spec = (true_negatives / total_negatives)
    else:
        pat_spec = None
    
    """
    total_predicted_positives = np.sum(pred >= thresh)
    total_predicted_negatives = np.sum(pred < thresh)
    """
    total_predicted_positives = len(pred[pred ==1])
    total_predicted_negatives = len(pred[pred ==0])
    if total_predicted_positives > 0:
        pat_prec = (true_positives / total_predicted_positives)
       
    else:
        pat_prec = None
        
    if total_predicted_negatives > 0 :    
        pat_npv = true_negatives/total_predicted_negatives
    else:
        pat_npv = None
        #print ("NPV None, pred neg =0")
    if pat_prec is not None and pat_sens is not None and (pat_prec + pat_sens)>0:
       pat_F1score = 2*((pat_prec * pat_sens)/(pat_prec + pat_sens))
    else:
         pat_F1score = None
    pat_acc = (true_positives + true_negatives) / (total_positives + total_negatives)
    
    # compute dice coefficient if a cancer label exists
    if true_label_prostate is not None:
        pat_dice = (1 - dice_dist(true_label_prostate, pred_label_prostate))
    else:
        pat_dice = None
    return pat_roc_auc, pat_pr_auc, pat_sens, pat_spec, pat_prec, pat_dice, pat_npv, pat_F1score, pat_acc, total_positives, total_negatives
  

    
#%%
def generate_lesions(ref_vol, label_np_source, volume_thresh, m_disks):
        """
        generating connected lesions from pixel-level annotations
        ***inputs to function***
        -------------------------------------------------------------------------------------------------------------------------------------
        ref_vol: reference volume to resample to, can be the prostate mask/t2 image with resampled pixel spacing, direction, etc
        label_np_source: the source label to process to generate lesions
        volume_threshold: volume threshold to discard small pixelated labels...
        (...refer to labels paper to see why discarding small lesions are needed for some labels: https://aapm.onlinelibrary.wiley.com/doi/full/10.1002/mp.15777)
        m_disks: the morphological processing disks
        
        ------------------------------------------------------------------------------------------------------------------------------------------
        *** outputs from function****
        ---------------------------------------------------------------------------------------------------------------------------------------
        lesions: generated 3D lesions
        num_lesions: number of generated lesions
        -----------------------------------------------------------------------------------------------------------------------------------------
        """
        # morphological processing, connected components, and thresholding
        # use for unprocessed labels
        label_np = np.copy(label_np_source)

        label_np[label_np != 0] = 1

        spacing = ref_vol.GetSpacing()

        margin = m_disks[0] / spacing[0]
        strel = disk(int(margin))

        margin = m_disks[1] / spacing[0]
        strel2 = disk(int(margin))
        strel2 = np.pad(strel2, int(m_disks[2]), 'constant')

        strel_total = np.stack([strel2, strel, strel2])

        connected = closing(label_np, strel_total)

        lesions, num_lesions = label(connected, return_num=True, connectivity=2)

        for lesion in range(num_lesions):
            # if generated lesion volume is less than volume threshold, discard that lesion
            if spacing[0] * spacing[1] * spacing[2] * np.sum(lesions == (lesion + 1)) < volume_thresh:
                lesions[lesions == (lesion + 1)] = 0

        lesions_mask = np.copy(lesions)
        lesions_mask[lesions_mask > 0] = 1
        
        # from skimage.measure import label
        lesions, num_lesions = label(lesions_mask, return_num=True, connectivity=2)

        return lesions, num_lesions
    
def generate_lesions_bx(label_np):  
     """
     for bx cohort, no need of morphological processing, just do connected component 
     """
     lesions, num_lesions = label(label_np, return_num=True, connectivity=2)
     return lesions, num_lesions

def threshold_lesions(ref_vol, label_np_source, volume_thresh):
        """
        if labels are already morphologically processed, and we just need to threshold based on volume, 
        use this function instead of generate_lesions
        """
        #connected components and thresholding
        # use for processed labels which already have undergone morphological processing
        
        label_np = np.copy(label_np_source)

        label_np[label_np != 0] = 1
        spacing = ref_vol.GetSpacing()
        lesions, num_lesions = label(label_np, return_num=True, connectivity=2)

        for lesion in range(num_lesions):
            if spacing[0] * spacing[1] * spacing[2] * np.sum(lesions == (lesion + 1)) < volume_thresh:
                lesions[lesions == (lesion + 1)] = 0

        lesions_mask = np.copy(lesions)
        lesions_mask[lesions_mask > 0] = 1
        lesions, num_lesions = label(lesions_mask, return_num=True, connectivity=2)

        return lesions, num_lesions
        



def check_if_cs(lesion_label_np, aggcancer_np, agg_thresh, num_lesions):
        
    """
    
    # check if the lesion is cs by checking the % agg component -- for RP cohort mixed lesions
    *** inputs to function ***
    -------------------------------------------------------------------------------------------------------------------------------------
    lesion_label_np: lesion labels
    aggcancer_np: agg cancer pixel-level labels (for digitial pathologist labels, we have pixel-level agg labels)
    agg_thresh: threshold of agg cancer pixels within a lesion for it to be considered clinically significant (cs)
    num_lesions: num of lesions for a case
    
    ------------------------------------------------------------------------------------------------------------------------------------------
    *** outputs from function****
    ---------------------------------------------------------------------------------------------------------------------------------------
    aggles:vector of 0 or 1, 1 indicating lesion is cs, 0 indicating it is not cs
    cs_lesions: labels of only cs lesions (if there are indolent lesions in label map, they are discarded)
    num_cs_lesions: number of cs lesions
    percentagg_lesions: percentage of agg cancer pixels in cs lesions
    -----------------------------------------------------------------------------------------------------------------------------------------
    
    """
    les_unique_vals = np.unique(lesion_label_np)[1:]
    cs_lesions_np = np.zeros_like(lesion_label_np)
     
    aggles = []
    percentagg_lesions = []
    # for each lesion, check the % agg cancer pixels, if greater than threshold, lesion is cs
    for les_num in range(0, num_lesions):
        print (les_num, les_unique_vals[les_num])
        les_unique_val = les_unique_vals[les_num]
        cancerpixels = len(lesion_label_np[lesion_label_np == les_unique_val])
        aggpixels = np.sum(aggcancer_np[lesion_label_np == les_unique_val])
        print (cancerpixels, aggpixels)
       
        if float(aggpixels)/cancerpixels >= agg_thresh:
            percentagg_in_les = float(aggpixels)/cancerpixels
            print ("agg percent:", float(aggpixels/cancerpixels))
            aggles.append(1)
            percentagg_lesions.append(percentagg_in_les)
            cs_lesions_np[lesion_label_np == les_unique_val] = les_unique_val
        else:
            aggles.append(0)
            
    cs_lesions, num_cs_lesions = label(cs_lesions_np, return_num=True, connectivity=2)
    
    return aggles, cs_lesions, num_cs_lesions, percentagg_lesions    



def check_if_cs_bx(lesion_label_np, num_lesions):
    # check if a lesion in the bx cohort is clinically significant (considers processed lesions with label values either 1 or 2)
    # different from RP cohort, since the labels are different
    # can be converted to a single function in combination with check_with_cs function above, which works only for RP cohort
    les_unique_vals = np.unique(lesion_label_np)[1:]
    cs_lesions_np = np.zeros_like(lesion_label_np)
     
    aggles = []
    for les_num in range(0, num_lesions):
        print (les_num, les_unique_vals[les_num])
        les_unique_val = les_unique_vals[les_num]
        cancerpixels = lesion_label_np[lesion_label_np == les_unique_val]
        
        if np.max(cancerpixels)>=2:
            
            aggles.append(1)
            cs_lesions_np[lesion_label_np == les_unique_val] = les_unique_val
        else:
            aggles.append(0)
            
    cs_lesions, num_cs_lesions = label(cs_lesions_np, return_num=True, connectivity=2)
    
    return aggles, cs_lesions, num_cs_lesions   


def check_if_ind_bx(lesion_label_np, num_lesions):
    # check if a lesion in the bx cohort is indolent
    les_unique_vals = np.unique(lesion_label_np)[1:]
    ind_lesions_np = np.zeros_like(lesion_label_np)
     
    indles = []
    for les_num in range(0, num_lesions):
        print (les_num, les_unique_vals[les_num])
        les_unique_val = les_unique_vals[les_num]
        cancerpixels = lesion_label_np[lesion_label_np == les_unique_val]
        
        if np.max(cancerpixels)==1: # indolent lesions in bx cohort marked as 1
            
            indles.append(1)
            ind_lesions_np[lesion_label_np == les_unique_val] = les_unique_val
        else:
            indles.append(0)
            
    ind_lesions, num_ind_lesions = label(ind_lesions_np, return_num=True, connectivity=2)
    
    return indles, ind_lesions, num_ind_lesions 

def compute_lesion_vols(ref_vol, lesion_label_np, num_lesions):
    """
    computes lesion volumes
    """
    les_unique_vals = np.unique(lesion_label_np)[1:]
    les_vol = []
    spacing = ref_vol.GetSpacing()
    for les_num in range(0, num_lesions):
    
        #print (les_num, les_unique_vals[les_num])
        les_vol.append(spacing[0] * spacing[1] * spacing[2] * np.sum(lesion_label_np == les_unique_vals[les_num]))
        
    return les_vol

    
def form_filteredlabels(all_label_path, ind_label_path, agg_label_path, case_id, mask_np, median_filter_labels):
    """
    **** this function was used to generate labels for the COrrSIgnIA paper, may be ignored for anything post labels paper***
    
    considers both DB and ck labels
    ck labels without DB labels are not considered during eval, but used as 50-50 ind-agg during training
    filteredlabels was what was used for the CorrSigNIA MEDIA paper, but was not used later for labels paper where morphological processing was used for DB labels
    """
    
    
    all_label_file = case_id + '_res_cancer_label.nii'
    ind_label_file = case_id + '_res_ind_cancer_label.nii'
    agg_label_file = case_id + '_res_agg_cancer_label.nii'
    
    if 'NP' not in case_id:
        if os.path.exists(os.path.join(all_label_path, all_label_file)):
            all_label = sitk.ReadImage(os.path.join(all_label_path, all_label_file))
            label_np = sitk.GetArrayFromImage(all_label).astype('float32')
            label_np[label_np > 0] = 3 # sets Dr. Kunder pixels to 3, will be overwritten by any grade labels
				
            if os.path.exists(os.path.join(ind_label_path, ind_label_file)):
                ind_label = sitk.ReadImage(os.path.join(ind_label_path, ind_label_file))
                ind_label_np = sitk.GetArrayFromImage(ind_label)
                label_np[ind_label_np > 0] = 1  # sets grade3 pixels to 1
            
            if os.path.exists(os.path.join(agg_label_path, agg_label_file)):
                agg_label = sitk.ReadImage(os.path.join(agg_label_path, agg_label_file))
                agg_label_np = sitk.GetArrayFromImage(agg_label)
                label_np[agg_label_np > 0] = 2 
                
            if median_filter_labels == True:
                filtered_label = np.zeros_like(label_np)     
                for slicenum in range(0, np.shape(label_np)[0]):
                    filtered_label[slicenum,:,:] = ndimage.median_filter(label_np[slicenum], size=(3,3))
                label_np = np.copy(filtered_label)
            return label_np
        else:
            return None
    elif 'NP' in case_id:
        label_np = np.zeros_like(mask_np).astype('float32')
        return label_np
#%%
def relevant_slices(case_id, label_np, mask_np, pred_label, agg_pred_prob, ind_pred_prob, norm_pred_prob, cancer_only = True):       
    """
    if cancer_only parameter is set True, selects only slices with cancer labels
    *** inputs to function ***
    -------------------------------------------------------------------------------------------------------------------------------------
    case_id : case name
    label_np: label
    mask_np: prostate segmentation
    pred_label: predicted label map
    agg_pred_prob: predicted aggressive probability map
    ind_pred_prob: predicted indolent probability map
    norm_pred_prob: predicted normal prob map
    cancer_only = True/False, if true, selects relevant slices with cancer labels, if false, selects all slices
    
    ------------------------------------------------------------------------------------------------------------------------------------------
    *** outputs from function****
    ---------------------------------------------------------------------------------------------------------------------------------------
    label_np, mask_np, pred_label, agg_pred_prob,ind_pred_prob, norm_pred_prob, cancer
    
    -----------------------------------------------------------------------------------------------------------------------------------------
    """
    
    if cancer_only and 'NP' not in case_id and label_np is not None:
        coor = np.nonzero(label_np)[0]
    else:
        coor = np.nonzero(mask_np)[0]
    cancer = np.unique(coor)
    label_np = label_np[cancer,:,:]		
    mask_np = mask_np[cancer, :, :]
    pred_label = pred_label[cancer,:,:]
    agg_pred_prob = agg_pred_prob[cancer,:,:]
    ind_pred_prob = ind_pred_prob[cancer,:,:]
    norm_pred_prob = norm_pred_prob[cancer,:,:]
    return label_np, mask_np, pred_label, agg_pred_prob,ind_pred_prob, norm_pred_prob, cancer
#%%

def relevant_slices_xyz(case_id, label_np, mask_np, pred_label, agg_pred_prob, ind_pred_prob, norm_pred_prob, cancer_only = True):       
    """
    if cancer_only parameter is set True, selects only slices with cancer labels
    *** inputs to function ***
    -------------------------------------------------------------------------------------------------------------------------------------
    case_id : case name
    label_np: label
    mask_np: prostate segmentation
    pred_label: predicted label map
    agg_pred_prob: predicted aggressive probability map
    ind_pred_prob: predicted indolent probability map
    norm_pred_prob: predicted normal prob map
    cancer_only = True/False, if true, selects relevant slices with cancer labels, if false, selects all slices
    
    ------------------------------------------------------------------------------------------------------------------------------------------
    *** outputs from function****
    ---------------------------------------------------------------------------------------------------------------------------------------
    label_np, mask_np, pred_label, agg_pred_prob,ind_pred_prob, norm_pred_prob, cancer
    
    -----------------------------------------------------------------------------------------------------------------------------------------
    """
    
    if cancer_only and 'NP' not in case_id and label_np is not None:
        coor = np.nonzero(label_np)[2]
    else:
        coor = np.nonzero(mask_np)[2]
    cancer = np.unique(coor)
    label_np = label_np[:,:, cancer]		
    mask_np = mask_np[:, :, cancer]
    pred_label = pred_label[:,:, cancer]
    agg_pred_prob = agg_pred_prob[:,:, cancer]
    ind_pred_prob = ind_pred_prob[:,:, cancer]
    norm_pred_prob = norm_pred_prob[:,:, cancer]
    return label_np, mask_np, pred_label, agg_pred_prob,ind_pred_prob, norm_pred_prob, cancer
#%%
def evaluate(case, mask, mask_np, label_np, pred, cancer_threshold, percentagg, cancer_only, model_name='ProViDNet', cancerOnly=True, volume_thresh=250): #
    pred_np   = pred[:,2]
    agg_pred  = pred[:,2]
    ind_pred  = pred[:,2]
    norm_pred = pred[:,0]
    im_spacing = mask.GetSpacing()
    
    # Filtered not radiologist annotated slices
    filtered_label_np, mask_np, pred_np, agg_pred, ind_pred, norm_pred, cancer = \
        relevant_slices(case, label_np, mask_np, pred_np, agg_pred, ind_pred, norm_pred, cancer_only)
    
    ind_pred[mask_np == 0] = 0
    agg_pred[mask_np == 0] = 0
    norm_pred[mask_np == 0] =0
    filtered_label_np[mask_np == 0] = 0
    agg_label_np = np.zeros_like(filtered_label_np)
    ind_label_np = np.zeros_like(filtered_label_np)
    cancer_label_np = np.zeros_like(filtered_label_np)
    
    agg_label_np[filtered_label_np == 1] = 1 ############### Because I don't care agg / ind
    ind_label_np[filtered_label_np == 1] = 1 # cancer_label_np = agg_label_np + ind_label_np
    cancer_label_np[filtered_label_np>0] = 1 # considers CK labels without DB labels also
    
    pred_np[mask_np == 0] = 0
    pred_agg_label = np.zeros_like(pred_np).astype('float')
    pred_ind_label = np.zeros_like(pred_np).astype('float')
    pred_cancer_label = np.zeros_like(pred_np).astype('float')
    pred_agg_label[pred_np == 1] = 1
    pred_ind_label[pred_np == 1] = 1
    #pred_cancer_label = pred_agg_label + pred_ind_label
    pred_cancer_label[pred_np > cancer_threshold] = 1
    pred_aggind_label = pred_cancer_label
    #print ("unique vals in pred_aggind_label:", np.unique(pred_aggind_label))
    
    cohort = 'MRI_bx'
    pred_type = 'cancer'
    
    label_np = cancer_label_np
    pred_label = pred_cancer_label
    pred_prob = agg_pred + ind_pred
    detect_cs = False
    # CUDA_VISIBLE_DEVICES=6 nohup nnUNetv2_train 500 2d 0 --npz > train_log_2d_fold0.out 2>&1 &
    detect_ind = False
    
    lesions, num_lesions = threshold_lesions(mask, label_np, volume_thresh)
    # lesions, num_lesions = generate_lesions_bx(label_np)
    lesions[mask_np==0] = 0
    label_np_copy = np.copy(lesions)
    
    lesion_vols = compute_lesion_vols(mask, label_np_copy, num_lesions)
    pat_true, pat_pred, pat_predprob, percent_FP, len_correct_pred_grades, vol_FP_sextant,  percent_agg_FP, FP_agg_vol = \
        lesion_classifier(label_np_copy, pred_label, pred_aggind_label, pred_prob, mask_np, cohort, im_spacing, percentagg, pred_type)

    true_prostate_label = label_np_copy[mask_np>0]
    pred_prostate_label = pred_label[mask_np>0]
    pat_roc_auc, pat_pr_auc, pat_sens, pat_spec, pat_prec, pat_dice, pat_npv, pat_F1score, pat_acc, pat_pos, pat_neg = eval_per_patient(pat_true, pat_pred, pat_predprob, true_prostate_label, pred_prostate_label)
    pat_stats = {'Model': model_name,'Patient id': case, 'ROC AUC': pat_roc_auc, 'PR AUC': pat_pr_auc,'Sensitivity':pat_sens,'Specificity':pat_spec,'Precision':pat_prec,'NPV':pat_npv, 'Dice':pat_dice,'Accuracy':pat_acc, 'Pred_type': pred_type, 'Lesion_vols': lesion_vols,  'les_pos':pat_pos, 'les_neg':pat_neg, 'les correct pred grades': len_correct_pred_grades}
    return pat_stats
#%%


