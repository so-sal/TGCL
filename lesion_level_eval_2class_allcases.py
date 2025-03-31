# -*- coding: utf-8 -*-
"""
Created on Tue Sep  20 19:55:07 2023

@author: Indrani + Sulaiman

lesion-level evaluation based on sextants code used for labels paper eval
evaluate indolent, aggressive and cancer
"""
import csv
import numpy as np
import pandas as pd
import argparse
from glob import glob
import os

import sys
from evaluation_support_functions import *
from tqdm import tqdm
# results_RP_CS
# results_RP_Reader
# results_RP_Reader_CS

def return_stats_lesion_extra(args):
    """
    ******************sextant-based lesion-level classification*************
    TP and FN are based on actual lesion labels
    TN and FP are based on sextants
    computes metrics on a per-patient level, and saves per patient eval metrics in a csv file
    """
    # Create the save folder if it doesn't exist
    os.makedirs(args.savefolder, exist_ok=True)
    # Load data
 
    pred_filenames = glob( os.path.join(args.pred_folder, "*" + args.pred_label_suffix + "*"))
    case_ids = [os.path.basename(i).split(args.pred_label_suffix)[0] for i in pred_filenames]
    tot_num_les = 0 # Calculate the total number of lesions (assuming you have a function for that)

    final_results = dict()
    thresholds = [i/100 for i in list(range(0, 100, 2))]
    for threshold in thresholds: final_results[threshold] = []

    print("save filename:", os.path.join(args.savefolder, file_savename)) # Print save folder and file name
    print("number of files:", len(case_ids))
    for idx, case in enumerate(case_ids):
        # Setting the path (prostate gland, cancer label, prediction probability, prediction label)
        prostate_path = os.path.join(args.gland_label_folder, case + args.prostate_name_suffix)
        label_path = os.path.join(args.cancer_label_folder, case + args.lesions_name_suffix)
        cancer_prob_path = os.path.join(args.pred_folder, case + args.pred_label_suffix)

        if not os.path.exists(prostate_path):
            print("no prostate label file:", prostate_path)
            continue
        elif not os.path.exists(label_path):
            print("no cancer label file:", label_path)
            continue
        elif not os.path.exists(cancer_prob_path):
            print("no cancer probability file:", cancer_prob_path)
            continue
        # elif not os.path.exists(cancer_pred_label_path):
        #     print("no cancer prediction:", cancer_pred_label_path)
        #     continue
        
        # Read the dataset: prostate gland (mask), cancer label (label_np), prediction probability (pred_prob), prediction label (pred_label)
        mask = sitk.ReadImage(prostate_path)
        mask_np_origin = np.array(sitk.GetArrayFromImage(mask).astype(np.uint8))
        mask_np_origin[mask_np_origin > 0] = 1
        label_np_origin = np.array(sitk.GetArrayFromImage(sitk.ReadImage(label_path))).astype(np.uint8)
        pred_prob_origin = sitk.GetArrayFromImage(sitk.ReadImage(cancer_prob_path))
        print(np.unique(label_np_origin))

        max_GG = np.unique(label_np_origin).max()
        if 'RP' in args.save_suffix:
            label_np_origin[ label_np_origin == 1] = 2
        
        #pred_np = np.array(sitk.GetArrayFromImage(sitk.ReadImage(cancer_pred_label_path))).astype(np.uint8)
        
        for threshold in tqdm(thresholds):
            pred_np = (pred_prob_origin > threshold).astype(np.float32)
            NP = len(np.unique(label_np_origin)) == 1 # NP means.. no prostate cancer?? maybe?
            if np.max(pred_np) >= 255.0: pred_np = pred_np / 255.0
            label_np, mask_np, pred_np, pred_prob, cancer = relevant_slices_2class(case, label_np_origin, mask_np_origin, pred_np, pred_prob_origin, args.cancer_only)
            label_np = np.where(label_np >= 2, 1, 0)

            label_np[mask_np == 0] = 0
            pred_np[mask_np == 0] = 0
            pred_prob[mask_np == 0] = 0
            
            pred_label = pred_np.copy()
                    
            lesions, num_lesions = threshold_lesions(mask, label_np, args.volume_thresh)
            lesions[mask_np == 0] = 0
            label_np_copy = np.copy(lesions)
                    
            tot_num_les += num_lesions
            lesion_vols = compute_lesion_vols(mask, label_np_copy, num_lesions)
            pat_true, pat_pred, pat_predprob, percent_FP, len_correct_pred_grades, vol_FP_sextant, percent_agg_FP, FP_agg_vol = \
                lesion_classifier(label_np_copy, pred_label, pred_label, pred_prob, mask_np, "Cohort", args.morph_disks, args.percentagg, args.pred_type)
            
            #print("This is the true and pred labels:", pat_true, pat_pred)
            true_prostate_label = label_np_copy[mask_np > 0]
            pred_prostate_label = pred_label[mask_np > 0]
            pat_roc_auc, pat_pr_auc, pat_sens, pat_spec, pat_prec, pat_dice, pat_npv, pat_F1score, pat_acc, pat_pos, pat_neg = eval_per_patient(pat_true, pat_pred, pat_predprob, true_prostate_label, pred_prostate_label)
            Prostate_vol = mask_np.sum() * np.prod( args.morph_disks )

            if NP:
                pat_stats = {'Model': args.model_name, 'Patient id': case,
                            'ROC AUC': None, 'PR AUC': None,
                            'Sensitivity': None, 'Specificity': pat_spec,
                            'Precision': None, 'NPV': pat_npv,
                            'Dice': None, 'Accuracy': pat_acc,
                            'pred_type': args.pred_type,
                            'Lesion_vols': lesion_vols, 'les_pos': pat_pos, 'les_neg': pat_neg,
                            'les correct pred grades': len_correct_pred_grades,
                            'pat_true': pat_true, 'pat_predprob': pat_predprob, 'max_GG': max_GG, 'Prostate_vol': Prostate_vol}
            else:
                pat_stats = {'Model': args.model_name, 'Patient id': case,
                            'ROC AUC': pat_roc_auc, 'PR AUC': pat_pr_auc,
                            'Sensitivity': pat_sens, 'Specificity': pat_spec,
                            'Precision': pat_prec, 'NPV': pat_npv,
                            'Dice': pat_dice, 'Accuracy': pat_acc,
                            'pred_type': args.pred_type,
                            'Lesion_vols': lesion_vols, 'les_pos': pat_pos, 'les_neg': pat_neg,
                            'les correct pred grades': len_correct_pred_grades,
                            'pat_true': pat_true, 'pat_predprob': pat_predprob, 'max_GG': max_GG, 'Prostate_vol': Prostate_vol}
            print(pat_stats)
            final_results[threshold].append(pat_stats)
    
    file_savename = "EvaluationResults/" + args.model_name + "_" + args.save_suffix + ".csv"
    pd.DataFrame(final_results).to_csv(file_savename, index=False)

    res = pd.DataFrame(final_results)
    res['ROC AUC'].mean().round(3)
    res['PR AUC'].mean().round(3)
    res['Sensitivity'].mean().round(3)
    res['Specificity'].mean().round(3)
    res['Precision'].mean().round(3)
    res['NPV'].mean().round(3)
    res['Dice'].loc[res['les_pos']>0].mean().round(3)
    res['Accuracy'].mean().round(3)

if __name__ == '__main__':
    # Create an ArgumentParser object
    parser = argparse.ArgumentParser(description='Your program description here.')
    # Define command-line arguments based on the YAML configuration with default values
    parser.add_argument('--model_name', type=str, default='ProViCNet', help='Model name')
    parser.add_argument('--pred_type', type=str, default='cancer', help='Prediction type')
    parser.add_argument('--morph_disks', type=float, nargs='+', default=[0.5, 0.5, 0.5], help='Morphological disks')
    parser.add_argument('--volume_thresh', type=int, default=250, help='Volume threshold')
    parser.add_argument('--cancer_only', type=bool, default=False, help='Cancer only')
    
    parser.add_argument('--percentagg', type=float, default=0.10, help='Percentage aggregation')    
    parser.add_argument('--pred_folder', type=str, default='/home/sosal/student_projects/JeongHoonLee/RAUS/results/Validation_T2/', help='Main prediction folder')
    parser.add_argument('--gland_label_folder', type=str, default='/home/sosal/Data/TRUS/TRUS_Prostate_Label/', help='Main prediction folder')
    parser.add_argument('--cancer_label_folder', type=str, default='/home/sosal/Data/TRUS/TRUS_ROI_Bxconfirmed_Label/', help='Main prediction folder')

    parser.add_argument('--savefolder', type=str, default='./results/evaluation/', help='Save folder')
    parser.add_argument('--save_suffix', type=str, default='val_ADC', help='Save suffix')
    parser.add_argument('--pred_label_suffix', type=str, default='_pred.nii.gz', help='Prediction label suffix')

    parser.add_argument('--clinically_significant', type=bool, default=True, help='Clinically significant')
    parser.add_argument('--lesions_name_suffix', type=str, default='_trus_roi_bxconfirmed_label.nii.gz', help='Lesions name suffix')
    parser.add_argument('--prostate_name_suffix', type=str, default='_trus_prostate_label.nii.gz', help='Lesions name suffix')

    args = parser.parse_args()
    return_stats_lesion_extra(args)

