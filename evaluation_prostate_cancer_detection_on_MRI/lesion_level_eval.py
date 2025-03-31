# -*- coding: utf-8 -*-
"""
Created on Tue Oct  5 19:55:07 2021

@author: Indrani

lesion-level evaluation based on sextants
code used for labels paper eval
evaluate indolent, aggressive and cancer 


"""
import numpy as np
import SimpleITK as sitk
import os,fnmatch
from sklearn import metrics
from scipy import interp
import matplotlib.pyplot as plt
import pandas as pd
from skimage.measure import label
from scipy.spatial.distance import dice as dice_dist
import csv
import argparse
import yaml
from scipy.stats import iqr
from skimage.morphology import dilation, disk, erosion, closing
from scipy import ndimage
import math
from sklearn.metrics import average_precision_score
from natsort import natsorted
from evaluation_support_functions import *

  

def return_stats_lesion_extra(args):
    """
    ******************sextant-based lesion-level classification*************
    TP and FN are based on actual lesion labels
    TN and FP are based on sextants 
    computes metrics on a per-patient level, and saves per patient eval metrics in a csv file
    """
    with open(args.config, 'r') as stream:
        config = yaml.safe_load(stream)
    
    total_pred = []
    all_F1score = []
    all_roc_auc = []
    all_pr_auc = []
    all_sens = []
    all_spec = []
    all_prec = []
    all_npv = []
    all_dice = []
    all_acc = []
    all_pos = []
    all_neg = []
    all_aggoverlap = []
    all_lesvols = []
    total_true = []
    all_percentaggles = []
    all_percent_FP = []
    all_vol_FP_sextant = []
    all_percent_agg_FP = []
    all_FP_agg_vol = []
    all_les_pred_grades = []
    
    model_name = config['model_name']
    pred_type = config['pred_type']
    predf_type = config['predf_type']
    volume_thresh = config['volume_thresh']
    cohort = config['cohort']
    morph_disks = config['morph_disks']
    cancer_only = config['cancer_only']
    atleast_onelesion = config['atleast_onelesion']
    percentagg = config['percentagg']
    processed_labels = config['processed_labels']
    
    mainpredfolder = config['read_preds']['mainpredfolder']
    pred_probfolder = mainpredfolder
    pred_labelfolder = pred_probfolder
    file_savename = cohort + '_' + model_name + '_lesion_' + pred_type + config['save_results']['save_suffix']+'.csv'
    main_dir = config['read_paths']['main_dir']
    prostate_path = config['read_paths']['prostate_mask']
    filtered_labelpath = config['read_paths']['filtered_labelpath']
    
    savefolder = os.path.join(config['save_results']['savefolder'],model_name)
    savefolder = "./"
    if not os.path.exists(savefolder):
        os.makedirs(savefolder)
    
    case_ids = [case.split('_')[0] for case in os.listdir(prostate_path)]
    if cohort =='Bx':
        exclude_cases = config['exclude_cases']
        bx_exclues = []
        for (k,v) in exclude_cases.items():
            bx_exclues += v
        case_ids = [x for x in case_ids if x not in bx_exclues]
        case_ids = natsorted(case_ids)
    # IB -- can be modularized and paths can be provided as input via json/commandline during running script
      
    tot_num_les = 0
    print (savefolder, file_savename)
    
    with open(savefolder + file_savename, mode='w', newline='') as csv_file:
    #with open(os.path.join(savefolder, file_savename), mode='w', newline='') as csv_file:

        fieldnames = ['Model','Patient id', 'ROC AUC','PR AUC', 'Sensitivity', 'Specificity', 'Precision', 'NPV', 'Dice', 'Accuracy', 'Pred_type', 'Lesion_vols', 'les_pos', 'les_neg', 'les correct pred grades' ]
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        
        
        writer.writeheader()
       

        # for miccai rebuttal 
        print (len(case_ids))
        for case in case_ids:
            # reads prostate mask
            print (case)
           
            if os.path.exists(pred_labelfolder + case +config['read_preds']['pred_label_suffix']) == True:
                # reads the predictions
                pred_label = sitk.ReadImage(pred_labelfolder + case +config['read_preds']['pred_label_suffix']) # 224, 224, 6
                pred_np = sitk.GetArrayFromImage(pred_label) # 6, 224, 224
                agg_pred = sitk.GetArrayFromImage(sitk.ReadImage(pred_probfolder + case + config['read_preds']['pred_agg_suffix']))
                ind_pred = sitk.GetArrayFromImage(sitk.ReadImage(pred_probfolder + case + config['read_preds']['pred_ind_suffix']))
                norm_pred = sitk.GetArrayFromImage(sitk.ReadImage(pred_probfolder + case + config['read_preds']['pred_norm_suffix']))

            else:
                pred_np = None
            if pred_np is not None:   
                if cohort == 'RP':
                    #if 'NP' not in case and os.path.exists(os.path.join(agg_label_path, case + '_res_agg_cancer_label.nii')) and os.path.exists(os.path.join(all_label_path, case + '_res_cancer_label.nii')):
                    if 'NP' not in case and os.path.exists(os.path.join(filtered_labelpath, case + config['read_paths']['lesions_name_suffix'])):
                        
                        print (case)
                        mask = sitk.ReadImage(prostate_path + case + config['read_paths']['prostate_mask_suffix'])
                        mask_np = sitk.GetArrayFromImage(mask)
                        mask_np[mask_np>0] = 1
                        im_spacing = mask.GetSpacing()
                        filtered_label_np = sitk.GetArrayFromImage(sitk.ReadImage(filtered_labelpath + case + config['read_paths']['lesions_name_suffix']))
                    else:
                        filtered_label_np = None
                
                elif cohort == 'Bx':
                    # mask = sitk.ReadImage(prostate_path + case + '_res_mask_label.nii')
                    mask_file = fnmatch.filter(os.listdir(prostate_path), case+'_')[0]
                    mask = sitk.ReadImage(mask_file)
                    im_spacing = mask.GetSpacing()
                    mask_np = sitk.GetArrayFromImage(mask)
                    mask_np[mask_np>0] = 1
                    if os.path.exists(filtered_labelpath + case + config['read_paths']['lesions_name_suffix']):
                        filtered_label_np = sitk.GetArrayFromImage(sitk.ReadImage(filtered_labelpath + case + config['read_paths']['lesions_name_suffix']))
                    else:
                        filtered_label_np = None
                        print ("none label")
                    
            
                if filtered_label_np is not None and 'NP' not in case and np.max(np.unique(filtered_label_np))>0:  
                    print ("case, mask size, filtered_label_size:", case, np.shape(mask_np), np.shape(filtered_label_np), np.unique(filtered_label_np))    
                
                    if np.max(agg_pred)>1:
                        agg_pred = agg_pred/255.0
                        ind_pred = ind_pred/255.0
                        norm_pred = norm_pred/255.0
                    if np.max(pred_np)>=255.0:
                        pred_np = pred_np/255.0
                        
                    
                    # consider cancer only slices, or all slices based on cancer_only parameter
                   
                    filtered_label_np, mask_np, pred_np, agg_pred, ind_pred, norm_pred, cancer = relevant_slices(case, mask_np, mask_np, pred_np, agg_pred, ind_pred, norm_pred, cancer_only)
                    
                    ind_pred[mask_np == 0] = 0
                    agg_pred[mask_np == 0] = 0 
                    norm_pred[mask_np == 0] =0
                    
                    filtered_label_np[mask_np == 0] = 0
                    agg_label_np = np.zeros_like(filtered_label_np)
                    ind_label_np = np.zeros_like(filtered_label_np)
                    cancer_label_np = np.zeros_like(filtered_label_np)
                    
                    agg_label_np[filtered_label_np == 2] = 1                      
                    ind_label_np[filtered_label_np == 1] = 1
                    #cancer_label_np = agg_label_np + ind_label_np
                    cancer_label_np[filtered_label_np>0] = 1 # considers CK labels without DB labels also
                    
                    pred_np[mask_np == 0] = 0
                    pred_agg_label = np.zeros_like(pred_np)
                    pred_ind_label = np.zeros_like(pred_np)
                    pred_cancer_label = np.zeros_like(pred_np)
                    pred_agg_label[pred_np == 2] = 1
                    pred_ind_label[pred_np == 1] = 1
                    #pred_cancer_label = pred_agg_label + pred_ind_label
                    pred_cancer_label[pred_np>0] = 1
                    pred_aggind_label = np.copy(pred_np)
                    print ("unique vals in pred_aggind_label:", np.unique(pred_aggind_label))
                    #pred_cancer_label[pred_cancer_label>0] = 1
                    
                    
                    if pred_type == 'agg':
                        label_np = agg_label_np
                        pred_label = pred_agg_label
                        pred_prob = agg_pred
                        detect_cs = False
                        detect_ind = False
                    elif pred_type == 'cancer':
                        label_np = cancer_label_np
                        pred_label = pred_cancer_label
                        pred_prob = agg_pred + ind_pred
                        detect_cs = False
                        detect_ind = False
                    elif pred_type == 'cs': # detect clinically significant cancer
                        label_np = cancer_label_np
                        pred_label = pred_cancer_label
                        pred_prob = agg_pred + ind_pred
                        detect_cs = True
                        detect_ind = False
                    elif pred_type == 'ind':
                        label_np = ind_label_np
                        pred_label = pred_ind_label
                        pred_prob = ind_pred
                        detect_cs = False
                        detect_ind = True
                    
                    # generate lesions from pixel level labels
                    if cohort == 'RP' and processed_labels == True: # if lesions are already generated from labels
                        lesions, num_lesions = threshold_lesions(mask, label_np, volume_thresh) #for labels already gone through morph process
                    elif cohort == 'RP' and processed_labels == False:# unprocessed pixel-level labels
                        lesions, num_lesions = generate_lesions(mask, label_np, volume_thresh, morph_disks)
    
                    elif cohort == 'Bx':
                        lesions, num_lesions = generate_lesions_bx(label_np)
                    
                    lesions[mask_np==0] = 0
            
                    label_np_copy = np.copy(lesions)
                    # if clinically significant cancers  
                    if detect_cs == True:
                        if cohort == 'RP':
                            aggind, cs_lesions, num_cs_lesions, percentaggles = check_if_cs(label_np_copy, agg_label_np, percentagg, num_lesions)
                            label_np_copy = np.copy(cs_lesions)
                            num_lesions = num_cs_lesions
                        elif cohort == 'Bx':
                            aggind, cs_lesions, num_cs_lesions = check_if_cs_bx(label_np_copy, num_lesions)
                            label_np_copy = np.copy(cs_lesions)
                            num_lesions = num_cs_lesions 
                    """
                    if detect_ind == True:
                        if cohort == 'Bx':
                            aggind, ind_lesions, num_ind_lesions = check_if_ind_bx(label_np_copy, num_lesions)
                            label_np_copy = np.copy(ind_lesions)
                            num_lesions = num_ind_lesions 
                    """       
                            
                    tot_num_les = tot_num_les+num_lesions
                    lesion_vols = compute_lesion_vols(mask, label_np_copy, num_lesions)
                    pat_true, pat_pred, pat_predprob, percent_FP, len_correct_pred_grades, vol_FP_sextant,  percent_agg_FP, FP_agg_vol = lesion_classifier(label_np_copy, pred_label, pred_aggind_label, pred_prob, mask_np, cohort, im_spacing, percentagg, pred_type)
                    #try:
                    #----- try catch block--"
                    if len(np.unique(pat_true))>1: 
                     
                        true_prostate_label = label_np_copy[mask_np>0]
                        pred_prostate_label = pred_label[mask_np>0]
                        pat_roc_auc, pat_pr_auc, pat_sens, pat_spec, pat_prec, pat_dice, pat_npv, pat_F1score, pat_acc, pat_pos, pat_neg = eval_per_patient(pat_true, pat_pred, pat_predprob, true_prostate_label, pred_prostate_label)
                        if detect_cs == True:
                            agg_overlap = agg_overlap_cs_lesion(label_np_copy, agg_label_np, pred_agg_label)
                        else:
                            agg_overlap = 0
                            
                        if detect_ind == True:
                            ind_overlap = ind_overlap_ind_lesion(label_np_copy, ind_label_np, pred_agg_label)
                        else:
                            ind_overlap = 0
                            
                        
                        pat_stats = {'Model': model_name,'Patient id': case, 'ROC AUC': pat_roc_auc, 'PR AUC': pat_pr_auc,'Sensitivity':pat_sens,'Specificity':pat_spec,'Precision':pat_prec,'NPV':pat_npv, 'Dice':pat_dice,'Accuracy':pat_acc, 'Pred_type': pred_type, 'Lesion_vols': lesion_vols,  'les_pos':pat_pos, 'les_neg':pat_neg, 'les correct pred grades': len_correct_pred_grades}

                        print (pat_stats)
                        
                        writer.writerow(pat_stats)
                        
                        all_roc_auc.append(pat_roc_auc)
                        all_pr_auc.append(pat_pr_auc)
                        all_sens.append(pat_sens)
                        all_spec.append(pat_spec)
                        all_prec.append(pat_prec)
                        all_npv.append(pat_npv)
                        all_dice.append(pat_dice)
                        all_acc.append(pat_acc)
                        all_pos.append(pat_pos)
                        all_neg.append(pat_neg)
                      
                        all_lesvols.append(lesion_vols)
                        all_les_pred_grades.append(len_correct_pred_grades)
                        #if detect_cs == True:
                        
                        total_true.append(pat_true)
                        total_pred.append(pat_pred)
                    else:
                        print ("only 1 GT class, could not process:", case)
                
                    #except:
                    #    print (case, "something went wrong!")
                    #     continue
                else:
                    print ("no GT lesions")
            else:
                print ("no predictions")
            #mean stats    
        #print ("all auc:", all_auc)
        mod_npv = [0 if x == None else x for x in all_npv]
        mod_prec = [0 if x == None else x for x in all_prec]
        all_lesvols = np.concatenate(all_lesvols)
        mod_lesvols = [0 if x==None else x for x in all_lesvols]
        print ("all_percent_FP:", all_percent_FP)
         
      
        mean_stats = {'Model': model_name,'Patient id': 'Mean_cases', 'ROC AUC': np.mean(all_roc_auc), 'PR AUC':np.mean(all_pr_auc),'Sensitivity':np.mean(all_sens),'Specificity':np.mean(all_spec),'Dice':np.mean(all_dice),'Accuracy':np.mean(all_acc), 'Pred_type': pred_type, 'Lesion_vols': np.mean(mod_lesvols),'les_pos':np.mean(all_pos), 'les_neg':np.mean(all_neg)}
        mean_stats['NPV']= np.mean(mod_npv)
        mean_stats['Precision'] = np.mean(mod_prec)
        mean_stats['les correct pred grades'] = np.mean(all_les_pred_grades)
        print (mean_stats)
        
        writer.writerow(mean_stats)
        #std stats
        std_stats = {'Model': model_name,'Patient id': 'std_cases', 'ROC AUC': np.std(all_roc_auc), 'PR AUC':np.std(all_pr_auc), 'Sensitivity':np.std(all_sens),'Specificity':np.std(all_spec),'Dice':np.std(all_dice),'Accuracy':np.std(all_acc),'Pred_type': pred_type, 'Lesion_vols': np.std(mod_lesvols), 'les_pos':np.std(all_pos), 'les_neg':np.std(all_neg)}
        std_stats['NPV']= np.std(mod_npv)
        std_stats['Precision'] = np.std(mod_prec)
        std_stats['les correct pred grades'] = np.mean(all_les_pred_grades)

        print (std_stats)
        writer.writerow(std_stats)
        
        median_stats = {'Model': model_name,'Patient id': 'median_cases', 'ROC AUC': np.median(all_roc_auc), 'PR AUC':np.median(all_pr_auc),'Sensitivity':np.median(all_sens),'Specificity':np.median(all_spec),'Dice':np.median(all_dice),'Accuracy':np.median(all_acc),'Pred_type': pred_type, 'Lesion_vols': np.median(mod_lesvols), 'les_pos':np.median(all_pos), 'les_neg':np.median(all_neg)}
        median_stats['NPV']= np.median(mod_npv)
        median_stats['Precision'] = np.median(mod_prec)
        median_stats['les correct pred grades'] = np.mean(all_les_pred_grades)
        print (median_stats)
        print ("total number of lesions", tot_num_les)
              
        writer.writerow(median_stats)

        sum_stats = {'Model': model_name,'Patient id': 'sum cases', 'Pred_type': pred_type, 'les_pos':np.sum(all_pos), 'les_neg':np.sum(all_neg)}
        sum_stats['les correct pred grades'] = np.sum(all_les_pred_grades)

        print (sum_stats)
        writer.writerow(sum_stats)

        

    return mean_stats, std_stats, median_stats, tot_num_les, sum_stats

# #### running lesion-level eval on RP cohort -- example

# # --- IB comment -- some issue with bx eval, need to double check..
# model_name = 'SPCNet'
# #------------------inputs to the return_stats lesion folder -----------------
# cohort = 'RP'
# #mainpredfolder --> # path where model predictions are saved
# if cohort == 'RP':
#     mainpredfolder = 'V:/GE_project/Labels_paper_predictions/SPCNet/L4_trained/RP_preds/'
# elif cohort == 'Bx':
#     mainpredfolder = 'V:/GE_project/Labels_paper_predictions/SPCNet/L4_trained/Bx_preds/'
# # label_path --> path where labels are stored
# # specify label path, for rp,providing dig path labels, i.e. L4 labels from labels paper
# #main_dir --> the main dir with the test cases, needs to have the prostate masks for each case

# if cohort == 'RP':
#     main_dir = "V:/GE_project/Data/test_with_complete_l2to4/"
#     label_path = 'Z:/RadPathFusion/Indrani_RadPath/Projected_masked/Size224_unmasked/smoothed_sigma0.25_v2/flexpad/all_grades_data_01172021/morph_DB_labels_250/'
#     exclude_cases = []
# elif cohort == 'Bx':
#     main_dir = 'Y:/Stanford_Prostate/Radiology/StanfordBxResampled/'
#     label_path = main_dir + 'combined_labels/' 
#     # exclude any bx case that overlaps with RP cases and with only pirads 1/2 lesions, look at the labels_paper eval script
#     RP_overlap = ['Bx_39', 'Bx_78', 'Bx_272', 'Bx_72', 'Bx_94', 'Bx_83', 'Bx_103', 'Bx_97', 'Bx_60', 'Bx_476', 'Bx_380', 'Bx_440', 'Bx_65', 'Bx_307', 'Bx_287', 'Bx_305', 'Bx_444', 'Bx_11', 'Bx_529', 'Bx_20', 'Bx_483', 'Bx_294', 'Bx_291', 'Bx_311', 'Bx_282', 'Bx_292', 'Bx_395', 'Bx_360', 'Bx_365', 'Bx_334', 'Bx_194', 'Bx_192', 'Bx_279', 'Bx_194', 'Bx_292', 'Bx_507', 'Bx_96', 'Bx_104', 'Bx_206', 'Bx_366', 'Bx_399', 'Bx_368', 'Bx_327', 'Bx_514', 'Bx_447', 'Bx_13', 'Bx_371', 'Bx_39', 'Bx_340', 'Bx_386', 'Bx_429', 'Bx_401', 'Bx_336', 'Bx_486', 'Bx_387', 'Bx_296', 'Bx_129']
#     pirads12 = ['Bx_188', 'Bx_196', 'Bx_304', 'Bx_404', 'Bx_109', 'Bx_228', 'Bx_232','Bx_238','Bx_243', 'Bx_275', 'Bx_329', 'Bx_335', 'Bx_347','Bx_398', 'Bx_410', 'Bx_415', 'Bx_421', 'Bx_461']
    		
#     Bx_exclude = ['Bx_33', 'Bx_41','Bx_47', 'Bx_53', 'Bx_96', 'Bx_172', 'Bx_200', 'Bx_217', 'Bx_220', 'Bx_229', 'Bx_237', 'Bx_251', 'Bx_313', 'Bx_318', 'Bx_337', 'Bx_362', 'Bx_390', 'Bx_437', 'Bx_183', 'Bx_237', 'Bx_333', 'Bx_362', 'Bx_487']
#     exclude_cases = RP_overlap + Bx_exclude + pirads12
    
# #savefolder --> path where quantitative evaluations will be stored as csv files
# savemain_folder = "V:/GE_project/test_lesion_eval/" + model_name + "/"
# #savefolder will be a subfolder in this folder based on pred_type, i.e., cancer vs all or cs vs all
# #pred_type --> whether eval is cancer vs all or cs cancer vs all
# #pred_types = ['cancer', 'cs'] # evaluate for cancer vs. all and clinically significant cancer vs. all
# pred_types = ['cs']
# #filesavename --> name to save the quantitative eval metrics
# #vt --> volume threshold
# vt = 250
# #morph_disk --> morph_disks to form generate lesions from labels
# morph_disk = [1.5, 0.5, 4]                
# atleast_onelesion = False  # if we want to make sure each case in RP cohort has 1 lesion, even if volume of all lesions<250mm3, make this parameter True
# # cancer_only --> do we evaluate with cancer_only slices or all slices?
# cancer_only = True    
# if cancer_only == True:
#     suffix = '_co'
# else:
#     suffix = '_allslices'
# # processed_labels --> True if using labels already morphologically processed to be independent lesions, e.g. L3 and L4 labels in labels paper, --
# # False if using pixel-level labels and need to morphologically process them 
# processed_labels = True
# #------------------------------------------------------   

# print ("--------------------------------RP cohort eval---------------")  
# for pred_type in pred_types:
#     print (pred_type, vt)
#     savefolder = savemain_folder +  cohort + '/'+ pred_type + '/'      
#     if os.path.exists(savefolder) == False:
#         os.makedirs(savefolder)    
#     label_type = 'L4'
#     # to save mean, median,std stats for all test set patients and all models
#     all_model_filename = savefolder + 'allmodels_rp_lesion_' + pred_type +suffix + '.csv'
#     with open(all_model_filename, mode='w', newline="") as csv_file:

#         fieldnames = ['Model','Patient id', 'ROC AUC','PR AUC', 'Sensitivity', 'Specificity', 'Precision', 'NPV', 'F1_score', 'Dice', 'Accuracy', 'Pred_type', 'Lesion_vols', 'ind_overlap', 'les_pos', 'les_neg', 'les correct pred grades']

#         writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        
#         writer.writeheader()
#         # contains per-patient metrics 
#         filesavename = cohort + '_' + model_name + '_les_' + pred_type + suffix+'.csv'
#         print ("main pred folder:", mainpredfolder)   
#         mean_stats, std_stats, median_stats, tot_num_les, sum_stats = return_stats_lesion_extra(mainpredfolder, label_path, savefolder, pred_type, main_dir,filesavename, vt, cohort, morph_disk, exclude_cases, cancer_only, atleast_onelesion)
#         writer.writerow(mean_stats)
#         writer.writerow(std_stats)
#         writer.writerow(median_stats)  
#         writer.writerow(sum_stats)
                            

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='./lesions_eval_configs.yaml')
    args = parser.parse_args()
    return_stats_lesion_extra(args)