from sklearn import metrics
from sklearn import model_selection
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_curve
import pandas as pd
import cv2
import numpy as np
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
import seaborn as sns

from normalise_2 import normalise
from import_data_3 import probability_tissue_maps, read_data
from sample_4 import sample_lacunes, non_lacune_sampling, test_sampling, train_test_combine
from feature_generation_5 import feature_gen_train, feature_gen_test
from rf_build_6 import best_number_of_trees, cv_folds, find_mean_thresh
from plots_8 import trees_plot, plot_auc_roc_thresholds, feature_importance_plot, density_plots

def main():
	print("here")
	# Setting paths for different folders
	FLAIR_scan = '/home/z5209394/Data/forAudrey.tar/Normalised/FLAIRinT1space_withLacunes_35.tar'
	T1_Lacunes_Incorrect = '/home/z5209934/Data/forAudrey.tar/Original/lacune_T1space.tar'
	T1_Lacunes_Correct = '/home/z5209394/Data/forAudrey.tar/Original/lacune_T1space_JiyangCorrected20210920'
	T1_scan = '/home/z5209394/Data/forAudrey.tar/Normalised/T1_withLacunes_35.tar'
	T1_Soft_Tissue = '/home/z5209394/Data/forAudrey.tar/Normalised/T1softTiss_withLacunes_35.tar'
	T1_Soft_Tissue_Mask = '/home/z5209394/Data/forAudrey.tar/Original/T1softTissMask_withLacunes_35.tar'
	T1_Soft_Tissue_Binary_Mask = '/home/z5209394/Data/forAudrey.tar/Original/T1softTissMask_withLacunes_35_binary.tar'
	tissue_maps = '/home/z5209394/Data/forEllen20220725'
	mypath_og = '/home/z5209394/Data/forAudrey.tar/Original'

	# Prepare data
	# normalise(mypath_og)

	# Import normalised data into relevant variables
	#CSF, WM, GM = probability_tissue_maps(tissue_maps)
	print("maps read")
	#T1_scan_data, FLAIR_scan_data, Lacune_indicator_data, Soft_tiss_data \
	#	= read_data(T1_scan, FLAIR_scan, T1_Lacunes_Correct, T1_Soft_Tissue)
	print("scans read")
	# Sample Train - Lacunes
	#X_train_3D_lacune, Y_train_3D_lacune, Y_train_segment_3D_lacune, X_train_3D_nlacune, \
	#	Y_train_3D_nlacune, Y_train_segment_3D_nlacune = sample_lacunes(CSF, GM, WM, \
	#	T1_Soft_Tissue_Binary_Mask, T1_scan_data, FLAIR_scan_data, Lacune_indicator_data, Soft_tiss_data)
	print("lacunes sampled")
	
	# Sample Train - Non-Lacunes
	#X_train_3D_nlacune_func2, Y_train_3D_nlacune_func2, Y_train_segment_3D_nlacune_func2 \
	#	= non_lacune_sampling(CSF, GM, WM, T1_Soft_Tissue_Binary_Mask, T1_scan_data, \
	#	FLAIR_scan_data, Lacune_indicator_data, Soft_tiss_data)

	# Sample Test
	#X_test_3D_lacune, Y_test_3D_lacune, Y_test_segment_3D_lacune, X_test_3D_nlacune, \
	#	Y_test_3D_nlacune, Y_test_segment_3D_nlacune = test_sampling(CSF, GM, WM, \
	#	T1_Soft_Tissue_Binary_Mask, T1_scan_data, FLAIR_scan_data, Lacune_indicator_data, Soft_tiss_data)
	print("tests sampled")
	# Combine Train Test Results
	#X_train_3D_nlacune_all, Y_train_3D_nlacune_all, Y_train_segment_3D_nlacune_all, X_train, \
	#	Y_train, Y_train_segment, Y_test_segment, Y_test, X_test = train_test_combine(X_train_3D_lacune, \
	#	Y_train_3D_lacune, Y_train_segment_3D_lacune, X_train_3D_nlacune, Y_train_3D_nlacune, Y_train_segment_3D_nlacune, \
	#	X_train_3D_nlacune_func2, Y_train_3D_nlacune_func2, Y_train_segment_3D_nlacune_func2, X_test_3D_lacune, \
	#	Y_test_3D_lacune, Y_test_segment_3D_lacune, X_test_3D_nlacune, Y_test_3D_nlacune, Y_test_segment_3D_nlacune)

	# First filter
	#indices = []
	#X_test_filtered = [] 
	#Y_test_filtered = []
	#for iter in range(len(X_test)):
	#	input_image = X_test[iter][7][:, 15, :]*255
	#	filterSize =(16,16)
	#	kernel = cv2.getStructuringElement(cv2.MORPH_RECT,filterSize)
	#	tophat_img = cv2.morphologyEx(input_image, cv2.MORPH_BLACKHAT,kernel)
	#	arr = tophat_img[(tophat_img > 75) & (tophat_img < 140)]
	#	if len(arr) > 0:
	#		X_test_filtered.append(X_test[iter])
	#		Y_test_filtered.append(Y_test[iter])
	#		indices.append(iter)
	#	iter = iter+1
	#print(len(indices))
	#print(len(X_test_filtered))
	#print(sum(Y_test_filtered))			
	
	#np.save('/home/z5209394/Data/indices_used.npy', indices)
	#np.save('/home/z5209394/Data/Y_test_filtered.npy', Y_test_filtered)	
	print("first filter applied")
	indices = np.load('/home/z5209394/Data/indices_used.npy')
	Y_test_filtered = np.load('/home/z5209394/Data/Y_test_filtered.npy')

	# Generate Train Features
	#filterSize =(16,16)
	#kernel = cv2.getStructuringElement(cv2.MORPH_RECT,filterSize)
	#brain, min_T1, med_T1, mid_T1, mid_vsmall_ratio_T1, mid_small_ratio_T1, mid_med_ratio_T1, mid_large_ratio_T1, mid_vsmall_ratio_T1_inc, mid_small_ratio_T1_inc, mid_med_ratio_T1_inc, mid_large_ratio_T1_inc, mean_T1, max_T1, var_T1, range_T1, H_T1_e1, H_T1_e2, H_T1_e3, min_FLAIR, mid_FLAIR, med_FLAIR, mid_vsmall_ratio_FLAIR, mid_small_ratio_FLAIR, mid_med_ratio_FLAIR, mid_large_ratio_FLAIR, mid_vsmall_ratio_FLAIR_inc, mid_small_ratio_FLAIR_inc, mid_med_ratio_FLAIR_inc, mid_large_ratio_FLAIR_inc, mean_FLAIR, max_FLAIR, var_FLAIR, range_FLAIR, H_FLAIR_e1, H_FLAIR_e2, H_FLAIR_e3, density_diff, sum_soft_tiss_binary, sum_percent_soft_tiss, min_st, med_st, mid_st, mid_vsmall_ratio_st, mid_small_ratio_st, mid_med_ratio_st, mid_large_ratio_st, mid_vsmall_ratio_st_inc, mid_small_ratio_st_inc, mid_med_ratio_st_inc, mid_large_ratio_st_inc, mean_st, max_st, var_st, range_st, H_st_e1, H_st_e2, H_st_e3, min_stm, med_stm, mid_stm, mid_vsmall_ratio_stm, mid_small_ratio_stm, mid_med_ratio_stm, mid_large_ratio_stm, mid_vsmall_ratio_stm_inc, mid_small_ratio_stm_inc, mid_med_ratio_stm_inc, mid_large_ratio_stm_inc, mean_stm, max_stm, var_stm, range_stm, H_stm_e1, H_stm_e2, H_stm_e3, min_th_T1, med_th_T1, mid_th_T1, mid_vsmall_ratio_th_T1, mid_small_ratio_th_T1, mid_med_ratio_th_T1, mid_large_ratio_th_T1, mid_vsmall_ratio_th_T1_inc, mid_small_ratio_th_T1_inc, mid_med_ratio_th_T1_inc, mid_large_ratio_th_T1_inc, mean_th_T1, max_th_T1, var_th_T1, range_th_T1, H_th_T1_e1, H_th_T1_e2, H_th_T1_e3, min_th_FLAIR, med_th_FLAIR, mid_th_FLAIR, mid_vsmall_ratio_th_FLAIR, mid_small_ratio_th_FLAIR, mid_med_ratio_th_FLAIR, mid_large_ratio_th_FLAIR, mid_vsmall_ratio_th_FLAIR_inc, mid_small_ratio_th_FLAIR_inc, mid_med_ratio_th_FLAIR_inc, mid_large_ratio_th_FLAIR_inc, mean_th_FLAIR, max_th_FLAIR, var_th_FLAIR, range_th_FLAIR,H_th_FLAIR_e1, H_th_FLAIR_e2, H_th_FLAIR_e3, min_th_st, med_th_st, mid_th_st, mid_vsmall_ratio_th_st, mid_small_ratio_th_st, mid_med_ratio_th_st, mid_large_ratio_th_st, mid_vsmall_ratio_th_st_inc, mid_small_ratio_th_st_inc, mid_med_ratio_th_st_inc, mid_large_ratio_th_st_inc, mean_th_st, max_th_st, var_th_st, range_th_st, H_th_st_e1, H_th_st_e2, H_th_st_e3, min_bh_T1, med_bh_T1, mid_bh_T1, mid_vsmall_ratio_bh_T1,mid_small_ratio_bh_T1, mid_med_ratio_bh_T1, mid_large_ratio_bh_T1, mid_vsmall_ratio_bh_T1_inc, mid_small_ratio_bh_T1_inc, mid_med_ratio_bh_T1_inc, mid_large_ratio_bh_T1_inc, mean_bh_T1, max_bh_T1, var_bh_T1, range_bh_T1, H_bh_T1_e1, H_bh_T1_e2, H_bh_T1_e3, min_bh_FLAIR, med_bh_FLAIR, mid_bh_FLAIR, mid_vsmall_ratio_bh_FLAIR, mid_small_ratio_bh_FLAIR, mid_med_ratio_bh_FLAIR, mid_large_ratio_bh_FLAIR, mid_vsmall_ratio_bh_FLAIR_inc, mid_small_ratio_bh_FLAIR_inc, mid_med_ratio_bh_FLAIR_inc, mid_large_ratio_bh_FLAIR_inc, mean_bh_FLAIR, max_bh_FLAIR, var_bh_FLAIR, range_bh_FLAIR, H_bh_FLAIR_e1, H_bh_FLAIR_e2, H_bh_FLAIR_e3, min_bh_st, med_bh_st, mid_bh_st, mid_vsmall_ratio_bh_st, mid_small_ratio_bh_st, mid_med_ratio_bh_st, mid_large_ratio_bh_st, mid_vsmall_ratio_bh_st_inc, mid_small_ratio_bh_st_inc, mid_med_ratio_bh_st_inc, mid_large_ratio_bh_st_inc, mean_bh_st, max_bh_st, var_bh_st, range_bh_st, H_bh_st_e1, H_bh_st_e2, H_bh_st_e3, x, y, z, WMH_x, WMH_y, WMH_z, CSF_feat, GM_feat, WM_feat = feature_gen_train(filterSize, kernel, X_train)

	# Generate Test Features
	#brain_test, min_T1_test, med_T1_test, mid_T1_test, mid_vsmall_ratio_T1_test, mid_small_ratio_T1_test, mid_med_ratio_T1_test, mid_large_ratio_T1_test, mid_vsmall_ratio_T1_inc_test, mid_small_ratio_T1_inc_test, mid_med_ratio_T1_inc_test, mid_large_ratio_T1_inc_test, mean_T1_test, max_T1_test, var_T1_test, range_T1_test, H_T1_e1_test, H_T1_e2_test, H_T1_e3_test, min_FLAIR_test, mid_FLAIR_test, med_FLAIR_test, mid_vsmall_ratio_FLAIR_test, mid_small_ratio_FLAIR_test, mid_med_ratio_FLAIR_test, mid_large_ratio_FLAIR_test, mid_vsmall_ratio_FLAIR_inc_test, mid_small_ratio_FLAIR_inc_test, mid_med_ratio_FLAIR_inc_test, mid_large_ratio_FLAIR_inc_test, mean_FLAIR_test, max_FLAIR_test, var_FLAIR_test, range_FLAIR_test, H_FLAIR_e1_test, H_FLAIR_e2_test, H_FLAIR_e3_test, density_diff_test, sum_soft_tiss_binary_test, sum_percent_soft_tiss_test, min_st_test, med_st_test, mid_st_test, mid_vsmall_ratio_st_test, mid_small_ratio_st_test, mid_med_ratio_st_test, mid_large_ratio_st_test, mid_vsmall_ratio_st_inc_test, mid_small_ratio_st_inc_test, mid_med_ratio_st_inc_test, mid_large_ratio_st_inc_test, mean_st_test, max_st_test, var_st_test, range_st_test, H_st_e1_test, H_st_e2_test, H_st_e3_test, min_stm_test, med_stm_test, mid_stm_test, mid_vsmall_ratio_stm_test, mid_small_ratio_stm_test, mid_med_ratio_stm_test, mid_large_ratio_stm_test, mid_vsmall_ratio_stm_inc_test, mid_small_ratio_stm_inc_test, mid_med_ratio_stm_inc_test, mid_large_ratio_stm_inc_test, mean_stm_test, max_stm_test, var_stm_test, range_stm_test, H_stm_e1_test, H_stm_e2_test, H_stm_e3_test, min_th_T1_test, med_th_T1_test, mid_th_T1_test, mid_vsmall_ratio_th_T1_test, mid_small_ratio_th_T1_test, mid_med_ratio_th_T1_test, mid_large_ratio_th_T1_test, mid_vsmall_ratio_th_T1_inc_test, mid_small_ratio_th_T1_inc_test, mid_med_ratio_th_T1_inc_test, mid_large_ratio_th_T1_inc_test, mean_th_T1_test, max_th_T1_test, var_th_T1_test, range_th_T1_test, H_th_T1_e1_test, H_th_T1_e2_test, H_th_T1_e3_test, min_th_FLAIR_test, med_th_FLAIR_test, mid_th_FLAIR_test, mid_vsmall_ratio_th_FLAIR_test, mid_small_ratio_th_FLAIR_test, mid_med_ratio_th_FLAIR_test, mid_large_ratio_th_FLAIR_test, mid_vsmall_ratio_th_FLAIR_inc_test, mid_small_ratio_th_FLAIR_inc_test, mid_med_ratio_th_FLAIR_inc_test, mid_large_ratio_th_FLAIR_inc_test, mean_th_FLAIR_test, max_th_FLAIR_test, var_th_FLAIR_test, range_th_FLAIR_test, H_th_FLAIR_e1_test, H_th_FLAIR_e2_test, H_th_FLAIR_e3_test, min_th_st_test, med_th_st_test, mid_th_st_test, mid_vsmall_ratio_th_st_test, mid_small_ratio_th_st_test, mid_med_ratio_th_st_test, mid_large_ratio_th_st_test, mid_vsmall_ratio_th_st_inc_test, mid_small_ratio_th_st_inc_test, mid_med_ratio_th_st_inc_test, mid_large_ratio_th_st_inc_test, mean_th_st_test, max_th_st_test, var_th_st_test, range_th_st_test, H_th_st_e1_test, H_th_st_e2_test, H_th_st_e3_test, min_bh_T1_test, med_bh_T1_test, mid_bh_T1_test, mid_vsmall_ratio_bh_T1_test, mid_small_ratio_bh_T1_test, mid_med_ratio_bh_T1_test, mid_large_ratio_bh_T1_test, mid_vsmall_ratio_bh_T1_inc_test, mid_small_ratio_bh_T1_inc_test, mid_med_ratio_bh_T1_inc_test, mid_large_ratio_bh_T1_inc_test, mean_bh_T1_test, max_bh_T1_test, var_bh_T1_test, range_bh_T1_test, H_bh_T1_e1_test, H_bh_T1_e2_test, H_bh_T1_e3_test, min_bh_FLAIR_test, med_bh_FLAIR_test, mid_bh_FLAIR_test, mid_vsmall_ratio_bh_FLAIR_test, mid_small_ratio_bh_FLAIR_test, mid_med_ratio_bh_FLAIR_test, mid_large_ratio_bh_FLAIR_test, mid_vsmall_ratio_bh_FLAIR_inc_test, mid_small_ratio_bh_FLAIR_inc_test, mid_med_ratio_bh_FLAIR_inc_test, mid_large_ratio_bh_FLAIR_inc_test, mean_bh_FLAIR_test, max_bh_FLAIR_test, var_bh_FLAIR_test, range_bh_FLAIR_test, H_bh_FLAIR_e1_test, H_bh_FLAIR_e2_test, H_bh_FLAIR_e3_test, min_bh_st_test, med_bh_st_test, mid_bh_st_test, mid_vsmall_ratio_bh_st_test, mid_small_ratio_bh_st_test, mid_med_ratio_bh_st_test, mid_large_ratio_bh_st_test, mid_vsmall_ratio_bh_st_inc_test, mid_small_ratio_bh_st_inc_test, mid_med_ratio_bh_st_inc_test, mid_large_ratio_bh_st_inc_test, mean_bh_st_test, max_bh_st_test, var_bh_st_test, range_bh_st_test, H_bh_st_e1_test, H_bh_st_e2_test, H_bh_st_e3_test, x_test, y_test, z_test, WMH_x_test, WMH_y_test, WMH_z_test, CSF_feat_test, GM_feat_test, WM_feat_test = feature_gen_test(filterSize, kernel, X_test_filtered)

	# Dataset
	#dataset = pd.DataFrame({'brain': brain, 'min_T1': min_T1, 'med_T1': med_T1, 'mid_T1': mid_T1, 'mid_vsmall_ratio_T1': mid_vsmall_ratio_T1, 'mid_small_ratio_T1': mid_small_ratio_T1, 'mid_med_ratio_T1': mid_med_ratio_T1, 'mid_large_ratio_T1': mid_large_ratio_T1, 'mid_vsmall_ratio_T1_inc': mid_vsmall_ratio_T1_inc, 'mid_small_ratio_T1_inc': mid_small_ratio_T1_inc, 'mid_med_ratio_T1_inc': mid_med_ratio_T1_inc, 'mid_large_ratio_T1_inc': mid_large_ratio_T1_inc, 'mean_T1': mean_T1, 'max_T1': max_T1, 'var_T1': var_T1,'range_T1': range_T1, 'H_T1_e1': H_T1_e1, 'H_T1_e2': H_T1_e2, 'H_T1_e3': H_T1_e3, 'min_FLAIR': min_FLAIR, 'mid_FLAIR': mid_FLAIR, 'med_FLAIR': med_FLAIR, 'mid_vsmall_ratio_FLAIR': mid_vsmall_ratio_FLAIR, 'mid_small_ratio_FLAIR': mid_small_ratio_FLAIR, 'mid_med_ratio_FLAIR': mid_med_ratio_FLAIR, 'mid_large_ratio_FLAIR': mid_large_ratio_FLAIR, 'mid_vsmall_ratio_FLAIR_inc': mid_vsmall_ratio_FLAIR_inc, 'mid_small_ratio_FLAIR_inc': mid_small_ratio_FLAIR_inc, 'mid_med_ratio_FLAIR_inc': mid_med_ratio_FLAIR_inc, 'mid_large_ratio_FLAIR_inc': mid_large_ratio_FLAIR_inc, 'mean_FLAIR': mean_FLAIR, 'max_FLAIR': max_FLAIR, 'var_FLAIR': var_FLAIR,'range_FLAIR': range_FLAIR,'H_FLAIR_e1': H_FLAIR_e1, 'H_FLAIR_e2': H_FLAIR_e2, 'H_FLAIR_e3': H_FLAIR_e3,'density_diff': density_diff, 'sum_soft_tiss_binary': sum_soft_tiss_binary, 'sum_percent_soft_tiss': sum_percent_soft_tiss, 'min_st': min_st, 'med_st': med_st, 'mid_st': mid_st, 'mid_vsmall_ratio_st': mid_vsmall_ratio_st, 'mid_small_ratio_st': mid_small_ratio_st,'mid_med_ratio_st': mid_med_ratio_st, 'mid_large_ratio_st': mid_large_ratio_st, 'mid_vsmall_ratio_st_inc': mid_vsmall_ratio_st_inc,'mid_small_ratio_st_inc': mid_small_ratio_st_inc, 'mid_med_ratio_st_inc': mid_med_ratio_st_inc, 'mid_large_ratio_st_inc': mid_large_ratio_st_inc, 'mean_st': mean_st, 'max_st': max_st, 'var_st': var_st,'range_st': range_st,'H_st_e1': H_st_e1, 'H_st_e2': H_st_e2, 'H_st_e3': H_st_e3,'min_stm': min_stm, 'med_stm': med_stm, 'mid_stm': mid_stm, 'mid_vsmall_ratio_stm': mid_vsmall_ratio_stm, 'mid_small_ratio_stm': mid_small_ratio_stm,'mid_med_ratio_stm': mid_med_ratio_stm, 'mid_large_ratio_stm': mid_large_ratio_stm, 'mid_vsmall_ratio_stm_inc': mid_vsmall_ratio_stm_inc,'mid_small_ratio_stm_inc': mid_small_ratio_stm_inc, 'mid_med_ratio_stm_inc': mid_med_ratio_stm_inc, 'mid_large_ratio_stm_inc': mid_large_ratio_stm_inc, 'mean_stm': mean_stm, 'max_stm': max_stm,'var_stm': var_stm,'range_stm': range_stm, 'H_stm_e1': H_stm_e1, 'H_stm_e2': H_stm_e2, 'H_stm_e3': H_stm_e3,'min_th_T1': min_th_T1, 'med_th_T1': med_th_T1, 'mid_th_T1': mid_th_T1, 'mid_vsmall_ratio_th_T1': mid_vsmall_ratio_th_T1, 'mid_small_ratio_th_T1': mid_small_ratio_th_T1, 'mid_med_ratio_th_T1': mid_med_ratio_th_T1, 'mid_large_ratio_th_T1': mid_large_ratio_th_T1, 'mid_vsmall_ratio_th_T1_inc': mid_vsmall_ratio_th_T1_inc, 'mid_small_ratio_th_T1_inc': mid_small_ratio_th_T1_inc, 'mid_med_ratio_th_T1_inc': mid_med_ratio_th_T1_inc, 'mid_large_ratio_th_T1_inc': mid_large_ratio_th_T1_inc, 'mean_th_T1': mean_th_T1, 'max_th_T1': max_th_T1, 'var_th_T1': var_th_T1,'range_th_T1': range_th_T1,'H_th_T1_e1': H_th_T1_e1, 'H_th_T1_e2': H_th_T1_e2, 'H_th_T1_e3': H_th_T1_e3,'min_th_FLAIR': min_th_FLAIR, 'med_th_FLAIR': med_th_FLAIR, 'mid_th_FLAIR': mid_th_FLAIR, 'mid_vsmall_ratio_th_FLAIR': mid_vsmall_ratio_th_FLAIR, 'mid_small_ratio_th_FLAIR': mid_small_ratio_th_FLAIR, 'mid_med_ratio_th_FLAIR': mid_med_ratio_th_FLAIR, 'mid_large_ratio_th_FLAIR': mid_large_ratio_th_FLAIR, 'mid_vsmall_ratio_th_FLAIR_inc': mid_vsmall_ratio_th_FLAIR_inc, 'mid_small_ratio_th_FLAIR_inc': mid_small_ratio_th_FLAIR_inc, 'mid_med_ratio_th_FLAIR_inc': mid_med_ratio_th_FLAIR_inc, 'mid_large_ratio_th_FLAIR_inc': mid_large_ratio_th_FLAIR_inc, 'mean_th_FLAIR': mean_th_FLAIR, 'max_th_FLAIR': max_th_FLAIR, 'var_th_FLAIR': var_th_FLAIR,'range_th_FLAIR': range_th_FLAIR,'H_th_FLAIR_e1': H_th_FLAIR_e1, 'H_th_FLAIR_e2': H_th_FLAIR_e2, 'H_th_FLAIR_e3': H_th_FLAIR_e3,'min_th_st': min_th_st, 'med_th_st': med_th_st, 'mid_th_st': mid_th_st, 'mid_vsmall_ratio_th_st': mid_vsmall_ratio_th_st,'mid_small_ratio_th_st': mid_small_ratio_th_st, 'mid_med_ratio_th_st': mid_med_ratio_th_st, 'mid_large_ratio_th_st': mid_large_ratio_th_st, 'mid_vsmall_ratio_th_st_inc': mid_vsmall_ratio_th_st_inc,'mid_small_ratio_th_st_inc': mid_small_ratio_th_st_inc, 'mid_med_ratio_th_st_inc': mid_med_ratio_th_st_inc, 'mid_large_ratio_th_st_inc': mid_large_ratio_th_st_inc, 'mean_th_st': mean_th_st, 'max_th_st': max_th_st,'var_th_st': var_th_st,'range_th_st': range_th_st, 'H_th_st_e1': H_th_st_e1, 'H_th_st_e2': H_th_st_e2, 'H_th_st_e3': H_th_st_e3,'min_bh_T1': min_bh_T1, 'med_bh_T1': med_bh_T1, 'mid_bh_T1': mid_bh_T1, 'mid_vsmall_ratio_bh_T1': mid_vsmall_ratio_bh_T1,'mid_small_ratio_bh_T1': mid_small_ratio_bh_T1, 'mid_med_ratio_bh_T1': mid_med_ratio_bh_T1, 'mid_large_ratio_bh_T1': mid_large_ratio_bh_T1, 'mid_vsmall_ratio_bh_T1_inc': mid_vsmall_ratio_bh_T1_inc,'mid_small_ratio_bh_T1_inc': mid_small_ratio_bh_T1_inc, 'mid_med_ratio_bh_T1_inc': mid_med_ratio_bh_T1_inc, 'mid_large_ratio_bh_T1_inc': mid_large_ratio_bh_T1_inc, 'mean_bh_T1': mean_bh_T1, 'max_bh_T1': max_bh_T1, 'var_bh_T1': var_bh_T1,'range_bh_T1': range_bh_T1, 'H_bh_T1_e1': H_bh_T1_e1, 'H_bh_T1_e2': H_bh_T1_e2, 'H_bh_T1_e3': H_bh_T1_e3,'min_bh_FLAIR': min_bh_FLAIR, 'med_bh_FLAIR': med_bh_FLAIR, 'mid_bh_FLAIR': mid_bh_FLAIR, 'mid_vsmall_ratio_bh_FLAIR': mid_vsmall_ratio_bh_FLAIR, 'mid_small_ratio_bh_FLAIR': mid_small_ratio_bh_FLAIR, 'mid_med_ratio_bh_FLAIR': mid_med_ratio_bh_FLAIR, 'mid_large_ratio_bh_FLAIR': mid_large_ratio_bh_FLAIR, 'mid_vsmall_ratio_bh_FLAIR_inc': mid_vsmall_ratio_bh_FLAIR_inc, 'mid_small_ratio_bh_FLAIR_inc': mid_small_ratio_bh_FLAIR_inc, 'mid_med_ratio_bh_FLAIR_inc': mid_med_ratio_bh_FLAIR_inc, 'mid_large_ratio_bh_FLAIR_inc': mid_large_ratio_bh_FLAIR_inc, 'mean_bh_FLAIR': mean_bh_FLAIR, 'max_bh_FLAIR': max_bh_FLAIR,'var_bh_FLAIR': var_bh_FLAIR,'range_bh_FLAIR': range_bh_FLAIR,'H_bh_FLAIR_e1': H_bh_FLAIR_e1, 'H_bh_FLAIR_e2': H_bh_FLAIR_e2, 'H_bh_FLAIR_e3': H_bh_FLAIR_e3,'min_bh_st': min_bh_st, 'med_bh_st': med_bh_st, 'mid_bh_st': mid_bh_st, 'mid_vsmall_ratio_bh_st': mid_vsmall_ratio_bh_st,'mid_small_ratio_bh_st': mid_small_ratio_bh_st, 'mid_med_ratio_bh_st': mid_med_ratio_bh_st, 'mid_large_ratio_bh_st': mid_large_ratio_bh_st, 'mid_vsmall_ratio_bh_st_inc': mid_vsmall_ratio_bh_st_inc,'mid_small_ratio_bh_st_inc': mid_small_ratio_bh_st_inc, 'mid_med_ratio_bh_st_inc': mid_med_ratio_bh_st_inc, 'mid_large_ratio_bh_st_inc': mid_large_ratio_bh_st_inc, 'mean_bh_st': mean_bh_st, 'max_bh_st': max_bh_st, 'var_bh_st': var_bh_st,'range_bh_st': range_bh_st, 'H_bh_st_e1': H_bh_st_e1, 'H_bh_st_e2': H_bh_st_e2, 'H_bh_st_e3': H_bh_st_e3, 'x': x, 'y': y, 'z': z, 'WMH_x': WMH_x, 'WMH_y' : WMH_y, 'WMH_z': WMH_z, 'CSF': CSF_feat, 'GM': GM_feat, 'WM': WM_feat})
	#dataset = np.nan_to_num(dataset, nan=-1, posinf=99, neginf=-99)
	#dataset_test = pd.DataFrame({'brain': brain_test, 'min_T1': min_T1_test, 'med_T1': med_T1_test, 'mid_T1': mid_T1_test, 'mid_vsmall_ratio_T1': mid_vsmall_ratio_T1_test, 'mid_small_ratio_T1': mid_small_ratio_T1_test, 'mid_med_ratio_T1': mid_med_ratio_T1_test, 'mid_large_ratio_T1': mid_large_ratio_T1_test, 'mid_vsmall_ratio_T1_inc': mid_vsmall_ratio_T1_inc_test, 'mid_small_ratio_T1_inc': mid_small_ratio_T1_inc_test, 'mid_med_ratio_T1_inc': mid_med_ratio_T1_inc_test, 'mid_large_ratio_T1_inc': mid_large_ratio_T1_inc_test, 'mean_T1': mean_T1_test, 'max_T1': max_T1_test, 'var_T1': var_T1_test,'range_T1': range_T1_test,'H_T1_e1': H_T1_e1_test, 'H_T1_e2': H_T1_e2_test, 'H_T1_e3': H_T1_e3_test,'min_FLAIR': min_FLAIR_test, 'mid_FLAIR': mid_FLAIR_test, 'med_FLAIR': med_FLAIR_test, 'mid_vsmall_ratio_FLAIR': mid_vsmall_ratio_FLAIR_test, 'mid_small_ratio_FLAIR': mid_small_ratio_FLAIR_test, 'mid_med_ratio_FLAIR': mid_med_ratio_FLAIR_test, 'mid_large_ratio_FLAIR': mid_large_ratio_FLAIR_test, 'mid_vsmall_ratio_FLAIR_inc': mid_vsmall_ratio_FLAIR_inc_test, 'mid_small_ratio_FLAIR_inc': mid_small_ratio_FLAIR_inc_test, 'mid_med_ratio_FLAIR_inc': mid_med_ratio_FLAIR_inc_test, 'mid_large_ratio_FLAIR_inc': mid_large_ratio_FLAIR_inc_test, 'mean_FLAIR': mean_FLAIR_test, 'max_FLAIR': max_FLAIR_test, 'var_FLAIR': var_FLAIR_test,'range_FLAIR': range_FLAIR_test,'H_FLAIR_e1': H_FLAIR_e1_test, 'H_FLAIR_e2': H_FLAIR_e2_test, 'H_FLAIR_e3': H_FLAIR_e3_test,'density_diff': density_diff_test, 'sum_soft_tiss_binary': sum_soft_tiss_binary_test, 'sum_percent_soft_tiss': sum_percent_soft_tiss_test, 'min_st': min_st_test, 'med_st': med_st_test, 'mid_st': mid_st_test, 'mid_vsmall_ratio_st': mid_vsmall_ratio_st_test, 'mid_small_ratio_st': mid_small_ratio_st_test,'mid_med_ratio_st': mid_med_ratio_st_test, 'mid_large_ratio_st': mid_large_ratio_st_test, 'mid_vsmall_ratio_st_inc': mid_vsmall_ratio_st_inc_test,'mid_small_ratio_st_inc': mid_small_ratio_st_inc_test, 'mid_med_ratio_st_inc': mid_med_ratio_st_inc_test, 'mid_large_ratio_st_inc': mid_large_ratio_st_inc_test, 'mean_st': mean_st_test, 'max_st': max_st_test, 'var_st': var_st_test,'range_st': range_st_test, 'H_st_e1': H_st_e1_test, 'H_st_e2': H_st_e2_test, 'H_st_e3': H_st_e3_test,'min_stm': min_stm_test, 'med_stm': med_stm_test, 'mid_stm': mid_stm_test, 'mid_vsmall_ratio_stm': mid_vsmall_ratio_stm_test, 'mid_small_ratio_stm': mid_small_ratio_stm_test,'mid_med_ratio_stm': mid_med_ratio_stm_test, 'mid_large_ratio_stm': mid_large_ratio_stm_test, 'mid_vsmall_ratio_stm_inc': mid_vsmall_ratio_stm_inc_test,'mid_small_ratio_stm_inc': mid_small_ratio_stm_inc_test, 'mid_med_ratio_stm_inc': mid_med_ratio_stm_inc_test, 'mid_large_ratio_stm_inc': mid_large_ratio_stm_inc_test, 'mean_stm': mean_stm_test, 'max_stm': max_stm_test,'var_stm': var_stm_test,'range_stm': range_stm_test, 'H_stm_e1': H_stm_e1_test, 'H_stm_e2': H_stm_e2_test, 'H_stm_e3': H_stm_e3_test, 'min_th_T1': min_th_T1_test, 'med_th_T1': med_th_T1_test, 'mid_th_T1': mid_th_T1_test, 'mid_vsmall_ratio_th_T1': mid_vsmall_ratio_th_T1_test, 'mid_small_ratio_th_T1': mid_small_ratio_th_T1_test, 'mid_med_ratio_th_T1': mid_med_ratio_th_T1_test, 'mid_large_ratio_th_T1': mid_large_ratio_th_T1_test, 'mid_vsmall_ratio_th_T1_inc': mid_vsmall_ratio_th_T1_inc_test, 'mid_small_ratio_th_T1_inc': mid_small_ratio_th_T1_inc_test, 'mid_med_ratio_th_T1_inc': mid_med_ratio_th_T1_inc_test, 'mid_large_ratio_th_T1_inc': mid_large_ratio_th_T1_inc_test, 'mean_th_T1': mean_th_T1_test, 'max_th_T1': max_th_T1_test, 'var_th_T1': var_th_T1_test,'range_th_T1': range_th_T1_test, 'H_th_T1_e1': H_th_T1_e1_test, 'H_th_T1_e2': H_th_T1_e2_test, 'H_th_T1_e3': H_th_T1_e3_test,'min_th_FLAIR': min_th_FLAIR_test, 'med_th_FLAIR': med_th_FLAIR_test, 'mid_th_FLAIR': mid_th_FLAIR_test, 'mid_vsmall_ratio_th_FLAIR': mid_vsmall_ratio_th_FLAIR_test, 'mid_small_ratio_th_FLAIR': mid_small_ratio_th_FLAIR_test, 'mid_med_ratio_th_FLAIR': mid_med_ratio_th_FLAIR_test, 'mid_large_ratio_th_FLAIR': mid_large_ratio_th_FLAIR_test, 'mid_vsmall_ratio_th_FLAIR_inc': mid_vsmall_ratio_th_FLAIR_inc_test, 'mid_small_ratio_th_FLAIR_inc': mid_small_ratio_th_FLAIR_inc_test, 'mid_med_ratio_th_FLAIR_inc': mid_med_ratio_th_FLAIR_inc_test, 'mid_large_ratio_th_FLAIR_inc': mid_large_ratio_th_FLAIR_inc_test, 'mean_th_FLAIR': mean_th_FLAIR_test, 'max_th_FLAIR': max_th_FLAIR_test, 'var_th_FLAIR': var_th_FLAIR_test,'range_th_FLAIR': range_th_FLAIR_test, 'H_th_FLAIR_e1': H_th_FLAIR_e1_test, 'H_th_FLAIR_e2': H_th_FLAIR_e2_test, 'H_th_FLAIR_e3': H_th_FLAIR_e3_test,'min_th_st': min_th_st_test, 'med_th_st': med_th_st_test, 'mid_th_st': mid_th_st_test, 'mid_vsmall_ratio_th_st': mid_vsmall_ratio_th_st_test,'mid_small_ratio_th_st': mid_small_ratio_th_st_test, 'mid_med_ratio_th_st': mid_med_ratio_th_st_test, 'mid_large_ratio_th_st': mid_large_ratio_th_st_test, 'mid_vsmall_ratio_th_st_inc': mid_vsmall_ratio_th_st_inc_test,'mid_small_ratio_th_st_inc': mid_small_ratio_th_st_inc_test, 'mid_med_ratio_th_st_inc': mid_med_ratio_th_st_inc_test, 'mid_large_ratio_th_st_inc': mid_large_ratio_th_st_inc_test, 'mean_th_st': mean_th_st_test, 'max_th_st': max_th_st_test,'var_th_st': var_th_st_test,'range_th_st': range_th_st_test, 'H_th_st_e1': H_th_st_e1_test, 'H_th_st_e2': H_th_st_e2_test, 'H_th_st_e3': H_th_st_e3_test,'min_bh_T1': min_bh_T1_test, 'med_bh_T1': med_bh_T1_test, 'mid_bh_T1': mid_bh_T1_test, 'mid_vsmall_ratio_bh_T1': mid_vsmall_ratio_bh_T1_test,'mid_small_ratio_bh_T1': mid_small_ratio_bh_T1_test, 'mid_med_ratio_bh_T1': mid_med_ratio_bh_T1_test, 'mid_large_ratio_bh_T1': mid_large_ratio_bh_T1_test, 'mid_vsmall_ratio_bh_T1_inc': mid_vsmall_ratio_bh_T1_inc_test,'mid_small_ratio_bh_T1_inc': mid_small_ratio_bh_T1_inc_test, 'mid_med_ratio_bh_T1_inc': mid_med_ratio_bh_T1_inc_test, 'mid_large_ratio_bh_T1_inc': mid_large_ratio_bh_T1_inc_test, 'mean_bh_T1': mean_bh_T1_test, 'max_bh_T1': max_bh_T1_test, 'var_bh_T1': var_bh_T1_test,'range_bh_T1': range_bh_T1_test, 'H_bh_T1_e1': H_bh_T1_e1_test, 'H_bh_T1_e2': H_bh_T1_e2_test, 'H_bh_T1_e3': H_bh_T1_e3_test,'min_bh_FLAIR': min_bh_FLAIR_test, 'med_bh_FLAIR': med_bh_FLAIR_test, 'mid_bh_FLAIR': mid_bh_FLAIR_test, 'mid_vsmall_ratio_bh_FLAIR': mid_vsmall_ratio_bh_FLAIR_test,'mid_small_ratio_bh_FLAIR': mid_small_ratio_bh_FLAIR_test, 'mid_med_ratio_bh_FLAIR': mid_med_ratio_bh_FLAIR_test, 'mid_large_ratio_bh_FLAIR': mid_large_ratio_bh_FLAIR_test, 'mid_vsmall_ratio_bh_FLAIR_inc': mid_vsmall_ratio_bh_FLAIR_inc_test,'mid_small_ratio_bh_FLAIR_inc': mid_small_ratio_bh_FLAIR_inc_test, 'mid_med_ratio_bh_FLAIR_inc': mid_med_ratio_bh_FLAIR_inc_test, 'mid_large_ratio_bh_FLAIR_inc': mid_large_ratio_bh_FLAIR_inc_test, 'mean_bh_FLAIR': mean_bh_FLAIR_test, 'max_bh_FLAIR': max_bh_FLAIR_test,'var_bh_FLAIR': var_bh_FLAIR_test,'range_bh_FLAIR': range_bh_FLAIR_test, 'H_bh_FLAIR_e1': H_bh_FLAIR_e1_test, 'H_bh_FLAIR_e2': H_bh_FLAIR_e2_test, 'H_bh_FLAIR_e3': H_bh_FLAIR_e3_test,'min_bh_st': min_bh_st_test, 'med_bh_st': med_bh_st_test, 'mid_bh_st': mid_bh_st_test, 'mid_vsmall_ratio_bh_st': mid_vsmall_ratio_bh_st_test,'mid_small_ratio_bh_st': mid_small_ratio_bh_st_test, 'mid_med_ratio_bh_st': mid_med_ratio_bh_st_test, 'mid_large_ratio_bh_st': mid_large_ratio_bh_st_test, 'mid_vsmall_ratio_bh_st_inc': mid_vsmall_ratio_bh_st_inc_test,'mid_small_ratio_bh_st_inc': mid_small_ratio_bh_st_inc_test, 'mid_med_ratio_bh_st_inc': mid_med_ratio_bh_st_inc_test, 'mid_large_ratio_bh_st_inc': mid_large_ratio_bh_st_inc_test, 'mean_bh_st': mean_bh_st_test, 'max_bh_st': max_bh_st_test, 'var_bh_st': var_bh_st_test,'range_bh_st': range_bh_st_test, 'H_bh_st_e1': H_bh_st_e1_test, 'H_bh_st_e2': H_bh_st_e2_test, 'H_bh_st_e3': H_bh_st_e3_test,'x': x_test, 'y': y_test, 'z': z_test, 'WMH_x': WMH_x_test, 'WMH_y' : WMH_y_test, 'WMH_z': WMH_z_test, 'CSF': CSF_feat_test, 'GM': GM_feat_test, 'WM': WM_feat_test})
	#dataset_test = np.nan_to_num(dataset_test, nan=-1, posinf=99, neginf=-99)
	
	#np.save('/home/z5209394/Data/dataset.npy', dataset)
	#np.save('/home/z5209394/Data/dataset_test.npy', dataset_test)
	#np.save('/home/z5209394/Data/y_train.npy', Y_train)
	#np.save('/home/z5209394/Data/y_test.npy', Y_test)
	print('features saved')
	dataset_b = np.load('/home/z5209394/Data/dataset.npy')
	dataset_test_b = np.load('/home/z5209394/Data/dataset_test.npy')
	Y_train = np.load('/home/z5209394/Data/y_train.npy')
	Y_test = np.load('/home/z5209394/Data/y_test.npy')

	# Remove brain from datafram
	dataset = dataset_b[:,1:]	
	dataset_test = dataset_test_b[:,1:]		

	# Best Number of Trees
	#accuracy_list = []
	#error_list = []
	#no_trees = [250, 500, 1000, 2500, 5000, 7500, 10000, 15000, 20000, 30000, 50000]
	#for i in no_trees:
	#	print(i)
	#	clf = RandomForestClassifier(n_estimators = i, criterion = 'gini', oob_score = True, bootstrap = True, n_jobs = 16)
	#	clf.fit(dataset, Y_train)
	#	y_pred = (clf.oob_decision_function_[:,1] >= 0.50).astype(bool)
	#	accuracy = metrics.accuracy_score(Y_train, y_pred)
	#	accuracy_list.append(accuracy)
	#	error_list.append(1-accuracy)

	# Generate plot
	#trees_plot(error_list, no_trees)
	print("no trees plotted")

	# CV-Folds
	cv_1_data, cv_2_data, cv_3_data, cv_4_data, cv_5_data, cv_1_idx, cv_2_idx, cv_3_idx, cv_4_idx, cv_5_idx, cv_1_brains, cv_2_brains, cv_3_brains, cv_4_brains, cv_5_brains, cv_1_train, cv_2_train, cv_3_train, cv_4_train, cv_5_train = cv_folds(dataset_b, Y_train)																																																																																																																																						

	# Randomized Search CV 
	# Number of trees in Random Forest
	#n_estimators = 15000

	# Number of features to consider at every split
	#rf_max_features = ['sqrt', 9, 10, 11, 12, 13, 14,15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25]

	# Minimum number of samples required to split a node
	#rf_min_samples_split = [int(x) for x in np.linspace(2, 20, 19)]

	# Create the grid
	#rf_grid = {'criterion' : ["gini"],
	#	'n_estimators': [n_estimators],
	#	'bootstrap': [True],
	#	'oob_score': [True],
	#	'max_features': rf_max_features,
	#	'min_samples_split': rf_min_samples_split}

	# Custom CV splits
	#cv_splits = [(cv_1_idx, cv_1_train), (cv_2_idx, cv_2_train), (cv_3_idx, cv_3_train), (cv_4_idx, cv_4_train), (cv_5_idx, cv_5_train)]

	# Add trees here
	#rf_base = RandomForestClassifier()

	#def custom_oob_score(estimator, dataset, Y_train):
	#	return estimator.oob_score_

	#scoring_list = {'oob_error': custom_oob_score}

	# Create the random search Random Forest
	#rf_random = model_selection.RandomizedSearchCV(estimator = rf_base, scoring = scoring_list, param_distributions = rf_grid, refit = 'oob_error',
	#	n_iter = 50, cv = cv_splits, verbose = 3, random_state = 42, 
	#	n_jobs = 16)
	# Fit the random search model
	#rf_random.fit(dataset, Y_train)

	# Save results
	#results_params = pd.DataFrame(rf_random.cv_results_)
	#results_params.to_numpy()

	# View the best parameters from the random search model
	#print(rf_random.best_params_)

	# Classification and ROC analysis

	# Model defined by best parameters
	#classifier = RandomForestClassifier()
	#classifier.set_params(**rf_random.best_params_)

	# Saved classifier
	classifier = RandomForestClassifier()
	params = {'oob_score': True, 'n_estimators': 15000, 'min_samples_split': 3, 'max_features': 10, 'criterion': 'gini', 'bootstrap': True, 'n_jobs': 16}
	classifier.set_params(**params)
	mean_thresh = 0.213214
	classifier.fit(dataset, Y_train)

	# Define cv splits to find best threshold
	cv_splits = [(np.concatenate([cv_2_idx, cv_3_idx, cv_4_idx, cv_5_idx]), cv_1_idx), 
			(np.concatenate([cv_1_idx, cv_3_idx, cv_4_idx, cv_5_idx]), cv_2_idx), 
			(np.concatenate([cv_1_idx, cv_2_idx, cv_4_idx, cv_5_idx]), cv_3_idx), 
			(np.concatenate([cv_1_idx, cv_2_idx, cv_3_idx, cv_5_idx]), cv_4_idx), 
			(np.concatenate([cv_1_idx, cv_2_idx, cv_3_idx, cv_4_idx]), cv_5_idx)]

	#mean_thresh = find_mean_thresh(classifier, cv_splits, dataset, Y_train)

	# Run classifier with cross-validation and plot ROC curves
	#plot_auc_roc_thresholds(classifier, dataset, Y_train, cv_splits, mean_thresh)

	dataset_pd = pd.DataFrame(dataset, columns = ['min_T1', 'med_T1', 'mid_T1', 'mid_vsmall_ratio_T1', 'mid_small_ratio_T1', 'mid_med_ratio_T1', 'mid_large_ratio_T1', 'mid_vsmall_ratio_T1_inc', 'mid_small_ratio_T1_inc', 'mid_med_ratio_T1_inc', 'mid_large_ratio_T1_inc', 'mean_T1', 'max_T1', 'var_T1','range_T1', 'H_T1_e1', 'H_T1_e2', 'H_T1_e3', 'min_FLAIR', 'mid_FLAIR', 'med_FLAIR', 'mid_vsmall_ratio_FLAIR', 'mid_small_ratio_FLAIR', 'mid_med_ratio_FLAIR', 'mid_large_ratio_FLAIR', 'mid_vsmall_ratio_FLAIR_inc', 'mid_small_ratio_FLAIR_inc', 'mid_med_ratio_FLAIR_inc', 'mid_large_ratio_FLAIR_inc','mean_FLAIR', 'max_FLAIR', 'var_FLAIR','range_FLAIR','H_FLAIR_e1', 'H_FLAIR_e2', 'H_FLAIR_e3','density_diff', 'sum_soft_tiss_binary', 'sum_percent_soft_tiss', 'min_st', 'med_st', 'mid_st', 'mid_vsmall_ratio_st', 'mid_small_ratio_st','mid_med_ratio_st', 'mid_large_ratio_st', 'mid_vsmall_ratio_st_inc','mid_small_ratio_st_inc', 'mid_med_ratio_st_inc', 'mid_large_ratio_st_inc', 'mean_st', 'max_st', 'var_st','range_st','H_st_e1', 'H_st_e2', 'H_st_e3','min_stm', 'med_stm', 'mid_stm', 'mid_vsmall_ratio_stm', 'mid_small_ratio_stm','mid_med_ratio_stm', 'mid_large_ratio_stm', 'mid_vsmall_ratio_stm_inc','mid_small_ratio_stm_inc', 'mid_med_ratio_stm_inc', 'mid_large_ratio_stm_inc', 'mean_stm', 'max_stm','var_stm','range_stm', 'H_stm_e1', 'H_stm_e2', 'H_stm_e3','min_th_T1', 'med_th_T1', 'mid_th_T1', 'mid_vsmall_ratio_th_T1', 'mid_small_ratio_th_T1', 'mid_med_ratio_th_T1', 'mid_large_ratio_th_T1', 'mid_vsmall_ratio_th_T1_inc', 'mid_small_ratio_th_T1_inc', 'mid_med_ratio_th_T1_inc', 'mid_large_ratio_th_T1_inc', 'mean_th_T1', 'max_th_T1', 'var_th_T1','range_th_T1','H_th_T1_e1', 'H_th_T1_e2', 'H_th_T1_e3','min_th_FLAIR', 'med_th_FLAIR', 'mid_th_FLAIR', 'mid_vsmall_ratio_th_FLAIR', 'mid_small_ratio_th_FLAIR', 'mid_med_ratio_th_FLAIR', 'mid_large_ratio_th_FLAIR', 'mid_vsmall_ratio_th_FLAIR_inc', 'mid_small_ratio_th_FLAIR_inc', 'mid_med_ratio_th_FLAIR_inc', 'mid_large_ratio_th_FLAIR_inc', 'mean_th_FLAIR', 'max_th_FLAIR', 'var_th_FLAIR','range_th_FLAIR','H_th_FLAIR_e1', 'H_th_FLAIR_e2', 'H_th_FLAIR_e3','min_th_st', 'med_th_st', 'mid_th_st', 'mid_vsmall_ratio_th_st','mid_small_ratio_th_st', 'mid_med_ratio_th_st', 'mid_large_ratio_th_st', 'mid_vsmall_ratio_th_st_inc','mid_small_ratio_th_st_inc', 'mid_med_ratio_th_st_inc', 'mid_large_ratio_th_st_inc', 'mean_th_st', 'max_th_st','var_th_st','range_th_st', 'H_th_st_e1', 'H_th_st_e2', 'H_th_st_e3','min_bh_T1', 'med_bh_T1', 'mid_bh_T1', 'mid_vsmall_ratio_bh_T1','mid_small_ratio_bh_T1', 'mid_med_ratio_bh_T1', 'mid_large_ratio_bh_T1', 'mid_vsmall_ratio_bh_T1_inc','mid_small_ratio_bh_T1_inc', 'mid_med_ratio_bh_T1_inc', 'mid_large_ratio_bh_T1_inc', 'mean_bh_T1', 'max_bh_T1', 'var_bh_T1','range_bh_T1', 'H_bh_T1_e1', 'H_bh_T1_e2', 'H_bh_T1_e3','min_bh_FLAIR', 'med_bh_FLAIR', 'mid_bh_FLAIR', 'mid_vsmall_ratio_bh_FLAIR', 'mid_small_ratio_bh_FLAIR', 'mid_med_ratio_bh_FLAIR', 'mid_large_ratio_bh_FLAIR', 'mid_vsmall_ratio_bh_FLAIR_inc', 'mid_small_ratio_bh_FLAIR_inc', 'mid_med_ratio_bh_FLAIR_inc', 'mid_large_ratio_bh_FLAIR_inc', 'mean_bh_FLAIR', 'max_bh_FLAIR','var_bh_FLAIR','range_bh_FLAIR','H_bh_FLAIR_e1', 'H_bh_FLAIR_e2', 'H_bh_FLAIR_e3','min_bh_st', 'med_bh_st', 'mid_bh_st', 'mid_vsmall_ratio_bh_st','mid_small_ratio_bh_st', 'mid_med_ratio_bh_st', 'mid_large_ratio_bh_st', 'mid_vsmall_ratio_bh_st_inc','mid_small_ratio_bh_st_inc', 'mid_med_ratio_bh_st_inc', 'mid_large_ratio_bh_st_inc', 'mean_bh_st', 'max_bh_st', 'var_bh_st','range_bh_st', 'H_bh_st_e1', 'H_bh_st_e2', 'H_bh_st_e3', 'x', 'y', 'z', 'WMH_x', 'WMH_y' , 'WMH_z', 'CSF', 'GM', 'WM'])
	dataset_test_pd = pd.DataFrame(dataset_test, columns = ['min_T1_test', 'med_T1_test', 'mid_T1_test', 'mid_vsmall_ratio_T1_test', 'mid_small_ratio_T1_test', 'mid_med_ratio_T1_test', 'mid_large_ratio_T1_test', 'mid_vsmall_ratio_T1_inc_test', 'mid_small_ratio_T1_inc_test', 'mid_med_ratio_T1_inc_test', 'mid_large_ratio_T1_inc_test', 'mean_T1_test', 'max_T1_test', 'var_T1_test','range_T1_test', 'H_T1_e1_test', 'H_T1_e2_test', 'H_T1_e3_test', 'min_FLAIR_test', 'mid_FLAIR_test', 'med_FLAIR_test', 'mid_vsmall_ratio_FLAIR_test', 'mid_small_ratio_FLAIR_test', 'mid_med_ratio_FLAIR_test', 'mid_large_ratio_FLAIR_test', 'mid_vsmall_ratio_FLAIR_inc_test', 'mid_small_ratio_FLAIR_inc_test', 'mid_med_ratio_FLAIR_inc_test', 'mid_large_ratio_FLAIR_inc_test','mean_FLAIR_test', 'max_FLAIR_test', 'var_FLAIR_test','range_FLAIR_test','H_FLAIR_e1_test', 'H_FLAIR_e2_test', 'H_FLAIR_e3_test','density_diff_test', 'sum_soft_tiss_binary_test', 'sum_percent_soft_tiss_test', 'min_st_test', 'med_st_test', 'mid_st_test', 'mid_vsmall_ratio_st_test', 'mid_small_ratio_st_test','mid_med_ratio_st_test', 'mid_large_ratio_st_test', 'mid_vsmall_ratio_st_inc_test','mid_small_ratio_st_inc_test', 'mid_med_ratio_st_inc_test', 'mid_large_ratio_st_inc_test', 'mean_st_test', 'max_st_test', 'var_st_test','range_st_test','H_st_e1_test', 'H_st_e2_test', 'H_st_e3_test','min_stm_test', 'med_stm_test', 'mid_stm_test', 'mid_vsmall_ratio_stm_test', 'mid_small_ratio_stm_test','mid_med_ratio_stm_test', 'mid_large_ratio_stm_test', 'mid_vsmall_ratio_stm_inc_test','mid_small_ratio_stm_inc_test', 'mid_med_ratio_stm_inc_test', 'mid_large_ratio_stm_inc_test', 'mean_stm_test', 'max_stm_test','var_stm_test','range_stm_test', 'H_stm_e1_test', 'H_stm_e2_test', 'H_stm_e3_test','min_th_T1_test', 'med_th_T1_test', 'mid_th_T1_test', 'mid_vsmall_ratio_th_T1_test', 'mid_small_ratio_th_T1_test', 'mid_med_ratio_th_T1_test', 'mid_large_ratio_th_T1_test', 'mid_vsmall_ratio_th_T1_inc_test', 'mid_small_ratio_th_T1_inc_test', 'mid_med_ratio_th_T1_inc_test', 'mid_large_ratio_th_T1_inc_test', 'mean_th_T1_test', 'max_th_T1_test', 'var_th_T1_test','range_th_T1_test','H_th_T1_e1_test', 'H_th_T1_e2_test', 'H_th_T1_e3_test','min_th_FLAIR_test', 'med_th_FLAIR_test', 'mid_th_FLAIR_test', 'mid_vsmall_ratio_th_FLAIR_test', 'mid_small_ratio_th_FLAIR_test', 'mid_med_ratio_th_FLAIR_test', 'mid_large_ratio_th_FLAIR_test', 'mid_vsmall_ratio_th_FLAIR_inc_test', 'mid_small_ratio_th_FLAIR_inc_test', 'mid_med_ratio_th_FLAIR_inc_test', 'mid_large_ratio_th_FLAIR_inc_test', 'mean_th_FLAIR_test', 'max_th_FLAIR_test', 'var_th_FLAIR_test','range_th_FLAIR_test','H_th_FLAIR_e1_test', 'H_th_FLAIR_e2_test', 'H_th_FLAIR_e3_test','min_th_st_test', 'med_th_st_test', 'mid_th_st_test', 'mid_vsmall_ratio_th_st_test','mid_small_ratio_th_st_test', 'mid_med_ratio_th_st_test', 'mid_large_ratio_th_st_test', 'mid_vsmall_ratio_th_st_inc_test','mid_small_ratio_th_st_inc_test', 'mid_med_ratio_th_st_inc_test', 'mid_large_ratio_th_st_inc_test', 'mean_th_st_test', 'max_th_st_test','var_th_st_test','range_th_st_test', 'H_th_st_e1_test', 'H_th_st_e2_test', 'H_th_st_e3_test','min_bh_T1_test', 'med_bh_T1_test', 'mid_bh_T1_test', 'mid_vsmall_ratio_bh_T1_test','mid_small_ratio_bh_T1_test', 'mid_med_ratio_bh_T1_test', 'mid_large_ratio_bh_T1_test', 'mid_vsmall_ratio_bh_T1_inc_test','mid_small_ratio_bh_T1_inc_test', 'mid_med_ratio_bh_T1_inc_test', 'mid_large_ratio_bh_T1_inc_test', 'mean_bh_T1_test', 'max_bh_T1_test', 'var_bh_T1_test','range_bh_T1_test', 'H_bh_T1_e1_test', 'H_bh_T1_e2_test', 'H_bh_T1_e3_test','min_bh_FLAIR_test', 'med_bh_FLAIR_test', 'mid_bh_FLAIR_test', 'mid_vsmall_ratio_bh_FLAIR_test', 'mid_small_ratio_bh_FLAIR_test', 'mid_med_ratio_bh_FLAIR_test', 'mid_large_ratio_bh_FLAIR_test', 'mid_vsmall_ratio_bh_FLAIR_inc_test', 'mid_small_ratio_bh_FLAIR_inc_test', 'mid_med_ratio_bh_FLAIR_inc_test', 'mid_large_ratio_bh_FLAIR_inc_test', 'mean_bh_FLAIR_test', 'max_bh_FLAIR_test','var_bh_FLAIR_test','range_bh_FLAIR_test','H_bh_FLAIR_e1_test', 'H_bh_FLAIR_e2_test', 'H_bh_FLAIR_e3_test','min_bh_st_test', 'med_bh_st_test', 'mid_bh_st_test', 'mid_vsmall_ratio_bh_st_test','mid_small_ratio_bh_st_test', 'mid_med_ratio_bh_st_test', 'mid_large_ratio_bh_st_test', 'mid_vsmall_ratio_bh_st_inc_test','mid_small_ratio_bh_st_inc_test', 'mid_med_ratio_bh_st_inc_test', 'mid_large_ratio_bh_st_inc_test', 'mean_bh_st_test', 'max_bh_st_test', 'var_bh_st_test','range_bh_st_test', 'H_bh_st_e1_test', 'H_bh_st_e2_test', 'H_bh_st_e3_test', 'x_test', 'y_test', 'z_test', 'WMH_x_test', 'WMH_y_test' , 'WMH_z_test', 'CSF_test', 'GM_test', 'WM_test'])
	
	# Feature Importance Plot
	result_columns, perm_sorted_idx = feature_importance_plot(classifier, dataset_pd, Y_train, cv_splits)
	result_columns.to_numpy()
	perm_sorted_idx.to_numpy()
	np.save('/home/z5209394/Data/result_columns.npy', result_columns)
	np.save('/home/z5209394/Data/perm_sorted_idx.npy', perm_sorted_idx)
	print("perm importance acquired")
	#perm_idx_dataframe = np.load('/home/z5209394/Data/perm_idx_dataframe.npy')

	
	variable_arr = []
	for i in range(len(result_columns)):
		if (perm_sorted_idx[i]) > 0:
			variable_arr.append(str(result_columns[i]))
	print(variable_arr)
	len(variable_arr)

	# Density Plot 
	#dataset_combined = dataset_pd.copy()
	#dataset_combined['Lacune'] = Y_train
	#density_plots(perm_sorted_idx, dataset_combined)
	print("densities plotted")

	# Subset features
	dataset_subset = dataset_pd[variable_arr]

	variable_test_arr = []
	for variable in variable_arr:
		variable_test_arr.append(variable + '_test')

	dataset_test_subset = dataset_test_pd[variable_test_arr]
	classifier.fit(dataset_subset, Y_train)

	# Classifier predict
	predictions = (classifier.predict_proba(dataset_test_subset)[:,1] >= 0.15).astype(bool)
	print(confusion_matrix(Y_test_filtered, predictions))

	predictions = (classifier.predict_proba(dataset_test_subset)[:,1] >= mean_thresh).astype(bool)
	print(confusion_matrix(Y_test_filtered, predictions))
	np.save('/home/z5209394/Data/predictions.npy', predictions)
	np.save('/home/z5209394/Data/y_test_actual.npy', Y_test_filtered)
	
	predictions = (classifier.predict_proba(dataset_test_subset)[:,1] >= 0.5).astype(bool)
	print(confusion_matrix(Y_test_filtered, predictions))																																																																																																																																																																																																																																																																																																																															

if __name__ == '__main__':
	print("hello")
	main()
