import numpy as np
from numpy import linalg as LA
import cv2

def feature_gen_train(filterSize, kernel, X_train):
		brain = []
		min_T1 = []
		med_T1 = []
		mid_T1 = []
		mid_vsmall_ratio_T1 = []
		mid_small_ratio_T1 = []
		mid_med_ratio_T1 = []
		mid_large_ratio_T1 = []
		mid_vsmall_ratio_T1_inc = []
		mid_small_ratio_T1_inc = []
		mid_med_ratio_T1_inc = []
		mid_large_ratio_T1_inc = []
		mean_T1 = []
		max_T1 = []
		var_T1 = []
		range_T1 = []
		H_T1_e1 = []
		H_T1_e2 = []
		H_T1_e3 = []

		min_FLAIR = []
		med_FLAIR = []
		mid_FLAIR = []
		mid_vsmall_ratio_FLAIR = []
		mid_small_ratio_FLAIR = []
		mid_med_ratio_FLAIR = []
		mid_large_ratio_FLAIR = []
		mid_vsmall_ratio_FLAIR_inc = []
		mid_small_ratio_FLAIR_inc = []

		mid_med_ratio_FLAIR_inc = []
		mid_large_ratio_FLAIR_inc = []
		mean_FLAIR = []
		max_FLAIR = []
		var_FLAIR = []
		range_FLAIR = []
		H_FLAIR_e1 = []
		H_FLAIR_e2 = []
		H_FLAIR_e3 = []

		density_diff = []
		sum_soft_tiss_binary = []
		sum_percent_soft_tiss = []

		min_st = []
		med_st = []
		mid_st = []
		mid_vsmall_ratio_st = []
		mid_small_ratio_st = []
		mid_med_ratio_st = []
		mid_large_ratio_st = []
		mid_vsmall_ratio_st_inc = []
		mid_small_ratio_st_inc = []
		mid_med_ratio_st_inc = []
		mid_large_ratio_st_inc = []
		mean_st = []
		max_st = []
		var_st = []
		range_st = []
		H_st_e1 = []
		H_st_e2 = []
		H_st_e3 = []

		min_stm = []
		med_stm = []
		mid_stm = []
		mid_vsmall_ratio_stm = []
		mid_small_ratio_stm = []
		mid_med_ratio_stm = []
		mid_large_ratio_stm = []
		mid_vsmall_ratio_stm_inc = []
		mid_small_ratio_stm_inc = []
		mid_med_ratio_stm_inc = []
		mid_large_ratio_stm_inc = []
		mean_stm = []
		max_stm = []
		var_stm = []
		range_stm = []
		H_stm_e1 = []
		H_stm_e2 = []
		H_stm_e3 = []

		min_th_T1 = []
		med_th_T1 = []
		mid_th_T1 = []
		mid_vsmall_ratio_th_T1 = []
		mid_small_ratio_th_T1 = []
		mid_med_ratio_th_T1 = []
		mid_large_ratio_th_T1 = []
		mid_vsmall_ratio_th_T1_inc = []
		mid_small_ratio_th_T1_inc = []
		mid_med_ratio_th_T1_inc = []
		mid_large_ratio_th_T1_inc = []
		mean_th_T1 = []
		max_th_T1 = []
		var_th_T1 = []
		range_th_T1 = []
		H_th_T1_e1 = []
		H_th_T1_e2 = []
		H_th_T1_e3 = []


		min_th_FLAIR = []
		med_th_FLAIR = []
		mid_th_FLAIR = []
		mid_vsmall_ratio_th_FLAIR = []
		mid_small_ratio_th_FLAIR = []
		mid_med_ratio_th_FLAIR = []
		mid_large_ratio_th_FLAIR = []
		mid_vsmall_ratio_th_FLAIR_inc = []
		mid_small_ratio_th_FLAIR_inc = []
		mid_med_ratio_th_FLAIR_inc = []
		mid_large_ratio_th_FLAIR_inc = []
		mean_th_FLAIR = []
		max_th_FLAIR = []
		var_th_FLAIR = []
		range_th_FLAIR = []
		H_th_FLAIR_e1 = []
		H_th_FLAIR_e2 = []
		H_th_FLAIR_e3 = []


		min_th_st = []
		med_th_st = []
		mid_th_st = []
		mid_vsmall_ratio_th_st = []
		mid_small_ratio_th_st = []
		mid_med_ratio_th_st = []
		mid_large_ratio_th_st = []
		mid_vsmall_ratio_th_st_inc = []
		mid_small_ratio_th_st_inc = []
		mid_med_ratio_th_st_inc = []
		mid_large_ratio_th_st_inc = []
		mean_th_st = []
		max_th_st = []
		var_th_st = []
		range_th_st = []
		H_th_st_e1 = []
		H_th_st_e2 = []
		H_th_st_e3 = []

		min_bh_T1 = []
		med_bh_T1 = []
		mid_bh_T1 = []
		mid_vsmall_ratio_bh_T1 = []
		mid_small_ratio_bh_T1 = []
		mid_med_ratio_bh_T1 = []
		mid_large_ratio_bh_T1 = []
		mid_vsmall_ratio_bh_T1_inc = []
		mid_small_ratio_bh_T1_inc = []
		mid_med_ratio_bh_T1_inc = []
		mid_large_ratio_bh_T1_inc = []
		mean_bh_T1 = []
		max_bh_T1 = []
		var_bh_T1 = []
		range_bh_T1 = []
		H_bh_T1_e1 = []
		H_bh_T1_e2 = []
		H_bh_T1_e3 = []

		min_bh_FLAIR = []
		med_bh_FLAIR = []
		mid_bh_FLAIR = []
		mid_vsmall_ratio_bh_FLAIR = []
		mid_small_ratio_bh_FLAIR = []
		mid_med_ratio_bh_FLAIR = []
		mid_large_ratio_bh_FLAIR = []
		mid_vsmall_ratio_bh_FLAIR_inc = []
		mid_small_ratio_bh_FLAIR_inc = []
		mid_med_ratio_bh_FLAIR_inc = []
		mid_large_ratio_bh_FLAIR_inc = []
		mean_bh_FLAIR = []
		max_bh_FLAIR = []
		var_bh_FLAIR = []
		range_bh_FLAIR = []
		H_bh_FLAIR_e1 = []
		H_bh_FLAIR_e2 = []
		H_bh_FLAIR_e3 = []

		min_bh_st = []
		med_bh_st = []
		mid_bh_st = []
		mid_vsmall_ratio_bh_st = []
		mid_small_ratio_bh_st = []
		mid_med_ratio_bh_st = []
		mid_large_ratio_bh_st = []
		mid_vsmall_ratio_bh_st_inc = []
		mid_small_ratio_bh_st_inc = []
		mid_med_ratio_bh_st_inc = []
		mid_large_ratio_bh_st_inc = []
		mean_bh_st = []
		max_bh_st = []
		var_bh_st = []
		range_bh_st = []
		H_bh_st_e1 = []
		H_bh_st_e2 = []
		H_bh_st_e3 = []

		x = []
		y = []
		z = []

		WMH_x = []
		WMH_y = []
		WMH_z = []

		CSF_feat = []
		GM_feat = []
		WM_feat = []


		for index in range(len(X_train)):
				tophat_img_T1 = cv2.morphologyEx(X_train[index][4], cv2.MORPH_TOPHAT,kernel)
				tophat_img_FLAIR = cv2.morphologyEx(X_train[index][5], cv2.MORPH_TOPHAT,kernel)
				tophat_img_st = cv2.morphologyEx(X_train[index][7], cv2.MORPH_TOPHAT,kernel)
				blackhat_img_T1 = cv2.morphologyEx(X_train[index][4], cv2.MORPH_BLACKHAT,kernel)
				blackhat_img_FLAIR = cv2.morphologyEx(X_train[index][5], cv2.MORPH_BLACKHAT,kernel)
				blackhat_img_st = cv2.morphologyEx(X_train[index][7], cv2.MORPH_BLACKHAT,kernel)
				
				min_T1.append(np.min(X_train[index][4]))
				med_T1.append(np.median(X_train[index][4]))
				mid_T1.append(X_train[index][4][10,10,10])
				mid_vsmall_ratio_T1.append(np.mean(X_train[index][4][9:11, 9:11, 9:11])/((sum(sum(sum(X_train[index][4][8:12, 8:12, 8:12]))) - sum(sum(sum(X_train[index][4][9:11, 9:11, 9:11]))))/56))
				mid_small_ratio_T1.append(np.mean(X_train[index][4][8:12, 8:12, 8:12])/((sum(sum(sum(X_train[index][4][7:13, 7:13, 7:13]))) - sum(sum(sum(X_train[index][4][8:12, 8:12, 8:12]))))/152))
				mid_med_ratio_T1.append(np.mean(X_train[index][4][7:13, 7:13, 7:13])/((sum(sum(sum(X_train[index][4][6:14, 6:14, 6:14]))) - sum(sum(sum(X_train[index][4][7:13, 7:13, 7:13]))))/296))
				mid_large_ratio_T1.append(np.mean(X_train[index][4][6:14, 6:14, 6:14])/((sum(sum(sum(X_train[index][4][3:17, 3:17, 3:17]))) - sum(sum(sum(X_train[index][4][6:14, 6:14, 6:14]))))/2232))
				mid_vsmall_ratio_T1_inc.append(np.mean(X_train[index][4][9:11, 9:11, 9:11])/((sum(sum(sum(X_train[index][4][8:12, 8:12, 8:12]))))/64))
				mid_small_ratio_T1_inc.append(np.mean(X_train[index][4][8:12, 8:12, 8:12])/((sum(sum(sum(X_train[index][4][7:13, 7:13, 7:13]))))/216))
				mid_med_ratio_T1_inc.append(np.mean(X_train[index][4][7:13, 7:13, 7:13])/((sum(sum(sum(X_train[index][4][6:14, 6:14, 6:14]))))/512))
				mid_large_ratio_T1_inc.append(np.mean(X_train[index][4][6:14, 6:14, 6:14])/((sum(sum(sum(X_train[index][4][3:17, 3:17, 3:17]))))/2744))
				mean_T1.append(np.mean(X_train[index][4]))
				max_T1.append(np.max(X_train[index][4]))
				var_T1.append(np.var(X_train[index][4]))
				range_T1.append(np.max(X_train[index][4]) - np.min(X_train[index][4]))
				H_all_T1 = np.array([np.gradient(i) for i in np.gradient(X_train[index][4])]).transpose(2,3,4,0,1)
				H_T1_e1.append(LA.eig(H_all_T1[10, 10, 10])[0][0])
				H_T1_e2.append(LA.eig(H_all_T1[10, 10, 10])[0][1])
				H_T1_e3.append(LA.eig(H_all_T1[10, 10, 10])[0][2])
				
				min_FLAIR.append(np.min(X_train[index][5]))
				med_FLAIR.append(np.median(X_train[index][5]))
				mid_FLAIR.append(X_train[index][5][10,10,10])
				mid_vsmall_ratio_FLAIR.append(np.mean(X_train[index][5][9:11, 9:11, 9:11])/((sum(sum(sum(X_train[index][5][8:12, 8:12, 8:12]))) - sum(sum(sum(X_train[index][5][9:11, 9:11, 9:11]))))/56))
				mid_small_ratio_FLAIR.append(np.mean(X_train[index][5][8:12, 8:12, 8:12])/((sum(sum(sum(X_train[index][5][7:13, 7:13, 7:13]))) - sum(sum(sum(X_train[index][5][8:12, 8:12, 8:12]))))/152))
				mid_med_ratio_FLAIR.append(np.mean(X_train[index][5][7:13, 7:13, 7:13])/((sum(sum(sum(X_train[index][5][6:14, 6:14, 6:14]))) - sum(sum(sum(X_train[index][5][7:13, 7:13, 7:13]))))/296))
				mid_large_ratio_FLAIR.append(np.mean(X_train[index][5][6:14, 6:14, 6:14])/((sum(sum(sum(X_train[index][5][3:17, 3:17, 3:17]))) - sum(sum(sum(X_train[index][5][6:14, 6:14, 6:14]))))/2232))
				mid_vsmall_ratio_FLAIR_inc.append(np.mean(X_train[index][5][9:11, 9:11, 9:11])/((sum(sum(sum(X_train[index][5][8:12, 8:12, 8:12]))))/64))
				mid_small_ratio_FLAIR_inc.append(np.mean(X_train[index][5][8:12, 8:12, 8:12])/((sum(sum(sum(X_train[index][5][7:13, 7:13, 7:13]))))/216))
				mid_med_ratio_FLAIR_inc.append(np.mean(X_train[index][5][7:13, 7:13, 7:13])/((sum(sum(sum(X_train[index][5][6:14, 6:14, 6:14]))))/512))
				mid_large_ratio_FLAIR_inc.append(np.mean(X_train[index][5][6:14, 6:14, 6:14])/((sum(sum(sum(X_train[index][5][3:17, 3:17, 3:17]))))/2744))
				mean_FLAIR.append(np.mean(X_train[index][5]))
				max_FLAIR.append(np.max(X_train[index][5]))
				var_FLAIR.append(np.var(X_train[index][5]))
				range_FLAIR.append(np.max(X_train[index][5]) - np.min(X_train[index][5]))
				H_all_FLAIR = np.array([np.gradient(i) for i in np.gradient(X_train[index][5])]).transpose(2,3,4,0,1)
				H_FLAIR_e1.append(LA.eig(H_all_FLAIR[10, 10, 10])[0][0])
				H_FLAIR_e2.append(LA.eig(H_all_FLAIR[10, 10, 10])[0][1])
				H_FLAIR_e3.append(LA.eig(H_all_FLAIR[10, 10, 10])[0][2])
				
				density_diff.append(X_train[index][5][10,10,10] - X_train[index][4][10,10,10])
				sum_soft_tiss_binary.append(sum(sum(sum(X_train[index][6]))))
				sum_percent_soft_tiss.append(sum(sum(sum(X_train[index][7]))))
				
				min_st.append(np.min(X_train[index][6]))
				med_st.append(np.median(X_train[index][6]))
				mid_st.append(X_train[index][6][10,10,10])
				mid_vsmall_ratio_st.append(np.mean(X_train[index][6][9:11, 9:11, 9:11])/((sum(sum(sum(X_train[index][6][8:12, 8:12, 8:12]))) - sum(sum(sum(X_train[index][6][9:11, 9:11, 9:11]))))/56))
				mid_small_ratio_st.append(np.mean(X_train[index][6][8:12, 8:12, 8:12])/((sum(sum(sum(X_train[index][6][7:13, 7:13, 7:13]))) - sum(sum(sum(X_train[index][6][8:12, 8:12, 8:12]))))/152))
				mid_med_ratio_st.append(np.mean(X_train[index][6][7:13, 7:13, 7:13])/((sum(sum(sum(X_train[index][6][6:14, 6:14, 6:14]))) - sum(sum(sum(X_train[index][6][7:13, 7:13, 7:13]))))/296))
				mid_large_ratio_st.append(np.mean(X_train[index][6][6:14, 6:14, 6:14])/((sum(sum(sum(X_train[index][6][3:17, 3:17, 3:17]))) - sum(sum(sum(X_train[index][6][6:14, 6:14, 6:14]))))/2232))
				mid_vsmall_ratio_st_inc.append(np.mean(X_train[index][6][9:11, 9:11, 9:11])/((sum(sum(sum(X_train[index][6][8:12, 8:12, 8:12]))))/64))
				mid_small_ratio_st_inc.append(np.mean(X_train[index][6][8:12, 8:12, 8:12])/((sum(sum(sum(X_train[index][6][7:13, 7:13, 7:13]))))/216))
				mid_med_ratio_st_inc.append(np.mean(X_train[index][6][7:13, 7:13, 7:13])/((sum(sum(sum(X_train[index][6][6:14, 6:14, 6:14]))))/512))
				mid_large_ratio_st_inc.append(np.mean(X_train[index][6][6:14, 6:14, 6:14])/((sum(sum(sum(X_train[index][6][3:17, 3:17, 3:17]))))/2744))
				mean_st.append(np.mean(X_train[index][6]))
				max_st.append(np.max(X_train[index][6]))
				var_st.append(np.var(X_train[index][6]))
				range_st.append(np.max(X_train[index][6]) - np.min(X_train[index][6]))
				H_all_st = np.array([np.gradient(i) for i in np.gradient(X_train[index][6])]).transpose(2,3,4,0,1)
				H_st_e1.append(LA.eig(H_all_st[10, 10, 10])[0][0])
				H_st_e2.append(LA.eig(H_all_st[10, 10, 10])[0][1])
				H_st_e3.append(LA.eig(H_all_st[10, 10, 10])[0][2])
				
				min_stm.append(np.min(X_train[index][7]))
				med_stm.append(np.median(X_train[index][7]))
				mid_stm.append(X_train[index][7][10,10,10])
				mid_vsmall_ratio_stm.append(np.mean(X_train[index][7][9:11, 9:11, 9:11])/((sum(sum(sum(X_train[index][7][8:12, 8:12, 8:12]))) - sum(sum(sum(X_train[index][7][9:11, 9:11, 9:11]))))/56))
				mid_small_ratio_stm.append(np.mean(X_train[index][7][8:12, 8:12, 8:12])/((sum(sum(sum(X_train[index][7][7:13, 7:13, 7:13]))) - sum(sum(sum(X_train[index][7][8:12, 8:12, 8:12]))))/152))
				mid_med_ratio_stm.append(np.mean(X_train[index][7][7:13, 7:13, 7:13])/((sum(sum(sum(X_train[index][7][6:14, 6:14, 6:14]))) - sum(sum(sum(X_train[index][7][7:13, 7:13, 7:13]))))/296))
				mid_large_ratio_stm.append(np.mean(X_train[index][7][6:14, 6:14, 6:14])/((sum(sum(sum(X_train[index][7][3:17, 3:17, 3:17]))) - sum(sum(sum(X_train[index][7][6:14, 6:14, 6:14]))))/2232))
				mid_vsmall_ratio_stm_inc.append(np.mean(X_train[index][7][9:11, 9:11, 9:11])/((sum(sum(sum(X_train[index][7][8:12, 8:12, 8:12]))))/64))
				mid_small_ratio_stm_inc.append(np.mean(X_train[index][7][8:12, 8:12, 8:12])/((sum(sum(sum(X_train[index][7][7:13, 7:13, 7:13]))))/216))
				mid_med_ratio_stm_inc.append(np.mean(X_train[index][7][7:13, 7:13, 7:13])/((sum(sum(sum(X_train[index][7][6:14, 6:14, 6:14]))))/512))
				mid_large_ratio_stm_inc.append(np.mean(X_train[index][7][6:14, 6:14, 6:14])/((sum(sum(sum(X_train[index][7][3:17, 3:17, 3:17]))))/2744))
				mean_stm.append(np.mean(X_train[index][7]))
				max_stm.append(np.max(X_train[index][7]))
				var_stm.append(np.var(X_train[index][7]))
				range_stm.append(np.max(X_train[index][7]) - np.min(X_train[index][7]))
				H_all_stm = np.array([np.gradient(i) for i in np.gradient(X_train[index][7])]).transpose(2,3,4,0,1)
				H_stm_e1.append(LA.eig(H_all_stm[10, 10, 10])[0][0])
				H_stm_e2.append(LA.eig(H_all_stm[10, 10, 10])[0][1])
				H_stm_e3.append(LA.eig(H_all_stm[10, 10, 10])[0][2])
				
				
				min_th_T1.append(np.min(tophat_img_T1))
				med_th_T1.append(np.median(tophat_img_T1))
				mid_th_T1.append(tophat_img_T1[10,10,10])
				mid_vsmall_ratio_th_T1.append(np.mean(tophat_img_T1[9:11, 9:11, 9:11])/((sum(sum(sum(tophat_img_T1[8:12, 8:12, 8:12]))) - sum(sum(sum(tophat_img_T1[9:11, 9:11, 9:11]))))/56))
				mid_small_ratio_th_T1.append(np.mean(tophat_img_T1[8:12, 8:12, 8:12])/((sum(sum(sum(tophat_img_T1[7:13, 7:13, 7:13]))) - sum(sum(sum(tophat_img_T1[8:12, 8:12, 8:12]))))/152))
				mid_med_ratio_th_T1.append(np.mean(tophat_img_T1[7:13, 7:13, 7:13])/((sum(sum(sum(tophat_img_T1[6:14, 6:14, 6:14]))) - sum(sum(sum(tophat_img_T1[7:13, 7:13, 7:13]))))/296))
				mid_large_ratio_th_T1.append(np.mean(tophat_img_T1[6:14, 6:14, 6:14])/((sum(sum(sum(tophat_img_T1[3:17, 3:17, 3:17]))) - sum(sum(sum(tophat_img_T1[6:14, 6:14, 6:14]))))/2232))
				mid_vsmall_ratio_th_T1_inc.append(np.mean(tophat_img_T1[9:11, 9:11, 9:11])/((sum(sum(sum(tophat_img_T1[8:12, 8:12, 8:12]))))/64))
				mid_small_ratio_th_T1_inc.append(np.mean(tophat_img_T1[8:12, 8:12, 8:12])/((sum(sum(sum(tophat_img_T1[7:13, 7:13, 7:13]))))/216))
				mid_med_ratio_th_T1_inc.append(np.mean(tophat_img_T1[7:13, 7:13, 7:13])/((sum(sum(sum(tophat_img_T1[6:14, 6:14, 6:14]))))/512))
				mid_large_ratio_th_T1_inc.append(np.mean(tophat_img_T1[6:14, 6:14, 6:14])/((sum(sum(sum(tophat_img_T1[3:17, 3:17, 3:17]))))/2744))
				mean_th_T1.append(np.mean(tophat_img_T1))
				max_th_T1.append(np.max(tophat_img_T1))
				var_th_T1.append(np.var(tophat_img_T1))
				range_th_T1.append(np.max(tophat_img_T1) - np.min(tophat_img_T1))
				H_all_th_T1 = np.array([np.gradient(i) for i in np.gradient(tophat_img_T1)]).transpose(2,3,4,0,1)
				H_th_T1_e1.append(LA.eig(H_all_th_T1[10, 10, 10])[0][0])
				H_th_T1_e2.append(LA.eig(H_all_th_T1[10, 10, 10])[0][1])
				H_th_T1_e3.append(LA.eig(H_all_th_T1[10, 10, 10])[0][2])
				
				min_th_FLAIR.append(np.min(tophat_img_FLAIR))
				med_th_FLAIR.append(np.median(tophat_img_FLAIR))
				mid_th_FLAIR.append(tophat_img_FLAIR[10,10,10])
				mid_vsmall_ratio_th_FLAIR.append(np.mean(tophat_img_FLAIR[9:11, 9:11, 9:11])/((sum(sum(sum(tophat_img_FLAIR[8:12, 8:12, 8:12]))) - sum(sum(sum(tophat_img_FLAIR[9:11, 9:11, 9:11]))))/56))
				mid_small_ratio_th_FLAIR.append(np.mean(tophat_img_FLAIR[8:12, 8:12, 8:12])/((sum(sum(sum(tophat_img_FLAIR[7:13, 7:13, 7:13]))) - sum(sum(sum(tophat_img_FLAIR[8:12, 8:12, 8:12]))))/152))
				mid_med_ratio_th_FLAIR.append(np.mean(tophat_img_FLAIR[7:13, 7:13, 7:13])/((sum(sum(sum(tophat_img_FLAIR[6:14, 6:14, 6:14]))) - sum(sum(sum(tophat_img_FLAIR[7:13, 7:13, 7:13]))))/296))
				mid_large_ratio_th_FLAIR.append(np.mean(tophat_img_FLAIR[6:14, 6:14, 6:14])/((sum(sum(sum(tophat_img_FLAIR[3:17, 3:17, 3:17]))) - sum(sum(sum(tophat_img_FLAIR[6:14, 6:14, 6:14]))))/2232))
				mid_vsmall_ratio_th_FLAIR_inc.append(np.mean(tophat_img_FLAIR[9:11, 9:11, 9:11])/((sum(sum(sum(tophat_img_FLAIR[8:12, 8:12, 8:12]))))/64))
				mid_small_ratio_th_FLAIR_inc.append(np.mean(tophat_img_FLAIR[8:12, 8:12, 8:12])/((sum(sum(sum(tophat_img_FLAIR[7:13, 7:13, 7:13]))))/216))
				mid_med_ratio_th_FLAIR_inc.append(np.mean(tophat_img_FLAIR[7:13, 7:13, 7:13])/((sum(sum(sum(tophat_img_FLAIR[6:14, 6:14, 6:14]))))/512))
				mid_large_ratio_th_FLAIR_inc.append(np.mean(tophat_img_FLAIR[6:14, 6:14, 6:14])/((sum(sum(sum(tophat_img_FLAIR[3:17, 3:17, 3:17]))))/2744))
				mean_th_FLAIR.append(np.mean(tophat_img_FLAIR))
				max_th_FLAIR.append(np.max(tophat_img_FLAIR))
				var_th_FLAIR.append(np.var(tophat_img_FLAIR))
				range_th_FLAIR.append(np.max(tophat_img_FLAIR) - np.min(tophat_img_FLAIR))
				H_all_th_FLAIR = np.array([np.gradient(i) for i in np.gradient(tophat_img_FLAIR)]).transpose(2,3,4,0,1)
				H_th_FLAIR_e1.append(LA.eig(H_all_th_FLAIR[10, 10, 10])[0][0])
				H_th_FLAIR_e2.append(LA.eig(H_all_th_FLAIR[10, 10, 10])[0][1])
				H_th_FLAIR_e3.append(LA.eig(H_all_th_FLAIR[10, 10, 10])[0][2])
				
				min_th_st.append(np.min(tophat_img_st))
				med_th_st.append(np.median(tophat_img_st))
				mid_th_st.append(tophat_img_st[10,10,10])
				mid_vsmall_ratio_th_st.append(np.mean(tophat_img_st[9:11, 9:11, 9:11])/((sum(sum(sum(tophat_img_st[8:12, 8:12, 8:12]))) - sum(sum(sum(tophat_img_st[9:11, 9:11, 9:11]))))/56))
				mid_small_ratio_th_st.append(np.mean(tophat_img_st[8:12, 8:12, 8:12])/((sum(sum(sum(tophat_img_st[7:13, 7:13, 7:13]))) - sum(sum(sum(tophat_img_st[8:12, 8:12, 8:12]))))/152))
				mid_med_ratio_th_st.append(np.mean(tophat_img_st[7:13, 7:13, 7:13])/((sum(sum(sum(tophat_img_st[6:14, 6:14, 6:14]))) - sum(sum(sum(tophat_img_st[7:13, 7:13, 7:13]))))/296))
				mid_large_ratio_th_st.append(np.mean(tophat_img_st[6:14, 6:14, 6:14])/((sum(sum(sum(tophat_img_st[3:17, 3:17, 3:17]))) - sum(sum(sum(tophat_img_st[6:14, 6:14, 6:14]))))/2232))
				mid_vsmall_ratio_th_st_inc.append(np.mean(tophat_img_st[9:11, 9:11, 9:11])/((sum(sum(sum(tophat_img_st[8:12, 8:12, 8:12]))))/64))
				mid_small_ratio_th_st_inc.append(np.mean(tophat_img_st[8:12, 8:12, 8:12])/((sum(sum(sum(tophat_img_st[7:13, 7:13, 7:13]))))/216))
				mid_med_ratio_th_st_inc.append(np.mean(tophat_img_st[7:13, 7:13, 7:13])/((sum(sum(sum(tophat_img_st[6:14, 6:14, 6:14]))))/512))
				mid_large_ratio_th_st_inc.append(np.mean(tophat_img_st[6:14, 6:14, 6:14])/((sum(sum(sum(tophat_img_st[3:17, 3:17, 3:17]))))/2744))
				mean_th_st.append(np.mean(tophat_img_st))
				max_th_st.append(np.max(tophat_img_st))
				var_th_st.append(np.var(tophat_img_st))
				range_th_st.append(np.max(tophat_img_st) - np.min(tophat_img_st))
				H_all_th_st = np.array([np.gradient(i) for i in np.gradient(tophat_img_st)]).transpose(2,3,4,0,1)
				H_th_st_e1.append(LA.eig(H_all_th_st[10, 10, 10])[0][0])
				H_th_st_e2.append(LA.eig(H_all_th_st[10, 10, 10])[0][1])
				H_th_st_e3.append(LA.eig(H_all_th_st[10, 10, 10])[0][2])
				
				min_bh_T1.append(np.min(blackhat_img_T1))
				med_bh_T1.append(np.median(blackhat_img_T1))
				mid_bh_T1.append(blackhat_img_T1[10,10,10])
				mid_vsmall_ratio_bh_T1.append(np.mean(blackhat_img_T1[9:11, 9:11, 9:11])/((sum(sum(sum(blackhat_img_T1[8:12, 8:12, 8:12]))) - sum(sum(sum(blackhat_img_T1[9:11, 9:11, 9:11]))))/56))
				mid_small_ratio_bh_T1.append(np.mean(blackhat_img_T1[8:12, 8:12, 8:12])/((sum(sum(sum(blackhat_img_T1[7:13, 7:13, 7:13]))) - sum(sum(sum(blackhat_img_T1[8:12, 8:12, 8:12]))))/152))
				mid_med_ratio_bh_T1.append(np.mean(blackhat_img_T1[7:13, 7:13, 7:13])/((sum(sum(sum(blackhat_img_T1[6:14, 6:14, 6:14]))) - sum(sum(sum(blackhat_img_T1[7:13, 7:13, 7:13]))))/296))
				mid_large_ratio_bh_T1.append(np.mean(blackhat_img_T1[6:14, 6:14, 6:14])/((sum(sum(sum(blackhat_img_T1[3:17, 3:17, 3:17]))) - sum(sum(sum(blackhat_img_T1[6:14, 6:14, 6:14]))))/2232))
				mid_vsmall_ratio_bh_T1_inc.append(np.mean(blackhat_img_T1[9:11, 9:11, 9:11])/((sum(sum(sum(blackhat_img_T1[8:12, 8:12, 8:12]))))/64))
				mid_small_ratio_bh_T1_inc.append(np.mean(blackhat_img_T1[8:12, 8:12, 8:12])/((sum(sum(sum(blackhat_img_T1[7:13, 7:13, 7:13]))))/216))
				mid_med_ratio_bh_T1_inc.append(np.mean(blackhat_img_T1[7:13, 7:13, 7:13])/((sum(sum(sum(blackhat_img_T1[6:14, 6:14, 6:14]))))/512))
				mid_large_ratio_bh_T1_inc.append(np.mean(blackhat_img_T1[6:14, 6:14, 6:14])/((sum(sum(sum(blackhat_img_T1[3:17, 3:17, 3:17]))))/2744))
				mean_bh_T1.append(np.mean(blackhat_img_T1))
				max_bh_T1.append(np.max(blackhat_img_T1))
				var_bh_T1.append(np.var(blackhat_img_T1))
				range_bh_T1.append(np.max(blackhat_img_T1) - np.min(blackhat_img_T1))
				H_all_bh_T1 = np.array([np.gradient(i) for i in np.gradient(blackhat_img_T1)]).transpose(2,3,4,0,1)
				H_bh_T1_e1.append(LA.eig(H_all_bh_T1[10, 10, 10])[0][0])
				H_bh_T1_e2.append(LA.eig(H_all_bh_T1[10, 10, 10])[0][1])
				H_bh_T1_e3.append(LA.eig(H_all_bh_T1[10, 10, 10])[0][2])
				
				min_bh_FLAIR.append(np.min(blackhat_img_FLAIR))
				med_bh_FLAIR.append(np.median(blackhat_img_FLAIR))
				mid_bh_FLAIR.append(blackhat_img_FLAIR[10,10,10])
				mid_vsmall_ratio_bh_FLAIR.append(np.mean(blackhat_img_FLAIR[9:11, 9:11, 9:11])/((sum(sum(sum(blackhat_img_FLAIR[8:12, 8:12, 8:12]))) - sum(sum(sum(blackhat_img_FLAIR[9:11, 9:11, 9:11]))))/56))
				mid_small_ratio_bh_FLAIR.append(np.mean(blackhat_img_FLAIR[8:12, 8:12, 8:12])/((sum(sum(sum(blackhat_img_FLAIR[7:13, 7:13, 7:13]))) - sum(sum(sum(blackhat_img_FLAIR[8:12, 8:12, 8:12]))))/152))
				mid_med_ratio_bh_FLAIR.append(np.mean(blackhat_img_FLAIR[7:13, 7:13, 7:13])/((sum(sum(sum(blackhat_img_FLAIR[6:14, 6:14, 6:14]))) - sum(sum(sum(blackhat_img_FLAIR[7:13, 7:13, 7:13]))))/296))
				mid_large_ratio_bh_FLAIR.append(np.mean(blackhat_img_FLAIR[6:14, 6:14, 6:14])/((sum(sum(sum(blackhat_img_FLAIR[3:17, 3:17, 3:17]))) - sum(sum(sum(blackhat_img_FLAIR[6:14, 6:14, 6:14]))))/2232))
				mid_vsmall_ratio_bh_FLAIR_inc.append(np.mean(blackhat_img_FLAIR[9:11, 9:11, 9:11])/((sum(sum(sum(blackhat_img_FLAIR[8:12, 8:12, 8:12]))))/64))
				mid_small_ratio_bh_FLAIR_inc.append(np.mean(blackhat_img_FLAIR[8:12, 8:12, 8:12])/((sum(sum(sum(blackhat_img_FLAIR[7:13, 7:13, 7:13]))))/216))
				mid_med_ratio_bh_FLAIR_inc.append(np.mean(blackhat_img_FLAIR[7:13, 7:13, 7:13])/((sum(sum(sum(blackhat_img_FLAIR[6:14, 6:14, 6:14]))))/512))
				mid_large_ratio_bh_FLAIR_inc.append(np.mean(blackhat_img_FLAIR[6:14, 6:14, 6:14])/((sum(sum(sum(blackhat_img_FLAIR[3:17, 3:17, 3:17]))))/2744))
				mean_bh_FLAIR.append(np.mean(blackhat_img_FLAIR))
				max_bh_FLAIR.append(np.max(blackhat_img_FLAIR))
				var_bh_FLAIR.append(np.var(blackhat_img_FLAIR))
				range_bh_FLAIR.append(np.max(blackhat_img_FLAIR) - np.min(blackhat_img_FLAIR))
				H_all_bh_FLAIR = np.array([np.gradient(i) for i in np.gradient(blackhat_img_FLAIR)]).transpose(2,3,4,0,1)
				H_bh_FLAIR_e1.append(LA.eig(H_all_bh_FLAIR[10, 10, 10])[0][0])
				H_bh_FLAIR_e2.append(LA.eig(H_all_bh_FLAIR[10, 10, 10])[0][1])
				H_bh_FLAIR_e3.append(LA.eig(H_all_bh_FLAIR[10, 10, 10])[0][2])
				
				min_bh_st.append(np.min(blackhat_img_st))
				med_bh_st.append(np.median(blackhat_img_st))
				mid_bh_st.append(blackhat_img_st[10,10,10])
				mid_vsmall_ratio_bh_st.append(np.mean(blackhat_img_st[9:11, 9:11, 9:11])/((sum(sum(sum(blackhat_img_st[8:12, 8:12, 8:12]))) - sum(sum(sum(blackhat_img_st[9:11, 9:11, 9:11]))))/56))
				mid_small_ratio_bh_st.append(np.mean(blackhat_img_st[8:12, 8:12, 8:12])/((sum(sum(sum(blackhat_img_st[7:13, 7:13, 7:13]))) - sum(sum(sum(blackhat_img_st[8:12, 8:12, 8:12]))))/152))
				mid_med_ratio_bh_st.append(np.mean(blackhat_img_st[7:13, 7:13, 7:13])/((sum(sum(sum(blackhat_img_st[6:14, 6:14, 6:14]))) - sum(sum(sum(blackhat_img_st[7:13, 7:13, 7:13]))))/296))
				mid_large_ratio_bh_st.append(np.mean(blackhat_img_st[6:14, 6:14, 6:14])/((sum(sum(sum(blackhat_img_st[3:17, 3:17, 3:17]))) - sum(sum(sum(blackhat_img_st[6:14, 6:14, 6:14]))))/2232))
				mid_vsmall_ratio_bh_st_inc.append(np.mean(blackhat_img_st[9:11, 9:11, 9:11])/((sum(sum(sum(blackhat_img_st[8:12, 8:12, 8:12]))))/64))
				mid_small_ratio_bh_st_inc.append(np.mean(blackhat_img_st[8:12, 8:12, 8:12])/((sum(sum(sum(blackhat_img_st[7:13, 7:13, 7:13]))))/216))
				mid_med_ratio_bh_st_inc.append(np.mean(blackhat_img_st[7:13, 7:13, 7:13])/((sum(sum(sum(blackhat_img_st[6:14, 6:14, 6:14]))))/512))
				mid_large_ratio_bh_st_inc.append(np.mean(blackhat_img_st[6:14, 6:14, 6:14])/((sum(sum(sum(blackhat_img_st[3:17, 3:17, 3:17]))))/2744))
				mean_bh_st.append(np.mean(blackhat_img_st))
				max_bh_st.append(np.max(blackhat_img_st))
				var_bh_st.append(np.var(blackhat_img_st))
				range_bh_st.append(np.max(blackhat_img_st) - np.min(blackhat_img_st))
				H_all_bh_st = np.array([np.gradient(i) for i in np.gradient(blackhat_img_st)]).transpose(2,3,4,0,1)
				H_bh_st_e1.append(LA.eig(H_all_bh_st[10, 10, 10])[0][0])
				H_bh_st_e2.append(LA.eig(H_all_bh_st[10, 10, 10])[0][1])
				H_bh_st_e3.append(LA.eig(H_all_bh_st[10, 10, 10])[0][2])
				
				WMH_x.append(np.count_nonzero(X_train[index][5][10,:,:]>0.45))
				WMH_y.append(np.count_nonzero(X_train[index][5][:,10,:]>0.45))
				WMH_z.append(np.count_nonzero(X_train[index][5][:,:,10]>0.45))
				
				brain.append(X_train[index][0])
				x.append(X_train[index][1])
				y.append(X_train[index][2])
				z.append(X_train[index][3])

				#edges = canny(tophat_img)
				CSF_feat.append(np.mean(X_train[index][8]))
				GM_feat.append(np.mean(X_train[index][10]))
				WM_feat.append(np.mean(X_train[index][9]))
				print(np.mean(X_train[index][10]))
				print(np.mean(X_train[index][9]))
		return brain, min_T1, med_T1, mid_T1, mid_vsmall_ratio_T1, mid_small_ratio_T1, mid_med_ratio_T1, mid_large_ratio_T1, mid_vsmall_ratio_T1_inc, mid_small_ratio_T1_inc, mid_med_ratio_T1_inc, mid_large_ratio_T1_inc, mean_T1, max_T1, var_T1, range_T1, H_T1_e1, H_T1_e2, H_T1_e3, min_FLAIR, mid_FLAIR, med_FLAIR, mid_vsmall_ratio_FLAIR, mid_small_ratio_FLAIR, mid_med_ratio_FLAIR, mid_large_ratio_FLAIR, mid_vsmall_ratio_FLAIR_inc, mid_small_ratio_FLAIR_inc, mid_med_ratio_FLAIR_inc, mid_large_ratio_FLAIR_inc, mean_FLAIR, max_FLAIR, var_FLAIR, range_FLAIR, H_FLAIR_e1, H_FLAIR_e2, H_FLAIR_e3, density_diff, sum_soft_tiss_binary, sum_percent_soft_tiss, min_st, med_st, mid_st, mid_vsmall_ratio_st, mid_small_ratio_st, mid_med_ratio_st, mid_large_ratio_st, mid_vsmall_ratio_st_inc, mid_small_ratio_st_inc, mid_med_ratio_st_inc, mid_large_ratio_st_inc, mean_st, max_st, var_st, range_st, H_st_e1, H_st_e2, H_st_e3, min_stm, med_stm, mid_stm, mid_vsmall_ratio_stm, mid_small_ratio_stm, mid_med_ratio_stm, mid_large_ratio_stm, mid_vsmall_ratio_stm_inc, mid_small_ratio_stm_inc, mid_med_ratio_stm_inc, mid_large_ratio_stm_inc, mean_stm, max_stm, var_stm, range_stm, H_stm_e1, H_stm_e2, H_stm_e3, min_th_T1, med_th_T1, mid_th_T1, mid_vsmall_ratio_th_T1, mid_small_ratio_th_T1, mid_med_ratio_th_T1, mid_large_ratio_th_T1, mid_vsmall_ratio_th_T1_inc, mid_small_ratio_th_T1_inc, mid_med_ratio_th_T1_inc, mid_large_ratio_th_T1_inc, mean_th_T1, max_th_T1, var_th_T1, range_th_T1, H_th_T1_e1, H_th_T1_e2, H_th_T1_e3, min_th_FLAIR, med_th_FLAIR, mid_th_FLAIR, mid_vsmall_ratio_th_FLAIR, mid_small_ratio_th_FLAIR, mid_med_ratio_th_FLAIR, mid_large_ratio_th_FLAIR, mid_vsmall_ratio_th_FLAIR_inc, mid_small_ratio_th_FLAIR_inc, mid_med_ratio_th_FLAIR_inc, mid_large_ratio_th_FLAIR_inc, mean_th_FLAIR, max_th_FLAIR, var_th_FLAIR, range_th_FLAIR,H_th_FLAIR_e1, H_th_FLAIR_e2, H_th_FLAIR_e3, min_th_st, med_th_st, mid_th_st, mid_vsmall_ratio_th_st, mid_small_ratio_th_st, mid_med_ratio_th_st, mid_large_ratio_th_st, mid_vsmall_ratio_th_st_inc, mid_small_ratio_th_st_inc, mid_med_ratio_th_st_inc, mid_large_ratio_th_st_inc, mean_th_st, max_th_st, var_th_st, range_th_st, H_th_st_e1, H_th_st_e2, H_th_st_e3, min_bh_T1, med_bh_T1, mid_bh_T1, mid_vsmall_ratio_bh_T1,mid_small_ratio_bh_T1, mid_med_ratio_bh_T1, mid_large_ratio_bh_T1, mid_vsmall_ratio_bh_T1_inc, mid_small_ratio_bh_T1_inc, mid_med_ratio_bh_T1_inc, mid_large_ratio_bh_T1_inc, mean_bh_T1, max_bh_T1, var_bh_T1, range_bh_T1, H_bh_T1_e1, H_bh_T1_e2, H_bh_T1_e3, min_bh_FLAIR, med_bh_FLAIR, mid_bh_FLAIR, mid_vsmall_ratio_bh_FLAIR, mid_small_ratio_bh_FLAIR, mid_med_ratio_bh_FLAIR, mid_large_ratio_bh_FLAIR, mid_vsmall_ratio_bh_FLAIR_inc, mid_small_ratio_bh_FLAIR_inc, mid_med_ratio_bh_FLAIR_inc, mid_large_ratio_bh_FLAIR_inc, mean_bh_FLAIR, max_bh_FLAIR, var_bh_FLAIR, range_bh_FLAIR, H_bh_FLAIR_e1, H_bh_FLAIR_e2, H_bh_FLAIR_e3, min_bh_st, med_bh_st, mid_bh_st, mid_vsmall_ratio_bh_st, mid_small_ratio_bh_st, mid_med_ratio_bh_st, mid_large_ratio_bh_st, mid_vsmall_ratio_bh_st_inc, mid_small_ratio_bh_st_inc, mid_med_ratio_bh_st_inc, mid_large_ratio_bh_st_inc, mean_bh_st, max_bh_st, var_bh_st, range_bh_st, H_bh_st_e1, H_bh_st_e2, H_bh_st_e3, x, y, z, WMH_x, WMH_y, WMH_z, CSF_feat, GM_feat, WM_feat

def feature_gen_test(filterSize, kernel, X_test):
		brain_test = []
		min_T1_test = []
		med_T1_test = []
		mid_T1_test = []
		mid_vsmall_ratio_T1_test = []
		mid_small_ratio_T1_test = []
		mid_med_ratio_T1_test = []
		mid_large_ratio_T1_test = []
		mid_vsmall_ratio_T1_inc_test = []
		mid_small_ratio_T1_inc_test = []
		mid_med_ratio_T1_inc_test = []
		mid_large_ratio_T1_inc_test = []
		mean_T1_test = []
		max_T1_test = []
		var_T1_test = []
		range_T1_test = []
		H_T1_e1_test = []
		H_T1_e2_test = []
		H_T1_e3_test = []

		min_FLAIR_test = []
		med_FLAIR_test = []
		mid_FLAIR_test = []
		mid_vsmall_ratio_FLAIR_test = []
		mid_small_ratio_FLAIR_test = []
		mid_med_ratio_FLAIR_test = []
		mid_large_ratio_FLAIR_test = []
		mid_vsmall_ratio_FLAIR_inc_test = []
		mid_small_ratio_FLAIR_inc_test = []
		mid_med_ratio_FLAIR_inc_test = []
		mid_large_ratio_FLAIR_inc_test = []
		mean_FLAIR_test = []
		max_FLAIR_test = []
		var_FLAIR_test = []
		range_FLAIR_test = []
		H_FLAIR_e1_test = []
		H_FLAIR_e2_test = []
		H_FLAIR_e3_test = []

		density_diff_test = []
		sum_soft_tiss_binary_test = []
		sum_percent_soft_tiss_test = []

		min_st_test = []
		med_st_test = []
		mid_st_test = []
		mid_vsmall_ratio_st_test = []
		mid_small_ratio_st_test = []
		mid_med_ratio_st_test = []
		mid_large_ratio_st_test = []
		mid_vsmall_ratio_st_inc_test = []
		mid_small_ratio_st_inc_test = []
		mid_med_ratio_st_inc_test = []
		mid_large_ratio_st_inc_test = []
		mean_st_test = []
		max_st_test = []
		var_st_test = []
		range_st_test = []
		H_st_e1_test = []
		H_st_e2_test = []
		H_st_e3_test = []

		min_stm_test = []
		med_stm_test = []
		mid_stm_test = []
		mid_vsmall_ratio_stm_test = []
		mid_small_ratio_stm_test = []
		mid_med_ratio_stm_test = []
		mid_large_ratio_stm_test = []
		mid_vsmall_ratio_stm_inc_test = []
		mid_small_ratio_stm_inc_test = []
		mid_med_ratio_stm_inc_test = []
		mid_large_ratio_stm_inc_test = []
		mean_stm_test = []
		max_stm_test = []
		var_stm_test = []
		range_stm_test = []
		H_stm_e1_test = []
		H_stm_e2_test = []
		H_stm_e3_test = []

		min_th_T1_test = []
		med_th_T1_test = []
		mid_th_T1_test = []
		mid_vsmall_ratio_th_T1_test = []
		mid_small_ratio_th_T1_test = []
		mid_med_ratio_th_T1_test = []
		mid_large_ratio_th_T1_test = []
		mid_vsmall_ratio_th_T1_inc_test = []
		mid_small_ratio_th_T1_inc_test = []
		mid_med_ratio_th_T1_inc_test = []
		mid_large_ratio_th_T1_inc_test = []
		mean_th_T1_test = []
		max_th_T1_test = []
		var_th_T1_test = []
		range_th_T1_test = []
		H_th_T1_e1_test = []
		H_th_T1_e2_test = []
		H_th_T1_e3_test = []

		min_th_FLAIR_test = []
		med_th_FLAIR_test = []
		mid_th_FLAIR_test = []
		mid_vsmall_ratio_th_FLAIR_test = []
		mid_small_ratio_th_FLAIR_test = []
		mid_med_ratio_th_FLAIR_test = []
		mid_large_ratio_th_FLAIR_test = []
		mid_vsmall_ratio_th_FLAIR_inc_test = []
		mid_small_ratio_th_FLAIR_inc_test = []
		mid_med_ratio_th_FLAIR_inc_test = []
		mid_large_ratio_th_FLAIR_inc_test = []
		mean_th_FLAIR_test = []
		max_th_FLAIR_test = []
		var_th_FLAIR_test = []
		range_th_FLAIR_test = []
		H_th_FLAIR_e1_test = []
		H_th_FLAIR_e2_test = []
		H_th_FLAIR_e3_test = []

		min_th_st_test = []
		med_th_st_test = []
		mid_th_st_test = []
		mid_vsmall_ratio_th_st_test = []
		mid_small_ratio_th_st_test = []
		mid_med_ratio_th_st_test = []
		mid_large_ratio_th_st_test = []
		mid_vsmall_ratio_th_st_inc_test = []
		mid_small_ratio_th_st_inc_test = []
		mid_med_ratio_th_st_inc_test = []
		mid_large_ratio_th_st_inc_test = []
		mean_th_st_test = []
		max_th_st_test = []
		var_th_st_test = []
		range_th_st_test = []
		H_th_st_e1_test = []
		H_th_st_e2_test = []
		H_th_st_e3_test = []

		min_bh_T1_test = []
		med_bh_T1_test = []
		mid_bh_T1_test = []
		mid_vsmall_ratio_bh_T1_test = []
		mid_small_ratio_bh_T1_test = []
		mid_med_ratio_bh_T1_test = []
		mid_large_ratio_bh_T1_test = []
		mid_vsmall_ratio_bh_T1_inc_test = []
		mid_small_ratio_bh_T1_inc_test = []
		mid_med_ratio_bh_T1_inc_test = []
		mid_large_ratio_bh_T1_inc_test = []
		mean_bh_T1_test = []
		max_bh_T1_test = []
		var_bh_T1_test = []
		range_bh_T1_test = []
		H_bh_T1_e1_test = []
		H_bh_T1_e2_test = []
		H_bh_T1_e3_test = []

		min_bh_FLAIR_test = []
		med_bh_FLAIR_test = []
		mid_bh_FLAIR_test = []
		mid_vsmall_ratio_bh_FLAIR_test = []
		mid_small_ratio_bh_FLAIR_test = []
		mid_med_ratio_bh_FLAIR_test = []
		mid_large_ratio_bh_FLAIR_test = []
		mid_vsmall_ratio_bh_FLAIR_inc_test = []
		mid_small_ratio_bh_FLAIR_inc_test = []
		mid_med_ratio_bh_FLAIR_inc_test = []
		mid_large_ratio_bh_FLAIR_inc_test = []
		mean_bh_FLAIR_test = []
		max_bh_FLAIR_test = []
		var_bh_FLAIR_test = []
		range_bh_FLAIR_test = []
		H_bh_FLAIR_e1_test = []
		H_bh_FLAIR_e2_test = []
		H_bh_FLAIR_e3_test = []

		min_bh_st_test = []
		med_bh_st_test = []
		mid_bh_st_test = []
		mid_vsmall_ratio_bh_st_test = []
		mid_small_ratio_bh_st_test = []
		mid_med_ratio_bh_st_test = []
		mid_large_ratio_bh_st_test = []
		mid_vsmall_ratio_bh_st_inc_test = []
		mid_small_ratio_bh_st_inc_test = []
		mid_med_ratio_bh_st_inc_test = []
		mid_large_ratio_bh_st_inc_test = []
		mean_bh_st_test = []
		max_bh_st_test = []
		var_bh_st_test = []
		range_bh_st_test = []
		H_bh_st_e1_test = []
		H_bh_st_e2_test = []
		H_bh_st_e3_test = []

		WMH_x_test = []
		WMH_y_test = []
		WMH_z_test = []

		x_test = []
		y_test = []
		z_test = []

		CSF_feat_test = []
		GM_feat_test = []
		WM_feat_test = []

		for index in range(len(X_test)):
				tophat_img_T1_test = cv2.morphologyEx(X_test[index][4], cv2.MORPH_TOPHAT,kernel)
				tophat_img_FLAIR_test = cv2.morphologyEx(X_test[index][5], cv2.MORPH_TOPHAT,kernel)
				tophat_img_st_test = cv2.morphologyEx(X_test[index][7], cv2.MORPH_TOPHAT,kernel)
				blackhat_img_T1_test = cv2.morphologyEx(X_test[index][4], cv2.MORPH_BLACKHAT,kernel)
				blackhat_img_FLAIR_test = cv2.morphologyEx(X_test[index][5], cv2.MORPH_BLACKHAT,kernel)
				blackhat_img_st_test = cv2.morphologyEx(X_test[index][7], cv2.MORPH_BLACKHAT,kernel)
				
				min_T1_test.append(np.min(X_test[index][4]))
				med_T1_test.append(np.median(X_test[index][4]))
				mid_T1_test.append(X_test[index][4][10,10,10])
				mid_vsmall_ratio_T1_test.append(np.mean(X_test[index][4][9:11, 9:11, 9:11])/((sum(sum(sum(X_test[index][4][8:12, 8:12, 8:12]))) - sum(sum(sum(X_test[index][4][9:11, 9:11, 9:11]))))/56))
				mid_small_ratio_T1_test.append(np.mean(X_test[index][4][8:12, 8:12, 8:12])/((sum(sum(sum(X_test[index][4][7:13, 7:13, 7:13]))) - sum(sum(sum(X_test[index][4][8:12, 8:12, 8:12]))))/152))
				mid_med_ratio_T1_test.append(np.mean(X_test[index][4][7:13, 7:13, 7:13])/((sum(sum(sum(X_test[index][4][6:14, 6:14, 6:14]))) - sum(sum(sum(X_test[index][4][7:13, 7:13, 7:13]))))/296))
				mid_large_ratio_T1_test.append(np.mean(X_test[index][4][6:14, 6:14, 6:14])/((sum(sum(sum(X_test[index][4][3:17, 3:17, 3:17]))) - sum(sum(sum(X_test[index][4][6:14, 6:14, 6:14]))))/2232))
				mid_vsmall_ratio_T1_inc_test.append(np.mean(X_test[index][4][9:11, 9:11, 9:11])/((sum(sum(sum(X_test[index][4][8:12, 8:12, 8:12]))))/64))
				mid_small_ratio_T1_inc_test.append(np.mean(X_test[index][4][8:12, 8:12, 8:12])/((sum(sum(sum(X_test[index][4][7:13, 7:13, 7:13]))))/216))
				mid_med_ratio_T1_inc_test.append(np.mean(X_test[index][4][7:13, 7:13, 7:13])/((sum(sum(sum(X_test[index][4][6:14, 6:14, 6:14]))))/512))
				mid_large_ratio_T1_inc_test.append(np.mean(X_test[index][4][6:14, 6:14, 6:14])/((sum(sum(sum(X_test[index][4][3:17, 3:17, 3:17]))))/2744))
				mean_T1_test.append(np.mean(X_test[index][4]))
				max_T1_test.append(np.max(X_test[index][4]))
				var_T1_test.append(np.var(X_test[index][4]))
				range_T1_test.append(np.max(X_test[index][4]) - np.min(X_test[index][4]))
				H_all_T1_test = np.array([np.gradient(i) for i in np.gradient(X_test[index][4])]).transpose(2,3,4,0,1)
				H_T1_e1_test.append(LA.eig(H_all_T1_test[10, 10, 10])[0][0])
				H_T1_e2_test.append(LA.eig(H_all_T1_test[10, 10, 10])[0][1])
				H_T1_e3_test.append(LA.eig(H_all_T1_test[10, 10, 10])[0][2])
				
				min_FLAIR_test.append(np.min(X_test[index][5]))
				med_FLAIR_test.append(np.median(X_test[index][5]))
				mid_FLAIR_test.append(X_test[index][5][10,10,10])
				mid_vsmall_ratio_FLAIR_test.append(np.mean(X_test[index][5][9:11, 9:11, 9:11])/((sum(sum(sum(X_test[index][4][8:12, 8:12, 8:12]))) - sum(sum(sum(X_test[index][4][9:11, 9:11, 9:11]))))/56))
				mid_small_ratio_FLAIR_test.append(np.mean(X_test[index][5][8:12, 8:12, 8:12])/((sum(sum(sum(X_test[index][5][7:13, 7:13, 7:13]))) - sum(sum(sum(X_test[index][5][8:12, 8:12, 8:12]))))/152))
				mid_med_ratio_FLAIR_test.append(np.mean(X_test[index][5][7:13, 7:13, 7:13])/((sum(sum(sum(X_test[index][5][6:14, 6:14, 6:14]))) - sum(sum(sum(X_test[index][5][7:13, 7:13, 7:13]))))/296))
				mid_large_ratio_FLAIR_test.append(np.mean(X_test[index][5][6:14, 6:14, 6:14])/((sum(sum(sum(X_test[index][5][3:17, 3:17, 3:17]))) - sum(sum(sum(X_test[index][5][6:14, 6:14, 6:14]))))/2232))
				mid_vsmall_ratio_FLAIR_inc_test.append(np.mean(X_test[index][5][9:11, 9:11, 9:11])/((sum(sum(sum(X_test[index][4][8:12, 8:12, 8:12]))))/64))
				mid_small_ratio_FLAIR_inc_test.append(np.mean(X_test[index][5][8:12, 8:12, 8:12])/((sum(sum(sum(X_test[index][5][7:13, 7:13, 7:13]))))/216))
				mid_med_ratio_FLAIR_inc_test.append(np.mean(X_test[index][5][7:13, 7:13, 7:13])/((sum(sum(sum(X_test[index][5][6:14, 6:14, 6:14]))))/512))
				mid_large_ratio_FLAIR_inc_test.append(np.mean(X_test[index][5][6:14, 6:14, 6:14])/((sum(sum(sum(X_test[index][5][3:17, 3:17, 3:17]))))/2744))
				mean_FLAIR_test.append(np.mean(X_test[index][5]))
				max_FLAIR_test.append(np.max(X_test[index][5]))
				var_FLAIR_test.append(np.var(X_test[index][5]))
				range_FLAIR_test.append(np.max(X_test[index][5]) - np.min(X_test[index][5]))
				H_all_FLAIR_test = np.array([np.gradient(i) for i in np.gradient(X_test[index][5])]).transpose(2,3,4,0,1)
				H_FLAIR_e1_test.append(LA.eig(H_all_FLAIR_test[10, 10, 10])[0][0])
				H_FLAIR_e2_test.append(LA.eig(H_all_FLAIR_test[10, 10, 10])[0][1])
				H_FLAIR_e3_test.append(LA.eig(H_all_FLAIR_test[10, 10, 10])[0][2])
				
				
				density_diff_test.append(X_test[index][5][10,10,10]/X_test[index][4][10,10,10])
				sum_soft_tiss_binary_test.append(sum(sum(sum(X_test[index][6]))))
				sum_percent_soft_tiss_test.append(sum(sum(sum(X_test[index][7]))))
				
				min_st_test.append(np.min(X_test[index][6]))
				med_st_test.append(np.median(X_test[index][6]))
				mid_st_test.append(X_test[index][6][10,10,10])
				mid_vsmall_ratio_st_test.append(np.mean(X_test[index][6][9:11, 9:11, 9:11])/((sum(sum(sum(X_test[index][4][8:12, 8:12, 8:12]))) - sum(sum(sum(X_test[index][4][9:11, 9:11, 9:11]))))/56))
				mid_small_ratio_st_test.append(np.mean(X_test[index][6][8:12, 8:12, 8:12])/((sum(sum(sum(X_test[index][6][7:13, 7:13, 7:13]))) - sum(sum(sum(X_test[index][6][8:12, 8:12, 8:12]))))/152))
				mid_med_ratio_st_test.append(np.mean(X_test[index][6][7:13, 7:13, 7:13])/((sum(sum(sum(X_test[index][6][6:14, 6:14, 6:14]))) - sum(sum(sum(X_test[index][6][7:13, 7:13, 7:13]))))/296))
				mid_large_ratio_st_test.append(np.mean(X_test[index][6][6:14, 6:14, 6:14])/((sum(sum(sum(X_test[index][6][3:17, 3:17, 3:17]))) - sum(sum(sum(X_test[index][6][6:14, 6:14, 6:14]))))/2232))
				mid_vsmall_ratio_st_inc_test.append(np.mean(X_test[index][6][9:11, 9:11, 9:11])/((sum(sum(sum(X_test[index][4][8:12, 8:12, 8:12]))))/64))
				mid_small_ratio_st_inc_test.append(np.mean(X_test[index][6][8:12, 8:12, 8:12])/((sum(sum(sum(X_test[index][6][7:13, 7:13, 7:13]))))/216))
				mid_med_ratio_st_inc_test.append(np.mean(X_test[index][6][7:13, 7:13, 7:13])/((sum(sum(sum(X_test[index][6][6:14, 6:14, 6:14]))))/512))
				mid_large_ratio_st_inc_test.append(np.mean(X_test[index][6][6:14, 6:14, 6:14])/((sum(sum(sum(X_test[index][6][3:17, 3:17, 3:17]))))/2744))
				mean_st_test.append(np.mean(X_test[index][6]))
				max_st_test.append(np.max(X_test[index][6]))
				var_st_test.append(np.var(X_test[index][6]))
				range_st_test.append(np.max(X_test[index][6]) - np.min(X_test[index][6]))
				H_all_st_test = np.array([np.gradient(i) for i in np.gradient(X_test[index][6])]).transpose(2,3,4,0,1)
				H_st_e1_test.append(LA.eig(H_all_st_test[10, 10, 10])[0][0])
				H_st_e2_test.append(LA.eig(H_all_st_test[10, 10, 10])[0][1])
				H_st_e3_test.append(LA.eig(H_all_st_test[10, 10, 10])[0][2])
				
				min_stm_test.append(np.min(X_test[index][7]))
				med_stm_test.append(np.median(X_test[index][7]))
				mid_stm_test.append(X_test[index][7][10,10,10])
				mid_vsmall_ratio_stm_test.append(np.mean(X_test[index][7][9:11, 9:11, 9:11])/((sum(sum(sum(X_test[index][4][8:12, 8:12, 8:12]))) - sum(sum(sum(X_test[index][4][9:11, 9:11, 9:11]))))/56))
				mid_small_ratio_stm_test.append(np.mean(X_test[index][7][8:12, 8:12, 8:12])/((sum(sum(sum(X_test[index][6][7:13, 7:13, 7:13]))) - sum(sum(sum(X_test[index][6][8:12, 8:12, 8:12]))))/152))
				mid_med_ratio_stm_test.append(np.mean(X_test[index][7][7:13, 7:13, 7:13])/((sum(sum(sum(X_test[index][6][6:14, 6:14, 6:14]))) - sum(sum(sum(X_test[index][6][7:13, 7:13, 7:13]))))/296))
				mid_large_ratio_stm_test.append(np.mean(X_test[index][7][6:14, 6:14, 6:14])/((sum(sum(sum(X_test[index][6][3:17, 3:17, 3:17]))) - sum(sum(sum(X_test[index][6][6:14, 6:14, 6:14]))))/2232))
				mid_vsmall_ratio_stm_inc_test.append(np.mean(X_test[index][7][9:11, 9:11, 9:11])/((sum(sum(sum(X_test[index][4][8:12, 8:12, 8:12]))))/64))
				mid_small_ratio_stm_inc_test.append(np.mean(X_test[index][7][8:12, 8:12, 8:12])/((sum(sum(sum(X_test[index][6][7:13, 7:13, 7:13]))))/216))
				mid_med_ratio_stm_inc_test.append(np.mean(X_test[index][7][7:13, 7:13, 7:13])/((sum(sum(sum(X_test[index][6][6:14, 6:14, 6:14]))))/512))
				mid_large_ratio_stm_inc_test.append(np.mean(X_test[index][7][6:14, 6:14, 6:14])/((sum(sum(sum(X_test[index][6][3:17, 3:17, 3:17]))))/2744))
				mean_stm_test.append(np.mean(X_test[index][7]))
				max_stm_test.append(np.max(X_test[index][7]))
				var_stm_test.append(np.var(X_test[index][7]))
				range_stm_test.append(np.max(X_test[index][7]) - np.min(X_test[index][7]))
				H_all_stm_test = np.array([np.gradient(i) for i in np.gradient(X_test[index][7])]).transpose(2,3,4,0,1)
				H_stm_e1_test.append(LA.eig(H_all_stm_test[10, 10, 10])[0][0])
				H_stm_e2_test.append(LA.eig(H_all_stm_test[10, 10, 10])[0][1])
				H_stm_e3_test.append(LA.eig(H_all_stm_test[10, 10, 10])[0][2])
				
				min_th_T1_test.append(np.min(tophat_img_T1_test))
				med_th_T1_test.append(np.median(tophat_img_T1_test))
				mid_th_T1_test.append(tophat_img_T1_test[10,10,10])
				mid_vsmall_ratio_th_T1_test.append(np.mean(tophat_img_T1_test[9:11, 9:11, 9:11])/((sum(sum(sum(tophat_img_T1_test[8:12, 8:12, 8:12]))) - sum(sum(sum(tophat_img_T1_test[9:11, 9:11, 9:11]))))/56))
				mid_small_ratio_th_T1_test.append(np.mean(tophat_img_T1_test[8:12, 8:12, 8:12])/((sum(sum(sum(tophat_img_T1_test[7:13, 7:13, 7:13]))) - sum(sum(sum(tophat_img_T1_test[8:12, 8:12, 8:12]))))/152))
				mid_med_ratio_th_T1_test.append(np.mean(tophat_img_T1_test[7:13, 7:13, 7:13])/((sum(sum(sum(tophat_img_T1_test[6:14, 6:14, 6:14]))) - sum(sum(sum(tophat_img_T1_test[7:13, 7:13, 7:13]))))/296))
				mid_large_ratio_th_T1_test.append(np.mean(tophat_img_T1_test[6:14, 6:14, 6:14])/((sum(sum(sum(tophat_img_T1_test[3:17, 3:17, 3:17]))) - sum(sum(sum(tophat_img_T1_test[6:14, 6:14, 6:14]))))/2232))
				mid_vsmall_ratio_th_T1_inc_test.append(np.mean(tophat_img_T1_test[9:11, 9:11, 9:11])/((sum(sum(sum(tophat_img_T1_test[8:12, 8:12, 8:12]))))/64))
				mid_small_ratio_th_T1_inc_test.append(np.mean(tophat_img_T1_test[8:12, 8:12, 8:12])/((sum(sum(sum(tophat_img_T1_test[7:13, 7:13, 7:13]))))/216))
				mid_med_ratio_th_T1_inc_test.append(np.mean(tophat_img_T1_test[7:13, 7:13, 7:13])/((sum(sum(sum(tophat_img_T1_test[6:14, 6:14, 6:14]))))/512))
				mid_large_ratio_th_T1_inc_test.append(np.mean(tophat_img_T1_test[6:14, 6:14, 6:14])/((sum(sum(sum(tophat_img_T1_test[3:17, 3:17, 3:17]))))/2744))
				mean_th_T1_test.append(np.mean(tophat_img_T1_test))
				max_th_T1_test.append(np.max(tophat_img_T1_test))
				var_th_T1_test.append(np.var(tophat_img_T1_test))
				range_th_T1_test.append(np.max(tophat_img_T1_test) - np.min(tophat_img_T1_test))
				H_all_th_T1_test = np.array([np.gradient(i) for i in np.gradient(tophat_img_T1_test)]).transpose(2,3,4,0,1)
				H_th_T1_e1_test.append(LA.eig(H_all_th_T1_test[10, 10, 10])[0][0])
				H_th_T1_e2_test.append(LA.eig(H_all_th_T1_test[10, 10, 10])[0][1])
				H_th_T1_e3_test.append(LA.eig(H_all_th_T1_test[10, 10, 10])[0][2])
				
				min_th_FLAIR_test.append(np.min(tophat_img_FLAIR_test))
				med_th_FLAIR_test.append(np.median(tophat_img_FLAIR_test))
				mid_th_FLAIR_test.append(tophat_img_FLAIR_test[10,10,10])
				mid_vsmall_ratio_th_FLAIR_test.append(np.mean(tophat_img_FLAIR_test[9:11, 9:11, 9:11])/((sum(sum(sum(tophat_img_FLAIR_test[8:12, 8:12, 8:12]))) - sum(sum(sum(tophat_img_FLAIR_test[9:11, 9:11, 9:11]))))/56))
				mid_small_ratio_th_FLAIR_test.append(np.mean(tophat_img_FLAIR_test[8:12, 8:12, 8:12])/((sum(sum(sum(tophat_img_FLAIR_test[7:13, 7:13, 7:13]))) - sum(sum(sum(tophat_img_FLAIR_test[8:12, 8:12, 8:12]))))/152))
				mid_med_ratio_th_FLAIR_test.append(np.mean(tophat_img_FLAIR_test[7:13, 7:13, 7:13])/((sum(sum(sum(tophat_img_FLAIR_test[6:14, 6:14, 6:14]))) - sum(sum(sum(tophat_img_FLAIR_test[7:13, 7:13, 7:13]))))/296))
				mid_large_ratio_th_FLAIR_test.append(np.mean(tophat_img_FLAIR_test[6:14, 6:14, 6:14])/((sum(sum(sum(tophat_img_FLAIR_test[3:17, 3:17, 3:17]))) - sum(sum(sum(tophat_img_FLAIR_test[6:14, 6:14, 6:14]))))/2232))
				mid_vsmall_ratio_th_FLAIR_inc_test.append(np.mean(tophat_img_FLAIR_test[9:11, 9:11, 9:11])/((sum(sum(sum(tophat_img_FLAIR_test[8:12, 8:12, 8:12]))))/64))
				mid_small_ratio_th_FLAIR_inc_test.append(np.mean(tophat_img_FLAIR_test[8:12, 8:12, 8:12])/((sum(sum(sum(tophat_img_FLAIR_test[7:13, 7:13, 7:13]))))/216))
				mid_med_ratio_th_FLAIR_inc_test.append(np.mean(tophat_img_FLAIR_test[7:13, 7:13, 7:13])/((sum(sum(sum(tophat_img_FLAIR_test[6:14, 6:14, 6:14]))))/512))
				mid_large_ratio_th_FLAIR_inc_test.append(np.mean(tophat_img_FLAIR_test[6:14, 6:14, 6:14])/((sum(sum(sum(tophat_img_FLAIR_test[3:17, 3:17, 3:17]))))/2744))
				mean_th_FLAIR_test.append(np.mean(tophat_img_FLAIR_test))
				max_th_FLAIR_test.append(np.max(tophat_img_FLAIR_test))
				var_th_FLAIR_test.append(np.var(tophat_img_FLAIR_test))
				range_th_FLAIR_test.append(np.max(tophat_img_FLAIR_test) - np.min(tophat_img_FLAIR_test))
				H_all_th_FLAIR_test = np.array([np.gradient(i) for i in np.gradient(tophat_img_FLAIR_test)]).transpose(2,3,4,0,1)
				H_th_FLAIR_e1_test.append(LA.eig(H_all_th_FLAIR_test[10, 10, 10])[0][0])
				H_th_FLAIR_e2_test.append(LA.eig(H_all_th_FLAIR_test[10, 10, 10])[0][1])
				H_th_FLAIR_e3_test.append(LA.eig(H_all_th_FLAIR_test[10, 10, 10])[0][2])
				
				min_th_st_test.append(np.min(tophat_img_st_test))
				med_th_st_test.append(np.median(tophat_img_st_test))
				mid_th_st_test.append(tophat_img_st_test[10,10,10])
				mid_vsmall_ratio_th_st_test.append(np.mean(tophat_img_st_test[9:11, 9:11, 9:11])/((sum(sum(sum(tophat_img_st_test[8:12, 8:12, 8:12]))) - sum(sum(sum(tophat_img_st_test[9:11, 9:11, 9:11]))))/56))
				mid_small_ratio_th_st_test.append(np.mean(tophat_img_st_test[8:12, 8:12, 8:12])/((sum(sum(sum(tophat_img_st_test[7:13, 7:13, 7:13]))) - sum(sum(sum(tophat_img_st_test[8:12, 8:12, 8:12]))))/152))
				mid_med_ratio_th_st_test.append(np.mean(tophat_img_st_test[7:13, 7:13, 7:13])/((sum(sum(sum(tophat_img_st_test[6:14, 6:14, 6:14]))) - sum(sum(sum(tophat_img_st_test[7:13, 7:13, 7:13]))))/296))
				mid_large_ratio_th_st_test.append(np.mean(tophat_img_st_test[6:14, 6:14, 6:14])/((sum(sum(sum(tophat_img_st_test[3:17, 3:17, 3:17]))) - sum(sum(sum(tophat_img_st_test[6:14, 6:14, 6:14]))))/2232))
				mid_vsmall_ratio_th_st_inc_test.append(np.mean(tophat_img_st_test[9:11, 9:11, 9:11])/((sum(sum(sum(tophat_img_st_test[8:12, 8:12, 8:12]))))/64))
				mid_small_ratio_th_st_inc_test.append(np.mean(tophat_img_st_test[8:12, 8:12, 8:12])/((sum(sum(sum(tophat_img_st_test[7:13, 7:13, 7:13]))))/216))
				mid_med_ratio_th_st_inc_test.append(np.mean(tophat_img_st_test[7:13, 7:13, 7:13])/((sum(sum(sum(tophat_img_st_test[6:14, 6:14, 6:14]))))/512))
				mid_large_ratio_th_st_inc_test.append(np.mean(tophat_img_st_test[6:14, 6:14, 6:14])/((sum(sum(sum(tophat_img_st_test[3:17, 3:17, 3:17]))))/2744))
				mean_th_st_test.append(np.mean(tophat_img_st_test))
				max_th_st_test.append(np.max(tophat_img_st_test))
				var_th_st_test.append(np.var(tophat_img_st_test))
				range_th_st_test.append(np.max(tophat_img_st_test) - np.min(tophat_img_st_test))
				H_all_th_st_test = np.array([np.gradient(i) for i in np.gradient(tophat_img_st_test)]).transpose(2,3,4,0,1)
				H_th_st_e1_test.append(LA.eig(H_all_th_st_test[10, 10, 10])[0][0])
				H_th_st_e2_test.append(LA.eig(H_all_th_st_test[10, 10, 10])[0][1])
				H_th_st_e3_test.append(LA.eig(H_all_th_st_test[10, 10, 10])[0][2])
				
				min_bh_T1_test.append(np.min(blackhat_img_T1_test))
				med_bh_T1_test.append(np.median(blackhat_img_T1_test))
				mid_bh_T1_test.append(blackhat_img_T1_test[10,10,10])
				mid_vsmall_ratio_bh_T1_test.append(np.mean(blackhat_img_T1_test[9:11, 9:11, 9:11])/((sum(sum(sum(blackhat_img_T1_test[8:12, 8:12, 8:12]))) - sum(sum(sum(blackhat_img_T1_test[9:11, 9:11, 9:11]))))/56))
				mid_small_ratio_bh_T1_test.append(np.mean(blackhat_img_T1_test[8:12, 8:12, 8:12])/((sum(sum(sum(blackhat_img_T1_test[7:13, 7:13, 7:13]))) - sum(sum(sum(blackhat_img_T1_test[8:12, 8:12, 8:12]))))/152))
				mid_med_ratio_bh_T1_test.append(np.mean(blackhat_img_T1_test[7:13, 7:13, 7:13])/((sum(sum(sum(blackhat_img_T1_test[6:14, 6:14, 6:14]))) - sum(sum(sum(blackhat_img_T1_test[7:13, 7:13, 7:13]))))/296))
				mid_large_ratio_bh_T1_test.append(np.mean(blackhat_img_T1_test[6:14, 6:14, 6:14])/((sum(sum(sum(blackhat_img_T1_test[3:17, 3:17, 3:17]))) - sum(sum(sum(blackhat_img_T1_test[6:14, 6:14, 6:14]))))/2232))
				mid_vsmall_ratio_bh_T1_inc_test.append(np.mean(blackhat_img_T1_test[9:11, 9:11, 9:11])/((sum(sum(sum(blackhat_img_T1_test[8:12, 8:12, 8:12]))))/64))
				mid_small_ratio_bh_T1_inc_test.append(np.mean(blackhat_img_T1_test[8:12, 8:12, 8:12])/((sum(sum(sum(blackhat_img_T1_test[7:13, 7:13, 7:13]))))/216))
				mid_med_ratio_bh_T1_inc_test.append(np.mean(blackhat_img_T1_test[7:13, 7:13, 7:13])/((sum(sum(sum(blackhat_img_T1_test[6:14, 6:14, 6:14]))))/512))
				mid_large_ratio_bh_T1_inc_test.append(np.mean(blackhat_img_T1_test[6:14, 6:14, 6:14])/((sum(sum(sum(blackhat_img_T1_test[3:17, 3:17, 3:17]))))/2744))
				mean_bh_T1_test.append(np.mean(blackhat_img_T1_test))
				max_bh_T1_test.append(np.max(blackhat_img_T1_test))
				var_bh_T1_test.append(np.var(blackhat_img_T1_test))
				range_bh_T1_test.append(np.max(blackhat_img_T1_test) - np.min(blackhat_img_T1_test))
				H_all_bh_T1_test = np.array([np.gradient(i) for i in np.gradient(blackhat_img_T1_test)]).transpose(2,3,4,0,1)
				H_bh_T1_e1_test.append(LA.eig(H_all_bh_T1_test[10, 10, 10])[0][0])
				H_bh_T1_e2_test.append(LA.eig(H_all_bh_T1_test[10, 10, 10])[0][1])
				H_bh_T1_e3_test.append(LA.eig(H_all_bh_T1_test[10, 10, 10])[0][2])
				
				min_bh_FLAIR_test.append(np.min(blackhat_img_FLAIR_test))
				med_bh_FLAIR_test.append(np.median(blackhat_img_FLAIR_test))
				mid_bh_FLAIR_test.append(blackhat_img_FLAIR_test[10,10,10])
				mid_vsmall_ratio_bh_FLAIR_test.append(np.mean(blackhat_img_FLAIR_test[9:11, 9:11, 9:11])/((sum(sum(sum(blackhat_img_FLAIR_test[8:12, 8:12, 8:12]))) - sum(sum(sum(blackhat_img_FLAIR_test[9:11, 9:11, 9:11]))))/56))
				mid_small_ratio_bh_FLAIR_test.append(np.mean(blackhat_img_FLAIR_test[8:12, 8:12, 8:12])/((sum(sum(sum(blackhat_img_FLAIR_test[7:13, 7:13, 7:13]))) - sum(sum(sum(blackhat_img_FLAIR_test[8:12, 8:12, 8:12]))))/152))
				mid_med_ratio_bh_FLAIR_test.append(np.mean(blackhat_img_FLAIR_test[7:13, 7:13, 7:13])/((sum(sum(sum(blackhat_img_FLAIR_test[6:14, 6:14, 6:14]))) - sum(sum(sum(blackhat_img_FLAIR_test[7:13, 7:13, 7:13]))))/296))
				mid_large_ratio_bh_FLAIR_test.append(np.mean(blackhat_img_FLAIR_test[6:14, 6:14, 6:14])/((sum(sum(sum(blackhat_img_FLAIR_test[3:17, 3:17, 3:17]))) - sum(sum(sum(blackhat_img_FLAIR_test[6:14, 6:14, 6:14]))))/2232))
				mid_vsmall_ratio_bh_FLAIR_inc_test.append(np.mean(blackhat_img_FLAIR_test[9:11, 9:11, 9:11])/((sum(sum(sum(blackhat_img_FLAIR_test[8:12, 8:12, 8:12]))))/64))
				mid_small_ratio_bh_FLAIR_inc_test.append(np.mean(blackhat_img_FLAIR_test[8:12, 8:12, 8:12])/((sum(sum(sum(blackhat_img_FLAIR_test[7:13, 7:13, 7:13]))))/216))
				mid_med_ratio_bh_FLAIR_inc_test.append(np.mean(blackhat_img_FLAIR_test[7:13, 7:13, 7:13])/((sum(sum(sum(blackhat_img_FLAIR_test[6:14, 6:14, 6:14]))))/512))
				mid_large_ratio_bh_FLAIR_inc_test.append(np.mean(blackhat_img_FLAIR_test[6:14, 6:14, 6:14])/((sum(sum(sum(blackhat_img_FLAIR_test[3:17, 3:17, 3:17]))))/2744))
				mean_bh_FLAIR_test.append(np.mean(blackhat_img_FLAIR_test))
				max_bh_FLAIR_test.append(np.max(blackhat_img_FLAIR_test))
				var_bh_FLAIR_test.append(np.var(blackhat_img_FLAIR_test))
				range_bh_FLAIR_test.append(np.max(blackhat_img_FLAIR_test) - np.min(blackhat_img_FLAIR_test))
				H_all_bh_FLAIR_test = np.array([np.gradient(i) for i in np.gradient(blackhat_img_FLAIR_test)]).transpose(2,3,4,0,1)
				H_bh_FLAIR_e1_test.append(LA.eig(H_all_bh_FLAIR_test[10, 10, 10])[0][0])
				H_bh_FLAIR_e2_test.append(LA.eig(H_all_bh_FLAIR_test[10, 10, 10])[0][1])
				H_bh_FLAIR_e3_test.append(LA.eig(H_all_bh_FLAIR_test[10, 10, 10])[0][2])
				
				min_bh_st_test.append(np.min(blackhat_img_st_test))
				med_bh_st_test.append(np.median(blackhat_img_st_test))
				mid_bh_st_test.append(blackhat_img_st_test[10,10,10])
				mid_vsmall_ratio_bh_st_test.append(np.mean(blackhat_img_st_test[9:11, 9:11, 9:11])/((sum(sum(sum(blackhat_img_st_test[8:12, 8:12, 8:12]))) - sum(sum(sum(blackhat_img_st_test[9:11, 9:11, 9:11]))))/56))
				mid_small_ratio_bh_st_test.append(np.mean(blackhat_img_st_test[8:12, 8:12, 8:12])/((sum(sum(sum(blackhat_img_st_test[7:13, 7:13, 7:13]))) - sum(sum(sum(blackhat_img_st_test[8:12, 8:12, 8:12]))))/152))
				mid_med_ratio_bh_st_test.append(np.mean(blackhat_img_st_test[7:13, 7:13, 7:13])/((sum(sum(sum(blackhat_img_st_test[6:14, 6:14, 6:14]))) - sum(sum(sum(blackhat_img_st_test[7:13, 7:13, 7:13]))))/296))
				mid_large_ratio_bh_st_test.append(np.mean(blackhat_img_st_test[6:14, 6:14, 6:14])/((sum(sum(sum(blackhat_img_st_test[3:17, 3:17, 3:17]))) - sum(sum(sum(blackhat_img_st_test[6:14, 6:14, 6:14]))))/2232))
				mid_vsmall_ratio_bh_st_inc_test.append(np.mean(blackhat_img_st_test[9:11, 9:11, 9:11])/((sum(sum(sum(blackhat_img_st_test[8:12, 8:12, 8:12]))))/64))
				mid_small_ratio_bh_st_inc_test.append(np.mean(blackhat_img_st_test[8:12, 8:12, 8:12])/((sum(sum(sum(blackhat_img_st_test[7:13, 7:13, 7:13]))))/216))
				mid_med_ratio_bh_st_inc_test.append(np.mean(blackhat_img_st_test[7:13, 7:13, 7:13])/((sum(sum(sum(blackhat_img_st_test[6:14, 6:14, 6:14]))))/512))
				mid_large_ratio_bh_st_inc_test.append(np.mean(blackhat_img_st_test[6:14, 6:14, 6:14])/((sum(sum(sum(blackhat_img_st_test[3:17, 3:17, 3:17]))))/2744))
				mean_bh_st_test.append(np.mean(blackhat_img_st_test))
				max_bh_st_test.append(np.max(blackhat_img_st_test))
				var_bh_st_test.append(np.var(blackhat_img_st_test))
				range_bh_st_test.append(np.max(blackhat_img_st_test) - np.min(blackhat_img_st_test))
				H_all_bh_st_test = np.array([np.gradient(i) for i in np.gradient(blackhat_img_FLAIR_test)]).transpose(2,3,4,0,1)
				H_bh_st_e1_test.append(LA.eig(H_all_bh_st_test[10, 10, 10])[0][0])
				H_bh_st_e2_test.append(LA.eig(H_all_bh_st_test[10, 10, 10])[0][1])
				H_bh_st_e3_test.append(LA.eig(H_all_bh_st_test[10, 10, 10])[0][2])
				
				brain_test.append(X_test[index][0])
				x_test.append(X_test[index][1])
				y_test.append(X_test[index][2])
				z_test.append(X_test[index][3])

				WMH_x_test.append(np.count_nonzero(X_test[index][5][10,:,:]>0.45))
				WMH_y_test.append(np.count_nonzero(X_test[index][5][:,10,:]>0.45))
				WMH_z_test.append(np.count_nonzero(X_test[index][5][:,:,10]>0.45))
				
				CSF_feat_test.append(np.mean(X_test[index][8]))
				GM_feat_test.append(np.mean(X_test[index][10]))
				WM_feat_test.append(np.mean(X_test[index][9]))
		return brain_test, min_T1_test, med_T1_test, mid_T1_test, mid_vsmall_ratio_T1_test, mid_small_ratio_T1_test, mid_med_ratio_T1_test, mid_large_ratio_T1_test, mid_vsmall_ratio_T1_inc_test, mid_small_ratio_T1_inc_test, mid_med_ratio_T1_inc_test, mid_large_ratio_T1_inc_test, mean_T1_test, max_T1_test, var_T1_test, range_T1_test, H_T1_e1_test, H_T1_e2_test, H_T1_e3_test, min_FLAIR_test, mid_FLAIR_test, med_FLAIR_test, mid_vsmall_ratio_FLAIR_test, mid_small_ratio_FLAIR_test, mid_med_ratio_FLAIR_test, mid_large_ratio_FLAIR_test, mid_vsmall_ratio_FLAIR_inc_test, mid_small_ratio_FLAIR_inc_test, mid_med_ratio_FLAIR_inc_test, mid_large_ratio_FLAIR_inc_test, mean_FLAIR_test, max_FLAIR_test, var_FLAIR_test, range_FLAIR_test, H_FLAIR_e1_test, H_FLAIR_e2_test, H_FLAIR_e3_test, density_diff_test, sum_soft_tiss_binary_test, sum_percent_soft_tiss_test, min_st_test, med_st_test, mid_st_test, mid_vsmall_ratio_st_test, mid_small_ratio_st_test, mid_med_ratio_st_test, mid_large_ratio_st_test, mid_vsmall_ratio_st_inc_test, mid_small_ratio_st_inc_test, mid_med_ratio_st_inc_test, mid_large_ratio_st_inc_test, mean_st_test, max_st_test, var_st_test, range_st_test, H_st_e1_test, H_st_e2_test, H_st_e3_test, min_stm_test, med_stm_test, mid_stm_test, mid_vsmall_ratio_stm_test, mid_small_ratio_stm_test, mid_med_ratio_stm_test, mid_large_ratio_stm_test, mid_vsmall_ratio_stm_inc_test, mid_small_ratio_stm_inc_test, mid_med_ratio_stm_inc_test, mid_large_ratio_stm_inc_test, mean_stm_test, max_stm_test, var_stm_test, range_stm_test, H_stm_e1_test, H_stm_e2_test, H_stm_e3_test, min_th_T1_test, med_th_T1_test, mid_th_T1_test, mid_vsmall_ratio_th_T1_test, mid_small_ratio_th_T1_test, mid_med_ratio_th_T1_test, mid_large_ratio_th_T1_test, mid_vsmall_ratio_th_T1_inc_test, mid_small_ratio_th_T1_inc_test, mid_med_ratio_th_T1_inc_test, mid_large_ratio_th_T1_inc_test, mean_th_T1_test, max_th_T1_test, var_th_T1_test, range_th_T1_test, H_th_T1_e1_test, H_th_T1_e2_test, H_th_T1_e3_test, min_th_FLAIR_test, med_th_FLAIR_test, mid_th_FLAIR_test, mid_vsmall_ratio_th_FLAIR_test, mid_small_ratio_th_FLAIR_test, mid_med_ratio_th_FLAIR_test, mid_large_ratio_th_FLAIR_test, mid_vsmall_ratio_th_FLAIR_inc_test, mid_small_ratio_th_FLAIR_inc_test, mid_med_ratio_th_FLAIR_inc_test, mid_large_ratio_th_FLAIR_inc_test, mean_th_FLAIR_test, max_th_FLAIR_test, var_th_FLAIR_test, range_th_FLAIR_test, H_th_FLAIR_e1_test, H_th_FLAIR_e2_test, H_th_FLAIR_e3_test, min_th_st_test, med_th_st_test, mid_th_st_test, mid_vsmall_ratio_th_st_test, mid_small_ratio_th_st_test, mid_med_ratio_th_st_test, mid_large_ratio_th_st_test, mid_vsmall_ratio_th_st_inc_test, mid_small_ratio_th_st_inc_test, mid_med_ratio_th_st_inc_test, mid_large_ratio_th_st_inc_test, mean_th_st_test, max_th_st_test, var_th_st_test, range_th_st_test, H_th_st_e1_test, H_th_st_e2_test, H_th_st_e3_test, min_bh_T1_test, med_bh_T1_test, mid_bh_T1_test, mid_vsmall_ratio_bh_T1_test, mid_small_ratio_bh_T1_test, mid_med_ratio_bh_T1_test, mid_large_ratio_bh_T1_test, mid_vsmall_ratio_bh_T1_inc_test, mid_small_ratio_bh_T1_inc_test, mid_med_ratio_bh_T1_inc_test, mid_large_ratio_bh_T1_inc_test, mean_bh_T1_test, max_bh_T1_test, var_bh_T1_test, range_bh_T1_test, H_bh_T1_e1_test, H_bh_T1_e2_test, H_bh_T1_e3_test, min_bh_FLAIR_test, med_bh_FLAIR_test, mid_bh_FLAIR_test, mid_vsmall_ratio_bh_FLAIR_test, mid_small_ratio_bh_FLAIR_test, mid_med_ratio_bh_FLAIR_test, mid_large_ratio_bh_FLAIR_test, mid_vsmall_ratio_bh_FLAIR_inc_test, mid_small_ratio_bh_FLAIR_inc_test, mid_med_ratio_bh_FLAIR_inc_test, mid_large_ratio_bh_FLAIR_inc_test, mean_bh_FLAIR_test, max_bh_FLAIR_test, var_bh_FLAIR_test, range_bh_FLAIR_test, H_bh_FLAIR_e1_test, H_bh_FLAIR_e2_test, H_bh_FLAIR_e3_test, min_bh_st_test, med_bh_st_test, mid_bh_st_test, mid_vsmall_ratio_bh_st_test, mid_small_ratio_bh_st_test, mid_med_ratio_bh_st_test, mid_large_ratio_bh_st_test, mid_vsmall_ratio_bh_st_inc_test, mid_small_ratio_bh_st_inc_test, mid_med_ratio_bh_st_inc_test, mid_large_ratio_bh_st_inc_test, mean_bh_st_test, max_bh_st_test, var_bh_st_test, range_bh_st_test, H_bh_st_e1_test, H_bh_st_e2_test, H_bh_st_e3_test, x_test, y_test, z_test, WMH_x_test, WMH_y_test, WMH_z_test, CSF_feat_test, GM_feat_test, WM_feat_test
