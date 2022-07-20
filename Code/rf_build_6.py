from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_curve
import numpy as np
from sklearn.metrics import plot_roc_curve
import matplotlib.pyplot as plt

def best_number_of_trees(dataset, Y_train):
	accuracy_list = []
	error_list = []
	no_trees = [50, 100, 200, 300, 500, 1000, 2000, 5000, 10000]
	for i in no_trees:
		from sklearn.ensemble import RandomForestClassifier
		#max features, min sample split 
		clf = RandomForestClassifier(n_estimators = i, max_features = 'sqrt', min_samples_split = 2, criterion = 'gini', oob_score = True, bootstrap = True, random_state = 30)
		clf.fit(dataset, Y_train)
		y_pred = (clf.oob_decision_function_[:,1] >= 0.50).astype(bool)
		accuracy = accuracy_score(Y_train, y_pred)
		accuracy_list.append(accuracy)
	for i in accuracy_list:
		error_list.append(1-i)
	return accuracy_list, error_list

def cv_folds(dataset_b, Y_train):
	# Cv list
	cv_1_data = []
	cv_2_data = []
	cv_3_data = []
	cv_4_data = []
	cv_5_data = []
	cv_1_idx = []
	cv_2_idx = []
	cv_3_idx = []
	cv_4_idx = []
	cv_5_idx = []
	cv_1_brains = [1183, 2208, 1448, 2733, 1070]
	cv_2_brains = [430, 46, 6324, 5568, 1477]
	cv_3_brains = [1224, 1873, 4689, 2777, 4442]
	cv_4_brains = [1243, 1242, 2396, 4837, 4968]
	cv_5_brains = [4848, 3318, 4602, 891, 1535]
	cv_1_train = []
	cv_2_train = []
	cv_3_train = []
	cv_4_train = []
	cv_5_train = []
	
	for i in range(dataset_b.shape[0]):
			if dataset_b[i][0] in cv_1_brains:
					cv_1_idx.append(i)
					cv_1_data.append(np.delete(dataset_b[i], 0))
					cv_1_train.append(Y_train[i])
			elif dataset_b[i][0] in cv_2_brains:
					cv_2_idx.append(i)
					cv_2_data.append(np.delete(dataset_b[i], 0))
					cv_2_train.append(Y_train[i])
			elif dataset_b[i][0] in cv_3_brains:
					cv_3_idx.append(i)
					cv_3_data.append(np.delete(dataset_b[i], 0))
					cv_3_train.append(Y_train[i])
			elif dataset_b[i][0] in cv_4_brains:
					cv_4_idx.append(i)
					cv_4_data.append(np.delete(dataset_b[i], 0))
					cv_4_train.append(Y_train[i])
			elif dataset_b[i][0] in cv_5_brains:
					cv_5_idx.append(i)
					cv_5_data.append(np.delete(dataset_b[i], 0))
					cv_5_train.append(Y_train[i])
	return cv_1_data, cv_2_data, cv_3_data, cv_4_data, cv_5_data, cv_1_idx, cv_2_idx, cv_3_idx, cv_4_idx, cv_5_idx, cv_1_brains, cv_2_brains, cv_3_brains, cv_4_brains, cv_5_brains, cv_1_train, cv_2_train, cv_3_train, cv_4_train, cv_5_train 

def find_mean_thresh(classifier, cv_splits, dataset, Y_train, rf_random):
	tprs = []
	aucs = []
	mean_fpr = np.linspace(0, 1, 100)
	threshold = []
	# Get best thresholds
	for j in range(0,10):
		fig, ax = plt.subplots()
		for i, (train, test) in enumerate(cv_splits):
			classifier.fit(dataset[train],Y_train[train])
			viz = plot_roc_curve(classifier, dataset[test], Y_train[test],
								name='ROC fold {}'.format(i),
								alpha=0.3, lw=1, ax=ax)
			#y_pred_scores = (classifier.predict_proba(X_test)[:,1] >= 0.3).astype(bool) # set threshold as 0.3
			y_pred_scores = classifier.predict_proba(dataset[test])
			fpr, tpr, thresholds = roc_curve(Y_train[test], y_pred_scores[:, -1])
			optimal_idx = np.argmax(tpr-fpr)
			optimal_threshold = thresholds[optimal_idx]
			threshold.append(optimal_threshold)
			print("Threshold value is:", optimal_threshold)
	
	mean_thresh = np.mean(threshold)
	return tprs, aucs, mean_thresh
