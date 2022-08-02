import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import auc
from sklearn.metrics import plot_roc_curve
from sklearn.inspection import permutation_importance
import seaborn as sns
import pandas as pd

def trees_plot(error_list, no_trees):
	index = np.arange(0,len(no_trees))
	plt.plot(index, error_list, label = "accuracy", linestyle="--")
	plt.xlabel('No. of Trees')
	plt.ylabel('OOB_Error')
	plt.xticks(index, no_trees)
	plt.title('Performance of Random Forest on different number of trees')
	plt.legend()
	plt.savefig('/home/z5209394/ew-honoursthesis/Graphs/no_trees.png')

def plot_auc_roc_thresholds(classifier, dataset, Y_train, cv_splits, mean_thresh):
	tprs = []
	aucs = []
	mean_fpr = np.linspace(0, 1, 100)
	thresholds = [0.213214]
	#thresholds.append(mean_thresh)

	for thresh in thresholds:
		fig, ax = plt.subplots()
		for i, (train, test) in enumerate(cv_splits):
			classifier.fit(dataset[train],Y_train[train])
			viz = plot_roc_curve(classifier, dataset[test], Y_train[test],
								name='ROC fold {}'.format(i),
								alpha=0.3, lw=1, ax=ax)
			interp_tpr = np.interp(mean_fpr, viz.fpr, viz.tpr)
			interp_tpr[0] = 0.0
			tprs.append(interp_tpr)
			aucs.append(viz.roc_auc)

		ax.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r',
				label='Chance', alpha=.8)

		mean_tpr = np.mean(tprs, axis=0)
		mean_tpr[-1] = 1.0
		mean_auc = auc(mean_fpr, mean_tpr)
		std_auc = np.std(aucs)
		ax.plot(mean_fpr, mean_tpr, color='b',
				label=r'Mean ROC (AUC = %0.2f $\pm$ %0.2f)' % (mean_auc, std_auc),
				lw=2, alpha=.8)

		std_tpr = np.std(tprs, axis=0)
		tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
		tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
		ax.fill_between(mean_fpr, tprs_lower, tprs_upper, color='grey', alpha=.2,
						label=r'$\pm$ 1 std. dev.')

		ax.set(xlim=[-0.05, 1.05], ylim=[-0.05, 1.05],
			title="Receiver operating characteristic example")
		ax.legend(loc="lower right")
		plt.savefig('/home/z5209394/ew-honoursthesis/Graphs/roc-auc + '+ str(thresh) +' +.png')

def feature_importance_plot(clf, dataset_pd, Y_train, cv_splits):
	# Feature Importance
	importances_sorted_all = []
	for i, (train, test) in enumerate(cv_splits):
		importance_sorted = []
		clf.fit(dataset_pd.iloc[train],Y_train[train])
		importance_sorted_idx = np.argsort(clf.feature_importances_)
		importance_sorted.append(dataset_pd.columns[importance_sorted_idx])
		importance_sorted.append(clf.feature_importances_[importance_sorted_idx])
		importances_sorted_all.append(importance_sorted)

	perm_sorted_all = []
	for i, (train, test) in enumerate(cv_splits):
		print(i)
		perm_sorted = []
		clf.fit(dataset_pd.iloc[train],Y_train[train])
		result = permutation_importance(clf, dataset_pd.iloc[test], Y_train[test], n_repeats=10)
		perm_sorted_idx = result.importances_mean.argsort()
		perm_sorted.append(dataset_pd.columns[perm_sorted_idx])
		perm_sorted.append(result.importances[perm_sorted_idx])
		perm_sorted_all.append(perm_sorted)

	perm_dict = {}
	for permutation in perm_sorted_all:
		for i in range(len(permutation[0])):
			if str(permutation[0][i]) not in perm_dict:
				perm_dict[str(permutation[0][i])] = []
				perm_dict[str(permutation[0][i])].append(permutation[1][i])
			else:
				perm_dict[str(permutation[0][i])].append(permutation[1][i])

	importance_dict = {}
	for importance in importances_sorted_all:
		for i in range(len(importance[0])):
			if str(importance[0][i]) not in importance_dict:
				importance_dict[str(importance[0][i])] = []
				importance_dict[str(importance[0][i])].append(importance[1][i])
			else:
				importance_dict[str(importance[0][i])].append(importance[1][i])

	column_perm = []
	perm_idx = []
	perm_idx_average = []
	for key, value in perm_dict.items():
		column_perm.append(key)
		perm_idx.append(value)
		value = np.reshape(value, -1)
		perm_idx_average.append(float(sum(value)/len(value)))
		
	column_importances = []
	importances_idx = []
	for key, value in importance_dict.items():
		column_importances.append(key)
		importances_idx.append(float(sum(value)/len(value)))

	perm_idx_reshaped = []
	for index in perm_idx:
		reshaped = np.reshape(index, -1)
		perm_idx_reshaped.append(reshaped)
		
	results_perm = pd.DataFrame()
	results_perm['feature'] = column_perm
	results_perm['perm_idx_average'] = perm_idx_average
	results_perm['perm_idx_reshaped'] = perm_idx_reshaped
	results_perm = results_perm.sort_values('perm_idx_average')

	results_importance = pd.DataFrame()
	results_importance['feature'] = column_importances
	results_importance['importances_idx'] = importances_idx
	results_importance = results_importance.sort_values('importances_idx')

	tree_indices = np.arange(0, len(clf.feature_importances_)) + 0.5
	fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 8))
	ax1.barh(tree_indices[-40:], results_importance['importances_idx'][-40:], height=0.7)
	ax1.set_yticks(tree_indices[-40:])
	ax1.set_yticklabels(results_importance['feature'][-40:])
	ax2.boxplot(
		results_perm['perm_idx_reshaped'][-40:].to_numpy(),
		vert=False,
		labels=(results_perm['feature'][-40:]).to_numpy(),
	)
	fig.tight_layout()
	plt.show()
	plt.savefig('/home/z5209394/ew-honoursthesis/Graphs/feature_importance.png')
	return results_perm['feature'], results_perm['perm_idx_average']


def density_plots(perm_sorted_idx, dataset_combined):
	for variable in perm_sorted_idx:
		plt.figure(figsize=(15,8))
		feature = str(variable)
		ax = sns.kdeplot(dataset_combined[feature][dataset_combined.Lacune == 1], color="darkturquoise", shade=True)
		sns.kdeplot(dataset_combined[feature][dataset_combined.Lacune == 0], color="lightcoral", shade=True)
		plt.legend(['Lacune', 'Non-Lacune'])
		title = "Density Plot of " + feature + " for lacunes and non-lacunes"
		plt.title(title)
		ax.set(xlabel='Unit Measurement of ' + feature)
		ax.set(ylabel='Density')
		plt.savefig('/home/z5209394/ew-honoursthesis/Graphs/' + feature + '.png')

