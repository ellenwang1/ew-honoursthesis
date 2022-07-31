import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import auc
from sklearn.metrics import plot_roc_curve
from sklearn.inspection import permutation_importance
import seaborn as sns

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

def feature_importance_plot(clf, dataset_pd, Y_train):
	# Feature Importance
	result = permutation_importance(clf, dataset_pd, Y_train, n_repeats=10)
	perm_sorted_idx = result.importances_mean.argsort()

	tree_importance_sorted_idx = np.argsort(clf.feature_importances_)
	tree_indices = np.arange(0, len(clf.feature_importances_)) + 0.5

	fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 8))
	ax1.barh(tree_indices[-40:], clf.feature_importances_[tree_importance_sorted_idx][-40:], height=0.7)
	ax1.set_yticks(tree_indices[-40:])
	ax1.set_yticklabels(dataset_pd.columns[tree_importance_sorted_idx][-40:])
	ax2.boxplot(
		result.importances[perm_sorted_idx][-40:].T,
		vert=False,
		labels=dataset_pd.columns[perm_sorted_idx][-40:],
	)
	fig.tight_layout()
	plt.savefig('/home/z5209394/ew-honoursthesis/Graphs/feature_importance.png')
	return list(dataset_pd.columns[perm_sorted_idx][-40:]), list(result.importances[perm_sorted_idx][-40])

def least_important_feature_plot(clf, dataset_pd, Y_train):
	# Feature Importance
	result = permutation_importance(clf, dataset_pd, Y_train, n_repeats=10)
	perm_sorted_idx = result.importances_mean.argsort()

	tree_importance_sorted_idx = np.argsort(clf.feature_importances_)
	tree_indices = np.arange(0, len(clf.feature_importances_)) + 0.5

	fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 8))
	ax1.barh(tree_indices[20:], clf.feature_importances_[tree_importance_sorted_idx][20:], height=0.7)
	ax1.set_yticks(tree_indices[20:])
	ax1.set_yticklabels(dataset_pd.columns[tree_importance_sorted_idx][20:])
	ax2.boxplot(
		result.importances[perm_sorted_idx][20:].T,
		vert=False,
		labels=dataset_pd.columns[perm_sorted_idx][20:],
	)
	fig.tight_layout()
	plt.savefig('/home/z5209394/ew-honoursthesis/Graphs/n_feature_importance.png')
	return list(dataset_pd.columns[perm_sorted_idx][20:]), list(result.importances[perm_sorted_idx][20])


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

