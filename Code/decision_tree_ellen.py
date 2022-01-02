import pandas as pd
import numpy as np
import math 
from random import Random, randrange
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from pprint import pprint

def entropy_func(c, n):
	"""
	The math formula
	"""
	return -(c*1.0/n)*math.log(c*1.0/n, 2)

def entropy_cal(c1, c2):
	"""
	Returns entropy of a group of data
	c1: count of one class
	c2: count of another class
	"""
	if c1== 0 or c2 == 0:  # when there is only one class in the group, entropy is 0
		return 0
	return entropy_func(c1, c1+c2) + entropy_func(c2, c1+c2)

# get the entropy of one big circle showing above
def entropy_of_one_division(division): 
	"""
	Returns entropy of a divided group of data
	Data may have multiple classes
	"""
	s = 0
	n = len(division)
	classes = set(division)
	for c in classes:   # for each class, get entropy
		n_c = sum(division==c)
		e = n_c*1.0/n * entropy_cal(sum(division==c), sum(division!=c)) # weighted avg
		s += e
	return s, n

# The whole entropy of two big circles combined
def get_entropy(y_predict, y_real):
	"""
	Returns entropy of a split
	y_predict is the split decision, True/Fasle, and y_true can be multi class
	"""
	if len(y_predict) != len(y_real):
		print('They have to be the same length')
		return None
	n = len(y_real)
	s_true, n_true = entropy_of_one_division(y_real[y_predict]) # left hand side entropy
	s_false, n_false = entropy_of_one_division(y_real[~y_predict]) # right hand side entropy
	s = n_true*1.0/n * s_true + n_false*1.0/n * s_false # overall entropy, again weighted average
	return s

def data_preparation(dataset):
	dataset = dataset.drop('PassengerId',axis=1)
	dataset = dataset.drop('Name', axis=1)
	dataset = dataset.drop('Cabin',axis=1)
	dataset = dataset.drop('Ticket',axis=1)
	dataset.Age.fillna(value=dataset.Age.mean(), inplace=True)
	dataset.Fare.fillna(value=dataset.Fare.mean(), inplace=True)
	dataset.Embarked.fillna(value=(dataset.Embarked.value_counts().idxmax()), inplace=True)
	enc = OneHotEncoder()
	enc.fit(dataset)
	dataset = dataset.replace({'male': 0, 'female': 1})
	dataset = dataset.replace({'C': 0, 'Q': 1, 'S': 2})
	feature_names = [i for i in dataset.columns if i != target_name]
	return dataset, feature_names

def train_test (dataset, target_name, train_size, shuffle):
	features = [i for i in dataset.columns if i != target_name]
	X = dataset[features]
	X = X.to_numpy()
	y = dataset[target_name]
	y = y.to_numpy()
	X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=train_size, shuffle = shuffle)
	return X_train, X_test, y_train, y_test


class DecisionTreeClassifier(object):
	def __init__(self, max_depth, min_samples_split):
		self.depth = 0
		self.max_depth = max_depth
		self.min_samples_split = min_samples_split
	
	def fit(self, x, y, feature_names, par_node={}, depth=0):
		if par_node is None: 
			return None
		elif len(y) < self.min_samples_split:
			return None
		elif self.all_same(y):
			return {'val':y[0]}
		elif depth >= self.max_depth:
			return None
		else: 
			col, cutoff, entropy = self.find_best_split_of_all(x, y)    # find one split given an information gain 
			print('Splitting feature', feature_names[col], 'on cutoff', cutoff)
			y_left = y[x[:, col] < cutoff]
			y_right = y[x[:, col] >= cutoff]
			# deciding if code should proceed given count in further left right splits.
			print(len(y_left))
			print(len(y_right))
			par_node = {'col': feature_names[col], 'index_col':col,
						'cutoff':cutoff,
					   'val': np.round(np.mean(y))}
			par_node['left'] = self.fit(x[x[:, col] < cutoff], y_left, feature_names, {}, depth+1)
			par_node['right'] = self.fit(x[x[:, col] >= cutoff], y_right, feature_names, {}, depth+1)
			self.depth += 1 
			self.trees = par_node
			return par_node
	
	def find_best_split_of_all(self, x, y):
		col = None
		min_entropy = 1
		cutoff = None
		for i, c in enumerate(x.T):
			entropy, cur_cutoff = self.find_best_split(c, y)
			if entropy == 0:    # find the first perfect cutoff. Stop Iterating
				return i, cur_cutoff, entropy
			elif entropy <= min_entropy:
				min_entropy = entropy
				col = i
				cutoff = cur_cutoff
		return col, cutoff, min_entropy
	
	def find_best_split(self, col, y):
		min_entropy = 10
		n = len(y)
		for value in set(col):
			y_predict = col < value
			my_entropy = get_entropy(y_predict, y)
			if my_entropy <= min_entropy:
				min_entropy = my_entropy
				cutoff = value
		return min_entropy, cutoff
	
	def all_same(self, items):
		return all(x == items[0] for x in items)
										   
	def predict(self, x):
		tree = self.trees
		results = np.array([0]*len(x))
		for i, c in enumerate(x):
			results[i] = self._get_prediction(c)
		return results
	
	def _get_prediction(self, row):
		cur_layer = self.trees
		while cur_layer.get('cutoff'):
			if row[cur_layer['index_col']] < cur_layer['cutoff']:
				if cur_layer['left'] is None:
					return cur_layer.get('val')
				else:
					cur_layer = cur_layer['left']
			else:
				if cur_layer['right'] is None:
						return cur_layer.get('val')
				else:
					cur_layer = cur_layer['right']
		else:
			return cur_layer.get('val')

#Import the dataset
dataset = pd.read_csv('../Data/Data Disaster.csv', header=0)
target_name = 'Survived'
#Dataset should return in numpy format
dataset, feature_names = data_preparation(dataset)
X_train, X_test, y_train, y_test = train_test(dataset, target_name, train_size = 0.8, shuffle = True)
#clf = DecisionTreeClassifier(max_depth=5, min_samples_split = 10)
#m = clf.fit(X_train, y_train, feature_names)
#pprint(m)
#y_pred = clf.predict(X_test)
#accuracy = accuracy_score(y_test, y_pred)
#print('Ellens Random Forest has a prediction accuracy of ', accuracy, '%')

class RandomForestClassifier(object):
	def __init__(self, max_depth, min_samples_split, n_trees, sample_size):
		self.depth = 0
		self.max_depth = max_depth
		self.min_samples_split = min_samples_split
		self.n_trees = n_trees
		self.sample_size = sample_size
	
	def fit(self, dataset, target_name, shuffle):
		trees = list()
		for i in range(self.n_trees):
			X_train, X_test, y_train, y_test = train_test(dataset, target_name, train_size = self.sample_size, shuffle = shuffle)
			clf = DecisionTreeClassifier(max_depth=5, min_samples_split = 10)
			tr = clf.fit(X_train, y_train, feature_names)
			trees.append(tr)
		self.trees = trees
		return trees
	
	def predict(self, x):
		trees = self.trees
		print(trees)
		results = np.array([0]*len(x))
		for i, tree in enumerate(trees):
			for j, data in enumerate(x):
				results[i][j] = self._get_prediction(tree, data)
		print(results)
		results = max(set(results), key=results.count)
		return results
	
	def _get_prediction(self, tree, row):
		cur_layer = tree
		while cur_layer.get('cutoff'):
			if row[cur_layer['index_col']] < cur_layer['cutoff']:
				if cur_layer['left'] is None:
					return cur_layer.get('val')
				else:
					cur_layer = cur_layer['left']
			else:
				if cur_layer['right'] is None:
						return cur_layer.get('val')
				else:
					cur_layer = cur_layer['right']
		else:
			return cur_layer.get('val')
 
# Random Forest Algorithm
#def random_forest(train, test, max_depth, min_size, sample_size, n_trees, n_features):
rf = RandomForestClassifier(max_depth = 5, min_samples_split = 10, n_trees = 2, sample_size = 0.8)
rf.fit(dataset, target_name, shuffle = True)
rf_pred = rf.predict(X_test)

## Running Sklearn Decision Tree Classifier
from sklearn import tree
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
import matplotlib as plt

"""
Split the data into a training and a testing set
"""
dataset = pd.read_csv('../Data/Data Disaster.csv', header=0)
target_name = 'Survived'
dataset, feature_names = data_preparation(dataset)
X = dataset[feature_names]
target = dataset[target_name]
X_train, X_test, y_train, y_test = train_test_split(dataset, target, test_size=0.2, random_state=42)
SKLN_tree = DecisionTreeClassifier(criterion = 'entropy', max_depth = 5, random_state = 0)
SKLN_tree_fit = SKLN_tree.fit(X_train, y_train)
prediction = SKLN_tree.predict(X_test)
pprint(SKLN_tree_fit)
print("Sklearn's prediction accuracy is: ",SKLN_tree_fit.score(X_test,y_test)*100,"%")