# IMPORT MODULES

import pandas as pd
import numpy as np
import sys
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import MinMaxScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import StratifiedKFold, train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import BaggingClassifier, AdaBoostClassifier, GradientBoostingClassifier, RandomForestClassifier
from sklearn.svm import SVC

# PROCESS COMMAND LINE ARGUMENTS

data = sys.argv[1]  # Path to data
algo = sys.argv[2]  # Algorithm to use
if len(sys.argv) == 4:  # Path to file with parameter values (for algorithms that require them)
	param_file = sys.argv[3]

# DEFINE FUNCTIONS

def kNNClassifier(X, y, K):
	cvKFold = StratifiedKFold(n_splits=10, shuffle=True, random_state=0)
	scores = []
	for train_index, test_index in cvKFold.split(X, y):
		X_train, X_test = X[train_index], X[test_index]
		y_train, y_test = y[train_index], y[test_index]
		knn = KNeighborsClassifier(n_neighbors=K)
		knn.fit(X_train, y_train)
		y_pred = knn.predict(X_test)
		scores.append(accuracy_score(y_test, y_pred))
	avg_score = sum(scores)/len(scores)
	return scores, avg_score

def logregClassifier(X, y):
	cvKFold = StratifiedKFold(n_splits=10, shuffle=True, random_state=0)
	scores = []
	for train_index, test_index in cvKFold.split(X, y):
		X_train, X_test = X[train_index], X[test_index]
		y_train, y_test = y[train_index], y[test_index]
		logreg = LogisticRegression(random_state=0)
		logreg.fit(X_train, y_train)
		y_pred = logreg.predict(X_test)
		scores.append(accuracy_score(y_test, y_pred))
	avg_score = sum(scores)/len(scores)
	return scores, avg_score

def nbClassifier(X, y):
	cvKFold = StratifiedKFold(n_splits=10, shuffle=True, random_state=0)
	scores = []
	for train_index, test_index in cvKFold.split(X, y):
		X_train, X_test = X[train_index], X[test_index]
		y_train, y_test = y[train_index], y[test_index]
		nb = GaussianNB()
		nb.fit(X_train, y_train)
		y_pred = nb.predict(X_test)
		scores.append(accuracy_score(y_test, y_pred))
	avg_score = sum(scores)/len(scores)
	return scores, avg_score

def dtClassifier(X, y):
	cvKFold = StratifiedKFold(n_splits=10, shuffle=True, random_state=0)
	scores = []
	for train_index, test_index in cvKFold.split(X, y):
		X_train, X_test = X[train_index], X[test_index]
		y_train, y_test = y[train_index], y[test_index]
		dt = DecisionTreeClassifier(criterion='entropy', random_state=0)
		dt.fit(X_train, y_train)
		y_pred = dt.predict(X_test)
		scores.append(accuracy_score(y_test, y_pred))
	avg_score = sum(scores)/len(scores)
	return scores, avg_score

def bagDTClassifier(X, y, n_estimators, max_samples, max_depth):
	cvKFold = StratifiedKFold(n_splits=10, shuffle=True, random_state=0)
	scores = []
	for train_index, test_index in cvKFold.split(X, y):
		X_train, X_test = X[train_index], X[test_index]
		y_train, y_test = y[train_index], y[test_index]
		dt = DecisionTreeClassifier(criterion='entropy', random_state=0, max_depth=max_depth)
		bag = BaggingClassifier(
			base_estimator=dt, n_estimators=n_estimators, 
			max_samples=max_samples, bootstrap=True, random_state=0)
		bag.fit(X_train, y_train)
		y_pred = bag.predict(X_test)
		scores.append(accuracy_score(y_test, y_pred))
	avg_score = sum(scores)/len(scores)
	return scores, avg_score

def adaDTClassifier(X, y, n_estimators, learning_rate, max_depth):
	cvKFold = StratifiedKFold(n_splits=10, shuffle=True, random_state=0)
	scores = []
	for train_index, test_index in cvKFold.split(X, y):
		X_train, X_test = X[train_index], X[test_index]
		y_train, y_test = y[train_index], y[test_index]
		dt = DecisionTreeClassifier(criterion='entropy', random_state=0, max_depth=max_depth)
		ada = AdaBoostClassifier(base_estimator=dt, n_estimators=n_estimators, 
								learning_rate=learning_rate, random_state=0)
		ada.fit(X_train, y_train)
		y_pred = ada.predict(X_test)
		scores.append(accuracy_score(y_test, y_pred))
	avg_score = sum(scores)/len(scores)
	return scores, avg_score

def gbClassifier(X, y, n_estimators, learning_rate):
	cvKFold = StratifiedKFold(n_splits=10, shuffle=True, random_state=0)
	scores = []
	for train_index, test_index in cvKFold.split(X, y):
		X_train, X_test = X[train_index], X[test_index]
		y_train, y_test = y[train_index], y[test_index]
		gb = GradientBoostingClassifier(n_estimators=n_estimators, learning_rate=learning_rate, random_state=0)
		gb.fit(X_train, y_train)
		y_pred = gb.predict(X_test)
		scores.append(accuracy_score(y_test, y_pred))
	avg_score = sum(scores)/len(scores)
	return scores, avg_score

def bestRFClassifier(X, y):
	X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, random_state=0)
	param_grid = {'n_estimators': [10, 20, 50, 100], 
				'max_features': ['auto', 'sqrt', 'log2'],
				'max_leaf_nodes': [10, 20, 30]}
	rf = RandomForestClassifier(criterion='entropy', random_state=0)
	cvKFold = StratifiedKFold(n_splits=10, shuffle=True, random_state=0)
	grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, cv=cvKFold, return_train_score=True)
	grid_search.fit(X_train, y_train)
	print(grid_search.best_params_['n_estimators'])
	print(grid_search.best_params_['max_features'])
	print(grid_search.best_params_['max_leaf_nodes'])
	print("{:.4f}".format(grid_search.best_score_))
	print("{:.4f}".format(grid_search.score(X_test, y_test))) 

def bestLinClassifier(X, y):
	X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, random_state=0)
	param_grid = {'C': [0.001, 0.01, 0.1, 1, 10, 100], 'gamma': [0.001, 0.01, 0.1, 1, 10, 100]}
	svm = SVC(kernel='linear', random_state=0)
	cvKFold = StratifiedKFold(n_splits=10, shuffle=True, random_state=0)
	grid_search = GridSearchCV(estimator=svm, param_grid=param_grid, cv=cvKFold, return_train_score=True)
	grid_search.fit(X_train, y_train)
	print(grid_search.best_params_['C'])
	print(grid_search.best_params_['gamma'])
	print("{:.4f}".format(grid_search.best_score_))
	print("{:.4f}".format(grid_search.score(X_test, y_test)))

# PRINT RESULTS

# Preprocess data and print to console
if algo == 'P':

	# Read in data
	df = pd.read_csv(data)

	# Clean data
	df.replace(to_replace='?', value = np.nan, inplace=True )
	df.replace(to_replace='class1', value = 0, inplace=True)
	df.replace(to_replace='class2', value = 1, inplace=True )
	
	# Split data into attributes and labels
	X = df[df.columns[:-1]]
	y = df[df.columns[-1]]

	# Impute
	simp = SimpleImputer(strategy='mean')
	simp.fit(X)
	imp_X = simp.transform(X)

	# Normalise
	scaler = MinMaxScaler()
	scaler.fit(imp_X)
	norm_X = scaler.transform(imp_X)

	# Print
	dfX = pd.DataFrame(data=norm_X)
	proc_data = pd.concat([dfX, y], axis=1)
	num_rows = len(proc_data.index)
	for row in proc_data.itertuples():
		for value in row[1:-1]:
			print(f'{value:.4f}' + ',', end='')
		if row[0] == num_rows-1:
			print(row[-1], end='')
		else:
			print(row[-1])

# Train classification model (assumes preprocessed data)
else:

	# Read in data
	df = pd.read_csv(data)

	# Split data and convert to numpy arrays
	X = np.array(df[df.columns[:-1]])
	y = np.array(df[df.columns[-1]])

	if algo == 'NN':

		# Read parameters
		params = pd.read_csv(param_file)
		K = params['K'][0]

		# Print results
		scores, avg_score = kNNClassifier(X, y, K)
		print(f'{avg_score:.4f}', end='') 

	elif algo == 'LR':
		scores, avg_score = logregClassifier(X, y)
		print(f'{avg_score:.4f}', end='')

	elif algo =='NB':
		scores, avg_score = nbClassifier(X, y)
		print(f'{avg_score:.4f}', end='')

	elif algo == 'DT':
		scores, avg_score = dtClassifier(X, y)
		print(f'{avg_score:.4f}', end='')

	elif algo == 'BAG':
		params = pd.read_csv(param_file)
		n_estimators = params['n_estimators'][0]
		max_samples = params['max_samples'][0]
		max_depth = params['max_depth'][0]

		scores, avg_score = bagDTClassifier(X, y, n_estimators, max_samples, max_depth)
		print(f'{avg_score:.4f}', end='')

	elif algo == 'ADA':
		params = pd.read_csv(param_file)
		n_estimators = params['n_estimators'][0]
		learning_rate = params['learning_rate'][0]
		max_depth = params['max_depth'][0]

		scores, avg_score = adaDTClassifier(X, y, n_estimators, learning_rate, max_depth)
		print(f'{avg_score:.4f}', end='')

	elif algo == 'GB':
		params = pd.read_csv(param_file)
		n_estimators = params['n_estimators'][0]
		learning_rate = params['learning_rate'][0]

		scores, avg_score = gbClassifier(X, y, n_estimators, learning_rate)
		print(f'{avg_score:.4f}', end='')

	elif algo == 'RF':
		bestRFClassifier(X,y)

	elif algo == 'SVM':
		bestLinClassifier(X, y)




























