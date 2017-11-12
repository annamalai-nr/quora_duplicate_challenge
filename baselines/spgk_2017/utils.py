# -*- coding: utf-8 -*-
"""
Created on Fri Apr 14 2017

@author: g.nikolentzos

"""

import numpy as np
import string
import re
from sklearn.svm import SVC
from sklearn.metrics import (accuracy_score,precision_recall_fscore_support)
from sklearn.model_selection import (StratifiedKFold,StratifiedShuffleSplit)


def load_file(filename):
	""" 
	Read the file containing the docs.
	
	"""
	docs =[]

	with open(filename) as f:
		for line in f:
			content = unicode(line, errors='ignore')
			docs.append(content.decode('utf-8'))
    
	return docs
	
	
def clean_str(string):
    """
    Tokenization/string cleaning for all datasets.
    """
    string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)     
    string = re.sub(r"\'s", " \'s", string) 
    string = re.sub(r"\'ve", " \'ve", string) 
    string = re.sub(r"n\'t", " n\'t", string) 
    string = re.sub(r"\'re", " \'re", string) 
    string = re.sub(r"\'d", " \'d", string) 
    string = re.sub(r"\'ll", " \'ll", string) 
    string = re.sub(r",", " , ", string) 
    string = re.sub(r"!", " ! ", string) 
    string = re.sub(r"\(", " \( ", string) 
    string = re.sub(r"\)", " \) ", string) 
    string = re.sub(r"\?", " \? ", string) 
    string = re.sub(r"\s{2,}", " ", string)

    return string.strip().split()

    
def preprocessing(docs):
	""" 
	Permorm data preprocessing.
	
	"""	   
	preprocessed_docs = []
  	
	for doc in docs:
		preprocessed_docs.append(clean_str(doc))

	return preprocessed_docs
	

def learn_model_and_predict_k_fold(K,labels):
	"""
	Given a kernel matrix, performs 10-fold cross-validation using an SVM and returns classification accuracy.
	At each iteration the optimal value of parameter C is determined using again cross-validation.
	
  	"""
  	print "\nStarted 10-fold cross validation:"

	# Number of instances
	n = len(labels)

	# Specify range of C values
	C_range = 10. ** np.arange(1,5,1)

	# Number of folds
	cv = 10

	# Output variables
	result = {}
	result["opt_c"] = np.zeros(cv)
	result["accuracy"] = np.zeros(cv)
	result["f1_score"] = np.zeros(cv)

	kf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=None)
	
	# Current iteration
	iteration = 0

	#Perform k-fold cv
	for train_indices_kf, test_indices_kf in kf.split(K,labels):
		
		labels_current = labels[train_indices_kf]
		 
		K_train = K[np.ix_(train_indices_kf, train_indices_kf)]
		labels_train = labels[train_indices_kf]

		K_test = K[np.ix_(test_indices_kf, train_indices_kf)]
		labels_test = labels[test_indices_kf]

		# Optimize parameter C
		sss = StratifiedShuffleSplit(n_splits=1, test_size=0.1, random_state=None)
		for train_index, test_index in sss.split(K_train, labels_train):
			K_C_train = K[np.ix_(train_index, train_index)]
			labels_C_train = labels[train_index]

			K_C_test = K[np.ix_(test_index, train_index)]
			labels_C_test = labels[test_index]

			best_C_acc = -1
			for i in range(C_range.shape[0]):
				C = C_range[i]
				clf = SVC(C=C,kernel='precomputed')
				clf.fit(K_C_train, labels_C_train) 
				labels_predicted = clf.predict(K_C_test)
				if accuracy_score(labels_C_test, labels_predicted) > best_C_acc:
					best_C_acc = accuracy_score(labels_C_test, labels_predicted)
					result["opt_c"][iteration] = C

		clf = SVC(C=result["opt_c"][iteration],kernel='precomputed')
		clf.fit(K_train, labels_train) 
		labels_predicted = clf.predict(K_test)
		result["accuracy"][iteration] = accuracy_score(labels_test, labels_predicted)
		result["f1_score"][iteration] = precision_recall_fscore_support(labels_test, labels_predicted, pos_label=None, average='macro')[2]
		iteration += 1
		print "Iteration " + str(iteration) + " complete"

	result["mean_accuracy"] = np.mean(result["accuracy"]) 
	result["mean_f1_score"] = np.mean(result["f1_score"])
	result["std"] = np.std(result["accuracy"])

	print "\nAverage accuracy: ", result["mean_accuracy"]
	print "Average macro f1-score: ", result["mean_f1_score"]
	print "-------------------------------------------------"