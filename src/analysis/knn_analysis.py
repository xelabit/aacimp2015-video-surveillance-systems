import numpy as np
import pandas as pd

import subprocess
import os

def load_data_from_file(x_train_fname,y_train_fname,x_test_fname,y_test_fname):
	X_test_original = np.load(x_test_fname)
	X_train_original = np.load(x_train_fname)
	Y_test_original = np.load(y_test_fname)
	Y_train_original = np.load(y_train_fname)
	return X_train_original,Y_train_original,X_test_original,Y_test_original


def data_reshape(X_train,Y_train,X_test,Y_test,frame_step):
	size = 1
	for mult in X_train[0].shape:
		size = size * mult
	X_train_new = np.array(np.zeros( shape = (len(X_train)/frame_step , size) , dtype='uint8'))
	Y_train_new = np.array(np.zeros( shape = (len(Y_train)/frame_step) , dtype='uint8'))
	for index in range(len(X_train) / frame_step):
   		X_train_new[index] = X_train[index * frame_step].reshape(size)
   		Y_train_new[index] = Y_train[index * frame_step]
	X_test_new = np.array(np.zeros( shape = (len(X_test)/frame_step , size) , dtype='uint8'))
	Y_test_new = np.array(np.zeros( shape = (len(Y_test)/frame_step) , dtype='uint8'))
	for index in range(len(X_test) / frame_step):
   		X_test_new[index] = X_test[index * frame_step].reshape(size)
   		Y_test_new[index] = Y_test[index * frame_step]
	return X_train_new,Y_train_new,X_test_new,Y_test_new

def knn_analysis(x_train_fname,y_train_fname,x_test_fname,y_test_fname,max_neighbors=7,step=2,each_x_frame=10):
	"""Analysis for images. 
	step : for number of neighbors in KNN. 
	each_x_frame : for train and test will be taken only each x_th frame from starting set.
	Output is list of tuples with scores: (number_of_neighbors, accuracy, precision , recall)"""
	X_train_t , Y_train_t , X_test_t , Y_test_t = load_data_from_file(x_train_fname,y_train_fname,x_test_fname,y_test_fname)
	X_train , Y_train , X_test , Y_test = data_reshape(X_train_t , Y_train_t , X_test_t , Y_test_t , each_x_frame)
	import sklearn
	scores = []
	from sklearn.neighbors import KNeighborsClassifier
	for k in range(1,max_neighbors+1,step):
		model = KNeighborsClassifier(k)
		model.fit(X_train, Y_train)
		y_pred = model.predict(X_test)
		scores.append((k,sklearn.metrics.accuracy_score(Y_test, y_pred),sklearn.metrics.precision_score(Y_test, y_pred),sklearn.metrics.recall_score(Y_test, y_pred)))
	return scores