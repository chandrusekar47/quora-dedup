from __future__ import print_function
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Dropout
import keras.backend as K
import csv
import os.path
import sys
import common
import keras.models

USE_CACHED_MODEL = True

def fmeasure(y_true, y_pred):
	print (y_true)
	return f1_score(y_true, y_pred, pos_label ='1')

# assumes ids are the first column of test_data
def generate_predictions(model, test_file_name, prediction_file_name):
	data = common.read_file(test_file_name)
	submission = open(prediction_file_name, 'w')
	ids = data[:, 0]
	x_test = data[:, 2:]
	prediction_score = model.predict_proba(x_test)
	print("test_id,is_duplicate", file = submission)
	for ind, question_id in enumerate(ids):
		print("%d,%s"%(question_id, prediction_score[ind][0]), file = submission)
	submission.close()

def train_model(training_file_name):
	data = common.read_file(training_file_name)
	x_train = data[:, 2:]
	y_train = data[:, 1]
	model_file_name = "neural-network-model-45.h5"
	if  not USE_CACHED_MODEL or not os.path.isfile(model_file_name):
		model = Sequential()
		model.add(Dense(35, input_dim=18, activation='relu'))
		model.add(Dropout(0.45))
		model.add(Dense(1, activation='sigmoid'))
		model.compile(loss='binary_crossentropy',
		              optimizer='rmsprop',
		              metrics=['accuracy'])
		model.fit(x_train, y_train, epochs=100, validation_split = 0.2)
		model.save(model_file_name)
	model = keras.models.load_model(model_file_name)
	return model
