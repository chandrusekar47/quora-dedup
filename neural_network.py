from __future__ import print_function
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Dropout
import keras.backend as K
import csv
import os.path
import sys
import common

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
	model = Sequential()
	model.add(Dense(35, input_dim=9, activation='relu'))
	model.add(Dense(1, activation='sigmoid'))
	model.compile(loss='binary_crossentropy',
	              optimizer='rmsprop',
	              metrics=['accuracy'])
	model.fit(x_train, y_train, epochs=100, validation_split = 0.2)
	model_output_file = open("neural-network-model.json", "w")
	model_output_file.write(model.to_json())
	model_output_file.close()
	return model
