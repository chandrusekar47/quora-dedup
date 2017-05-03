from __future__ import print_function
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Dropout
from sklearn.ensemble import AdaBoostClassifier
from sklearn.model_selection import cross_val_score
import keras.backend as K
import csv
import os.path
import sys
import common
import keras.models
from xgboost import XGBClassifier

USE_CACHED_MODEL = True
features = [2 , 3 , 4 , 5 , 6 , 7 , 8 , 9 , 10, 11, 12, 15, 16, 19]
# assumes ids are the first column of test_data
def generate_predictions(model, test_file_name, prediction_file_name):
	data = common.read_file(test_file_name)
	submission = open(prediction_file_name, 'w')
	ids = data[:, 0]
	x_test = data[:, features]
	prediction_score = model.predict_proba(x_test)
	print("test_id,is_duplicate", file = submission)
	for ind, question_id in enumerate(ids):
		print("%d,%s"%(question_id, prediction_score[ind][0]), file = submission)
	submission.close()

def train_model(training_file_name):
	data = common.read_file(training_file_name)
	x_train = data[:, features]
	y_train = data[:, 1]
	xgb = XGBClassifier()
	xgb.classes_ = [0, 1]
	xgb = xgb.fit(x_train, y_train)
	scores = cross_val_score(xgb, x_train, y_train, cv = 10, scoring='accuracy')
	print("Scores gotten using AdaBoost classifier (# of estimators="+str(25)+")")
	print(scores)
	print(np.mean(scores))
	return xgb