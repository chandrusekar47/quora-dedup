from __future__ import print_function
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Dropout
import keras.backend as K
import csv
import os.path
import sys

def read_file(input_file):
	records = []
	with open(input_file, "rb") as file:
		reader = csv.reader(file, delimiter = ",", )
		next(reader, None)
		for line in reader:
			if line[1].strip() == "":
				line[1] = "0"
			record = np.array(line).astype('float')
			records.append(record)
	return np.array(records)

def fmeasure(y_true, y_pred):
	print (y_true)
	return f1_score(y_true, y_pred, pos_label ='1')

# assumes ids are the first column of test_data
def generate_predictions(model, test_file_name, prediction_file_name):
	data = read_file(test_file_name)
	submission = open(prediction_file_name, 'w')
	ids = data[:, 0]
	x_test = data[:, 2:]
	prediction_score = model.predict_proba(x_test)
	print("test_id,is_duplicate", file = submission)
	for ind, question_id in enumerate(ids):
		print("%d,%s"%(question_id, prediction_score[ind][0]), file = submission)
	submission.close()

def main():
	if len(sys.argv) == 2:
		print("Only training the model")
		should_test = False
		training_file_name = sys.argv[1]
	elif len(sys.argv) == 4:
		print("Training the model and generating predictions for test set")
		should_test = True
		training_file_name = sys.argv[1]
		test_file_name = sys.argv[2]
		prediction_file_name = sys.argv[3]
	else:
		print("Usage: python classifier.py <training-csv-file> [<testing-csv-file> <predictions-output-file>]")
		exit(-1)

	if not os.path.isfile(training_file_name):
		print(training_file_name + " not present")
		exit(-1)
	elif should_test and not os.path.isfile(test_file_name):
		print(test_file_name + " not present")
		exit(-1)

	data = read_file(training_file_name)
	x_train = data[:, 2:]
	y_train = data[:, 1]

	model = Sequential()
	model.add(Dense(15, input_dim=9, activation='relu'))
	model.add(Dense(1, activation='sigmoid'))

	model.compile(loss='binary_crossentropy',
	              optimizer='rmsprop',
	              metrics=['accuracy'])

	model.fit(x_train, y_train, epochs=100, validation_split = 0.2)
	if should_test:
		generate_predictions(model, test_file_name, prediction_file_name)

if __name__ == '__main__':
	main()
