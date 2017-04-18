import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Dropout
import keras.backend as K
import csv

def read_file(input_file):
	records = []
	with open(input_file, "rb") as file:
		reader = csv.reader(file, delimiter = ",", )
		next(reader, None)
		for line in reader:
			record = np.array(line).astype('float')
			records.append(record)
	return np.array(records)

def fmeasure(y_true, y_pred):
	print (y_true)
	return f1_score(y_true, y_pred, pos_label ='1')

def main():
	# Generate dummy data
	data = read_file("data/sample_10000.csv")
	input_features = [2,4,5,6,8,9]
	x_train = data[:, input_features]
	y_train = data[:, 1]

	model = Sequential()
	model.add(Dense(15, input_dim=9, activation='relu'))
	model.add(Dense(1, activation='sigmoid'))

	model.compile(loss='binary_crossentropy',
	              optimizer='rmsprop',
	              metrics=['accuracy'])

	model.fit(x_train, y_train, epochs=40, validation_split = 0.2)
	# score = model.evaluate(x_test, y_test, batch_size=128)
	# print score


if __name__ == '__main__':
	main()
