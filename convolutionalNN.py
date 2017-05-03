from __future__ import print_function

from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.layers import Embedding
from keras.layers import Conv1D, GlobalMaxPooling1D, AveragePooling1D
from keras.datasets import imdb
import keras
import common
import time
import numpy as np
import pickle
import sys
import h5py
import csv

# set parameters:
maxlen = 7
embedding_dims = 300
filters = 80
kernel_size_1 = 3
kernel_size_2 = maxlen-((kernel_size_1-1)/2) # will not work for even kernel_size_1
hidden_dims = 250
batch_size = 32
epochs = 10

def sentence2vec(words_in_sentence, model):
	array_of_vectors = map(lambda (x): common.vec(x, model), words_in_sentence)
	filtered = np.array([x for x in array_of_vectors if len(x) != 0])
	return filtered.transpose()

def reload_h5dump(file_name):
	h5f = h5py.File(file_name, 'r')
	embeddings = h5f['embeddings'][:]
	h5f.close()
	return embeddings

def get_embedded_sentence(question_pairs, model):
    # dataset_x = []
    # dataset_y = []
    # for ind, question_pair in enumerate(question_pairs):
    #     v1 = sentence2vec(question_pair.question_1, model)
    #     v2 = sentence2vec(question_pair.question_2, model)
    #     v3 = sequence.pad_sequences(v1.tolist(), dtype='float', maxlen=maxlen)
    #     v4 = sequence.pad_sequences(v2.tolist(), dtype='float', maxlen=maxlen)
    #     if(len(v3)==300 and len(v4)==300):
    #         v5 = np.concatenate((v3,v4),axis=1)
    #         dataset_x.append(v5.transpose())
    #         dataset_y.append(question_pair.is_duplicate)
    # return (np.array(dataset_x),np.array(dataset_y))
    return reload_h5dump('data/train_qn_embeddings.h5')


def build_model():
	model = Sequential()

	# adding a Convolution1D, which will learn filters
	model.add(Conv1D(filters, kernel_size_1, padding='valid', activation='relu', strides=1, input_shape=(2*maxlen,embedding_dims)))
	model.add(Dropout(0.3))

	# TODO might want to add a pooling layer
	# adding a second Convolution1D to learn patterns specific to each question.
	model.add(Conv1D(10, kernel_size_2, padding='valid', activation='relu', strides=kernel_size_2))
	model.add(Dropout(0.3))
	# using global max pooling:
	model.add(GlobalMaxPooling1D())

	# # We add a vanilla hidden layer:
	model.add(Dense(hidden_dims))
	model.add(Dropout(0.2))
	model.add(Activation('relu'))

	# We project onto a single unit output layer, and squash it with a sigmoid:
	model.add(Dense(1))
	model.add(Activation('sigmoid'))
	model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
	return model

def main():
	# cmdline_args = sys.argv[1:]
	# if len(cmdline_args) == 0:
	# 	print("usage python convolutionalNN.py sample|total")
	# 	exit(-1)
	# dataset = cmdline_args[0]
	# if dataset == "sample":
	# 	time1 = time.time()
	# 	training_questions = pickle.load(open( "data/train_40k_qn_pairs.p", "rb" ))
	# 	time2 = time.time()
	# elif operation == "total":
	# 	time1 = time.time()
	# 	training_questions = pickle.load(open( "data/train_qn_pairs.p", "rb" ))
	# 	time2 = time.time()
	# else:
	# 	print("usage python convolutionalNN.py  sample|total")
	# 	exit(-1)
	# print ("Loaded Pickle : %f min" % ((time2 - time1)/60))
	# google_model = common.load_model("google")
	# time3 = time.time()
	# print ("Loaded model: %f min" % ((time3 - time2)/60))
	data = common.read_file('data/train_cleaned_features.csv')
	y_train = data[:, 1]
	x_train = get_embedded_sentence(None, None)
	# time4 = time.time()
	# print ("Obtained Embeddings: %f min" % ((time4 - time3)/60))
	cnn_model = build_model()
	cnn_model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, validation_split=0.3)
	cnn_model.save('cnn_model.h5')

def sentence2vec(words_in_sentence, model):
	array_of_vectors = map(lambda (x): common.vec(x, model), words_in_sentence)
	filtered = np.array([x for x in array_of_vectors if len(x) != 0])
	return filtered

def get_embedded_sentence(question_pairs, model):
	dataset_x = np.zeros((len(question_pairs),2*maxlen,embedding_dims))
	# dataset_y = np.zeros(len(question_pairs))
	for ind, question_pair in enumerate(question_pairs):
		v1 = sentence2vec(question_pair.question_1, model)
		v2 = sentence2vec(question_pair.question_2, model)
		if(len(v1)>0 and len(v2)>0):
			dataset_x[ind,:min(maxlen,len(v1)),:] = v1[:maxlen,:]
			dataset_x[ind,maxlen:(maxlen+min(maxlen,len(v2))),:] = v2[:maxlen,:]
			# dataset_y[ind] = float(question_pair.is_duplicate)
	return dataset_x

def run_on_test_data():
	cnn_model = keras.models.load_model('cnn_model.h5')
	test_questions = common.read_qp_dump('data/test_qn_pairs.p')
	model = common.load_model("google")
	predictions = np.zeros(len(test_questions))
	submission = open('cnn-predictions.csv', 'w')
	print("test_id,is_duplicate", file = submission)
	for ind, question_pair in enumerate(test_questions):
		combined = np.zeros((1,2*maxlen,embedding_dims))
		v1 = sentence2vec(question_pair.question_1, model)
		v2 = sentence2vec(question_pair.question_2, model)
		if(len(v1)>0 and len(v2)>0):
			combined[0,:min(maxlen,len(v1)),:] = v1[:maxlen,:]
			combined[0, maxlen:(maxlen+min(maxlen,len(v2))),:] = v2[:maxlen,:]
			print("%s,%f"%(question_pair.id, cnn_model.predict_proba(combined, verbose = False)[0][0]), file = submission)
		else:
			print("%s,0.37"%(question_pair.id, ), file = submission)
	submission.close()

def fix_submissions():
	with open('cnn-predictions.csv', "r") as file:
		all_lines_predictions = list(csv.reader(file, delimiter = ",", ))
	all_lines_predictions.pop(0)
	all_lines_predictions = np.array(all_lines_predictions)
	with open('data/test.csv', 'r') as file:
		all_question_lines = list(csv.reader(file, delimiter = ",", ))
	all_question_lines.pop(0)
	all_question_lines = np.array(all_question_lines)
	orig_ids = set(all_question_lines[:, 0])
	predicted_ids = set(all_lines_predictions[:, 0])
	missing = orig_ids.difference(predicted_ids)
	submission = open('cnn-predictions-new.csv', 'w')
	print("test_id,is_duplicate", file = submission)
	for line in all_lines_predictions:
		print("%s,%s"%(line[0], line[1]), file = submission)
	for item in missing:
		print("%s,0.16754"%(item), file = submission)
	submission.close()

if __name__ == '__main__':
	# run_on_test_data()
	fix_submissions()
