from __future__ import print_function

from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.layers import Embedding
from keras.layers import Conv1D, GlobalMaxPooling1D, AveragePooling1D
from keras.datasets import imdb
import common
import time
import numpy as np
import pickle
import sys

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

def get_embedded_sentence(question_pairs, model):
    dataset_x = []
    dataset_y = []
    for ind, question_pair in enumerate(question_pairs):
        v1 = sentence2vec(question_pair.question_1, model)
        v2 = sentence2vec(question_pair.question_2, model)
        v3 = sequence.pad_sequences(v1.tolist(), dtype='float', maxlen=maxlen)
        v4 = sequence.pad_sequences(v2.tolist(), dtype='float', maxlen=maxlen)
        if(len(v3)==300 and len(v4)==300):
            v5 = np.concatenate((v3,v4),axis=1)
            dataset_x.append(v5.transpose())
            dataset_y.append(question_pair.is_duplicate)
    return (np.array(dataset_x),np.array(dataset_y))

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
	cmdline_args = sys.argv[1:]
	if len(cmdline_args) == 0:
		print("usage python convolutionalNN.py sample|total")
		exit(-1)
	dataset = cmdline_args[0]
	if dataset == "sample":
		time1 = time.time()
		training_questions = pickle.load(open( "data/train_40k_qn_pairs.p", "rb" ))
		time2 = time.time()
	elif operation == "total":
		time1 = time.time()
		training_questions = pickle.load(open( "data/train_qn_pairs.p", "rb" ))
		time2 = time.time()
	else:
		print("usage python convolutionalNN.py  sample|total")
		exit(-1)
	print ("Loaded Pickle : %f min" % ((time2 - time1)/60))
	google_model = common.load_model("google")
	time3 = time.time()
	print ("Loaded model: %f min" % ((time3 - time2)/60))
	x_train, y_train = get_embedded_sentence(training_questions, google_model)
	time4 = time.time()
	print ("Obtained Embeddings: %f min" % ((time4 - time3)/60))
	cnn_model = build_model()
	cnn_model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, validation_split=0.3)

if __name__ == '__main__':
	main()
