from __future__ import print_function

from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.layers import Embedding
from keras.layers import Conv1D, GlobalMaxPooling1D
from keras.datasets import imdb
import common
import numpy as np
import pickle
import time
import sys
import gc
import h5py

maxlen = 7
embedding_dims = 300

def generate_training_sample(file_name, num_records,is_training_data=False):
	if num_records!=0:
		lines = np.array(common.read_lines_from_file(file_name))
		sampled_lines = lines[np.random.randint(len(lines), size = num_records), :]
	else:
		sampled_lines = np.array(common.read_lines_from_file(file_name))
	return common.convert_lines_to_question_pairs(sampled_lines.tolist(),is_training_data)

def saveData(file_name,out_file_name):
	training_questions = generate_training_sample(file_name, 0)
	pickle.dump( training_questions, open(out_file_name , "wb" ) )
	return training_questions

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

def saveData2(file_name,out_file_name):
	model = common.load_model("google")
	time1 = time.time()
	train_qn_pairs = pickle.load(open( file_name, "rb" ))
	time2 = time.time()
	print ("Loaded Pickle : %f min" % ((time2 - time1)/60))
	train_data = get_embedded_sentence(train_qn_pairs, model)
	time3 = time.time()
	model = None
	train_qn_pairs = None
	gc.collect()
	print ("Obtained Embeddings: %f min" % ((time3 - time2)/60))
	h5f = h5py.File(out_file_name, 'w')
	h5f.create_dataset('embeddings', train_data.shape, data = train_data)
	h5f.close()
	# pickle.dump( train_data, open( out_file_name, "wb" ) )

def reload_h5dump(file_name):
	h5f = h5py.File(file_name, 'r')
	embeddings = h5f['embeddings'][:]
	h5f.close()
	return embeddings

if __name__ == '__main__':
	# for saving test question pairs
	# test_qn_pairs = saveData("data/new_test.csv", "data/test_qn_pairs.p")
	# for saving test word embeddings
	# test_data = get_embedded_sentence(test_qn_pairs, model)
	# pickle.dump( test_data, open( "data/test_qn_embeddings.p", "wb" ) )
	# for saving train word embeddings
	# saveData2("data/train_qn_pairs.p","data/train_qn_embeddings.p", model)
	# saveData2(sys.argv[1], sys.argv[2])
	balh = reload_h5dump('data/train_qn_embeddings.h5')
