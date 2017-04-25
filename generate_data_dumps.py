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
maxlen = 7
def generate_training_sample(file_name, num_records,is_training_data=True):
    if num_records!=0:
        lines = np.array(common.read_lines_from_file(file_name))
        sampled_lines = lines[np.random.randint(len(lines), size = num_records), :]
    else:
        sampled_lines = np.array(common.read_lines_from_file(file_name))
    return common.convert_lines_to_question_pairs(sampled_lines.tolist(),is_training_data)

def saveData(file_name,out_file_name):
    training_questions = generate_training_sample(file_name, 0)
    pickle.dump( training_questions, open(out_file_name , "wb" ) )

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

def saveData2(file_name,out_file_name, model):
    time1 = time.time()
    train_qn_pairs = pickle.load(open( file_name, "rb" ))
    time2 = time.time()
    print ("Loaded Pickle : %f min" % ((time2 - time1)/60))
	train_data = get_embedded_sentence(train_qn_pairs, model)
	time3 = time.time()
	print ("Obtained Embeddings: %f min" % ((time3 - time2)/60))
    pickle.dump( train_data, open( out_file_name, "wb" ) )

if __name__ == '__main__':
    # for saving test question pairs
    test_qn_pairs = saveData("data/test.csv", "data/test_qn_pairs.p")
    # for saving test word embeddings
    model = common.load_model("google")
    test_data = get_embedded_sentence(test_qn_pairs, model)
    pickle.dump( test_data, open( "data/test_qn_embeddings.p", "wb" ) )
    # for saving train word embeddings
    saveData2("data/train_qn_pairs.p","data/train_qn_embeddings.p", model)
