from __future__ import print_function
import csv
from gensim.models import *
import re
import numpy as np
from scipy import spatial
from collections import namedtuple
from sklearn import metrics
from sklearn.metrics import *
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import time
import editdistance
from fuzzywuzzy import fuzz
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer

DEBUG = False
MODEL_USED = "wiki" # "google"
REMOVE_STOP_WORDS = True

QuestionPair = namedtuple("QuestionPair", "id, q1_id, q2_id, question_1, question_2, is_duplicate, q1_str, q2_str")
wordnet_lemmatizer = WordNetLemmatizer()

def load_model(model_name):
	if model_name == "wiki":
		return KeyedVectors.load('data/wiki_en_1000_no_stem/en.model')
	elif model_name == "google":
		return KeyedVectors.load_word2vec_format('data/google-vec.bin', binary = True)
	elif model_name == "quora":
		return KeyedVectors.load_word2vec_format('data/quora_embeddings.bin', binary = True)
	return None

def load_word_embeddings():
	(embeddings, headers) = read_lines_from_file('data/quora_embeddings.csv')
	wordvec_map = {}
	for row in embeddings:
		wordvec_map[row[0]] = np.array(row[1:]).astype('float')
	return wordvec_map

def read_lines_from_file(input_file):
	lines = []
	with open(input_file, "rb") as file:
		reader = csv.reader(file, delimiter = ",", )
		next(reader, None)
		for line in reader:
			lines.append(line)
	return lines

def convert_lines_to_question_pairs(lines, is_training_data):
	question_pairs = []
	for line in lines:
		if is_training_data:
			q1_str = line[-3]
			q2_str = line[-2]
			line[-2] = to_words(q1_str)
			line[-3] = to_words(q2_str)
			question_pair = QuestionPair(*(line[1:] + [q1_str,q2_str]))
		else:
			q1_str = line[1]
			q2_str = line[2]
			line[1] = to_words(q1_str)
			line[2] = to_words(q2_str)
			question_pair = QuestionPair(line[0], "", "", line[1], line[2], '0', q1_str, q2_str)
		question_pairs.append(question_pair)
	return question_pairs

# is_training_data - the test data contains only four columns. this is to handle that
def read_file_as_questions(input_file, is_training_data = True):
	lines = read_lines_from_file(input_file)
	return convert_lines_to_question_pairs(lines,is_training_data)

def read_qp_dump(pickle_qp_dump):
	return pickle.load(open(pickle_qp_dump, 'rb'))

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

def generate_training_sample(file_name, num_records,is_training_data=True):
	lines = np.array(read_lines_from_file(file_name))
	sampled_lines = lines[np.random.randint(len(lines), size = num_records), :]
	return convert_lines_to_question_pairs(sampled_lines.tolist(),is_training_data)

def to_words(sent):
	# remove non alphanumeric characters except period
	sent = re.sub("[^a-zA-Z\d\s.]", " ", sent)
	# remove all periods except the ones in the numbers
	sent = re.sub("([^\d])\.([^\d])",r"\1 \2",sent)
	words = [x.lower().strip() for x in sent.split(" ") if x.strip() != ""]
	if REMOVE_STOP_WORDS:
		words = [wordnet_lemmatizer.lemmatize(x) for x in words if x not in stopwords.words('english')]
	return words

def vec(word, model):
	return [] if word not in model.vocab else model[word]

def distance_to_similarity(distance_value):
	return 1.0/(1+distance_value)

def sentence2vec(words_in_sentence, model, vectorizer):
	word_vectors_array = []
	for word in words_in_sentence:
		if word in model.vocab:
			if vectorizer.vocabulary_.has_key(word):
				weights_of_word = vectorizer.idf_[vectorizer.vocabulary_[word]]
				word_vectors_array.append(weights_of_word * model[word])
	return [] if len(word_vectors_array) == 0 else np.mean(word_vectors_array, axis = 0)

def compute_num_common_words(q1,q2):
	a = set(q1)
	b = set(q2)
	return len(a.intersection(b))

def compute_num_words(q1):
	return len(q1)

def compute_word_movers_dist(q1,q2,model):
	return model.wmdistance(q1,q2)

def compute_len_of_question(q):
	total = 0
	for x in q:
		total = total + len(x)
	return total
#ADDED THIS
def compute_chars_without_spaces(s1):
	return len(s1.replace(' ',''))

def compute_difference_length(l1,l2):
	return l1-l2

def compute_levenstein_score(s1,s2): #questions as strings
	return int(editdistance.eval(s1,s2))

def compute_partial_token_ratio(s1,s2):
	return fuzz.partial_ratio(s1,s2)

def generate_scores(question_pairs, model):
	scores = []
	vectorizer = get_tfidf_vectorizer()
	for ind, question_pair in enumerate(question_pairs):
		v1 = sentence2vec(question_pair.question_1, model, vectorizer)
		v2 = sentence2vec(question_pair.question_2, model, vectorizer)
		#questions as arrays of strings
		q1 = question_pair.question_1
		q2 = question_pair.question_2

		common_words = compute_num_common_words(q1,q2)
		q1_len = compute_len_of_question(q1)
		q2_len = compute_len_of_question(q2)
		#ADDED THIS
		q1_chars = compute_chars_without_spaces(question_pair.q1_str)
		q2_chars = compute_chars_without_spaces(question_pair.q2_str)

		diff_len = compute_difference_length(q1_len,q2_len)
		q1_num_of_words = compute_num_words(q1)
		q2_num_of_words = compute_num_words(q2)
		wmd_dist = compute_word_movers_dist(q1,q2,model)
		levenstein = compute_levenstein_score(question_pair.q1_str, question_pair.q2_str)
		partial_token_ratio = compute_partial_token_ratio(question_pair.q1_str, question_pair.q2_str)
		#compute levenstein distance.. need q's as strings for that!
		#compute fuzzy partial token ratio
		if len(v1) == 0 or len(v2) == 0:
			scores.append((question_pair.id,question_pair.is_duplicate,
				q1_len,
				q2_len,
				q1_chars,
				q2_chars,
				diff_len,
				q1_num_of_words,
				q2_num_of_words,
				common_words,
				levenstein,
				partial_token_ratio,
				0,0,0,0))
		else:
			scores.append((question_pair.id,
					question_pair.is_duplicate,
					q1_len,
					q2_len,
					q1_chars,
					q2_chars,
					diff_len,
					q1_num_of_words,
					q2_num_of_words,
					common_words,
					levenstein,
					partial_token_ratio,
					((spatial.distance.cosine(v1, v2)-1)*-1),
					distance_to_similarity(spatial.distance.euclidean(v1, v2)),
					distance_to_similarity(spatial.distance.minkowski(v1, v2, 3)),
					distance_to_similarity(wmd_dist)))
	return scores

def get_tfidf_vectorizer():
	return pickle.load(open('models/tfidf_vectorizer.p', 'rb'))

def save_tfidf_vectorizer(training_data):
	vectorizer = TfidfVectorizer(analyzer = "word", 
                    tokenizer = nltk.word_tokenize, 
                    preprocessor = None, 
                    stop_words = set(stopwords.words('english')), 
                    max_features = 10000, 
                    lowercase = True)
	vectorizer.fit(np.concatenate((training_data[:, -3], training_data[:, -2])))
	pickle.dump(vectorizer, open('models/tfidf_vectorizer.p', 'wb'))
	transformed_question_1 = vectorizer.transform(training_data[:, -3])
	transformed_question_2 = vectorizer.transform(training_data[:, -2])
	return (transformed_question_1, transformed_question_2)
