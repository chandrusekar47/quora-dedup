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
import time
#YOLOYL
MODEL_USED = "wiki" # "google"
REMOVE_STOP_WORDS = True

nltk.download('stopwords')
QuestionPair = namedtuple("QuestionPair", "id, q1_id, q2_id, question_1, question_2, is_duplicate")

def load_word_vectors(filename):
	return KeyedVectors.load_word2vec_format(filename, binary = True)

def load_model(filename):
	return KeyedVectors.load(filename)

t0 = time.time()
Model = load_word_vectors('data/google-vec.bin')
t1 = time.time()
print("Loading the model took %f seconds" % ( t1 - t0))
# Model = load_model('data/wiki_en_1000_no_stem/en.model')

def compute_distance(points, centroid):
	return np.sqrt(np.sum((points - centroid)**2, axis=1))

def read_file(input_file):
	question_pairs = []
	with open(input_file, "rb") as file:
		reader = csv.reader(file, delimiter = ",", )
		next(reader, None)
		for line in reader:
			# line[-1] = True if line[-1] == "1" else False
			line[-2] = to_words(line[-2])
			line[-3] = to_words(line[-3])
			question_pair = QuestionPair(*line[1:])
			question_pairs.append(question_pair)
	return question_pairs

def to_words(sent):
	# remove non alphanumeric characters except period
	sent = re.sub("[^a-zA-Z\d\s.]", " ", sent)
	# remove all periods except the ones in the numbers
	sent = re.sub("([^\d])\.([^\d])",r"\1 \2",sent)
	return [x.lower().strip() for x in sent.split(" ") if x.strip() != "" and x not in stopwords.words('english')]

def vec(word):
	return [] if word not in Model.vocab else Model[word]

def distance_to_similarity(distance_value):
	return 1.0/(1+distance_value)

def sentence2vec(words_in_sentence):
	array_of_vectors = map(vec, words_in_sentence)
	filtered = np.array([x for x in array_of_vectors if len(x) != 0])
	return [] if len(filtered) == 0 else filtered.mean(axis = 0)

def generate_scores(question_pairs):
	scores = []
	for ind, question_pair in enumerate(question_pairs):
		v1 = sentence2vec(question_pair.question_1)
		v2 = sentence2vec(question_pair.question_2)
		wmd_dist = Model.wmdistance(question_pair.question_1, question_pair.question_2)
		if len(v1) == 0 or len(v2) == 0:
			scores.append((question_pair.id,question_pair.is_duplicate, 0,0,0,0))
		else:
			scores.append((question_pair.id,
					question_pair.is_duplicate,
					metrics.pairwise.cosine_similarity(v1, v2),
					distance_to_similarity(spatial.distance.euclidean(v1, v2)),
					distance_to_similarity(spatial.distance.minkowski(v1, v2, 3)),
					distance_to_similarity(wmd_dist)))
	return scores

def gen_auc_metrics(true_class, scores, similarity_measure = ''):
	fpr, tpr, thresholds = metrics.roc_curve(true_class, scores, pos_label='1')
	scores = np.array(scores)
	points = np.zeros((len(fpr), 2))
	points[:, 0] = fpr
	points[:, 1] = tpr
	best_threshold = thresholds[np.argmin(compute_distance(points, np.array([0, 1.0])))]
	predicted_classes = scores >= best_threshold
	predicted_classes = ['1' if x else '0' for x in predicted_classes]
	print("%s \nF1 - Score for ROC best threshold: %0.4f\nPrecision: %0.4f\nRecall: %0.4f"%(similarity_measure, f1_score(true_class, predicted_classes, pos_label='1'), precision_score(true_class, predicted_classes, pos_label='1'), recall_score(true_class, predicted_classes, pos_label='1')))


def print_scores(scores):
	print("id,is_duplicate,cosine,euclidean,minkowski")
	for score in scores:
		print("%s,%s,%0.4f,%0.4f,%0.4f"%(score[0],score[1],score[2],score[3],score[4]))

def main():

	t0 = time.time()
	question_pairs = read_file("data/train_sample.csv")
	t1 = time.time()
	print("Loading the training sample took %f seconds" % ( t1 - t0))

	scores=generate_scores(question_pairs)
	print_scores(scores)
	true_classes = [ x.is_duplicate for x in question_pairs]
	cosine_scores = [ x[2] for x in scores]
	euc_scores = [ x[3] for x in scores]
	minko_scores = [ x[4] for x in scores]
	wmd_scores = [ x[5] for x in scores]
	gen_auc_metrics(true_classes, cosine_scores, 'cosine')
	gen_auc_metrics(true_classes, euc_scores, 'euclidean')
	gen_auc_metrics(true_classes, minko_scores, 'minkowski r=3')
	gen_auc_metrics(true_classes, wmd_scores, 'WMD score')


if __name__ == '__main__':
	main()
