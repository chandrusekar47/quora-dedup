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

DEBUG = False
MODEL_USED = "wiki" # "google"
REMOVE_STOP_WORDS = True
QuestionPair = namedtuple("QuestionPair", "id, q1_id, q2_id, question_1, question_2, is_duplicate")
wordnet_lemmatizer = WordNetLemmatizer()

def load_model(model_name):
	if model_name == "wiki":
		return KeyedVectors.load('data/wiki_en_1000_no_stem/en.model')
	elif model_name == "google":
		return KeyedVectors.load_word2vec_format('data/google-vec.bin', binary = True)
	return None

def compute_distance(points, centroid):
	return np.sqrt(np.sum((points - centroid)**2, axis=1))

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
			line[-2] = to_words(line[-2])
			line[-3] = to_words(line[-3])
			question_pair = QuestionPair(*line[1:])
		else:
			line[1] = to_words(line[1])
			line[2] = to_words(line[2])
			question_pair = QuestionPair(line[0], "", "", line[1], line[2], '')
		question_pairs.append(question_pair)
	return question_pairs

# is_training_data - the test data contains only four columns. this is to handle that
def read_file(input_file, is_training_data = True):
	lines = read_lines_from_file(input_file)
	return convert_lines_to_question_pairs(lines,is_training_data)

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

def sentence2vec(words_in_sentence, model):
	array_of_vectors = map(lambda (x): vec(x, model), words_in_sentence)
	filtered = np.array([x for x in array_of_vectors if len(x) != 0])
	return [] if len(filtered) == 0 else filtered.mean(axis = 0)

def generate_scores(question_pairs, model):
	scores = []
	for ind, question_pair in enumerate(question_pairs):
		v1 = sentence2vec(question_pair.question_1, model)
		v2 = sentence2vec(question_pair.question_2, model)
		#to compute number of common words
		a = set(str(question_pair.question_1).lower().split())
		b = set(str(question_pair.question_2).lower().split())
		common_words = len(a.intersection(b))
		wmd_dist = model.wmdistance(question_pair.question_1, question_pair.question_2)
		if len(v1) == 0 or len(v2) == 0:
			scores.append((question_pair.id,question_pair.is_duplicate, 0,0,0,0))
		else:
			scores.append((question_pair.id,
					question_pair.is_duplicate,
					common_words,
					((spatial.distance.cosine(v1, v2)-1)*-1),
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
	if DEBUG:
		predicted_classes = scores >= best_threshold
		predicted_classes = ['1' if x else '0' for x in predicted_classes]
		print("%s \nF1 - Score for ROC best threshold: %0.4f\nPrecision: %0.4f\nRecall: %0.4f\nAccuracy: %0.4f"%(similarity_measure, f1_score(true_class, predicted_classes, pos_label='1'), precision_score(true_class, predicted_classes, pos_label='1'), recall_score(true_class, predicted_classes, pos_label='1'),accuracy_score(true_class, predicted_classes)))
	return best_threshold

def print_scores(scores):
	if DEBUG:
		print("id,is_duplicate,cosine,euclidean,minkowski,wmd")
		for score in scores:
			print("%s,%s,%0.4f,%0.4f,%0.4f,%0.4f"%(score[0],score[1],score[2],score[3],score[4],score[5]))

def run_on_test_data(test_file_name, best_threshold):
	question_pairs = read_file(test_file_name, is_training_data = False)
	scores = generate_scores(question_pairs)
	wmd_scores = np.array([ x[5] for x in scores])
	predicted_classes = wmd_scores >= best_threshold
	predicted_classes = ['1' if x else '0' for x in predicted_classes]
	print("test_id,is_duplicate")
	for ind, question in enumerate(question_pairs):
		print("%s,%s"%(question.id, predicted_classes[ind]))

def main():
	wordnet_lemmatizer = WordNetLemmatizer()
	nltk.download('stopwords')
	QuestionPair = namedtuple("QuestionPair", "id, q1_id, q2_id, question_1, question_2, is_duplicate")
	t0 = time.time()
	Model = load_model("wiki")
	t1 = time.time()
	print("Loading the model took %f seconds" % ( t1 - t0))
	t0 = time.time()
	question_pairs = read_file("data/train_cleaned.csv")
	t1 = time.time()
	print("Loading the training sample took %f seconds" % ( t1 - t0))
	scores=generate_scores(question_pairs, Model)
	print_scores(scores)
	true_classes = [ x.is_duplicate for x in question_pairs]
	cosine_scores = [ x[2] for x in scores]
	euc_scores = [ x[3] for x in scores]
	minko_scores = [ x[4] for x in scores]
	wmd_scores = [ x[5] for x in scores]
	# gen_auc_metrics(true_classes, cosine_scores, 'cosine')
	# gen_auc_metrics(true_classes, euc_scores, 'euclidean')
	# gen_auc_metrics(true_classes, minko_scores, 'minkowski r=3')
	best_threshold = gen_auc_metrics(true_classes, wmd_scores, 'WMD score')
	run_on_test_data('data/test.csv', best_threshold)


if __name__ == '__main__':
	main()
