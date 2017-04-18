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
import common

DEBUG = False

def compute_distance(points, centroid):
	return np.sqrt(np.sum((points - centroid)**2, axis=1))

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

def generate_predictions(best_score_index, test_file_name, output_predictions_file):
	data = common.read_file(test_file_name)
	submission = open(output_predictions_file, 'w')
	ids = data[:, 0]
	best_scores = data[:, best_score_index]
	print("test_id,is_duplicate", file = submission)
	for ind, id in enumerate(ids):
		print("%d,%s"%(id, best_scores[ind]), file = submission)
	submission.close()

def train_model(training_file_name):
	return 10
	# model = common.load_model("wiki")
	# question_pairs = common.read_file(training_file_name)
	# scores=common.generate_scores(question_pairs, Model)
	# true_classes = [ x.is_duplicate for x in question_pairs]
	# wmd_scores = [ x[5] for x in scores]
	# best_threshold = gen_auc_metrics(true_classes, wmd_scores)
	# return best_threshold