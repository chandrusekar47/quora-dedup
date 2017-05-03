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
import sys
import common

def print_combined_scores(wiki_scores, google_scores, quora_scores, file = sys.stdout):
	print(','.join(["id","is_duplicate","q1_len","q2_len","q1_chars","q2_chars","diff_len","q1_num_of_words","q2_num_of_words","num_common_words", "levenstein", "partial_ratio",
			"cosine_wiki","euclidean_wiki","minkowski_wiki","wmd_wiki",
			"cosine_google","euclidean_google","minkowski_google","wmd_google",
			"cosine_quora","euclidean_quora","minkowski_quora","wmd_quora"]), file=file)
	for ind, wiki_score in enumerate(wiki_scores):
		float_vals = ",".join(map(lambda x: "{:0.4f}".format(x), wiki_score[2:] + google_scores[ind][12:] + quora_scores[ind][12:]))
		print("{},{},{}".format(wiki_score[0], wiki_score[1], float_vals), file = file)

def main(input_file, output_file, is_training = True):
	questions = common.read_qp_dump(input_file)
	output = open(output_file, "w")
	print("generating scores")
	model = common.load_model("wiki")
	wiki_scores = common.generate_scores(questions, model)
	model = common.load_model("google")
	google_scores = common.generate_scores(questions, model)
	model = common.load_model("quora")
	quora_scores = common.generate_scores(questions, model)
	print_combined_scores(wiki_scores, google_scores, quora_scores, file = output)
	output.close()