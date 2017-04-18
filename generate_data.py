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
import quora_dedup
import sys
import common

def print_combined_scores(wiki_scores, google_scores):
	print(','.join(["id","is_duplicate","q1_len","q2_len","diff_len","q1_num_of_words","q2_num_of_words","num_common_words","cosine_wiki","euclidean_wiki","minkowski_wiki","wmd_wiki","cosine_google","euclidean_google","minkowski_google","wmd_google"]), file=file)
	for ind, wiki_score in enumerate(wiki_scores):
		print("{},{},{:0.4f},{:0.4f},{:0.4f},{:0.4f},{:0.4f},{:0.4f},{:0.4f},{:0.4f},{:0.4f},{:0.4f},{:0.4f},{:0.4f},{:0.4f},{:0.4f}".format(*(wiki_score + google_scores[ind][8:])), file = file)

def main(input_file, output_file):
	training_questions = common.read_file_as_questions(input_file)
	train_output = open(output_file, "w")
	wiki_model = common.load_model("wiki")
	wiki_scores = common.generate_scores(training_questions, wiki_model)
	google_model = common.load_model("google")
	google_scores = common.generate_scores(training_questions, google_model)
	print_combined_scores(wiki_scores, google_scores, file = train_output)
	train_output.close()