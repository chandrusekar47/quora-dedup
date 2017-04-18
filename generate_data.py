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

def print_combined_scores(wiki_scores, google_scores):
	print(','.join(["id","is_duplicate","q1_len","q2_len","diff_len","q1_num_of_words","q2_num_of_words","num_common_words","cosine_wiki","euclidean_wiki","minkowski_wiki","wmd_wiki","cosine_google","euclidean_google","minkowski_google","wmd_google"]))
	for ind, wiki_score in enumerate(wiki_scores):
		print("{},{},{:0.4f},{:0.4f},{:0.4f},{:0.4f},{:0.4f},{:0.4f},{:0.4f},{:0.4f},{:0.4f},{:0.4f},{:0.4f},{:0.4f},{:0.4f},{:0.4f}".format(*(wiki_score + google_scores[ind][8:])))

if __name__ == '__main__':
	sampled_questions = quora_dedup.generate_training_sample('data/train_cleaned.csv',10)
	print("generated sample questions")
	wiki_scores = quora_dedup.generate_scores(sampled_questions, quora_dedup.load_model("wiki"))
	google_scores = quora_dedup.generate_scores(sampled_questions, quora_dedup.load_model("google"))
	print_combined_scores(wiki_scores, google_scores)



		