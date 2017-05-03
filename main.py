from __future__ import print_function
import threshold_classifier
import generate_data
import sys 
import os.path
import common
import numpy as np

def main():
	cmdline_args = sys.argv[1:]
	if len(cmdline_args) == 0:
		print("usage python main.py generate|classify")
		exit(-1)
	operation = cmdline_args[0]
	if operation == "generate":
		files = cmdline_args[1:]
		if len(files) > 2 or len(files) == 0:
			print("usage python main.py generate <training_file_csv> [<test_file_csv>]")
			exit(-1)
		for file in files:
			if not os.path.isfile(file):
				print(file + " not found")
				exit(-1)
		generate_data.main(files[0], os.path.splitext(files[0])[0]+"_features.csv")
		if len(files) == 2:
			generate_data.main(files[1], os.path.splitext(files[1])[0]+"_features.csv", is_training = False)
	elif operation == "classify":
		cmdline_args = cmdline_args[1:]
		classifier_to_use = cmdline_args[0]
		if len(cmdline_args) == 2:
			print("Only training the model")
			should_test = False
			training_file_name = cmdline_args[1]
		elif len(cmdline_args) == 4:
			print("Training the model and generating predictions for test set")
			should_test = True
			training_file_name = cmdline_args[1]
			test_file_name = cmdline_args[2]
			prediction_file_name = cmdline_args[3]
		else:
			print("Usage: python classify <classifier_file.py> <training-csv-file> [<testing-csv-file> <predictions-output-file>]")
			exit(-1)
		if not os.path.isfile(training_file_name):
			print(training_file_name + " not present")
			exit(-1)
		elif should_test and not os.path.isfile(test_file_name):
			print(test_file_name + " not present")
			exit(-1)
		classifier = __import__(classifier_to_use)
		model = classifier.train_model(training_file_name)
		if should_test:
			classifier.generate_predictions(model, test_file_name, prediction_file_name)
	else:
		print("usage python main.py generate|classify")
		exit(-1)

if __name__ == '__main__':
	main()
	# all_data = np.array(common.read_lines_from_file('data/train_sample.csv'))
	# transformed_before_saving = common.save_tfidf_vectorizer(all_data)
	# vectorizer = common.get_tfidf_vectorizer()
	# transformed_question_1 = vectorizer.transform(all_data[:, -3]).todense()
	# transformed_one_after = np.zeros(transformed_question_1.shape)
	# for ind, x in enumerate(all_data[:, -3]):
	# 	transformed_one_after[ind] = vectorizer.transform([x]).todense()
	# print(np.allclose(transformed_question_1, transformed_one_after))