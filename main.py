from __future__ import print_function
import threshold_classifier
import generate_data
import sys 
import os.path

def main():
	cmdline_args = sys.argv[1:]
	if len(cmdline_args) == 0:
		print("usage python main.py generate|classify")
		exit(-1)
	operation = cmdline_args[0]
	if operation == "generate":
		files = cmdline_args[1:]
		if len(files) == 0:
			print("usage python main.py generate <input_file_name1> <input_file_name2> ...")
			exit(-1)
		for file in files:
			if not os.path.isfile(file):
				print(file + " not found")
				exit(-1)
		for file in files:
			print("generating feature file for "+file)
			generate_data.main(file, os.path.splitext(file)[0]+"_features.csv")
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