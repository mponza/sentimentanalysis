from classifier.classifier import Classifier

from random import shuffle
import threading

class CrossValidator:

	def __init__(self, labeled_featuresets):
		"""
			Constructs a new CrossValidator instance builded on the files specified.
			The object can be used to test a Classifier using the standard cross
			validation algorithm.

			The labeled_featuresets parameter must have the format [([feature], label)]
		"""
		if not isinstance(labeled_featuresets, list):
			raise ValueError("labeled_featuresets must be an instance of list")
		self.__tweets = []
		self.__labelToItems = {}
		self.__lock = threading.Lock()

		# read each file of the array
		for labeled_featureset in labeled_featuresets:
			if not isinstance(labeled_featureset, tuple):
				raise ValueError("labeled_featuresets must be a list of tuples")
			if len(labeled_featureset) != 2:
				raise ValueError("labeled_featuresets must be a list of pairs")
			if not isinstance(labeled_featureset[0], list):
				raise ValueError("each featureset must be a list")
			if not isinstance(labeled_featureset[1], str):
				raise ValueError("the label must be a string")

			# add the tagged tex and the label to the internal data structures
			if not self.__labelToItems.has_key(labeled_featureset[1]):
				self.__labelToItems[labeled_featureset[1]] = [labeled_featureset]
			else:
				self.__labelToItems[labeled_featureset[1]].append(labeled_featureset)

			# store the labeled_featureset
			self.__tweets.append(labeled_featureset)


	def __compute_validation_result(self, r_labels, p_labels):
		correctClassified = 0
		confusionMatrix = { }
		# a tmp list to remember all the labels
		label_list = []
		j = len(p_labels)
		while j > 0:
			j = j - 1
			# Accuracy 
			if p_labels[j] == r_labels[j]:
				correctClassified += 1
			# Confusion Matrix
			# implemented as a dictionary of dictionaries, confusionmatrix[predictedlabel][reallabel] yields
			# the number of examples that were classified as predictedlabel that were actually reallabel
			predicted_label = p_labels[j]
                        real_label = r_labels[j]
			if predicted_label not in confusionMatrix:
                            confusionMatrix[predicted_label] = {}
                            label_list.append(predicted_label)
                            confusionMatrix[predicted_label][real_label] = 1
                        else:
                            if real_label not in confusionMatrix[predicted_label]:
                                confusionMatrix[predicted_label][real_label] = 1
                            else:
                                confusionMatrix[predicted_label][real_label] += 1
		
                precision = {}
                recall = {}
                f_measure = {}
		# compute precision, recall, and f_measure counting falsepositive and false negative for each label
                for label in label_list:
                    true_positive = confusionMatrix[label][label]
		    false_positive = 0
                    false_negative = 0
                    for other_label in label_list:
                        if not other_label == label:
			# wrongly predicted as label 
                            false_positive += confusionMatrix[label][other_label]
			# wrongly predicted as other label instead of label                        
			    false_negative += confusionMatrix[other_label][label]
                    precision[label] = float(true_positive)/float(true_positive+false_positive)
                    recall[label] = float(true_positive)/float(true_positive+false_negative)
                    f_measure[label] = 2 * (precision[label]*recall[label])/(precision[label]+recall[label])
		return (float(correctClassified) / float(len(self.__tweets)), confusionMatrix, precision, recall, f_measure)
		
	
	def get_labels(self):
		"""
			Returns the list of category labels in the data set
		"""
		return self.__labelToItems.keys()

	def get_data_set_size(self):
		"""
			Returns the size of the internal labeled set.
		"""
		return len(self.__tweets)

	def validate(self, classifier, foldsNumber):
		"""
			Validates the specified classifier on the internal set using the standard
			cross validation algorithm with stratified sampling and with foldsNumber bins.
			At the end trains the classifier on the overall dataset.

			The classifier parameter must be an instance of Classifier
			The foldsNumber is the number of folds to use
			Returns the accuracy of the classifier on the entire preloaded set
		"""
		# manage special cases
		if not isinstance(classifier, Classifier):
			raise ValueError("classifier must be an instance of Classifier")
		if foldsNumber <= 0:
			raise ValueError("foldsNumber must be greater than 0")
		if not len(self.__tweets):
			return 1

		# stratify the categories among the folds
		# http://en.wikipedia.org/wiki/Stratified_sampling
		folds = []
		for i in range(0,foldsNumber):
			folds.append([])

		i = 0
		for category in self.__labelToItems:
			items = self.__labelToItems[category][:]
			shuffle(items)
			for item in items:
				folds[i].append(item)
				i = (i + 1) % foldsNumber

		# real and predicted labels (used to compute the accuracy and the confusion matrix)
		r_labels = []
		p_labels = []

		# iterate on bins
		for i in range(0, foldsNumber):

			# create the training set
			trainingSet = []
			for j in range(0, foldsNumber):
				if j <> i:
					trainingSet = trainingSet + folds[j]

			# training phase
			classifier.train(trainingSet)

			# test phase
			testSet = []
			tmp_r_labels = []
			for featureset, label in folds[i]:
				tmp_r_labels.append(label)
				testSet.append(featureset)

			tmp_p_labels = classifier.classify_set(testSet)

			with self.__lock:
				r_labels += tmp_r_labels
				p_labels += tmp_p_labels

		# trains the classifier on the overall data set
		classifier.train(self.__tweets)

		# return the accuracy starting from correct classified items
		return self.__compute_validation_result(r_labels, p_labels)
