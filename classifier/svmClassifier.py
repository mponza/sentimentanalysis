from classifier import Classifier

from libsvm.svm import svm_parameter
from libsvm.svm import svm_problem
from libsvm.svmutil import svm_predict
from libsvm.svmutil import svm_train

class SVMClassifier(Classifier):

	def train(self, labeled_featuresets):
		"""
			Trains the classifier on the specified training set. Multiple calls to
			this method invalidates the previous ones.

			The labeled_featuresets parameter must have the format [([feature], label)]
		"""
		self.__features_ids = {}
		self.__last_feature_id = 0
		self.__labels_ids = {}
		self.__labels = []

		y, x = self.__adapt_labeled_featuresets(labeled_featuresets)
		prob  = svm_problem(y, x)
		param = svm_parameter('-c 1000 -q')
		self.__model = svm_train(prob, param)


	def classify_set(self, featuresets):
		"""
			Classifies the specified featuresets.

			The featuresets parameter must have the format [ [feature] ]
			Returns the most probable label of each item in according to this classifier,
			where the returned value has the format [label]
		"""
		x = [self.__adapt_featureset(featureset) for featureset in featuresets]
		# create a fake labels array for the library
		y = [0]*len(x)
		p_labels = svm_predict(y, x, self.__model, "-q")[0]

		# convert the label's ids into the original form
		return [self.__labels[int(p_label_id)] for p_label_id in p_labels]


	def __adapt_featureset(self, featureset):
		"""
			Adapts the featureset to the task.

			The featureset parameter must have the format [feature]
			Returns the adapted_featureset that is a dictionary with the couples
			(feature's id, frequency)
		"""
		frequency = {}
		for feature in featureset:
			# convert the features into ids
			if feature not in self.__features_ids:
				feature_id = self.__last_feature_id
				self.__last_feature_id += 1
				self.__features_ids[feature] = feature_id
			else:
				feature_id = self.__features_ids[feature]

			# update the features frequency
			if feature_id not in frequency:
				frequency[feature_id] = 1
			else:
				frequency[feature_id] += 1
		return frequency


	def __adapt_labeled_featuresets(self, labeled_featuresets):
		"""
			Adapts the labeled featureset to the task

			The labeled_featuresets parameter must have the format [([feature], label)]
			Returns a couple (adapted_labelset, adapted_featuresets) as required
			from the libsvm library
		"""
		target_list = []
		inputs_list = []

		for featureset, label in labeled_featuresets:
			# convert the labels into ids
			if label not in self.__labels_ids:
				label_id = len(self.__labels)
				self.__labels.append(label)
				self.__labels_ids[label] = label_id
			else:
				label_id = self.__labels_ids[label]

			target_list.append(label_id)
			inputs_list.append(self.__adapt_featureset(featureset))

		return (target_list, inputs_list)
