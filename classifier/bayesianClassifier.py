from classifier import Classifier

from nltk.classify import NaiveBayesClassifier

class BayesianClassifier(Classifier):

	def train(self, labeled_featuresets):
		"""
			Trains the classifier on the specified training set. Multiple calls to
			this method invalidates the previous ones.

			The labeled_featuresets parameter must have the format [([feature], label)]
		"""
		adapted_set = self.__adapt_labeled_featuresets(labeled_featuresets)
		self.__classifier = NaiveBayesClassifier.train(adapted_set)


	def classify_set(self, featuresets):
		"""
			Classifies the specified featuresets.

			The featuresets parameter must have the format [ [feature] ]
			Returns the most probable label of each item in according to this classifier,
			where the returned value has the format [label]
		"""
		result = []
		for item in featuresets:
			result.append(self.__classifier.classify(self.__adapt_featureset(item)))
		return result


	def __adapt_featureset(self, featureset):
		"""
			Adapts the featureset to the task.
		"""
		return dict([(feature, True) for feature in featureset])


	def __adapt_labeled_featuresets(self, labeled_featuresets):
		"""
			Adapts the labeled featureset to the task
		"""
		return [(self.__adapt_featureset(featuresets), label) for (featuresets, label) in labeled_featuresets]
