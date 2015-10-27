class Classifier:

	def train(self, labeled_featuresets):
		"""
			Trains the classifier on the specified training set. Multiple calls to
			this method invalidates the previous ones.

			The labeled_featuresets parameter must have the format [([feature], label)]
		"""
		raise NotImplementedError()

	def classify(self, featureset):
		"""
			Classifies the specified featureset.
			Notice: this method is already implemented, you don't need to redefine it.

			The featureset parameter must have the format [feature]
			Returns the most probable item's label in according to this classifier
		"""
		return self.classify_set([featureset])[0]

	def classify_set(self, featuresets):
		"""
			Classifies the specified featuresets.

			The featuresets parameter must have the format [ [feature] ]
			Returns the most probable label of each item in according to this classifier,
			where the returned value has the format [label]
		"""
		raise NotImplementedError()

	def classify_dict(self, featuresets):
		"""
			Classifies the specified featuresets.

			The featuresets parameter must have the format { key: [feature] }
			Returns the most probable label of each item in according to this classifier,
			where the returned value has the format { key: label }
		"""
		values = featuresets.values()
		labels = self.classify_set(values)
		return dict(zip(featuresets.keys(), labels))
