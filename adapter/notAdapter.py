from adapter.featuresAdapter import FeaturesAdapter


class NotAdapter(FeaturesAdapter):
	"""
		Manage 'not' tokens by propagating the negation	on token's head and neighbors'

	"""


	def __propagate_not(self, indexhead, features_list, childrenNot):
		"""
		Propagates 'not' on the indexhead's children recursively.

		"""
		for i in range(0, len(features_list)):
			lemma, tag, parent, dependency_label = features_list[i]
			# propagate on indexhead's children
			if parent == indexhead and dependency_label != "ROOT" and parent != i:
				childrenNot[i] += 1
				self.__propagate_not(i, features_list, childrenNot)

	def adapt_features_list(self, features_list):
		"""
			Adapts the specified list of features removing the stopwords.
			featrues_list in the form: [ (lemma, tag, dependency head, dependency label) ]
			Returns the adapted features list.
		"""

		notList = ["not", "no"]

		# childrenNot is a map { key: tuple index in features_list, value: number of 'not' propagated }
		childrenNot = dict( (i,0) for i in range(0, len(features_list)) )
		for lemma, tag, parent, dependency_label in features_list:
			# if the feature is 'not' I propagate on the parent and on my neighbors
			if dependency_label != "ROOT" and lemma.lower() in notList:
				childrenNot[parent] += 1
				self.__propagate_not(parent, features_list, childrenNot)

		for key in childrenNot.keys():
			if childrenNot[key] % 2 == 1:
				features_list[key] = ("N_" + features_list[key][0], features_list[key][1], features_list[key][2], features_list[key][3] )

		return features_list