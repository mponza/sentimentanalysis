from adapter.featuresAdapter import FeaturesAdapter

class TagRemover(FeaturesAdapter):
	"""
		Eliminates tags
	"""

	def adapt_features_list(self, features_list):
		"""
			Adapts the specified list of features removing the tag from each feature set

			Returns the adapted features list.
		"""
		result = []
		for features in features_list:
			features = list(features)
			features[1] = ""
			features = tuple(features)
			result.append(features)
		return result