from adapter.featuresAdapter import FeaturesAdapter

class TagFilter(FeaturesAdapter):
	"""
		Eliminates the features that match the specified tags
	"""

	def __init__(self, tagList):
		self.__tagList = []
		if not isinstance(tagList, list):
			raise ValueError("tagList must be an instance of list")
		for tag in tagList:
			if not isinstance(tag, str):
				raise ValueError("tagList must be a list of strings")
			self.__tagList.append(tag.lower())

	def adapt_features_list(self, features_list):
		"""
			Adapts the specified list of features removing the features matching the specified tags

			Returns the adapted features list.
		"""
		result = []
		for features in features_list:
			if features[1].lower() not in self.__tagList:
				result.append(features)
		return result
