from util.preprocessor import Preprocessor
from adapter.featuresAdapter import FeaturesAdapter

class PunctuationFilter(FeaturesAdapter):
	"""
		Eliminates punctuation in texts
	"""

	def __init__(self, preprocessor):
		if not isinstance(preprocessor, Preprocessor):
			raise ValueError("preprocessor must be an instance of Preprocessor")
		self.__preprocessor = preprocessor

	def adapt_features_list(self, features_list):
		"""
			Adapts the specified list of features replacing the urls with some fixed word

			Returns the adapted features list.
		"""
		result = []
		for features in features_list:
			replacement = self.__preprocessor.remove_punctuation(features[0])
			if features[0] != replacement:
				features = list(features)
				features[0] = replacement
				features = tuple(features)
			result.append(features)
		return result
