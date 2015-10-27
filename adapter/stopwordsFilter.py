from adapter.featuresAdapter import FeaturesAdapter
from nltk.corpus import stopwords

class StopwordsFilter(FeaturesAdapter):
	"""
		Eliminates the stopwords using the nltk corpus
	"""

	def __init__(self):
		self.__corpus = stopwords.words('english')
		self.__corpus.remove("not")
		self.__corpus.remove("no")

	def adapt_features_list(self, features_list):
		"""
			Adapts the specified list of features removing the stopwords.

			Returns the adapted features list.
		"""
		result = []
		for features in features_list:
			if features[0].lower().decode("utf8") not in self.__corpus:
				result.append(features)
		return result
