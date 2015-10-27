from tagger import Tagger

class DummyTagger(Tagger):

	def tag(self, tokenList):
		"""
		Tags the specified token list. This implementation tags each token with True

		The tokenList parameter must have the format [token]
		Returns a list of tagged tokens in the format [(token, tag)]
		"""
		return [(token, True) for token in tokenList]