from tagger import Tagger
from nltk import pos_tag

class NLTKTagger(Tagger):

	def tag(self, tokenList):
		"""
		Tags the specified token list using the nltk POS tagger.

		The tokenList parameter must have the format [token]
		Returns a list of tagged tokens in the format [(token, tag)]
		"""
		return pos_tag(tokenList)