from tokenizer import Tokenizer
from nltk.tokenize import word_tokenize

class NLTKTokenizer(Tokenizer):

	def tokenize(self, string):
		"""
		Tokenizes the specified string

		Returns a list of token
		"""
		return word_tokenize(string)
