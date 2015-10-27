from tokenizer import Tokenizer

class DummyTokenizer(Tokenizer):

	def tokenize(self, string):
		"""
		Tokenizes the specified string

		Returns a list of token
		"""
		return string.split()
