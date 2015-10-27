from tokenizer import Tokenizer
from filter.preprocessor import Preprocessor

class WordTokenizer(Tokenizer):

	def __init__(self):
		"""
		Creates a new WordTokenizer instance
		"""
		self.__preprocessor = Preprocessor()

	def tokenize(self, string):
		"""
		Tokenizes the specified string

		Returns a list of token
		"""
		cleanedText = self.__preprocessor.clean(string)
		
		return cleanedText.split()