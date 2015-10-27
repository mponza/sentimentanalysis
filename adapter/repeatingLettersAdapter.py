from util.preprocessor import Preprocessor
from adapter.tokensAdapter import TokensAdapter

class RepeatingLettersAdapter(TokensAdapter):
	"""
		Eliminates repeating letters
	"""

	def __init__(self, preprocessor):
		if not isinstance(preprocessor, Preprocessor):
			raise ValueError("preprocessor must be an instance of Preprocessor")
		self.__preprocessor = preprocessor

	def adapt_token_list(self, token_list):
		"""
			Adapts the specified list of token removing the repeated letters

			Returns the adapted token list.
		"""
		result = []
		for token in token_list:
			token = self.__preprocessor.replace_repeating_letters(token)
			if len(token) != 0:
				result.append(token)
		return result
