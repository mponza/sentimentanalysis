import types
from adapter.tokensAdapter import TokensAdapter
from adapter.featuresAdapter import FeaturesAdapter
from tagger.tagger import Tagger
from tagger.dummyTagger import DummyTagger
from tokenizer.tokenizer import Tokenizer
from tokenizer.dummyTokenizer import DummyTokenizer

class Pipeline():

	def __init__( self, tokenizer=None, tagger=None, token_adapters_list=[], features_adapters_list=[] ):
		if tokenizer == None:
			tokenizer = DummyTokenizer()
		if not isinstance(tokenizer, Tokenizer):
			raise ValueError("tokenizer must be an instance of Tokenizer")
		if tagger == None:
			tagger = DummyTagger()
		if not isinstance(tagger, Tagger):
			raise ValueError("tagger must be an instance of Tagger")
		if not isinstance(token_adapters_list, list):
			raise ValueError("token_adapters_list must be an instance of list")
		if not isinstance(features_adapters_list, list):
			raise ValueError("features_adapters_list must be an instance of list")

		self.__token_adapters = []
		self.__features_adapters =  []
		self.__tokenizer = tokenizer
		self.__tagger = tagger

		for f in token_adapters_list:
			if isinstance(f, types.FunctionType):
				self.__token_adapters.append(f)
			elif isinstance(f, TokensAdapter):
				self.__token_adapters.append(f.adapt_token_list)
			else:
				raise ValueError("token_adapters_list must contains TokensAdapter(s) or functions")

		for f in features_adapters_list:
			if isinstance(f, types.FunctionType):
				self.__features_adapters.append(f)
			elif isinstance(f, FeaturesAdapter):
				self.__features_adapters.append(f.adapt_features_list)
			else:
				raise ValueError("features_adapters_list must contains FeaturesAdapter(s) or functions")


	def adapt_token_list(self, token_list):
		"""
			Adapts the specified token_list in according to the internal list of adapters.

			token_list must be an instance of list in the form: [token]
			Returns the adapted token list
		"""
		if not isinstance(token_list, list):
			raise ValueError("token_list must be an instance of list")

		for fun in self.__token_adapters:
			token_list = fun( token_list )

			if not isinstance(token_list, list):
				raise ValueError("Some tokensAdapter doesn't return a list")
			for token in token_list:
				if not isinstance(token, str):
					raise ValueError("Some tokensAdapter doesn't return a list of strings")
				elif token == "":
					raise ValueError("Some adapted token list contains an empty string")

		return token_list


	def adapt_features_list(self, features_list):
		"""
			Adapts the specified features_list in according to the internal list of adapters.

			features_list must be an instance of list in the form: [(token,tag,...)]
			Returns [token + "_" + tag] where the items are taken from the
			adapted features list
		"""

		if not isinstance(features_list, list):
			raise ValueError("features_list must be an instance of list")
		if len(features_list) == 0:
			return features_list

		length = len(features_list[0])
		for features in features_list:
			if len(features) != length:
				raise ValueError("The features have different sizes")

		for fun in self.__features_adapters:
			features_list = fun( features_list )

			if not isinstance(features_list, list):
				raise ValueError("Some featuresAdapter doesn't return a list")
			for features in features_list:
				if not isinstance(features, tuple):
					raise ValueError("Some featuresAdapter doesn't return a list of tuples")
				elif len(features) != length:
					raise ValueError("Some adapted feature tuple have a different size")
	#	print ["_".join((features[:2])) for features in features_list]
	#	print features_list
	#	for features in features_list:
	#		print [features[0], str(features[1])]
		return ["_".join([features[0], str(features[1])]) for features in features_list]


	def text_to_features(self, text):
		"""
		Returns a list of [feature] built from the specified text using the following procedure:
		-------------		 ---------		  ------	   ---------------
		TokensAdapter  --->  Tokenizer  --->  Tagger  -->  FeaturesAdapter
		-------------		 ---------		  ------	   ---------------
		"""

		# tokenization
		tokens = self.__tokenizer.tokenize(text)

		# adapt the tokens
		tokens = [ tokenset for tokenset in self.adapt_token_list(tokens) ]

		# tagging
		taggedText = self.__tagger.tag(tokens)

		# adapt the features
		return [feature for feature in self.adapt_features_list(taggedText)]
