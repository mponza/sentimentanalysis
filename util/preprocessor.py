import string
import re


class Preprocessor:
	""" Classe che implmenta le varie operazioni di preprocessing """

	def replace_repeating_letters(self, word):
		""" Procedura che elimina le lettere ripetute in word, notare che si levano solo ripetizioni maggiori di tre per
			evitare l'introduzione di ulteriore rumore. In questo modo haaappy diventa happy e non hapy """
		return re.sub(r'(.)\1+', r'\1\1', word )

	def remove_punctuation(self, word):
		""" Procedura che torna word senza la punteggiatura """
		punctuation = list(string.punctuation)
		punctuation.remove('@')
		punctuation.remove('#')

		charList = list(word)
		result = ''
		for char in charList:
			if char in punctuation:
				result = result + ''
			else:
				result = result + char 
		
		return result

	def replace_url(self, word):
		if re.match('http[s]?://[\S]+', word):
			word = 'LINK'
		return word

	def replace_hash(self, word):
		if re.match('#\S+', word):
			word = 'HASHTAG'
		return word

	def replace_user (self, word):
		if re.match('@\S+', word):
			word = 'USER'
		return word
