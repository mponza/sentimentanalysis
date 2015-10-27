from tagger import Tagger
import subprocess
from sys import stderr

class POSTagger(Tagger):

	def __init__(self):
		self.__process = subprocess.Popen(["../../bin/posTagger", "-l", "../../resources/models/english.hmm"], bufsize=2048, stdin=subprocess.PIPE, stdout=subprocess.PIPE, universal_newlines=True)

	def __del__(self):
		self.__process.stdin.close()
		if self.__process.wait() <> 0:
			stderr.write("The subprocess POSTagger terminated with %d" % self.__process.returncode)

	def tag(self, tokenList):
		"""
		Tags the specified token list

		The tokenList parameter must have the format [token]
		Returns a list of tagged tokens in the format [(token, tag)]
		"""
		# send the token list to the process
		for token in tokenList:
			self.__process.stdin.write(token + '\n')
		self.__process.stdin.write('\n')
		self.__process.stdin.flush()

		# read the results
		res = []
		for token in tokenList:
			tmp = self.__process.stdout.readline().rstrip("\n")
			if tmp == "":
				continue

			token, tag, lemma  = tmp.split("	")
			if lemma <> "<unknown>":
				res.append((lemma, tag))
			else:
				res.append((token, tag))

		# consume the empty line after the result
		self.__process.stdout.readline()

		return res
