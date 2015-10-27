from tokenizer import Tokenizer
import subprocess
from sys import stderr

class POSTokenizer(Tokenizer):

	def __init__(self):
		self.__process = subprocess.Popen("../../bin/posTokenizer", bufsize=1024, stdin=subprocess.PIPE, stdout=subprocess.PIPE, universal_newlines=True)

	def __del__(self):
		self.__process.stdin.close()
		if self.__process.wait() <> 0:
			print >> stderr, "The subprocess POSTokenizer terminated with", self.__process.returncode

	def tokenize(self, string):
		"""
		Tokenizes the specified string

		Returns a list of token
		"""
		# adapt the string to the task
		string = string.replace("\n", " ")
		endToken = " END"
		if string.endswith(endToken):
			endToken = " ENDS"
		string = string + endToken
		endToken = endToken.lstrip(" ")

		# use the external process to tokenize the string
		res = []
		self.__process.stdin.write(string + '\n \n')
		self.__process.stdin.flush()
		while True:
			line = self.__process.stdout.readline().rstrip("\n")
			if line == endToken:
				break
			if line <> "":
				res.append(line)
		return res
