import time

class Timer:

	def __init__(self):
		"""
		Initializes a new Timer
		"""
		self.__startTime = 0

	def start(self):
		"""
		Starts the timing
		"""
		self.__startTime = time.time()

	def stop(self):
		"""
		Stops the timing

		Returns the elapsed time in ms
		"""
		return (time.time() - self.__startTime) * 1000