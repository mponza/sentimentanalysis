from util.pipeline import Pipeline


def __recognize_label(label):
	# use only 3 categories for our purpose
	# a more accurate check can avoid unexpected situations
	if label == '"positive"' or label == 'positive':
		return "positive"
	elif label == '"negative"' or label == 'negative':
		return "negative"
	else:
		return "neutral"

def __task_features(task):
	if task != "a" and task != "b":
		raise ValueError("task must be a single character between \"a\" and \"b\"")

	if task == "a":
		return {"minlen" : 5, "idlen" : 4}
	else:
		return {"minlen" : 3, "idlen" : 2}


def read_tweets_file(fileList, pipeline, task="b"):
	"""
		Reads the file from the list and apply the pipeline to the text to retrieve
		a labeled features sets

		The fileList parameter must be a list of file names. Each file must
		contains labeled tweets in the format: SID\tUID\tCLASS\tTWEET
		The pipeline must be an istance of Pipeline
		The task must be a single character between "a" and "b" (default "b")
		
		Returns a dictionary of labeled featuresets: { id: ([feature], label) }
	"""
	if not isinstance(fileList, list):
		raise ValueError("fileList must be an instance of list")
	if not isinstance(pipeline, Pipeline):
		raise ValueError("pipeline must be an instance of Pipeline")
	
	task = __task_features(task)
	result = {}

	# read each file of the array
	for filePath in fileList:

		# read line by line
		for line in open(filePath):

			# decode the line and fetch the line parts
			line = line.strip()
			parts = line.split("\t")

			if len(parts) < task["minlen"]:
				print "Line \"" + line + "\" has the wrong number of tokens"
			key = "\t".join(parts[:task["idlen"]])

			# skip items repeated multiple times
			if result.has_key(key):
				continue

			label = __recognize_label(parts[task["idlen"]])


			# use the pipeline
			featureset = pipeline.text_to_features( "\t".join(parts[task["idlen"]+1:]) )


			# store the labeled featureset
			result[key] = (featureset, label)

	return result


def read_conll_file(originalFileList, conllFileList, pipeline, task="b"):
	"""
		Reads the file from the list and retrieve the labeled feature set

		The originalFileList parameter must be a list of file names. Each file must
		contains labeled tweets in the format: SID\tUID\tCLASS\tTWEET
		The originalFileList parameter must be a list of file names of the same size
		of the previous parameter. It must respect the conll format.

		Returns a dictionary of labeled featuresets: { id: ([(lemma, tag, dependency head, dependency label)], label) }
	"""
	if not isinstance(originalFileList, list):
		raise ValueError("fileList must be an instance of list")
	if not isinstance(conllFileList, list):
		raise ValueError("fileList must be an instance of list")
	if len(originalFileList) != len(conllFileList):
		raise ValueError("originalFileList and conllFileList must be of the same size")
	if not isinstance(pipeline, Pipeline):
		raise ValueError("pipeline must be an instance of Pipeline")

	task = __task_features(task)
	result = {}

	# read each file of the array
	file_id = len(originalFileList)
	while file_id > 0:
		file_id -= 1

		conll_file = open(conllFileList[file_id])

		# read line by line
		for line in open(originalFileList[file_id]):

			# decode the line and fetch the line parts
			line = line.strip()
			parts = line.split("\t")

			if len(parts) < task["minlen"]:
				print "Line \"" + line + "\" has the wrong number of tokens"
			key = "\t".join(parts[:task["idlen"]])

			label = __recognize_label(parts[task["idlen"]])

			# compose the tagged text
			featureset = []
			while True:
				parts = conll_file.readline().strip().split("\t")
				if len(parts) < 10:
					if len(parts) > 1:
						print line
						print "\t".join(parts)
						raise Exception("bad line in conllFileList")
					break;

				featureset.append( (parts[2] ,parts[4], int(parts[6])-1, parts[7]) )

			if len(featureset) == 0:
				print line
				print "\t".join(parts)
				raise Exception("originalFileList and conllFileList aren't aligned")


			# use the pipeline
			featureset = pipeline.adapt_features_list( featureset )


			# store the labeled featureset
			if not result.has_key(key):
				result[key] = (featureset, label)

		conll_file.close()

	return result