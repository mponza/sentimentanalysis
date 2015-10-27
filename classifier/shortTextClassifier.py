from classifier import Classifier
from util.path import get_project_dir

from libshorttext.converter import Text2svmConverter
from libshorttext.classifier.classifier_impl import train_converted_text
from libshorttext.classifier.learner.learner_impl import predict

class ShortTextClassifier(Classifier):
	# static fields
	__tmpPath = (get_project_dir() + 'tmp/').replace(' ', '\ ')
	__svmFilePath = __tmpPath + 'svmFile'

	# library parameters
	__converter_arguments = '-stopword 0 -stemming 0 -feature 0'
	"""
		Preprocessor options.
		-stopword : 0=no stopword removal | 1=stopword removal
		-stemming : 0=no stemming | 1=stemming
		-feature : 0=unigram | 1=bigram
	"""
	__grid_arguments	  = '0'
	"""
		Grid search for the penalty parameter in linear classifiers.
		"0"   disable grid search (faster)
		"1"   enable grid search (slightly better results)
	"""
	__feature_arguments   = ''
	"""
		Feature representation. (default "-D 1 -N 1")
		"-D 1" : binary features
		"-D 0" : word count
		"-T 1" : term frequency
		"-I 1" : IDF (to use TF-IDF put "-D 0 -T 1 -I 1")

		"-N 1" : Instance-wise normalization before training/test.
	"""
	__liblinear_arguments = '-q'
	"""
		Classifier. (default "-s 4")
		"-s 4" : support vector classification by Crammer and Singer
		"-s 3" : L1-loss support vector classification
		"-s 1" : L2-loss support vector classification
		"-s 7" : logistic regression

		"-q" : quiet mode
	"""


	def train(self, labeled_featuresets):
		"""
			Trains the classifier on the specified training set. Multiple calls to
			this method invalidates the previous ones.

			The labeled_featuresets parameter must have the format [([feature], label)]
		"""

		# generate training file from labeled_featuresets
		self.__text_converter = Text2svmConverter(self.__converter_arguments)
		self.__convert_labeled_featuresets(labeled_featuresets, self.__svmFilePath)

		# train the model
		self.__model = train_converted_text(self.__svmFilePath, self.__text_converter, grid_arguments=self.__grid_arguments, feature_arguments=self.__feature_arguments, train_arguments=self.__liblinear_arguments)


	def classify_set(self, featuresets):
		"""
			Classifies the specified featuresets.

			The featuresets parameter must have the format [ [feature] ]
			Returns the most probable label of each item in according to this classifier,
			where the returned value has the format [label]
		"""

		#Generation test file form itemSet
		self.__convert_featuresets(featuresets, self.__svmFilePath)

		# classify the featuresets
		p_labels = predict(self.__svmFilePath, self.__model.svm_model, self.__liblinear_arguments)[0]
		p_labels = [self.__text_converter.getClassName(int(label)) for label in p_labels]

		return p_labels

	def __convert_featuresets(self, featuresets, output):
		"""
		Convert a text data to a LIBSVM-format data.

		The featuresets parameter must have the format [ [feature] ]
		The output parameter is the file path where the result will be stored
		"""

		if isinstance(output, str):
			output = open(output,'w')
		elif not isinstance(output, file):
			raise TypeError('output is a str or a file.')

		for featureset in featuresets:
			feat = self.__text_converter.toSVM(" ".join(featureset))
			feat = ''.join(' {0}:{1}'.format(f,feat[f]) for f in sorted(feat))

			output.write('-1 ' + feat + '\n')
		output.close()


	def __convert_labeled_featuresets(self, labeled_featuresets, output):
		"""
		Convert a text data to a LIBSVM-format data.

		The labeled_featuresets parameter must have the format [([feature], label)]
		The output parameter is the file path where the result will be stored
		"""

		if isinstance(output, str):
			output = open(output,'w')
		elif not isinstance(output, file):
			raise TypeError('output is a str or a file.')

		for featureset, label in labeled_featuresets:
			feat, label = self.__text_converter.toSVM(" ".join(featureset), label)
			feat = ''.join(' {0}:{1}'.format(f,feat[f]) for f in sorted(feat))
			if label == None:
				label = -1
			output.write(str(label) + ' ' + feat + '\n')
		output.close()
