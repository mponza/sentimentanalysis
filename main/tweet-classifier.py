import sys
sys.path.append('../')

import argparse
parser = argparse.ArgumentParser()

from util.timer import Timer
from util.path import get_project_dir
from util.read_file import read_tweets_file
from util.read_file import read_conll_file
from util.pipeline import Pipeline
from util.preprocessor import Preprocessor
from validator.crossValidator import CrossValidator

#dataset
parser.add_argument("datasetpath", help="Dataset path.")

parser.add_argument("--testfile", help="Specify a different test file to classify. Default 'twitter-test-input-B'",default='twitter-test-input-B')

#classifier
parser.add_argument("-c","--classifier", help="Choose a classifier. Possible choices are: SVM classifier ('svm'), Bayes ('bayes'), ShortTextClassifier ('short'). Default: ShortTextClassifier.",default='SVMshort')

#tokenizer
parser.add_argument("--tok",help="Choose a tokenizer. Possible choices are: POS tokenizer ('pos'), Nltk tokenizer ('nltk'), string.split() tokenizer ('split'). Default: POStokenizer.",default='pos')

#tagger
parser.add_argument("--tag",help="Choose a tagger. Possible choices are: POS tagger ('pos'), Nltk tagger ('nltk'), no tagger ('none'). Default: POStagger.",default='pos')

#CV
parser.add_argument("-v", help="Number of folds for cross validation. Default 10.",type = int, default=10)

#testfile
parser.add_argument("--predict", help="After training (without cross validation), predict the labels for the default test file.",action="store_true")

#filters
parser.add_argument("-p", help="Enables the punctuation filter.",action="store_true")
parser.add_argument("-e", help="Enables the tag remover.",action="store_true")
parser.add_argument("-r", help="Enables the repeating letters' filter.",action="store_true")
parser.add_argument("-s", help="Enables the stopwords' filter (using nltk's stopwords list).",action="store_true")
parser.add_argument("-t", help="Enables the stopwords' filter (using pos tag).",action="store_true")
parser.add_argument("-u", help="Enables the url filter.",action="store_true")
parser.add_argument("-n", help="Enables the not adapter.",action="store_true")
args = parser.parse_args()

######### Object creation


# timer used for timing
timer = Timer()

# a preprocessor
preprocess = Preprocessor( )

print ""
print ""
# Classifier selection
if (args.classifier == 'svm' ):
	from classifier.svmClassifier import SVMClassifier
	print "Classifier:	SVM Classifier."
	classifier = SVMClassifier()
else:
	if (args.classifier == 'bayes' ):
		from classifier.bayesianClassifier import BayesianClassifier
		print "Classifier:	Bayesian Classifier."
		classifier = BayesianClassifier()
	else:
		from classifier.shortTextClassifier import ShortTextClassifier
		print "Classifier:	Short Text Classifier."
		classifier = ShortTextClassifier()

print ""
print ""
# Tokenizer selection
if (args.tok == 'split' ):
	from tokenizer.dummyTokenizer import DummyTokenizer
	print "Tokenizer:	string.split()."
	tokenizer = DummyTokenizer()
else:
	if (args.tok == 'nltk' ):
		from tokenizer.nltkTokenizer import NLTKTokenizer
		print "Tokenizer:	Nltk Tokenizer."
		tokenizer =NLTKTokenizer()
	else:
		from tokenizer.posTokenizer import POSTokenizer
		print "Tokenizer:	Pos Tokenizer."
		tokenizer = POSTokenizer()

print ""
print ""	
# Tagger selection
if (args.tag == 'none' ):
	from tagger.dummyTagger import DummyTagger
	print "Tagger:		none."
	tagger = DummyTagger()
else:
	if (args.tag == 'nltk' ):
		from tagger.nltkTagger import NLTKTagger
		print "Tagger:		Nltk Tagger."
		tagger =NLTKTagger()
	else:
		from tagger.posTagger import POSTagger
		print "Tagger:		Pos Tagger."
		tagger = POSTagger()
#filters
prefilters = []
postfilters = []

print ""
print ""
print "Filtri attivati:\n["

if (args.n):
	from adapter.notAdapter import NotAdapter
	notadapt = NotAdapter()
	postfilters.append(notadapt)
	print "		Not adapter."
if (args.u):
	from adapter.urlAdapter import URLAdapter
	url = URLAdapter( preprocess )
	postfilters.append(url)
	print "		URLs elimination."
if (args.s):
	from adapter.stopwordsFilter import StopwordsFilter
	stopwrd = StopwordsFilter()
	postfilters.append(stopwrd)
	print "		Stopwords elimination."
if (args.t):
	from adapter.tagFilter import TagFilter
	tag_filter = TagFilter(['CC', 'CD', 'DT', 'EX', 'IN', 'LS', 'TO', 'WDT'])
	postfilters.append(tag_filter)
	print "		Stopwords elimination (using pos tag)."
if (args.p):
	from adapter.punctuationFilter import PunctuationFilter
	punc =  PunctuationFilter( preprocess )
	postfilters.append(punc)
	print "		Punctuation filter."
if (args.r):
	from adapter.repeatingLettersAdapter import RepeatingLettersAdapter
	rpt = RepeatingLettersAdapter( preprocess )
	prefilters.append(rpt)
	print "		RepeatingLetters filter."
if (args.e):
	from adapter.tagRemover import TagRemover
	tagrm = TagRemover()
	postfilters.append(tagrm)
	print "		TagRemover."

print "]"
print ""
print ""

pipeline = Pipeline( tokenizer, tagger, prefilters, postfilters )

file = ["tweeti-b", "tweeti-b.dev"]
if not args.n:
	# Load standard tweet file
	trainingfile = map(lambda path: args.datasetpath + path + ".tsv", file)
	labeled_featuresets = read_tweets_file(trainingfile, pipeline).values()
else:
	# If the not adapter filter has to be used, the program has to load the *.conll files instead
	# the conll files must be in the same dataset path specified by the user.
	trainingfile = map(lambda path: args.datasetpath + path + ".tsv", file)
	conllfile = map(lambda path: args.datasetpath + path + ".conll", file)
	labeled_featuresets = read_conll_file(trainingfile, conllfile,pipeline).values()

if not args.predict:
############ Cross Validation
	validator = CrossValidator(labeled_featuresets)
	timer.start()
	(acc, conf_matr, prec, recall, f_measure) = validator.validate(classifier, args.v)
	print "Accuracy:		%f" % acc
	print "Confusion Matrix:"
	for prec_label in conf_matr:
		for real_label in conf_matr[prec_label]:
			print "\tPredicted: "+prec_label + "\tReal: "+ real_label +"\t"+ str(conf_matr[prec_label][real_label])
	print "Precision:"
	for label in prec:
		print "\t"+ label + ":	%f" % prec[label]

	print "Recall:"
	for label in recall:
		print "\t"+ label + ":	%f" % recall[label]
	
	print "F-measure:"
	for label in f_measure:
		print "\t"+ label + ":	%f" % f_measure[label]
	print "  Elapsed time: %.0fs\n" % (timer.stop()/1000)

else:

############ Training with all the dataset and predict testfile
	timer.start()
	classifier.train( labeled_featuresets )
	print "  Elapsed time: %.0fs\n" % (timer.stop()/1000)
	testingfile =  map(lambda path: args.datasetpath + path + ".tsv", [args.testfile])
	# read the test file
	if not args.n:
		labeled_featuresets_test = read_tweets_file( testingfile, pipeline )
		#print labeled_featuresets_test[:5]
	else:
		conlltestfile = map(lambda path: args.datasetpath + path + ".conll",  [args.testfile])
		labeled_featuresets_test = read_conll_file(testingfile, conlltestfile,pipeline)
		
	for key in labeled_featuresets_test:
		labeled_featuresets_test[key] = labeled_featuresets_test[key][0]
	labeled_featuresets_test = classifier.classify_dict( labeled_featuresets_test )
	# output generation
	output = open( args.datasetpath + "/"+args.testfile+".out", 'w')
	for key, label in labeled_featuresets_test.iteritems():
  		output.write(key + "\t" + label + "\n")

	print "Output generato in " +args.datasetpath + "/"+args.testfile+".out"
	

	



















