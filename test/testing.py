from util.timer import Timer
from util.path import get_project_dir
from util.read_file import read_tweets_file
from util.read_file import read_conll_file
from util.pipeline import Pipeline
from util.preprocessor import Preprocessor

from validator.crossValidator import CrossValidator
from tokenizer.posTokenizer import POSTokenizer
from tagger.posTagger import POSTagger

from classifier.bayesianClassifier import BayesianClassifier
from classifier.svmClassifier import SVMClassifier
from classifier.shortTextClassifier import ShortTextClassifier

from adapter.punctuationFilter import PunctuationFilter
from adapter.repeatingLettersAdapter import RepeatingLettersAdapter
from adapter.stopwordsFilter import StopwordsFilter
from adapter.tagFilter import TagFilter
from adapter.tagRemover import TagRemover
from adapter.urlAdapter import URLAdapter
from adapter.notAdapter import NotAdapter


# parameters
file = ["tweeti-b", "tweeti-b.dev"]
tokenizer = POSTokenizer()
tagger = POSTagger()
numOfBins = 10

# timer used for timing
timer = Timer()

# a preprocessor
preprocess = Preprocessor( )

# different filters
punc =  PunctuationFilter( preprocess )
stopwrd = StopwordsFilter()
rpt = RepeatingLettersAdapter( preprocess )
tag_filter = TagFilter(['CC', 'CD', 'DT', 'EX', 'IN', 'LS', 'TO', 'WDT'])
tagrm = TagRemover()
url = URLAdapter( preprocess )
notadapt = NotAdapter()

# classifiers to test
classifiers = {"Bayes": BayesianClassifier(), "SVM": SVMClassifier(), "ShortTextClassifier": ShortTextClassifier()}

# to perform a quick test
#file = ["tweeti-b.dev"]
classifiers = {"ShortTextClassifier": ShortTextClassifier()}


# adjust all path
useConllFile = True
originalFile = map(lambda path: get_project_dir() + "resources/tweeti/" + path + ".tsv", file)
conllFile = map(lambda path: get_project_dir() + "resources/conll/" + path + ".conll", file)


# support function
def test_pipeline(pipeline):
	"""
		Support function used to test a pipeline using the specified testSet
	"""
	if not isinstance(pipeline, Pipeline):
		raise ValueError("pipeline must be an instance of Pipeline")

	timer.start()
	if not useConllFile:
		labeled_featuresets = read_tweets_file(originalFile, pipeline).values()
	else:
		labeled_featuresets = read_conll_file(originalFile, conllFile, pipeline).values()

	validator = CrossValidator(labeled_featuresets)
	print "Elapsed time for data set processing: %.0fs\n" % (timer.stop()/1000)

	# test the classifiers
	for classifierName in classifiers:
		timer.start()
		print "- %s " % classifierName,
		print "accuracy:	%f" % validator.validate(classifiers[classifierName], numOfBins)[0]
		print "  Elapsed time: %.0fs\n" % (timer.stop()/1000)


"""
			TEST
"""

print "	********** TEST 1 ***********   "
print "	stopwords: no"
print "	url: no"
print "	punctuation: no"
print "	repeating letters: no"
print "	tag removing: no"
print "	not adapter: no\n"

test_pipeline( Pipeline( tokenizer, tagger ) )



print "	********** TEST 2 ***********   "
print "	stopwords: yes"
print "	url: no"
print "	punctuation: no"
print "	repeating letters: no"
print "	tag removing: no"
print "	not adapter: no\n"

test_pipeline( Pipeline( tokenizer, tagger, [], [stopwrd] ) )


print "	********** TEST 3 ***********	"
print "	stopwords: yes"
print "	url: yes"
print "	punctuation: no"
print "	repeating letters: no"
print "	tag removing: no"
print "	not adapter: no\n"

test_pipeline( Pipeline( tokenizer, tagger, [], [url, stopwrd] ) )


print "	********** TEST 4 ***********	"
print "	stopwords: yes"
print "	url: yes"
print "	punctuation: yes"
print "	repeating letters: no"
print "	tag removing: no"
print "	not adapter: no\n"

test_pipeline( Pipeline( tokenizer, tagger, [], [url, stopwrd, punc] ) )


print "	********** TEST 5 ***********	"
print "	stopwords: yes"
print "	url: yes"
print "	punctuation: yes"
print "	repeating letters: yes"
print "	tag removing: no"
print "	not adapter: no\n"

test_pipeline( Pipeline( tokenizer, tagger, [rpt], [url, stopwrd, punc] ) )


print "	********** TEST 6 POST ***********	"
print "	stopwords: yes on tag"
print "	url: yes"
print "	punctuation: yes"
print "	repeating letters: no"
print "	tag removing: no"
print "	not adapter: no\n"

test_pipeline( Pipeline( tokenizer, tagger, [], [url, tag_filter, punc] ) )

print "	********** TEST 7 ***********	"
print "	stopwords: yes"
print "	url: yes"
print "	punctuation: no"
print "	repeating letters: yes"
print "	tag removing: yes"
print "	not adapter: no\n"

test_pipeline( Pipeline( tokenizer, tagger, [rpt], [url, stopwrd, tagrm] ) )

print "	********** TEST 8 ***********	"
print "	stopwords: no"
print "	url: no"
print "	punctuation: no"
print "	repeating letters: no"
print "	tag removing: no"
print "	not adapter: yes\n"

test_pipeline( Pipeline( tokenizer, tagger, [], [notadapt] ) )


print "	********** TEST 9 ***********   "
print "	stopwords: yes"
print "	url: no"
print "	punctuation: no"
print "	repeating letters: no"
print "	tag removing: no"
print "	not adapter: yes\n"

test_pipeline( Pipeline( tokenizer, tagger, [], [notadapt, stopwrd] ) )


print "	********** TEST 10 ***********	"
print "	stopwords: yes"
print "	url: yes"
print "	punctuation: no"
print "	repeating letters: no"
print "	tag removing: no"
print "	not adapter: yes\n"

test_pipeline( Pipeline( tokenizer, tagger, [], [notadapt, url, stopwrd] ) )


print "	********** TEST 11 ***********	"
print "	stopwords: yes"
print "	url: yes"
print "	punctuation: yes"
print "	repeating letters: no"
print "	tag removing: no"
print "	not adapter: yes\n"

test_pipeline( Pipeline( tokenizer, tagger, [], [notadapt, url, stopwrd, punc] ) )


print "	********** TEST 12 ***********	"
print "	stopwords: yes"
print "	url: yes"
print "	punctuation: yes"
print "	repeating letters: yes"
print "	tag removing: no"
print "	not adapter: yes\n"

test_pipeline( Pipeline( tokenizer, tagger, [rpt], [notadapt, url, stopwrd, punc] ) )


print "	********** TEST 13 POST ***********	"
print "	stopwords: yes on tag"
print "	url: yes"
print "	punctuation: yes"
print "	repeating letters: no"
print "	tag removing: no"
print "	not adapter: yes\n"

test_pipeline( Pipeline( tokenizer, tagger, [], [notadapt, url, tag_filter, punc] ) )


print "	********** TEST 14 ***********	"
print "	stopwords: yes"
print "	url: yes"
print "	punctuation: no"
print "	repeating letters: yes"
print "	tag removing: yes"
print "	not adapter: yes\n"

test_pipeline( Pipeline( tokenizer, tagger, [rpt], [notadapt, url, stopwrd, tagrm] ) )

print "FINE"