import sys
sys.path.append('../')

from util.path import get_project_dir
from util.read_file import read_tweets_file
from util.pipeline import Pipeline

from tokenizer.posTokenizer import POSTokenizer
from tagger.posTagger import POSTagger

# classifier
from classifier.shortTextClassifier import ShortTextClassifier
classifier =  ShortTextClassifier( )

# file paths
originalFile = map(lambda path: get_project_dir() + "resources/tweeti/" + path + ".tsv", ["tweeti-b", "tweeti-b.dev"])
testingFile =  map(lambda path: get_project_dir() + "resources/tweeti/" + path + ".tsv", ["twitter-test-input-B"])

# initialize the pipeline used to transform the tweets
tokenizer = POSTokenizer()
tagger = POSTagger()
pipeline = Pipeline( tokenizer, tagger, [], [] )

# read the training file
labeled_featuresets = read_tweets_file( originalFile, pipeline ).values( )

# training
classifier.train( labeled_featuresets )

# read the test file
labeled_featuresets_test = read_tweets_file( testingFile, pipeline )
for key in labeled_featuresets_test:
	labeled_featuresets_test[key] = labeled_featuresets_test[key][0]

# classification
labeled_featuresets_test = classifier.classify_dict( labeled_featuresets_test )

# output generation
output = open( get_project_dir() + "resources/twitter-test-input-B.out", 'w')
for key, label in labeled_featuresets_test.iteritems():
    output.write(key + "\t" + label + "\n")

print "FINISH"