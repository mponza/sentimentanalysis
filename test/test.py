import sys
sys.path.append("../")
#from validator.manageScorer import *
from util.timer import Timer
from util.path import get_project_dir
from util.read_file import read_conll_file
from util.pipeline import Pipeline
from validator.crossValidator import CrossValidator
from classifier.shortTextClassifier import ShortTextClassifier
from classifier.bayesianClassifier import BayesianClassifier
from classifier.svmClassifier import SVMClassifier
#from classifier.linearClassifier import LinearClassifier

# parameters
file = ["tweeti-b", "tweeti-b.dev"]
#file = ["tweeti-b.dev" ]
numOfBins = 10

# adjust all path
originalFile = map(lambda path: get_project_dir() + "resources/tweeti/" + path + ".tsv", file)
conllFile = map(lambda path: get_project_dir() + "resources/conll/" + path + ".conll", file)

# timer used for timing
timer = Timer()

# classifiers to test
classifiers = {"ShortTextClassifier": ShortTextClassifier(), "SVMClassifier": SVMClassifier(), "Bayes": BayesianClassifier()}
#classifiers = {"LinearClassifier": LinearClassifier()}

#classifiers = {"Bayes": BayesianClassifier()}
# loading and processing data set
timer.start()
labeled_featuresets = read_conll_file(originalFile, conllFile, Pipeline()).values()

validator = CrossValidator(labeled_featuresets)
print "Elapsed time for data set processing: %.0fs\n" % (timer.stop()/1000)

# test the classifiers
for classifierName in classifiers:
	timer.start()
	print "- %s " % classifierName
	(acc, conf_matr, prec, recall, f_measure) = validator.validate(classifiers[classifierName], numOfBins)
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









