Sentiment Analysis in Twitter
====================

sentimentanalysis is the final project of the course of Natural Language Processing, whose goal was to develop a system which classifies whether a tweet is positive, negative or neutral.

The software was developed by a team of three students: [Ilaria Ceppa](https://github.com/ilariaceppa), [Hind Chfouka] (https://github.com/chfouka), [Marco Ponza](https://github.com/mponza) and Roberto Trani.


Installation and Data Format
--------------------------------

On Unix systems, go into /src directory and type:

	$ make

to install the package.


Data Format
---------------
For training data, every line in the file have to be in the format:

	<SID><tab><UID><tab><TOPIC><tab><positive|negative|neutral|objective><tab><TWITTER_MESSAGE>

For test data, every line in the file have to be in the format:

	<SID><tab><UID><tab><TWITTER_MESSAGE>

Using the Tweet Classifier, the output file will be in the format:

	<SID><tab><UID><tab><positive|negative|neutral>


Quick Start
===========

Tweet Classifier provides a simple workflow:

1. training set => preprocessing => cross validation => performance
2. training set + test set => preprocessing => classification => labeled set

To perform the cross validation go into src/main and type:

	$ python tweet-classifier.py [options] datasetpath

where:

- `datasetpath`:			directory path which contains:
						
 * training set's files (tweeti-b.tsv and tweeti-b.dev.tsv files)
 * conll files for the training set
 * test file

- `[options]` have the following parameters:

 * `--testfile TESTFILE` Specifies a different test file to classify. Default: `twitter-test-input-B`.
 *	`-c CLASSIFIER`	Chooses a classifier. Possible choices are: SVM classifier (`svm`), Bayes (`bayes`), ShortTextClassifier (`short`). Default: `short`.
 * `--classifier CLASSIFIER` See -c CLASSIFIER.
 *  `--tok TOK` Chooses a tokenizer. Possible choices are: POS tokenizer (`pos`), NLTK tokenizer (`nltk`), `string.split()` tokenizer (`split`). Default: `POStokenizer`.

 * `--tag TAG` Choose a tagger. Possible choices are: POS tagger (`pos`), NLTK tagger (`nltk`), no tagger (`none`). Default: `POStagger`.
 * `-v V` Number of folds for cross validation. Default 10.
 * `--predict` After training (without cross validation), predict the labels for the default test file.
  * `--testfile TESTFILE` Specify a different test file to classify. Default: `twitter-test-input-B`.
 *  `-p` Enables the punctuation filter.
 * `-e` Enables the tag remover.
 * `-r` Enables the repeating letters' filter.
 * `-s` Enables the stopwords filter (by using NLTK stopword list).
 * `-t` Enables the stopwords' filter (using pos tag).
 * `-u` Enables the url filter.
 * `-n` Enables the not adapter.

Examples
========

Perform 10-fold cross validation on the default training set (files tweeti-b.tsv and tweeti-b.dev.tsv files) using Pos tagger, Pos tokenizer and the Short Text Classifier, without any filter:

	$ python tweet-classifier.py ../../resources/tweeti/
	
Perform 10-fold cross validation on the default training set not using a tagger, using the NLTK tokenizer and the Bayesian Classifier, with a couple of filters:

	$ python tweet-classifier.py ../../resources/tweeti/ --tok nltk --tag none -c bayes -p -s
	
Train the svm classifier on the dataset with the pos tagger and the pos tokenizer, using the not adapter and some other filters, and then classify the tweets in file "tweet-test.tsv" writing the output in "tweet-test.out".
	
	$ python tweet-classifier.py ../../resources/tweeti/ --predict --testfile tweet-test -n -p -s
