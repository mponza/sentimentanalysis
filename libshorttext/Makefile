
all: code 

code: stemmer learner

stemmer:
	make -C converter/stemmer

learner:
	make -C classifier/learner

clean:

cleanclean:
	rm -rf *.svm *.converter *.model *.config *.out *.pyc
	make -C doc clean
