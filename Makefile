all: short svm

short:
	make -C ./libshorttext

svm:
	make -C ./libsvm_files
	make lib -C ./libsvm_files

clean: cleanshort cleansvm 
	rm -f ../tmp/svmFile
	find . -name '*.pyc' -delete

cleanshort:
	make clean -C ./libshorttext/converter/stemmer
	make clean -C ./libshorttext/classifier/learner

cleansvm:
	make clean -C ./libsvm_files
