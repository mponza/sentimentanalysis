from classifier.classifier import Classifier
from pipeline.pipeline import Pipeline


def generatePredictionsFile( classifier, pipeline, testsetPath, predictionsPath ):
	"""
	Given a classifier and a testsetPath with the SemEval format <SID> <UID> <category> <tweet>
	the function generates a predictions file with the format <SID> <[(token,feature)]> <category>
	remark: the real predictions file must have the same format of predictions file
	"""
	if not isinstance(classifier, Classifier):
		raise ValueError("classifier must be an instance of Classifier")
	if not isinstance(pipeline, Pipeline):
		raise ValueError("pipeline must be an instance of Pipeline")

	# preparing tweets dictionary <key:SID><value:(category,[(token,feature)])> 
	allTweets = { }
	
	for line in open( testsetPath ):
		line = line.rstrip( '\n\r' )
		features = line.split( '	' )
		id = features[0] + "	" + features[1]
		if allTweets.has_key( id ):
			continue

		# apply the pipeline
		taggedText = pipeline.apply(features[3])

		if features[2] == '"positive"':
			label = 1
		if features[2] == '"negative"':
			label = -1
		if features[2] == '"neutral"':
			label = 0			
		
		# adding the label of the current tweet
		allTweets[id] = label


	# getting the classifier predictions. Assumption: the function classify_set preserves the order 
	itemSet = []
	for id, label in allTweets.items( ):
		itemSet.append( (item[1], item[0]) )
	labels = classifier.classify_set( itemSet ) 

	# generating classifier predictions file
	f = open( predictionsPath, 'w' )		   
	i = 0
	for id,item in allTweets.items( ):
		if labels[i] == str( 1 ):
			label = 'positive'
		if labels[i] == str( -1 ):
			label = 'negative'
		if labels[i] == str( 0 ):
			label = 'neutral'
		#(sid  [(token,feature)] category)
		f.write(sid + '	' + str( item[1] ) + '	' + label + '\n')
		i+=1
	
	f.close( )