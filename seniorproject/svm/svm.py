import matplotlib
import nltk
import numpy
import sys
import time
import os
import collections
import sklearn
import random
from nltk.parse.stanford import StanfordParser
from nltk.parse.stanford import StanfordDependencyParser
from nltk.parse.corenlp import CoreNLPDependencyParser
from nltk.parse.corenlp import CoreNLPServer
from nltk.internals import *
from sklearn import svm
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import GridSearchCV
from sklearn.decomposition import TruncatedSVD
from sklearn.manifold import TSNE
from sklearn import metrics
from matplotlib import figure
import matplotlib.pyplot as plt

corpus_directory = "/home/theotherpena/seniorproject/seniorproject/tweets/"

def read_corpus_file(author, directory):
	# Read file into tokens
	filename = corpus_directory + directory + "/" + author + "_tweets.txt"
	f = open(filename, "r")
	file_text = f.read()
	tweets = file_text.split('\n\t')
	f.close()
	g = open(corpus_directory + directory + "/" + author + "_parse.txt.predict", "r")
	parse_text = g.read()
	parses = parse_text.split('\n\n')
	g.close()
	corpus_dicts = []
	for tweet, parse in zip(tweets, parses):
		conll_lines = parse.split('\n')
		parse_data = []
		for line in conll_lines:
			conll_data = line.split('\t')
			if len(conll_data) >= 8:
				parse_data.append({'index': conll_data[0], 'form': conll_data[1], 'lemma': conll_data[2], 'utag': conll_data[3], 'xtag': conll_data[4], 'feats': conll_data[5], 'head': conll_data[6], 'deprel': conll_data[7]})
		corpus_dicts.append({'author': author, 'text': tweet, 'parse': parse_data})
	return corpus_dicts

# to be called in the analysis bit of countvectorizer
def analyse_features(tweet):
	#more of 3rd operation (dist(head, dep)): tweet tokenizer isn't working here, for whatever reason ("-2" is being parsed as "-", "2")
	
	tokenizer = nltk.tokenize.TweetTokenizer(strip_handles=False, reduce_len=False, preserve_case=True)
	tokens = tokenizer.tokenize(tweet)
	# tokens = tweet.split(' ')
	# print tokens
	#return tokens
	# Normal operation (word), or 1st (pos) or 2nd (word-pos)
	# tokenize each tweet using NLTK's TweetTokenizer; we want handles and idiosyncracies of case and of word length to remain.
	
	# First variation: Pos tokenizing
	#pos_tagged_tokens = nltk.pos_tag(tokens)
	#pos_tags = []
	#for tup in pos_tagged_tokens:
	#	pos_tags += tup[1]	
	#return pos_tags

	# Second variation: POS and Word combos
	pos_tagged_tokens = nltk.pos_tag(tokens)
	items = []
	for tup in pos_tagged_tokens:
		items.append(tup[0] + "-" + tup[1])
	return items
	
	
def extract_features(control_authors, trump_parodies):
	document_collection = []
	test_collection = []
	for author in control_authors:
		tweets_from_one_author = []
		tweets_from_one_author = read_corpus_file(author, "control_tweets")
		tweets_from_one_author = tweets_from_one_author[:-1]
		random.shuffle(tweets_from_one_author)
		test_collection += tweets_from_one_author[:500]
		document_collection += tweets_from_one_author[500:]

	
	for author in trump_parodies:
		tweets_from_one_parody = []
		tweets_from_one_parody = read_corpus_file(author, "trump_parodies")
		test_collection += tweets_from_one_parody[:-1]

	
	# strings feed the count vectorizer		
	tweet_list_train = []
	author_list_train = []
	parse_list_train = []
	for item in document_collection:
		author_list_train.append(item['author'])
		tweet_list_train.append(item['text']) #uncomment for 0, 1, 2
		#parse_list_train.append(item['parse']) #uncomment for #3 or #4?

	#THIRD VARIATION: dist. between head and dep
	#for parse in parse_list_train:
	#	vecstring=""
	#	for token in parse:
	#		if int(token['head']) == 0 or int(token['head']) == -1:
	#			dist = 0
	#		else:
	#			dist = int(token['index']) - int(token['head'])
	#		vecstring += str(dist)
	#		vecstring += " "
	#	tweet_list_train.append(vecstring)
	#END THIRD VARIATION. When uncommenting remember to re-include tweet_list_train above (or exclude it as need be).  

	#FOURTH VARIATION: dep., if any
	#for parse in parse_list_train:
	#	vecstring = ""
	#	for token in parse:
	#		vecstring += token['xtag']
	#		vecstring += " "
	#	tweet_list_train.append(vecstring.rstrip())
	# END FOURTH. Remember the uncomment above as well.

	#FIFTH VARIATION: dependency relation
	#for parse in parse_list_train:
	#	vecstring = ""
	#	for token in parse:
	#		vecstring += token['deprel']
	#		vecstring += " "
	#	tweet_list_train.append(vecstring.rstrip())
	#END FIFTH. Remember uncommenting above. 	
	
	print "vectorizer initialized"
	vectorizer = TfidfVectorizer(input='content', lowercase=False, preprocessor=None, analyzer='word', tokenizer=analyse_features, ngram_range=(2, 2))	
	X = vectorizer.fit_transform(tweet_list_train, author_list_train)
	y = []
	
	for author in author_list_train:
		y.append(control_authors.index(author))

	tweet_list_test = []
	author_list_test = []
	parse_list_test = []
	for item in test_collection:
		author_list_test.append(item['author'])
		tweet_list_test.append(item['text'])
		parse_list_test.append(item['parse'])

	X_test = vectorizer.transform(tweet_list_test)
	y_test = []
	for author in author_list_test:
		if author in control_authors:
			y_test.append(control_authors.index(author))
		else:
			y_test.append(2)

	
	# poly degree 3 works well on 2, 3
	# linear works 30% on 1-grams, a little worse on 2-grams and a little worse yet on 3-grams
	#C_range = 10. ** numpy.arange(-3, 3)
	#gamma_range = 10. ** numpy.arange(-3, 2)

	#param_grid = dict(gamma=gamma_range, C=C_range)
	#grid = GridSearchCV(svm.SVC(), param_grid=param_grid, cv=StratifiedKFold())	

	#appr. .775 w/ c=50, gamma=10


	# at c = 1, gamma = .001, the tests all appear to come out at about 75? Overfitting?
	
	#12/9 afternoon: c=250, gamma=.001

	print "classifier fitting"	
	classifier = svm.SVC(kernel="rbf", C=1000, gamma=.5, probability=True)
	classifier.fit(X, y)
	
	#grid.fit(X, y)

	#print("The best classifier is: \n", grid.best_estimator_)
	#print grid.grid_scores_

	#return
	
	# X is too big to hold in memory
	print "begin decomposition"
	X_decomposed = TruncatedSVD(n_components=50, random_state=0).fit_transform(X)
	print "decomp successful"

	#X is not too big to hold in memory
	#X_decomposed = X.toarray()

	#
	print "begin TSNE" 
	X_embedded = TSNE(n_components=2, perplexity=40, verbose=2).fit_transform(X_decomposed)
	print "plot TSNE"
	fig = plt.figure(figsize=(10, 10))
	ax = plt.axes(frameon=False)
	plt.setp(ax, xticks=(), yticks=())
	plt.subplots_adjust(left=0.0, bottom=0.0, right=1.0, top=0.9, wspace=0.0, hspace=0.0)
	plt.scatter(X_embedded[:, 0], X_embedded[:, 1], marker="x")
	plt.savefig("1000c05g_wordandpos_bigrams.png")
	# 

	numpy.set_printoptions(threshold=numpy.nan)
	
	predictions = classifier.predict(X_test)
	probabilities = classifier.predict_proba(X_test)
	
	#print len(probabilities), len(probabilities[0])
	#print "\n"
	#for pred, prob, author, tweet in zip(predictions, probabilities, author_list_test, tweet_list_test):
	#	print "@{3} - {2}, prediction: {0} ({1})\n".format(pred, control_authors[classifier.classes_[pred]], tweet, author)
	#	for i, class_rank in zip(range(6), prob):
	#		print "\t{0} score: {1}\n".format(control_authors[i], class_rank)

	all_authors = control_authors + trump_parodies
	print all_authors
	authors_breakdown = [[0 for a in range(11)] for b in range(11)] 

	for pred, auth in zip(predictions, author_list_test):
		if auth not in control_authors: #Parody tweet.
			auth_indx = all_authors.index(auth)
			guess_indx = pred
			authors_breakdown[auth_indx][guess_indx] += 1
		else:
			auth_indx = all_authors.index(auth)
			guess_indx = pred
			authors_breakdown[auth_indx][guess_indx] += 1

	print authors_breakdown	
	print classifier.score(X_test, y_test)
	

if __name__ == "__main__":
	control_authors = ["DouthatNYT", "jonathanchait", "realDonaldTrump", "mattyglesias", "GovMikeHuckabee", "juliaioffe"]
	trump_parodies = ["DungeonsDonald", "realDonalDrumpf", "RealDonaldTrFan", "realNFLTrump", "Writeintrump"]

	
	extract_features(control_authors, trump_parodies)
