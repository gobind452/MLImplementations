from nltk.corpus import stopwords
from nltk.tag import pos_tag
from nltk.stem import WordNetLemmatizer
import nltk
import json
import numpy as np 
import string
import os
import sys
from collections import Counter

tags = ['JJ','JJR','JJS','RB','RBR','RBS']
subset = str(sys.argv[1])

lemmatizer = WordNetLemmatizer()
stop_words = list(stopwords.words('english'))
stop_words = [e.translate(str.maketrans('','',string.punctuation)) for e in stop_words]
stop_words = set(stop_words)
stop_words.discard('few')
stop_words.discard('not')
toBeAdded = set()
for e in stop_words:
	if e.endswith('nt'):
		toBeAdded.add(e[:-2])
for e in toBeAdded:
	stop_words.add(e)

def getClassProbability(review,label):
	review = review.split()
	filtered = list()
	for i in range(len(review)):
		review[i] = review[i].translate(str.maketrans('','',string.punctuation))
		review[i] = review[i].lower()
		if review[i] in vocab:
			filtered.append(vocab[review[i]][label-1])
	bigram_filtered = list()
	bigrams = list(nltk.bigrams(review))
	for bigram in bigrams:
		key = bigram[0]+' '+bigram[1]
		if key in bigram_vocab:
			bigram_filtered.append(bigram_vocab[key][label-1])
	return sum(filtered)+5*sum(bigram_filtered) + np.log(1-class_prior[label-1])

def getClassProbabilityNoBigrams(review,label):
	review = review.split()
	filtered = list()
	for i in range(len(review)):
		review[i] = review[i].translate(str.maketrans('','',string.punctuation))
		review[i] = review[i].lower()
		if review[i] in vocab:
			filtered.append(vocab[review[i]][label-1])
	return sum(filtered)+ np.log(1-class_prior[label-1])

def makePrediction(review):
	probab = [getClassProbability(review,i+1) for i in range(5)]
	return (probab.index(min(probab))+1)

def makePredictionNoBigrams(review):
	probab = [getClassProbabilityNoBigrams(review,i+1) for i in range(5)]
	return (probab.index(min(probab))+1)

def calculateMetric(count,method):
	if method == 0:
		count = [float(count[i]+1)/(word_count[i]+vocab_size) for i in range(5)]
	elif method == 1:
		count = [float(count[i]+1)/(bigram_count[i]+bigram_vocab_size) for i in range(5)]
	count = sorted(count)
	return 10*(count[-1]-count[-2])+sum(count)

def vocabCleanup(fraction1,fraction2):
	global word_count,bigram_count,vocab_size,bigram_vocab_size
	vocab_size = len(vocab.keys())
	bigram_vocab_size = len(bigram_vocab.keys())
	metric = {}
	keys = list(vocab.keys())
	for e in keys:
		metric[e] = calculateMetric(vocab[e],0)
	metric = sorted(metric.items(), key=lambda x: x[1])
	length = int(fraction1*len(metric))
	for i in range(length):
		for j in range(5):
			word_count[j] = word_count[j] - vocab[metric[i][0]][j]
		del vocab[metric[i][0]]
	metric = {}
	keys = list(bigram_vocab.keys())
	for e in keys:
		metric[e] = calculateMetric(bigram_vocab[e],1)
	metric = sorted(metric.items(), key=lambda x: x[1])
	length = int(fraction2*len(metric))
	for i in range(length):
		for j in range(5):
			bigram_count[j] = bigram_count[j] - bigram_vocab[metric[i][0]][j]
		del bigram_vocab[metric[i][0]]

def determineFractions():
	fraction1 = int(vocab_size<10000)*0.5+int(vocab_size>10000 and vocab_size<100000)*0.7+int(vocab_size>100000)*0.8
	return fraction1,fraction1

def transform(fraction1,fraction2):
	global word_count,bigram_count,vocab,bigram_vocab
	for e in vocab.keys():
		if e not in stop_words and lemmatizer.lemmatize(e) not in stop_words:
			new_vocab[e] = [0,0,0,0,0]
			for j in range(5):
				new_vocab[e][j] = round(vocab[e][j],4)
				word_count[j] = word_count[j] + new_vocab[e][j]
	for bigram in bigram_vocab.keys():
		filtered = bigram.split()
		if filtered[0] in stop_words or filtered[1] in stop_words:
			continue
		new_bigram_vocab[bigram] = [0,0,0,0,0]
		for j in range(5):
			new_bigram_vocab[bigram][j] = round(bigram_vocab[bigram][j],4)
			bigram_count[j] = bigram_count[j]+new_bigram_vocab[bigram][j]
	del vocab
	del bigram_vocab
	vocab = new_vocab
	bigram_vocab = new_bigram_vocab
	vocabCleanup(fraction1,fraction2)
	vocab_size = len(vocab.keys())
	bigram_vocab_size = len(bigram_vocab.keys())
	for e in vocab.keys():
		copy = [0,0,0,0,0]
		for j in range(5):
			copy[j] = np.log((sum(vocab[e])-vocab[e][j]+4)/(sum(word_count)-word_count[j]))
		vocab[e] = copy[:]
	for bigram in bigram_vocab.keys():
		copy = [0,0,0,0,0]
		for j in range(5):
			copy[j] = np.log((sum(bigram_vocab[bigram])-bigram_vocab[bigram][j]+4)/(sum(bigram_count)-bigram_count[j]))
		bigram_vocab[bigram] = copy[:]

def testData(method,test_path,totalPapers=0):
	vocab_size = len(vocab.keys())
	count = 0
	correct = 0
	confusion_matrix = np.zeros([5,5])
	with open(test_path) as data:
		for line in data:
			info = json.loads(line)
			review = info['text']
			if method == 0:
				prediction = makePredictionNoBigrams(review)
			else:
				prediction = makePrediction(review)
			if prediction == int(info['stars']):
				correct = correct + 1
			confusion_matrix[prediction-1][int(info['stars'])-1] = confusion_matrix[prediction-1][int(info['stars'])-1] + 1
			count = count + 1
			if count == totalPapers:
				break
	print("Accuracy",float(correct)/(totalPapers))
	print("Confusion Matrix",confusion_matrix)
	f1_score = [0,0,0,0,0]
	for i in range(len(f1_score)):
		precision = float(confusion_matrix[i][i])/sum(confusion_matrix[i])
		recall = 0
		for j in range(5):
			recall = confusion_matrix[j][i] + recall
		recall = float(confusion_matrix[i][i])/recall
		f1_score[i] = round(2*precision*recall/(precision+recall),3)
	print("F1 Score",f1_score)
	print("Macro-F1 Score",sum(f1_score)/5)

if sys.argv[1] == 'saved':
	with open('metadata.txt') as meta:
		for line in meta:
			info = json.loads(line)
			info_subset = info["0"]
			class_prior = info_subset['prior']
			word_count = info_subset['count']
			totalPapers = info_subset['papers']
			bigram_count = info_subset['bigram']

	with open('vocab_advanced.txt') as vocabulary:
		for line in vocabulary:
			vocab = json.loads(line)

	with open('bigram_vocab.txt') as bigram_vocabulary:
		for line in bigram_vocabulary:
			bigram_vocab = json.loads(line)
	vocab_size = len(vocab.keys())
	bigram_vocab_size = len(bigram_vocab.keys())
	testData(1,'test.json',1000)

else:
	train_path = sys.argv[1]
	test_path = sys.argv[2]
	vocab = {}
	bigram_vocab = {}
	new_vocab = {}
	new_bigram_vocab = {}
	bigram_count = [0,0,0,0,0]
	class_prior = [0,0,0,0,0]
	word_count = [0,0,0,0,0]
	count = 0
	with open(train_path) as data:
	    for line in data:
	    	info = json.loads(line)
	    	label = int(info['stars'])
	    	count = count + 1
	    	class_prior[label-1] = class_prior[label-1]+1
	    	tokens = info['text'].split()
	    	tagged = dict(pos_tag(tokens))
	    	bigrams = list(nltk.bigrams(tokens))
	    	for i in range(len(tokens)):
	    		tokens[i] = tokens[i].translate(str.maketrans('','',string.punctuation))
	    		tokens[i] = tokens[i].lower()
	    	tokens = Counter(tokens)
	    	for token in tokens.keys():
	    		if token.isalpha() and token not in stop_words:
	    			add = np.log(1+tokens[token])
	    			word_count[label-1] = word_count[label-1]+add
	    			if token not in vocab.keys():
	    				vocab[token] = [0,0,0,0,0]
	    			vocab[token][label-1] = vocab[token][label-1]+add
	    	for bigram in bigrams:
	    		if tagged[bigram[0]] not in tags and tagged[bigram[1]] not in tags: # One adverb or adjective
	    			continue
	    		filtered = [e.translate(str.maketrans('','',string.punctuation)) for e in bigram] # Remove punctuations
	    		if filtered[0]!=bigram[0] or filtered[1]!=bigram[1]: # Different sentences
	    			continue 
	    		filtered = [e.lower() for e in filtered]
	    		if filtered[0] in stop_words or filtered[1] in stop_words: # Already in vocab
	    			continue
	    		key = filtered[0] + ' ' + filtered[1]
	    		if key not in bigram_vocab.keys():
	    			bigram_vocab[key] = [0,0,0,0,0]
	    		bigram_vocab[key][label-1] = bigram_vocab[key][label-1] + 1
	    		bigram_count[label-1] = bigram_count[label-1] + 1
	class_prior = [float(e+1)/(count+5) for e in class_prior]
	vocab_size = len(vocab.keys())
	bigram_vocab_size = len(bigram_vocab.keys())
	fraction1,fraction2 = determineFractions()
	word_count = [0,0,0,0,0]
	bigram_count = [0,0,0,0,0]
	transform(fraction1,fraction2)
	vocab_size = len(vocab.keys())
	bigram_vocab_size = len(bigram_vocab.keys())

# Two Models (One with bigrams and other one without bigrams)
if len(sys.argv) >=4:
	if sys.argv[3] == '1':
		testData(0,test_path,count)
		testData(1,test_path,count)
	else:
		testData(1,test_path,count)
