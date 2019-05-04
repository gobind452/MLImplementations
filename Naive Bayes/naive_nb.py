from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
import json
import numpy as np 
import string
import os
import sys

#Q1 Vanilla Naive Bayes

stemmer = SnowballStemmer("english",ignore_stopwords = True) # Stemmer

#Stop Word Manipulation
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

def getClassProbability(review,label,method): # Returns the log probability for a given class and a review
	logProbab = 0
	review = review.split()
	filtered = list()
	for i in range(len(review)):
		review[i] = review[i].translate(str.maketrans('','',string.punctuation))
		review[i] = review[i].lower()
		if method == 1: #Stemming
			review[i] = stemmer.stem(review[i])
		filtered.append(review[i])
	if method == 0:
		for word in filtered:
			if word in vocab:
				logProbab = logProbab + np.log((vocab[word][label-1]+1)/(word_count[label-1]+vocab_size))
	elif method == 1:
		for word in filtered:
			if word in stemmed_vocab:
				logProbab = logProbab + np.log((stemmed_vocab[word][label-1]+1)/(stemmed_word_count[label-1]+stemmed_vocab_size))
	logProbab = logProbab + np.log(class_prior[label-1])
	return logProbab

def determineFractions():
	if vocab_size < 1000:
		return 0.1
	elif vocab_size < 10000:
		fraction1 = 0.2
	elif vocab_size < 50000:
		fraction1 = 0.4
	elif vocab_size < 100000:
		fraction1 = 0.6
	else:
		fraction1 = 0.8
	return fraction1,fraction1

def makePrediction(review,method): # Makes a prediction given a review
	probab = [getClassProbability(review,i+1,method) for i in range(5)]
	return (probab.index(max(probab))+1)

def calculateMetric(count,method): # Calculates Metric For filtering out vocabulary using thresholding
	if method == 0:
		count = [float(count[i]+1)/(word_count[i]+vocab_size) for i in range(5)]
	elif method == 1:
		count = [float(count[i]+1)/(stemmed_word_count[i]+stemmed_vocab_size) for i in range(5)]
	count = sorted(count)
	return (count[-1]-count[-2]+count[-4]-count[-5])

def vocabCleanup(fraction1,fraction2): # Cleans up the vocabulary using thresholding
	global vocab_size,stemmed_vocab_size
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
	keys = list(stemmed_vocab.keys())
	for e in keys:
		metric[e] = calculateMetric(stemmed_vocab[e],1)
	metric = sorted(metric.items(), key=lambda x: x[1])
	length = int(fraction2*len(metric))
	for i in range(length):
		for j in range(5):
			stemmed_word_count[j] = stemmed_word_count[j] - stemmed_vocab[metric[i][0]][j]
		del stemmed_vocab[metric[i][0]]
	vocab_size = len(vocab.keys())
	stemmed_vocab_size = len(stemmed_vocab.keys())

def testData(method,test_path,totalPapers=0): # Tests Data
	vocab_size = len(vocab.keys())
	count = 0
	correct = 0
	confusion_matrix = np.zeros([5,5])
	with open(test_path) as data:
		for line in data:
			info = json.loads(line)
			review = info['text']
			prediction = makePrediction(review,method)
			if prediction == int(info['stars']):
				correct = correct + 1
			confusion_matrix[prediction-1][int(info['stars'])-1] = confusion_matrix[prediction-1][int(info['stars'])-1] + 1
			count = count + 1
			if count == totalPapers:
				break
	print("Accuracy",float(correct)/(count))
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
	print("Macro F1 Score",sum(f1_score)/5)

if sys.argv[1] == 'saved': # Run for my saved vocabulary
	with open('metadata.txt') as meta:
		for line in meta:
			info = json.loads(line)
			info_subset = info["4"]
			class_prior = info_subset['prior']
			word_count = info_subset['count']
			stemmed_word_count = info_subset['stemmed_count']
	with open('vocab_normal.txt') as vocabulary:
		for line in vocabulary:
			vocabulary = json.loads(line)
	vocab = vocabulary["vocab"]
	stemmed_vocab = vocabulary["stemmed_vocab"]
	vocab_size = len(vocab.keys())
	stemmed_vocab_size = len(stemmed_vocab.keys())
	testData(0,'test.json')
	sys.exit()

else:
	vocab = {}
	class_prior = [0,0,0,0,0]
	word_count = [0,0,0,0,0]
	stemmed_word_count = [0,0,0,0,0]
	stemmed_vocab = {}
	metadata = {}
	train_path = str(sys.argv[1])
	test_path = str(sys.argv[2])
	totalPapers = 0
	with open(train_path) as data:
		for line in data:
			info = json.loads(line)
			label = int(info['stars'])
			class_prior[label-1] = class_prior[label-1]+1
			tokens = info['text'].split()
			totalPapers = totalPapers + 1
			for i in range(len(tokens)):
				tokens[i] = tokens[i].translate(str.maketrans('','',string.punctuation))
			for token in tokens:
				if token.isalpha():
					word_count[label-1] = word_count[label-1]+1
					lower = token.lower()
					if lower not in vocab:
						vocab[lower] = [0,0,0,0,0]
					vocab[lower][label-1] = vocab[lower][label-1]+1
	class_prior = [float(e+1)/(totalPapers+5) for e in class_prior]
	stemmed_word_count = word_count[:]
	for e in vocab.keys():
		stem = stemmer.stem(e)
		if e in stop_words:
			for i in range(5):
				stemmed_word_count[i] = stemmed_word_count[i]-vocab[e][i]
		elif stem in stop_words:
			for i in range(5):
				stemmed_word_count[i] = stemmed_word_count[i]-vocab[e][i]
		else:
			if stem not in stemmed_vocab.keys():
				stemmed_vocab[stem] = vocab[e][:]
			else:
				for j in range(5):
					stemmed_vocab[stem][j]= stemmed_vocab[stem][j] + vocab[e][j]

	vocab_size = len(vocab.keys())
	stemmed_vocab_size = len(stemmed_vocab.keys())
	fraction1,fraction2 = determineFractions()
	vocabCleanup(fraction1,fraction2)
	
# Part A
print("Vanilla Naive Bayes")
print("Test Data")
testData(0,test_path,totalPapers)
print("Training Data")
testData(0,train_path,totalPapers)

majority = class_prior.index(max(class_prior))+1
random_correct = 0
majority_correct = 0
count = 0

with open(test_path) as data:
	for line in data:
		info = json.loads(line)
		stars = int(info['stars'])
		random_prediction = int(np.random.randint(1,6))
		if random_prediction == stars:
			random_correct = random_correct+1
		if majority == stars:
			majority_correct = majority_correct+1
		count = count+1

print("Random Accuracy",random_correct/(totalPapers+1),"Majority Accuracy",majority_correct/(totalPapers+1))

print("Stemmed and stopword removal")
print("Test Data")
testData(1,test_path,totalPapers)
print("Train Data")
testData(1,train_path,totalPapers)