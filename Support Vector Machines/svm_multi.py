import pandas as pd 
import numpy as np
from cvxopt import solvers,matrix
from svmutil import *
from sklearn.metrics import pairwise
import svm_binary
import sys

train_path = sys.argv[1]
test_path = sys.argv[2]
part = sys.argv[3]

solvers.options['show_progress'] = False
examples_count = 400
test_count = 5000
gamma = 0.05 #Gaussian Kernel parameter
cost = 1 #Cost parameter in lagrangian

def updateCount(predictions,number1,number2):
	for i in range(test_count):
		if predictions[i] > 0:
			count[i][number1] = count[i][number1]+1
			score[i][number1] = score[i][number1]+predictions[i]
		else:
			count[i][number2] = count[i][number2]+1
			score[i][number2] = score[i][number2]+np.abs(predictions[i])

training_data = pd.read_csv(train_path,header=None,index_col=False)
testing_data = pd.read_csv(test_path,header=None,index_col=False)[:test_count]
validation_data = training_data.sample(frac=0.1,random_state=1)
training_data.drop(validation_data.index)

if part == 'a':
	test_labels = list(testing_data[784])[:test_count]
	del testing_data[784]
	test_data = list(testing_data.values)[:test_count]
	del test_labels[784]
	del test_data[784]
	test_count = test_count-1
	test_data = np.reshape(np.asarray(test_data),(test_count,-1))
	test_labels = np.reshape(np.asarray(test_labels),(-1,1))
	count = np.zeros([test_count,10])
	score = np.zeros([test_count,10])
	for number1 in range(9):
		for number2 in range(number1+1,10):
			binary_train = pd.DataFrame(training_data.loc[training_data[784].isin([number1,number2])])
			train_labels = np.asarray(binary_train[784])[:examples_count]
			del binary_train[784]
			train_data = np.asarray(binary_train.values)[:examples_count]
			for i in range(examples_count):
				if train_labels[i] == number1:
					train_labels[i] = 1
				else:
					train_labels[i] = -1
			predictions = svm_binary.trainGaussianBinaryClassifier(train_data,train_labels,test_data,cost,gamma)
			updateCount(predictions,number1,number2)
			print("Done for ",number1,number2)
	confusion_matrix = np.zeros([10,10])
	correct = 0
	for i in range(test_count):
		label = test_labels[i]
		amax = max(count[i])
		predictions = [k for k,j in enumerate(count[i]) if j == amax]
		if len(predictions) == 1:
			prediction = predictions[0]
		else:
			prediction = predictions[0]
			for e in predictions[1:]:
				if score[i][e] > score[i][prediction]:
					prediction = e 
		if prediction == label:
			correct = correct + 1
		confusion_matrix[prediction][int(label)] = confusion_matrix[prediction][int(label)]+1
	print(confusion_matrix)
	print(correct/test_data.shape[0])

elif part == 'b' or part == 'c':
	#LIBSVM part
	examples_count = 1000
	test_count = 2000
	train_labels = list(training_data[784][:])
	del training_data[784]
	del train_labels[784]
	train_data = list(training_data.values)
	del train_data[784]
	for i in range(len(train_data)):
		train_data[i] = list(train_data[i])
	test_labels = list(testing_data[784][:])
	del test_labels[784]
	del testing_data[784]
	test_data = list(testing_data.values)
	del test_data[784]
	for i in range(len(test_data)):
		test_data[i] = list(test_data[i])
	prob = svm_problem(train_labels[:examples_count],train_data[:examples_count])
	gaussian_param = svm_parameter('-t 2 -g 0.05 -c 1 -q')
	model = svm_train(prob,gaussian_param)
	p_labels,p_acc,p_val = svm_predict(y=test_labels[:test_count],x=test_data[:test_count],m=model)
	confusion_matrix = np.zeros([10,10])
	for i in range(test_count):
		print(i)
		confusion_matrix[int(p_labels[i])][int(test_labels[i])] = confusion_matrix[int(p_labels[i])][int(test_labels[i])] + 1
	print(confusion_matrix)

elif part == 'd':
	examples_count = 1000
	test_count = 5000
	costs = [0.00001,0.001,1,5,10]
	gamma = 0.05
	train_labels = list(training_data[784][:])
	del training_data[784]
	del train_labels[784]
	train_data = list(training_data.values)
	del train_data[784]
	for i in range(len(train_data)):
		train_data[i] = list(train_data[i])
	test_labels = list(testing_data[784][:])
	del test_labels[784]
	del testing_data[784]
	test_data = list(testing_data.values)
	del test_data[784]
	for i in range(len(test_data)):
		test_data[i] = list(test_data[i])
	validation_labels = list(validation_data[784][:])
	del validation_labels[784]
	del validation_data[784]
	validation_data = list(validation_data.values)
	del validation_data[784]
	for i in range(len(validation_labels)):
		validation_labels[i] = int(validation_labels[i])
	for i in range(len(validation_labels)):
		validation_data[i] = list(validation_data[i])
	prob = svm_problem(train_labels[:examples_count],train_data[:examples_count])
	validation_accuracy = []
	test_accuracy = []
	for cost in costs:
		parameter = svm_parameter('-t 0 -g '+str(gamma)+' -c '+ str(cost)+' -q')
		model = svm_train(prob,parameter)
		print("Current c ",cost)
		print("Validation Accuracy")
		p_labels,p_acc,p_val = svm_predict(y=validation_labels,x=validation_data,m=model)
		validation_accuracy.append(p_acc[0])
		print("Test Accuracy")
		p_labels,p_acc,p_val = svm_predict(y=test_labels[:test_count],x=test_data[:test_count],m=model)
		test_accuracy.append(p_acc[0])
	print("Best c is ",costs[validation_accuracy.index(max(validation_accuracy))])
