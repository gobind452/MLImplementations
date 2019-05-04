import pandas as pd 
import numpy as np
from cvxopt import solvers,matrix
from svmutil import *
from sklearn.metrics import pairwise
import sys

def createKernelMatrix(train_data,train_labels,flag,gamma=0):
	if flag == 0: #Linear Kernel
		kernel = pairwise.linear_kernel(train_data,train_data)
		kernel = np.multiply(np.multiply(kernel,train_labels.reshape(-1,1)),train_labels.reshape(1,-1))
		return kernel
	elif flag == 1: # Gaussian Kernel
		kernel = pairwise.rbf_kernel(X=train_data,Y=train_data,gamma = gamma)
		kernel = np.multiply(np.multiply(kernel,train_labels.reshape(-1,1)),train_labels.reshape(1,-1))
		return kernel

def trainLinearBinaryClassifier(train_data,train_labels,test_data,cost,test_labels=-1):
	examples_count = train_data.shape[0]
	kernel = createKernelMatrix(train_data,train_labels,0)
	P = matrix(kernel) 
	q = matrix(-1*np.ones([examples_count,1])) 
	G = np.zeros([2*examples_count,examples_count])
	for i in range(examples_count):
		G[i][i] = -1
	for i in range(examples_count,2*examples_count):
		G[i][i-examples_count] = 1
	G = matrix(G)
	h = cost*np.ones([2*examples_count,1]) #Gx<h
	for i in range(examples_count):
		h[i] = 0
	h = matrix(h)
	A = np.ones([1,examples_count])
	for i in range(examples_count):
		A[0][i] = train_labels[i]
	A = matrix(A)
	b = np.zeros([1,1])
	b = matrix(b)
	sol = solvers.qp(P,q,G,h,A,b)
	alpha = np.array(sol['x'])
	for i in range(alpha.shape[0]):
		alpha[i][0] = round(alpha[i][0],5)
		alpha[i] = alpha[i][0]
	w = np.matmul(np.transpose(train_data),np.multiply(alpha,np.reshape(train_labels,[-1,1])))
	intercept = 0
	for i in range(alpha.shape[0]):
		if alpha[i]!=0 and alpha[i] < cost:
			intercept = train_labels[i]-np.matmul(np.transpose(w),train_data[i])
			break
	correct = 0
	intercept_vector = intercept*np.ones([test_data.shape[0],1])
	predictions = np.add(np.matmul(test_data,w),intercept_vector)
	if type(test_labels) == int:
		return predictions
	predictions = np.add(np.multiply(np.greater(predictions,0).astype(int),np.ones([test_data.shape[0],1])),np.multiply(np.less_equal(predictions,0).astype(int),-1*np.ones([test_data.shape[0],1])))
	predictions = np.equal(predictions,np.reshape(test_labels,(test_data.shape[0],1))).astype(int)
	correct = np.sum(predictions,axis=0)[0]
	print(correct/(test_data.shape[0]))

def trainGaussianBinaryClassifier(train_data,train_labels,test_data,cost,gamma,test_labels=-1):
	examples_count = train_data.shape[0]
	gaussian_kernel = createKernelMatrix(train_data,train_labels,1,gamma)
	P = matrix(gaussian_kernel)
	q = matrix(-1*np.ones([examples_count,1])) 
	G = np.zeros([2*examples_count,examples_count])
	for i in range(examples_count):
		G[i][i] = -1
	for i in range(examples_count,2*examples_count):
		G[i][i-examples_count] = 1
	G = matrix(G)
	h = cost*np.ones([2*examples_count,1]) #Gx<h
	for i in range(examples_count):
		h[i] = 0
	h = matrix(h)
	A = np.ones([1,examples_count])
	for i in range(examples_count):
		A[0][i] = train_labels[i]
	A = matrix(A)
	b = np.zeros([1,1])
	b = matrix(b)
	sol = solvers.qp(P,q,G,h,A,b)
	alpha = np.array(sol['x'])
	alpha = np.reshape(alpha,(examples_count,1))
	for i in range(alpha.shape[0]):
		alpha[i][0] = round(alpha[i][0],5)
	intercept = 0
	for i in range(alpha.shape[0]):
		if alpha[i][0] < cost and alpha[i][0]>0:
			prediction = pairwise.rbf_kernel(X=np.reshape(train_data[i],(1,-1)),Y=train_data,gamma = gamma)
			prediction = np.matmul(prediction,np.multiply(alpha,np.reshape(train_labels,(-1,1))))
			intercept = train_labels[i]-prediction[0][0]
			break
	intercept_vector = intercept*np.ones([test_data.shape[0],1])
	correct = 0
	predictions = pairwise.rbf_kernel(X=test_data,Y=train_data[:examples_count],gamma = gamma)
	predictions = np.add(np.matmul(predictions,np.multiply(alpha,np.reshape(train_labels,(-1,1)))),intercept_vector)	
	if type(test_labels) == int:
		return predictions
	predictions = np.add(np.multiply(np.greater(predictions,0).astype(int),np.ones([test_data.shape[0],1])),np.multiply(np.less_equal(predictions,0).astype(int),-1*np.ones([test_data.shape[0],1])))
	predictions = np.equal(predictions,np.reshape(test_labels,(test_data.shape[0],1))).astype(int)
	correct = np.sum(predictions,axis=0)[0]
	print(correct,test_data.shape[0])
	print(correct/(test_data.shape[0]))

if __name__ == "__main__":
	examples_count = 4000
	train_path = sys.argv[1]
	test_path = sys.argv[2]
	part = sys.argv[3]
	number1 = 7
	number2 = 8
	training_data = pd.read_csv(train_path,header=None,index_col=False)
	testing_data = pd.read_csv(test_path,header=None,index_col=False)
	binary_train = pd.DataFrame(training_data.loc[training_data[784].isin([number1,number2])])[:examples_count]
	binary_test = pd.DataFrame(testing_data.loc[testing_data[784].isin([number1,number2])])
	train_labels = np.asarray(binary_train[784])[:examples_count]
	del binary_train[784]
	train_data = np.asarray(binary_train.values)
	for i in range(len(train_labels)):
		if train_labels[i] == number1:
			train_labels[i] = 1
		else:
			train_labels[i] = -1
	test_labels = np.asarray(binary_test[784])
	del binary_test[784]
	test_data = np.asarray(binary_test.values)
	for i in range(test_labels.shape[0]):
		if test_labels[i] == number1:
			test_labels[i] = 1
		else:
			test_labels[i] = -1
	if part == 'a': #Linear
		trainLinearBinaryClassifier(train_data,train_labels,test_data,1,test_labels)
	elif part == 'b': #Gaussian
		trainGaussianBinaryClassifier(train_data,train_labels,test_data,1,0.05,test_labels)
	elif part == 'c': # LibSVM 
		train_labels = list(train_labels)
		test_labels = list(test_labels)
		train_data = list(train_data)
		test_data = list(test_data)
		for i in range(len(test_labels)):
			test_data[i] = list(test_data[i])
		for i in range(len(train_labels)):
			train_data[i] = list(train_data[i])
		prob = svm_problem(train_labels,train_data)
		linear_param = svm_parameter('-t 0 -c 1')
		gaussian_param = svm_parameter('-t 2 -g 0.05 -c 1')
		model = svm_train(prob,linear_param)
		p_labels_linear,p_acc,p_val = svm_predict(x=test_data,y=test_labels,m=model)
		model = svm_train(prob,gaussian_param)
		p_labels_gaussian,p_acc,p_val = svm_predict(x=test_data,y=test_labels,m=model)
