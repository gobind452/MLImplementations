import numpy as np 
import pandas as pd
import sys
import string

tol = 0.0001

def preprocess(parameters):
	parameters[0] = int(parameters[0]) # Inputs
	parameters[1] = int(parameters[1]) # Outputs
	parameters[2] = int(parameters[2]) # Batch_Size
	parameters[3] = int(parameters[3]) # Number of hidden layers
	if parameters[3]>0:
		temp = parameters[4].split()
		for i in range(len(temp)):
			temp[i] = int(temp[i])
		parameters[4] = temp
	else:
		parameters[4] = []
	if parameters[5][:-1] == 'sigmoid':
		parameters[5] = 0
	else:
		parameters[5] = 1
	if parameters[6][:-1] == 'fixed':
		parameters[6] = 0.1
	else:
		parameters[6] = 0

def activationSigmoid(arg,flag):
	if flag == 0:
		check = np.greater(arg,0).astype(int)
		complement = np.subtract(1,check)
		input1 = np.multiply(check,arg) # Positive args
		input2 = np.multiply(complement,arg) # Negative args
		output = np.add(1/(1+np.exp(-input1)),np.subtract(1,1/(1+np.exp(input2))))
		output = np.subtract(output,0.5*np.ones(arg.shape))
		return output 
	elif flag == 1:
		return np.multiply(arg,np.subtract(1,arg))

def activationRELU(arg,flag): # Flag corresponds to derivative. Subgradient already in
	check = np.greater(arg,0).astype(int)
	if flag == 0:
		return np.multiply(check,arg)
	elif flag == 1:
		return check

def activation(arg,flag,func):
	if func == 0:
		return activationSigmoid(arg,flag)
	else:
		return activationRELU(arg,flag)

class Layer: 
	def __init__(self,input_size,units): # Input_size = Dimension of the inputs, Units = Number of inputs
		self.weight = np.random.normal(loc=0,scale=1/np.sqrt(input_size),size=(units,input_size+1)) # Add intercept term for the matrix
		self.input_size = input_size+1
		self.units = units
		self.outputs = []
		self.activation_func = 0

	def output(self,vector):
		if vector.shape[0] != self.input_size:
			print("Wrong input size")
			print(vector.shape,self.input_size)
			return
		output = activation(np.matmul(self.weight,vector),0,self.activation_func)
		self.outputs.append(output)
		return(np.vstack((np.ones([1,1]),output))) # Add intercept term for the next layer

	def learn(self,gradient,learning_rate):
		self.weight = np.add(self.weight,learning_rate*gradient)

	def cleanup(self): # Clean all previous outputs
		self.outputs[:] = [] 

class NeuralNetwork:
	def __init__(self,input_size,hidden,output_size):
		self.layers = []
		initial = input_size
		for e in hidden:
			self.layers.append(Layer(initial,e))
			initial = e
		self.layers.append(Layer(initial,output_size))
		self.input_size = input_size+1
		self.output_size = output_size

	def __propagate(self,x):
		if x.shape[0]!= self.input_size and x.shape[1]!=self.input_size:
			print("Wrong input size")
			return
		x = np.reshape(x,(-1,1)) # Reshape as a vector
		inter = np.copy(x)
		for layer in self.layers:
			inter = layer.output(inter)
		return(np.reshape(inter[1:],(self.output_size)))

	def cleanup(self):
		for layer in self.layers:
			layer.cleanup()

	def __forwardPropagate(self,X):
		temp = np.vectorize(self.__propagate,signature='(m)->(n)')
		return temp(X)

	def predict(self,X):
		temp = np.vectorize(self.__propagate,signature='(m)->(n)')
		prediction = temp(X)
		self.cleanup()
		return prediction

	def getError(self,X,y): # X = m x (1+n) , Y = m x classes
		predicted = self.predict(X)
		predicted = np.subtract(predicted,y)
		predicted = np.multiply(predicted,predicted)
		error = 0.5*np.sum(predicted)/train_X.shape[0]
		return error

	def getClassificationError(self,X,y):
		confusion_matrix = np.zeros([y.shape[1],y.shape[1]])
		predicted = self.predict(X)
		predicted = np.argmax(predicted,axis=1)
		y = np.argmax(y,axis=1)
		correct = np.equal(predicted,y).astype(int)
		for i in range(y.shape[0]):
			confusion_matrix[predicted[i]][y[i]] = confusion_matrix[predicted[i]][y[i]]+1
		print(confusion_matrix)
		return np.sum(correct)/y.shape[0]

	def checkConvergence(self,error1,error2):
		if error1 <= 0.05 and abs(error1-error2)/error1<=0.0001:
			return 1
		return 0

	def backPropagate(self,train_X,train_Y,learning_rate):
		outputs = self.__forwardPropagate(train_X)
		net = np.multiply(np.subtract(train_Y,outputs),activation(outputs,1,0))
		rev_layers = self.layers[::-1]
		for i in range(len(rev_layers)-1):
			rev_layers[i].cleanup()
			prev_layer_outputs = np.reshape(np.array(rev_layers[i+1].outputs),(-1,rev_layers[i+1].units)) # Outputs from prev layer
			inputs_into_this = np.hstack((np.ones([prev_layer_outputs.shape[0],1]),prev_layer_outputs)) # Inputs into this layer
			gradient = np.matmul(np.transpose(net),inputs_into_this)
			connect = np.transpose(np.transpose(rev_layers[i].weight)[1:]) # Connections of this layer to previous layer outputs
			net = np.matmul(net,connect)
			net = np.multiply(net,activation(prev_layer_outputs,1,rev_layers[i+1].activation_func))
			rev_layers[i].learn(gradient,learning_rate)
		inputs_into_this = train_X # Inputs to the first layer
		gradient = np.matmul(np.transpose(net),inputs_into_this)
		self.layers[0].learn(gradient,learning_rate)
		self.layers[0].cleanup()

	def trainNetwork(self,train_X,train_Y,epochs,batch_size=1,activation_func=0,learning_rate=0):
		adaptive = 0
		if learning_rate == 0:
			learning_rate = 0.1
			change = 0
			adaptive = 1
		if activation_func == 1:
			for layer in self.layers[:-1]:
				layer.activation_func = activation_func
		curr_error = self.getError(train_X,train_Y)
		number_batches = int(train_X.shape[0]/batch_size)+int(train_X.shape[0]%batch_size>0)
		if train_X.shape[0]%batch_size>0:
			for epoch in range(1,epochs+1):
				print("At epoch %s"%epoch,curr_error)
				for batch in range(number_batches-1):
					self.backPropagate(train_X[batch*batch_size:(batch+1)*batch_size],train_Y[batch*batch_size:(batch+1)*batch_size],learning_rate)
				self.backPropagate(train_X[(number_batches-1)*batch_size:],train_Y[(number_batches-1)*batch_size:],learning_rate)
				new_error = self.getError(train_X,train_Y)
				if adaptive == 1:
					if new_error - curr_error >= tol:
						change = change + 1
					else:
						change = 0
					if change == 2:
						change = 0
						learning_rate = learning_rate/5
						print("Adapting")
				if self.checkConvergence(new_error,curr_error) == 1:
					print("Converged at %s"%epoch)
					print(new_error)
					return
				curr_error = new_error
		else:
			for epoch in range(1,epochs+1):
				print("At epoch %s"%epoch,curr_error)
				for batch in range(number_batches):
					self.backPropagate(train_X[batch*batch_size:(batch+1)*batch_size],train_Y[batch*batch_size:(batch+1)*batch_size],learning_rate)
				new_error = self.getError(train_X,train_Y)
				if adaptive == 1:
					if new_error - curr_error >= tol:
						change = change + 1
					else:
						change = 0
					if change == 2:
						change = 0
						learning_rate = learning_rate/5
						print("Adapting")
				if self.checkConvergence(new_error,curr_error) == 1:
					print("Converged at %s"%epoch)
					print(new_error)
					return
				curr_error = new_error
		return

config = sys.argv[1]

parameters = []
with open(config,'r') as file:
	for line in file:
		parameters.append(line)

preprocess(parameters)

training_data = pd.read_csv(sys.argv[2])

examples_count = training_data.shape[0]

train_Y = pd.Categorical(training_data['CLASS'])
train_Y = pd.get_dummies(train_Y,prefix='Y')
del training_data['CLASS']

df = pd.DataFrame([])
df['C'] = 1
training_data = pd.concat([df,training_data],axis=1)
training_data['C'] = 1
train_X = np.reshape(np.array(training_data.values),(examples_count,-1))
train_Y = np.reshape(np.array(train_Y.values),(examples_count,-1))

network = NeuralNetwork(parameters[0],parameters[4],train_Y.shape[1])
network.trainNetwork(train_X,train_Y,500,parameters[2],parameters[5],parameters[6])
print(network.getClassificationError(train_X,train_Y))

testing_data = pd.read_csv(sys.argv[3])
test_examples_count = testing_data.shape[0]

test_Y = pd.Categorical(testing_data['CLASS'])
test_Y = pd.get_dummies(test_Y,prefix='Y')
del testing_data['CLASS']

df = pd.DataFrame([])
df['C'] = 1
testing_data = pd.concat([df,testing_data],axis=1)
testing_data['C'] = 1
test_X = np.reshape(np.array(testing_data.values),(test_examples_count,-1))
test_Y = np.reshape(np.array(test_Y.values),(test_examples_count,-1))
print(network.getClassificationError(test_X,test_Y))
