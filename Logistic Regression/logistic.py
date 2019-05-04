import numpy as np
import pandas as pd 
from matplotlib import pyplot as plt 
import sys 

training_data = [] # Training Data Matrix
labels = []

def readData(filenameX,filenameY): # Read data from filenames
	global training_data
	global labels,mean,stddev
	training_data = pd.read_csv(filenameX,header=None)
	training_data[2] = training_data[1]
	training_data[1] = training_data[0]
	training_data[0] = 1
	labels = pd.read_csv(filenameY,header=None)

readData(sys.argv[1],sys.argv[2])

examples_count = training_data.shape[0]
dimensions = training_data.shape[1]
training_data = np.array(training_data.values,dtype='f')
labels = np.array(labels.values,dtype='f')
theta = np.zeros([dimensions,1])

def newtonMethodIteration(theta): # Calculates the change in theta per iteration
	sigmoid_vector = np.zeros([examples_count,1])
	for i in range(examples_count):
		sigmoid_vector[i] = 1/(1+np.exp(-1*np.matmul(np.transpose(theta),training_data[i])))
	gradient = np.matmul(np.transpose(training_data),np.subtract(labels,sigmoid_vector))
	sigmoid_matrix = np.zeros([examples_count,examples_count])
	for i in range(examples_count):
		sigmoid_matrix[i][i] = sigmoid_vector[i]*(1-sigmoid_vector[i])
	hessian = -1*np.matmul(np.transpose(training_data),np.matmul(sigmoid_matrix,training_data))
	return(np.matmul(np.linalg.inv(hessian),gradient))

while 1:
	change = newtonMethodIteration(theta)
	theta = np.subtract(theta,change)
	if np.linalg.norm(change)/np.linalg.norm(theta) < 0.01: # Convergence condition
		break

zeros = {}
ones = {}
zeros["x1"] = []
zeros["x2"] = []
ones["x1"] = []
ones["x2"] = []

for i in range(examples_count):
	if labels[i] == 1:
		ones["x1"].append(training_data[i][1])  # Separate the data into classes
		ones["x2"].append(training_data[i][2])
	elif labels[i] == 0:
		zeros["x1"].append(training_data[i][1])
		zeros["x2"].append(training_data[i][2])


min1 = min([min(zeros["x1"]),min(ones["x1"])])
max1 = max([max(zeros["x1"]),max(ones["x1"])]) 
 
plt.scatter(zeros["x1"],zeros["x2"],c='b',marker='o',label='Date (Label 0)') # Plot the data
plt.scatter(ones["x1"],ones["x2"],c='r',marker='^',label='Date (Label 1)')

x1 = []
x2 = []
for i in range(examples_count):
	x1.append(training_data[i][1])
	x2.append(-1*(theta[0]+theta[1]*x1[-1])/theta[2]) # Get the predicted values

x1 = np.array(x1)
x2 = np.array(x2)


print("Theta obtained :",theta.flatten())

plt.plot(x1,x2,c='k',label='Decision Boundary') # Plot the decision boundary
plt.legend(loc='upper left')
plt.xlabel('X1')
plt.ylabel('X2')
plt.title('Logistic Regression')
plt.show()

