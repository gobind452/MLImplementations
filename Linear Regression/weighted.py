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
	training_data[1] = training_data[0]
	training_data[0] = 1
	labels = pd.read_csv(filenameY,header=None)

readData(sys.argv[1],sys.argv[2])

examples_count = training_data.shape[0] # Number of examples
dimensions = training_data.shape[1] #Dimensions
training_data = np.array(training_data.values,dtype='f')
labels = np.array(labels.values,dtype='f')
theta = np.zeros([dimensions,1]) # Parameters

theta = np.matmul(np.linalg.inv(np.matmul(np.transpose(training_data),training_data)),np.matmul(np.transpose(training_data),labels))
## Using the normal linear regression

realx_values = []
for e in training_data:
	realx_values.append(e[1])

plt.scatter(realx_values,labels,c='b',label='Data') # Plot data
line_x = np.linspace(min(realx_values),max(realx_values),50)
predicted = theta[0]+theta[1]*line_x
plt.plot(line_x,predicted,c='r',label='Linear Fit') # Plot linear fit
plt.xlabel('X')
plt.ylabel('Y')
plt.title('Linear Regression')
plt.legend(loc='upper left')
plt.show()

bandwidth = float(sys.argv[3]) # Read bandwidth 

y_values = []
for e in line_x: # Loop for computing the theta for each x and calculating the values
	weighted_matrix = np.zeros([examples_count,examples_count])
	for i in range(examples_count):
		weighted_matrix[i][i] = np.exp(-1*np.power(e-training_data[i][1],2)/(2*bandwidth*bandwidth)) # Weights
	theta = np.matmul(np.linalg.inv(np.matmul(np.transpose(training_data),np.matmul(weighted_matrix,training_data))),np.matmul(np.transpose(training_data),np.matmul(weighted_matrix,labels)))
	y_values.append(theta[0]+theta[1]*e)

y_values = np.array(y_values)

plt.scatter(realx_values,labels,c='r',label='Data')
line_x = np.linspace(min(realx_values),max(realx_values),50)
plt.plot(line_x,predicted,c='k',label='Linear Fit') # Plot linear fit
plt.plot(line_x,y_values,c='b',label='Weighted Fit') # Plot weighted fit
plt.xlabel('X')
plt.ylabel('Y')
plt.title('Weighted Fit With Bandwidth '+str(bandwidth))
plt.legend(loc='upper left')
plt.show()



