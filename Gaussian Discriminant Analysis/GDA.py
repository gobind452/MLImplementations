import numpy as np
import pandas as pd 
from matplotlib import pyplot as plt 
import sys 

training_data = []
classes = []

def readData(filenameX,filenameY): # Read data from filenames
	global training_data
	global classes
	training_data = pd.read_fwf(filenameX,header=None,usecols=[0,1])
	classes = pd.read_csv(filenameY,header=None,usecols=[0])

readData(sys.argv[1],sys.argv[2])

examples_count = training_data.shape[0]
dimensions = training_data.shape[1]
training_data = np.array(training_data.values,dtype='f')
training_data = np.reshape(training_data,[examples_count,dimensions])

labels = []

for i in range(examples_count): # Create a label vector
	if classes.values[i] == 'Alaska':
		labels.append(0)
	elif classes.values[i] == 'Canada':
		labels.append(1)

labels = np.reshape(np.array(labels,dtype='f'),[examples_count,1])
labels_comp = (labels+1)%2

covariance = np.zeros([dimensions,dimensions])
phi = sum(labels)/float(examples_count) #Bernoulli variable for y

mean_0 = np.matmul(np.transpose(training_data),labels_comp)/sum(labels_comp) #Mean for class 0
mean_1 = np.matmul(np.transpose(training_data),labels)/sum(labels) # Mean for class 1

for i in range(examples_count):
	add = np.matmul(np.subtract(np.reshape(training_data[i],[2,1]),mean_0*labels_comp[i]+mean_1*labels[i]),np.transpose(np.subtract(np.reshape(training_data[i],[2,1]),mean_0*labels_comp[i]+mean_1*labels[i])))
	covariance = np.add(covariance,add)
covariance = covariance/examples_count # Covariance matrix

intercept = np.matmul(np.transpose(mean_1),np.matmul(np.linalg.inv(covariance),mean_1))
intercept = intercept - np.matmul(np.transpose(mean_0),np.matmul(np.linalg.inv(covariance),mean_0))
intercept = (intercept - 2*np.log(phi/(1-phi))).flatten() # Compute the intercept term in linear equation

linear = (2*np.matmul(np.linalg.inv(covariance),np.subtract(mean_0,mean_1))).flatten()

zeros = {}
ones = {}
zeros["x1"] = []
zeros["x2"] = []
ones["x1"] = []
ones["x2"] = []

for i in range(examples_count): # Separate data into classes
	if labels[i] == 1:
		ones["x1"].append(training_data[i][0])
		ones["x2"].append(training_data[i][1])
	elif labels[i] == 0:
		zeros["x1"].append(training_data[i][0])
		zeros["x2"].append(training_data[i][1])

min1 = min([min(zeros["x1"]),min(ones["x1"])])
max1 = max([max(zeros["x1"]),max(ones["x1"])]) 

x1 = np.linspace(min1,max1,100)
x2 = -1*(intercept+linear[0]*x1)/linear[1]

plt.scatter(zeros["x1"],zeros["x2"],c='r',marker='o',label='Alaska') # Plot data
plt.scatter(ones["x1"],ones["x2"],c='b',marker='^',label='Canada')
plt.xlabel('Fresh Water Diameter')
plt.ylabel('Marine Water Diameter')
plt.legend(loc='upper right')
plt.title('Data')
plt.show()

if int(sys.argv[3]) == 0:
	print("Mean_0",mean_0)
	print("Mean_1",mean_1)
	print("Covariance",covariance)
	plt.scatter(zeros["x1"],zeros["x2"],c='r',marker='o',label='Alaska')
	plt.scatter(ones["x1"],ones["x2"],c='b',marker='^',label='Canada')
	plt.plot(x1,x2,c='g',label='Linear Boundary') # Plot linear boundary
	plt.xlabel('Fresh Water Diameter')
	plt.ylabel('Marine Water Diameter')
	plt.legend(loc='upper right')
	plt.title('Linear Seperator')
	plt.show()

covariance_0 = np.zeros([dimensions,dimensions]) # Different covariance matrix
covariance_1 = np.zeros([dimensions,dimensions])

for i in range(examples_count):
	add = np.matmul(np.subtract(np.reshape(training_data[i],[2,1]),mean_0*labels_comp[i]+mean_1*labels[i]),np.transpose(np.subtract(np.reshape(training_data[i],[2,1]),mean_0*labels_comp[i]+mean_1*labels[i])))
	covariance_0 = np.add(covariance_0,add*labels_comp[i])
	covariance_1 = np.add(covariance_1,add*labels[i])

covariance_0 = covariance_0/sum(labels_comp)
covariance_1 = covariance_1/sum(labels)

intercept = np.matmul(np.transpose(mean_1),np.matmul(np.linalg.inv(covariance_1),mean_1))
intercept = intercept - np.matmul(np.transpose(mean_0),np.matmul(np.linalg.inv(covariance_0),mean_0))
intercept = (intercept - 2*np.log(phi/(1-phi))).flatten() # Constant term in quadratic equation

linear = np.matmul(np.linalg.inv(covariance_0),mean_0)
linear = (2*np.subtract(linear,np.matmul(np.linalg.inv(covariance_1),mean_1))).flatten() # Linear terms

quadratic = np.subtract(np.linalg.inv(covariance_1),np.linalg.inv(covariance_0)) # Quadratic terms

z1 = np.copy(x1)

def solveforZ2(z1): # Solve for z2 given z1 using the quadratic equation
	c = intercept+quadratic[0][0]*z1*z1+linear[0]*z1
	b = linear[1]+2*quadratic[0][1]*z1
	a = quadratic[1][1]
	return((-b+np.sqrt(b*b-4*a*c))/(2*a))

z2 = solveforZ2(z1)

if int(sys.argv[3]) == 1:
	print("Mean_0",mean_0)
	print("Mean_1",mean_1)
	print("Covariance_0",covariance_0)
	print("Covariance_1",covariance_1)
	plt.scatter(zeros["x1"],zeros["x2"],c='r',marker='o',label='Alaska')
	plt.scatter(ones["x1"],ones["x2"],c='b',marker='^',label='Canada')
	plt.plot(x1,x2,c='g',label='Linear Boundary')
	plt.plot(z1,z2,c='y',label='Quadratic Boundary') # Plot the quadratic boundary
	plt.xlabel('Fresh Water Diameter')
	plt.ylabel('Marine Water Diameter')
	plt.legend(loc='upper right')
	plt.title('Quadratic Seperator')
	plt.show()