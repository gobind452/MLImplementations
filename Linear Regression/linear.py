import numpy as np
import pandas as pd 
from matplotlib import pyplot as plt 
from matplotlib import animation as animation
from mpl_toolkits.mplot3d import Axes3D
import sys 

training_data = [] 
labels = []

mean = 0 # Mean and standard deviation of the data
stddev = 0

def readData(filenameX,filenameY): # Reads data from the given filenames
	global training_data
	global labels,mean,stddev
	training_data = pd.read_csv(filenameX,header=None) 
	training_data[1] = training_data[0]
	training_data[0] = 1
	labels = pd.read_csv(filenameY,header=None)
	mean = training_data[1].mean()
	stddev = training_data[1].std()
	training_data[1] = (training_data[1]-mean)/float(stddev)

readData(sys.argv[1],sys.argv[2])

examples_count = training_data.shape[0] # Number of examples
dimensions = training_data.shape[1] # Dimensions of input space
training_data = np.array(training_data.values,dtype='f') 
labels = np.array(labels.values,dtype='f')
theta = np.zeros([dimensions,1])

def error(theta): # Computes error in the linear fit parametrized by theta
	diff = np.subtract(labels,np.matmul(training_data,theta))
	x = np.matmul(np.transpose(diff),diff)/(2*examples_count)
	return(x)

def gradient(theta): # Computes the gradient of the error function at the given theta
	diff = np.subtract(labels,np.matmul(training_data,theta))
	return(np.matmul(np.transpose(training_data),diff)/(examples_count))

def distance(x1,x2): # Metric distance between two points in parameter space
	return np.linalg.norm(np.subtract(x1,x2))

def checkConvergence(theta1,theta2): # Convergence condition for gradient descent
	error1 = error(theta1)
	error2 = error(theta2)
	if abs(error1-error2)/abs(error2) < 0.001:
		return 1
	return 0

visited_theta0 = [] # For storing the theta visited by the algorithm
visited_theta1 = []
visited_errors = []

def linearRegression(learning_rate): # Performs gradient descent
	temp_theta = np.zeros([dimensions,1])
	previous_theta = np.zeros([dimensions,1])
	iteration = 0
	while 1:
		temp_theta = np.copy(np.add(temp_theta,learning_rate*gradient(temp_theta)))
		iteration = iteration+1
		to = np.ndarray.flatten(np.copy(temp_theta))
		visited_theta0.append(list(to)[0])
		visited_theta1.append(list(to)[1])
		visited_errors.append(error(temp_theta))
		if checkConvergence(temp_theta,previous_theta) == 1:
			break
		previous_theta = np.copy(temp_theta)
	global theta
	theta = np.copy(temp_theta)


##Q1 
#Part A
linearRegression(float(sys.argv[3])) # Perform linear regression
print("Theta obtained : ",theta.flatten())

#Part2
x_values = []
for e in training_data:
	x_values.append(stddev*stddev*e[1]+mean)
plt.scatter(x_values,labels,label='Data')
line_x = np.linspace(min(x_values),max(x_values),100)
predicted = theta[0]+theta[1]*((line_x-mean)/stddev)
plt.plot(line_x,predicted,'r',label='Linear Fit') # Plot the linear fit
plt.legend(loc='upper right')
plt.xlabel('Acidity')
plt.ylabel('Density')
plt.title('Linear Regression')
plt.show()

#Part3
x = np.arange(-3,3,0.1)
y = np.arange(-3,3,0.1)
a,b = np.meshgrid(x,y)
z = [] # For storing error values for different thetas

for i in x:
	z.append([])
	for j in y:
		new = np.transpose(np.asarray([i,j]))
		new = np.reshape(new,(2,1))
		z[-1].append(error(new)) # Add error values
z = np.array(z)
z = np.resize(z,(x.shape[0],y.shape[0]))

fig1 = plt.figure()
ax1 = Axes3D(fig1)
surf = ax1.plot_surface(b, a, z, cmap='coolwarm') # Plot the error function
ax1.set_xlabel('theta0')
ax1.set_ylabel('theta1')
ax1.set_zlabel('Error')
ax1.view_init(60,40)

visited_theta0 = np.array(visited_theta0)
visited_theta1 = np.array(visited_theta1)
visited_errors = np.array(visited_errors)
visited_errors = np.resize(visited_errors,(visited_errors.shape[0],1))

line, = ax1.plot([], [], [], 'r-', label = 'Gradient descent', lw = 1.5) # Animated line object
point, = ax1.plot([], [], [], 'bo') # Animated point object
display_value = ax1.text(3., 3., 27.5, '', transform=ax1.transAxes)

def init(): # Initialise the graph
	line.set_data([], [])
	line.set_3d_properties([])
	point.set_data([], [])
	point.set_3d_properties([])
	return line, point,

def animate(i): # Animate Function
	line.set_data(list(visited_theta0[:i]),list(visited_theta1[:i])) # Add previous theta to show a path
	line.set_3d_properties(list(visited_errors[:i]))
	point.set_data(visited_theta0[i],visited_theta1[i]) # Add the current theta
	point.set_3d_properties(visited_errors[i])
	return line, point, 

ax1.legend(loc = 1)
anim = animation.FuncAnimation(fig1, animate, init_func=init,
                               frames=len(visited_errors), interval=int(1000*float(sys.argv[4])), 
                               repeat_delay=60, blit=True) # Blit the plot and call the animate every timestep
plt.show()


#Part4
fig1 = plt.figure()
ax = Axes3D(fig1)

ax.contour(b,a,z,cmap='winter',levels=np.logspace(-3,3,25)) # Plot the contours for the error function
ax.set_xlabel('theta0')
ax.set_ylabel('theta1')
ax.set_zlabel('error')
ax.view_init(60,40)

line, = ax.plot([], [], [], 'r-', label = 'Gradient descent', lw = 1.5) # Similar to the previous part
point, = ax.plot([], [], [], 'bo')
display_value = ax.text(2., 2., 27.5, '', transform=ax.transAxes)

def init_contour():
	line.set_data([], [])
	line.set_3d_properties([])
	point.set_data([], [])
	point.set_3d_properties([])
	return line, point,

def animate_contour(i):
	line.set_data(list(visited_theta0[:i]),list(visited_theta1[:i])) # Add the visited theta till this point to be plotted
	line.set_3d_properties(list(visited_errors[:i]))
	point.set_data(visited_theta0[i],visited_theta1[i]) # Current theta
	point.set_3d_properties(visited_errors[i])
	return line, point, 

anim = animation.FuncAnimation(fig1, animate_contour, init_func=init_contour,
                               frames=len(visited_errors), interval=int(1000*float(sys.argv[4])), 
                               repeat_delay=60, blit=True)
plt.show()
