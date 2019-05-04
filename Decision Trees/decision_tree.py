import numpy as np 
import pandas as pd 
from sklearn import tree
from sklearn import ensemble
import sys

dimensions = 23

train_path = sys.argv[2]
test_path = sys.argv[3]
validate_path = sys.argv[4]

train_data = pd.read_csv(train_path)
test_data = pd.read_csv(test_path)
validation_data = pd.read_csv(validate_path)

train_examples = train_data.shape[0]
test_examples = test_data.shape[0]
validation_examples = validation_data.shape[0]

train_Y = np.reshape(np.array(train_data['Y']),(-1,1))
test_Y = np.reshape(np.array(test_data['Y']),(-1,1))
validate_Y = np.reshape(np.array(validation_data['Y']),(-1,1))
del train_data['Y']
del test_data['Y']
del validation_data['Y']

train_X = train_data.values
test_X = test_data.values
validate_X = validation_data.values

if int(sys.argv[1]) == 4:
	clf = tree.DecisionTreeClassifier(criterion='entropy',max_depth=20,min_samples_leaf=100,min_samples_split=40)
	clf.fit(train_X,train_Y)
	predicted = np.reshape(np.array(clf.predict(train_X)),(-1,1))
	correct = np.equal(predicted,train_Y)
	accuracy = np.sum(correct,axis=0)/float(train_examples)
	print("Train Accuracy ",accuracy[0])
	n_nodes = clf.tree_.node_count
	print("Number of nodes are %s"%n_nodes)
	predicted = np.reshape(np.array(clf.predict(validate_X)),(-1,1))
	correct = np.equal(predicted,validate_Y)
	accuracy = np.sum(correct,axis=0)/float(validation_examples)
	print("Validation Accuracy ",accuracy[0])
	predicted = np.reshape(np.array(clf.predict(test_X)),(-1,1))
	correct = np.equal(predicted,test_Y)
	accuracy = np.sum(correct,axis=0)/float(test_examples)
	print("Test Accuracy  ",accuracy[0])

elif int(sys.argv[1]) == 5:
	clf = tree.DecisionTreeClassifier(criterion='entropy',min_samples_split=100,max_depth=20,min_samples_leaf=50)
	clf.fit(train_X,train_Y)
	predicted = np.reshape(np.array(clf.predict(train_X)),(-1,1))
	correct = np.equal(predicted,train_Y)
	accuracy = np.sum(correct,axis=0)/float(train_examples)
	print("Train Accuracy ",accuracy[0])
	n_nodes = clf.tree_.node_count
	print("Number of nodes are %s"%n_nodes)
	predicted = np.reshape(np.array(clf.predict(validate_X)),(-1,1))
	correct = np.equal(predicted,validate_Y)
	accuracy = np.sum(correct,axis=0)/float(validation_examples)
	print("Validation Accuracy ",accuracy[0])
	predicted = np.reshape(np.array(clf.predict(test_X)),(-1,1))
	correct = np.equal(predicted,test_Y)
	accuracy = np.sum(correct,axis=0)/float(test_examples)
	print("Test Accuracy  ",accuracy[0])

else:
	clf = ensemble.RandomForestClassifier(criterion='entropy',bootstrap=True,n_estimators=100,max_features=0.4,min_samples_split=100)
	clf.fit(train_X,np.reshape(train_Y,(train_Y.shape[0])))
	predicted = np.reshape(np.array(clf.predict(train_X)),(-1,1))
	correct = np.equal(predicted,train_Y)
	accuracy = np.sum(correct,axis=0)/float(train_examples)
	print("Train Accuracy ",accuracy[0])
	predicted = np.reshape(np.array(clf.predict(validate_X)),(-1,1))
	correct = np.equal(predicted,validate_Y)
	accuracy = np.sum(correct,axis=0)/float(validation_examples)
	print("Validation Accuracy ",accuracy[0])
	predicted = np.reshape(np.array(clf.predict(test_X)),(-1,1))
	correct = np.equal(predicted,test_Y)
	accuracy = np.sum(correct,axis=0)/float(test_examples)
	print("Test Accuracy  ",accuracy[0])