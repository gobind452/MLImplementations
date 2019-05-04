import numpy as np 
import pandas as pd 
import string
import sys

train_data = pd.read_csv(sys.argv[1])
test_data = pd.read_csv(sys.argv[2])

categorical = ["C5","S5","C4","S4","C3","S3","C2","S2","C1","S1"]

for variable in categorical:
	train_data[variable] = pd.Categorical(train_data[variable])
	df = pd.get_dummies(train_data[variable],prefix=variable)
	del train_data[variable]
	train_data = pd.concat([df,train_data],axis=1)

	test_data[variable] = pd.Categorical(test_data[variable])
	df = pd.get_dummies(test_data[variable],prefix=variable)
	del test_data[variable]
	test_data = pd.concat([df,test_data],axis=1)

train_data.to_csv(sys.argv[3],index=None)
test_data.to_csv(sys.argv[4],index=None)

