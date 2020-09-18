import numpy as np
def split(X,Y,splitratio):
	"""
	Here X and Y are two dimensional arrays which are not ndarrays.
	"""
	index = (splitratio*len(X))
	trainX = []
	trainY = []
	testX = []
	testY = []
	for i in range(len(X)):
		if(i+1>index):
			trainX.append(X[i])
			trainY.append(Y[i])
		else:
			testX.append(X[i])
			testY.append(Y[i])
	return np.array(trainX),np.array(trainY),np.array(testX),np.array(testY)
