import numpy as np

def MeanSquaredError(Y,Yprime):
	"""
	This is the mean squared error class.
	It computes the mean squared error and gives the error value and the error vector back
	"""
	errorVector = np.square(Y-Yprime)

	return np.sum(errorVector),errorVector

def CrossEntropy(Y,Yprime):
	"""
	This is the cross entropy class.
	It computes the cross entropy and gives the error value and the error vector back
	
	"""
	errorVector = []
	for i in range(len(Y)):
		errorVector.append(Y[i]*np.log(Yprime[i])+(1-Y[i])*np.log(1-Yprime[i]))
	errorVector = np.array(errorVector)
	return np.sum(errorVector)/len(Y)