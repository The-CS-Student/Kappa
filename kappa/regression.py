import numpy as np
from . import error
class LinearRegression:
	def __init__(self,X,Y):
		"""
		This is the base Linear Regression Class.
		Here X is the training input. X's dimensions are [num_of_sample,num__features]
		Here Y is the training output. Y's dimensions are [num_of_sample,1]

		"""
		self.X = np.array(X)
		self.Y = np.array(Y)
		"""
		Here are the Weights and Biases. 
		Weights have the dimension [num_of_features,1]
		Bias is just a random number
		"""
	
	def predict(self,x):
		return np.matmul(x,self.W.T)
	def returnWeights(self):
		return self.W
	

	def fit(self):
		"""
		Here each iteration goes through a predict -> error -> backpropagate loop
		Error and Average Error is printed out 
		"""
		self.W = np.matmul(self.Y.T,np.matmul(self.X,np.linalg.inv(np.matmul(self.X.T,self.X))))
	
class LogisticRegression:
	def __init__(self,X,Y):
		"""
		This is the base Linear Regression Class.
		Here X is the training input. X's dimensions are [num_of_sample,num__features]
		Here Y is the training output. Y's dimensions are [num_of_sample,1]

		"""
		self.X = np.array(X)
		self.Y = np.array(Y)
		"""
		Here are the Weights. 
		Weights have the dimension [num_of_features,1]
		"""
		self.W = np.zeros((self.X.shape[1],1))
	def sigmoid(self,prediction):
		"""
		The sigmoid function
		"""
		return 1/(1+np.exp(-prediction))
	
	
	def predict(self,x):
		return self.sigmoid(np.matmul(x,self.W))
	def returnWeights(self):
		return self.W
	def fit(self,iterations):
		"""
		Here each iteration goes through a predict -> error -> backpropagate loop
		Error and Average Error is printed out 
		"""
		avgerror = 0
		for i in range(iterations):
			predicted = self.predict(self.X)
			errorVal = error.CrossEntropy(self.Y,predicted)
			avgerror+=errorVal
			print('Iteration: {} Error: {}'.format(i+1, errorVal))
			W = np.tile(np.diagonal(predicted * (1 - predicted)),[self.X.shape[1], 1])

			hessian = (self.X.T*W)@self.X
			self.W+=np.matmul(np.linalg.inv(hessian),np.matmul(self.X.T,self.Y-predicted))
		print("Average Error: {}".format(avgerror/iterations))