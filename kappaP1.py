import numpy
import random
import pickle
import helperP
import matplotlib.pyplot as plt
import pydotplus
from graphviz import Digraph
class LinearRegression:
	def __init__(self,X,Y,**kwargs):
		self.X = X
		self.Y = Y
		self.optimizer = "SGD"
		self.iterations = 1000
		self.learningRate = 0.01
		self.epochs = len(X[0])
		self.error = "MSE"
		for key,value in kwargs.items():
			if(key=="learningRate"):
				self.learningRate = value
			elif(key=="optimizer"):
				self.optimizer = value
			elif(key=="epochs"):
				self.epochs = value
			elif(key=="error"):
				self.error = value
			elif(key=="iterations"):
				self.iterations = value
		self.weights = []
		self.bias = random.randint(0,10)
		for i in range(len(X[0])):
			self.weights.append(random.random())
	def fit(self):
		averageError = 0
		for i in range(self.iterations):
			
			print("Iteration : ",i+1," Average Error : ",averageError)
			averageError = 0
			for j in range(len(self.X)):

				result = 0

				for k in range(len(self.weights)):

					result+=self.X[j][k]*self.weights[k]

				result+=self.bias
				if(self.error=="MSE"):
					error = (self.Y[j]-result)**1
				averageError+=error
				for k in range(len(self.weights)):
					self.weights[k]+=2*(error)*self.X[j][k]*self.learningRate
				self.bias+=2*(error)*self.learningRate
				if(j%(len(self.X)/self.epochs)==0):

					print("Epochs : ",int(j/(len(self.X)/self.epochs))," Error : ",error)
			averageError = averageError/len(self.X[0])
	def save(self,filename):
		dictW = {"modelType":"LinearRegression","weights":self.weights,"bias":self.bias}
		pickle_out = open(filename,"wb")
		pickle.dump(dictW,pickle_out)
		pickle_out.close()
	

	def returnWeights(self):
		return self.weights

	def returnBias(self):
		return self.bias

	def returnError(self):
		return self.error

class LogisticRegression:
	def __init__(self,X,Y,**kwargs):
		self.X = X
		self.Y = Y
		self.optimizer = "SGD"
		self.iterations = 1000
		self.learningRate = 0.01
		self.epochs = len(X[0])
		self.error = "CrossEntropy"
		for key,value in kwargs.items():
			if(key=="learningRate"):
				self.learningRate = value
			elif(key=="optimizer"):
				self.optimizer = value
			elif(key=="epochs"):
				self.epochs = evalue
			elif(key=="error"):
				self.error = value
			elif(key=="iterations"):
				self.iterations = value
		self.weights = []
		self.bias = random.randint(0,10)
		for i in range(len(X[0])):
			self.weights.append(random.random())
	def fit(self):
		averageError = 0
		for i in range(self.iterations):
			
			print("Iteration : ",i+1," Average Error : ",averageError)
			averageError = 0
			for j in range(len(self.X)):

				result = 0
				for k in range(len(self.weights)):

					result+=self.X[j][k]*self.weights[k]

				result+=self.bias
				result = 1/(1+numpy.exp(-result))
				
				if(self.error=="CrossEntropy"):
					error = -(self.Y[j]*numpy.log(result+1e-6)+(1-self.Y[j])*numpy.log((1-result)+1e-6))
				averageError+=error
				for k in range(len(self.weights)):
					self.weights[k]-=self.X[j][k]*(result-self.Y[j])*self.learningRate
				self.bias-=(result-self.Y[j])*self.learningRate
				if(j%(len(self.X)/self.epochs)==0):

					print("Epochs : ",int(j/(len(self.X)/self.epochs))," Error : ",error)

			averageError = averageError/len(self.X[0])
	
	def save(self,filename):
		dictW = {"modelType":"LogisticRegression","weights":self.weights,"bias":self.bias}
		pickle_out = open(filename,"wb")
		pickle.dump(dictW,pickle_out)
		pickle_out.close()
	def returnWeights(self):
		return self.weights

	def returnBias(self):
		return self.bias

	def returnError(self):
		return self.error
class NaiveBayes:
	def __init__(self,X,Y):
		self.X = X
		self.Y = Y
		self.xDict = []
		self.yDict = {}
		self.yDictCount = {}
		for i in range(len(self.Y)):
			if Y[i] not in self.yDict:
				self.yDict[Y[i]] = 0
				self.yDictCount[Y[i]]=1
			else:
				self.yDictCount[Y[i]]+=1

		for i in range(len(self.X)):
			arr = []
			arr.append(X[i][0])
			arr.append({})
			for j in range(len(self.X[0])):
				if(j!=0):
					if X[i][j] not in arr[1]:
						arr[1][X[i][j]] = helperP.NameChange(self.yDict,X[i][j])
			self.xDict.append(arr)


	def fit(self):
		for i in range(len(self.X)):
			for j in range(len(self.X[0])):
				if(j!=0):
					self.xDict[i][1][self.X[i][j]][str(self.Y[j-1])+str(self.X[i][j])]+=1/self.yDictCount[self.Y[j-1]]

	def predict(self,array):
		
		prediction = self.yDict

		predictionC = list(prediction)
		for i in range(len(predictionC)):
			prediction[predictionC[i]]=1
		
		for i in range(len(array)):

			for j in range(len(list(self.xDict[i][1]))):
				
				if(array[i]==list(self.xDict[i][1])[j]):
					for k in range(len(predictionC)):
						prediction[predictionC[k]]*=self.xDict[i][1][array[i]][str(predictionC[k])+str(array[i])]
		for i in range(len(list(self.yDictCount))):
			
			prediction[list(self.yDictCount)[i]]*=self.yDictCount[list(self.yDictCount)[i]]/len(self.Y)
		sumP = 0
		
		for i in range(len(list(prediction))):
			sumP+=prediction[list(prediction)[i]]
		for i in range(len(list(prediction))):
			prediction[list(prediction)[i]] = prediction[list(prediction)[i]]/sumP
		return prediction
	def save(self,filename):
		dictW = {"xDict":self.xDict,"yDict":self.yDict,"yDictCount":self.yDictCount}
		pickle_out = open(filename,"wb")
		pickle.dump(dictW,pickle_out)
		pickle_out.close()




class KMeans:
	def __init__(self,X,k,**kwargs):

		self.X = X
		self.k = k
		self.centroids = []
		self.iterations = 20
		self.distance = "Euclid"
		for key,value in kwargs.items():
			if(key=="iterations"):
				self.iterations = value
		

		
	def fit(self,**kwargs):
		for key,value in kwargs.items():
			if(key=="Manhattan"):
				self.distance = value
		centroid = []
		variationArr = []
		for i in range(self.k-1):
			kLength = i+2
			randomPoints = helperP.KRandom(self.X,kLength)
			
			for j in range(self.iterations):
				centroidArr = []
				for k in range(kLength):
					centroidArr.append([randomPoints[k]])
				for k in range(len(self.X)):
					euclidD = []
					for m in range(len(randomPoints)):
						
						euclidD.append(helperP.EuclideanDistance(randomPoints[m],self.X[k]))
					
					maxV = max(euclidD)
					index = None
					for m in range(len(euclidD)):
						if(euclidD[m]==maxV):
							index = m

					centroidArr[index].append(self.X[k])

				for k in range(len(centroidArr)):
					randomPoints[k] = helperP.hyperMean(centroidArr[k])
			totalCount = len(self.X)/k
			variation = 0
			for j in range(len(centroidArr)):
				variation+=(len(centroidArr[j])-totalCount)**2
			variationArr.append(variation)
			centroid.append(randomPoints)
		minVar = min(variationArr)
		for i in range(len(variationArr)):
			if(variationArr[i]==minVar):
				self.centroids=centroid[i]
				break

	def returnCentroids(self):
		return self.centroids
	def save(self,filename):
		dictW = {"modelType":"KMeans","centroids":self.centroids}
		pickle_out = open(filename,"wb")
		pickle.dump(dictW,pickle_out)
		pickle_out.close()
	
	




			
class KNN:
	def __init__(self,X,Y,k):
		self.X = X
		self.Y = Y
		self.yDict = {}
		for i in range(len(self.Y)):
			if self.Y[i] not in self.yDict:
				self.yDict[Y[i]] = 0
		self.k = k
		self.correct = 0
		self.distance = "Euclidean"
	def fit(self,**kwargs):
		for key,value in kwargs.items():
			if(key=="Manhattan"):
				self.distance = value

		for i in range(len(self.X)):
			distanceArr = []
			for j in range(len(self.X)):
				distanceArr.append([helperP.EuclideanDistance(self.X[i],self.X[j]),j])
			
			distanceArr = helperP.quickSort(distanceArr,0,len(distanceArr)-1)
			
			karray = []
			for j in range(self.k+1):
				if(distanceArr[j][1]!=i):
					karray.append(distanceArr[j])

			d2 = {}
			for key in self.yDict:
				d2[key] = self.yDict[key]

			for j in range(len(karray)):
				d2[self.Y[karray[j][1]]]+=1

			maximumVal = None
			index = None
			for j in range(len(d2)):
				
				if(j==0):
					maximumVal = d2[j]
					index = 0
				else:
					if(d2[j]>maximumVal):
						maximumVal = d2[j]
						index = j
			listD2 = list(d2)
			if(self.Y[i]==listD2[index]):
				self.correct+=1

		print("Accuracy : ",(self.correct/len(self.X))*100)
	def save(self,filename):
		dictW = {"modelType":"KNN","k":self.k,"distance":self.distance,"yDict":self.yDict}
		pickle_out = open(filename,"wb")
		pickle.dump(dictW,pickle_out)
		pickle_out.close()

	def showTree(self,filename):
		
		g = Digraph('G', filename=filename)
		g.edge(self.HTree.name,self.HTree.node1.name)
		g.edge(self.HTree.name,self.HTree.node2.name)
		helperP.showHTree(g,self.HTree.node1,self.HTree.node1.name)
		helperP.showHTree(g,self.HTree.node2,self.HTree.node2.name)
		g.view()


class LoadModel:
	def __init__(self,filename,**kwargs):
		self.filename = filename
		self.modelType = None
		for key,value in kwargs.items():
			if(key=="X"):
				self.X = value
			elif(key=="Y"):
				self.Y = value
	def load(self):
		pickle_in = open(self.filename,"rb")
		saved = pickle.load(pickle_in)
		if(saved["modelType"]=="LinearRegression"):
			self.modelType = "LinearRegression"
			print("Loaded LinearRegression Model")
			self.weights = saved['weights']
			self.bias = saved['bias']
		elif(saved["modelType"]=="LogisticRegression"):
			self.modelType = "LogisticRegression"
			print("Loaded LogisticRegression Model")
			self.weights = saved['weights']
			self.bias = saved['bias']
		elif(saved["modelType"]=="KMeans"):

			self.modelType = "KMeans"
			print("Loaded KMeans Model")

			self.weights = saved['centroids']
		elif(saved["modelType"]=="KNN"):

			self.modelType = "KNN"
			print("Loaded KNN Model")

			self.k = saved['k']

			self.distance = saved['distance']
			self.yDict = saved['yDict']
	def predict(self,array):
		if(self.modelType=="LinearRegression"):
			solution = []
			for i in range(len(array)):
				prediction = 0
				for j in range(len(self.weights)):
					prediction+=self.weights[j]*array[i][j]
				prediction+=self.bias
				solution.append(prediction)
			return solution
		elif(self.modelType=="LogisticRegression"):
			solution = []
			for i in range(len(array)):
				prediction = 0

				for k in range(len(self.weights)):
					prediction+=self.weights[k]*array[i][k]
					prediction+=self.bias
				prediction = 1/(1+numpy.exp(-prediction))
				solution.append(prediction)
			return solution
		elif(self.modelType=="KMeans"):
			solution = []
			for i in range(len(array)):
				distanceC = []
				for j in range(len(self.weights)):
					distanceC.append(helperP.EuclideanDistance(self.weights[j],array[i]))
				minDist = None
				Index = None
				for j in range(len(distanceC)):
					if(j==0):
						minDist = distanceC[j]
						Index = j
					else:
						if(distanceC[j]<minDist):
							minDist = distanceC[j]
							Index = j
				solution.append("Cluster "+str(Index))
			return solution
		elif(self.modelType=="KNN"):
			solution = []
			for i in range(len(array)):
				distanceArr = []
				for j in range(len(self.X)):
					distanceArr.append([helperP.EuclideanDistance(array[i],self.X[j]),j])
				
				distanceArr = helperP.quickSort(distanceArr,0,len(distanceArr)-1)
				
				karray = []
				for j in range(self.k):
					
					karray.append(distanceArr[j])
				# print(self.yDict)
				# break
				d2 = {}
				for key in self.yDict:
					d2[key] = self.yDict[key]

				for j in range(len(karray)):
					
					d2[self.Y[karray[j][1]]]+=1

				maximumVal = None
				index = None
				for j in range(len(list(d2))):
					print(d2[j],maximumVal)
					if(j==0):
						maximumVal = d2[j]
						index = 0
					else:
						if(d2[j]>maximumVal):
							maximumVal = d2[j]
							index = j
							print(index)

				
				solution.append(list(d2)[index])
			return solution