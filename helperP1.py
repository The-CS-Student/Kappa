import numpy
import random


def KRandom(array,k):
	points = []
	for i in range(k):
		index = random.randint(0,len(array)-1)
		points.append(array[index])
		array.pop(index)
	return points
def EuclideanDistance(arr1,arr2):
	distance = 0
	for i in range(len(arr1)):
		distance+=(arr1[i]-arr2[i])**2
	distance = distance**0.5
	return distance
def hyperMean(array):
	meanArr = []

	for i in range(len(array[0])):
		mean = 0
		for j in range(len(array)):
			mean+=array[j][i]
		mean = mean/len(array)
		meanArr.append(mean)
	return meanArr


def partition(arr,low,high): 
	i = ( low-1 )
	pivot = arr[high][0]

	for j in range(low , high): 

		if arr[j][0] <= pivot: 
			i = i+1
			arr[i],arr[j] = arr[j],arr[i] 

	arr[i+1],arr[high] = arr[high],arr[i+1] 
	return ( i+1 ) 

def quickSort(arr,low,high): 
	if low < high: 

		pi = partition(arr,low,high) 

		
		quickSort(arr, low, pi-1) 
		quickSort(arr, pi+1, high) 
	return arr



def notIn(element,array,index):
	for i in range(len(array)):
		if(array[i][index]==element):
			return True
	return False


def NameChange(dictionary,name):
	x = list(dictionary)
	newdict = {}
	for i in range(len(x)):
		newdict[str(x[i])+str(name)] = dictionary[x[i]]
	return newdict

