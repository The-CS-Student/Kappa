import numpy as np
def rmse(testY,prediction):
	rmseSum = (np.sum(np.square(testY-prediction))/len(prediction))**0.5
	print("Root Mean Squared Error : {}".format(rmseSum))
def confusionmatrix(testY,prediction):
	TT = 0
	TF = 0
	FT = 0
	FF = 0
	for i in range(len(testY)):
		if(round(testY[i][0])==1 and round(prediction[i][0])==1):
			TT+=1
		elif(round(testY[i][0])==1 and round(prediction[i][0])==0):
			TF+=1
		elif(round(testY[i][0])==0 and round(prediction[i][0])==1):
			FT+=1
		elif(round(testY[i][0])==0 and round(prediction[i][0])==0):
			FF+=1
	print("Accuracy : {}".format((TT+FF)*100/(TT+FT+TF+FF)))