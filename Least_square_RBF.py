import math
import numpy as np
from scipy import signal
from matplotlib import pyplot as plt


# 3.1 Batch mode training using least squares
# - Supervised learning of network weights

def batchMode(rbfNodes):
	#Create a column vector containing the data points (patterns)
	stepSize = 0.1
	interval = [0, 2*math.pi]
	trainStart = 0
	testStart = 0.05
	#The points at which the functions should be evaluated
	trainX = np.arange(interval[0]+trainStart, interval[1], stepSize)
	testX = np.arange(interval[0]+testStart, interval[1], stepSize)

	# noise_train = np.random.normal(0,0.1,len(trainX))
	# noise_test = np.random.normal(0,0.1,len(testX))

	noise_train = 0
	noise_test = 0

	#sin(2x)
	trainSinY = np.sin(2*trainX) + noise_train
	testSinY = np.sin(2*testX) + noise_test
	# print(testSinY)

	#square(2x)
	trainSqY = np.ones(len(trainSinY))
	testSqY = np.ones(len(testSinY))
	trainSqY[np.where(trainSinY>=0)] = 1
	trainSqY[np.where(trainSinY < 0)]= -1
	testSqY[np.where(testSinY>=0)] = 1
	testSqY[np.where(testSinY < 0)]= -1

	# plt.plot(trainX,testSqY)
	# plt.show()

	#RBF nodes
	rbfSigma = 1
	intervalOffset = 0 #Just having fun with math. Don't know of this offset is necessary whatsoever.
	rbfMu = np.linspace(interval[0]+intervalOffset, interval[1]-intervalOffset, rbfNodes)

	# Create Phi
	Phi = np.zeros((trainX.shape[0], rbfMu.shape[0]))
	for i in range(trainX.shape[0]):
		for j in range(rbfMu.shape[0]):
			Phi[i][j] = gaussianTransfer(trainX[i], rbfMu[j], rbfSigma)
			# Phi[i][j] = 1/(1+ np.exp(Phi[i][j]))
		# print(Phi[i][j])

	# #Initiate weights (Maybe not what we should do)
	# w = np.ones((rbfMu.shape[0])) / rbfMu.shape[0]

	#Obtain the w by solving the system [Phi.T*Phi*w = Phi.T*f]
	PhiTPhi = np.dot(Phi.T, Phi)
	PhiTf = np.dot(Phi.T, trainSqY)
	# PhiTf = np.dot(Phi.T, trainSinY)
	w = np.dot(np.linalg.inv(PhiTPhi), PhiTf)
	# print(w)
	# totErr = np.power(np.linalg.norm(np.dot(Phi, w)-trainSinY),2)
	# print(totErr)

	Phi = np.zeros((testX.shape[0], rbfMu.shape[0]))
	for i in range(testX.shape[0]):
		for j in range(rbfMu.shape[0]):
			Phi[i][j] = gaussianTransfer(testX[i], rbfMu[j], rbfSigma)
			# For square only
			# Phi[i][j] = 1 / (1 + np.exp(Phi[i][j]))
		# print(Phi[i][j])
	# predicted = np.ones(len(testX))
	# predicted[np.where(np.dot(Phi, w) >=0)] = 1
	# predicted[np.where(np.dot(Phi, w) <0)] = -1

	# For sin only
	predicted = np.dot(Phi, w)

	plt.plot(testX, testSqY, 'k--', label='actual test output')
	plt.plot(testX, predicted, 'r.', label='predicted test output')
	plt.legend()
	title = 'Function approximation for', rbfNodes, 'nodes'
	plt.title(title)
	plt.show()
	return ((np.mean(np.abs(testSqY - predicted))))
	# return ((np.mean(np.abs(testSinY - predicted))))
	# print(np.dot(Phi, w)-trainSinY)



def gaussianTransfer(x, mu, sigma):
	phi_i = np.exp(np.divide(-np.power((x-mu), 2), 2*np.power(sigma, 2)))
	return phi_i

for i in (0,63):
	res_error = batchMode(i)
	print("RE",res_error)
	if res_error == 0 :
		print("Zero error", i)
		print("RE", res_error)
	elif res_error<0.001:
		print("LEss than 0.001", i)
		print("RE", res_error)
	elif res_error<0.01:
		print("Less than 0.01",i)
		print("RE", res_error)
	elif res_error<0.1:
		print("LEss than 0.1",i)
		print("RE", res_error)

