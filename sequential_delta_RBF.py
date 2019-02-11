import math
import numpy as np
from scipy import signal
from matplotlib import pyplot as plt


# 3.1 Batch mode training using least squares
# - Supervised learning of network weights
def gen_data(rbfNodes):
    stepSize = 0.1
    interval = [0, 2 * math.pi]
    trainStart = 0
    testStart = 0.05
    # The points at which the functions should be evaluated
    trainX = np.arange(interval[0] + trainStart, interval[1], stepSize)
    testX = np.arange(interval[0] + testStart, interval[1], stepSize)

    # noise_train = np.random.normal(0, 0.1, len(trainX))
    # noise_test = np.random.normal(0, 0.1, len(testX))

    noise_train = 0
    noise_test = 0

    # sin(2x)
    trainSinY = np.sin(2 * trainX) + noise_train
    testSinY = np.sin(2 * testX) + noise_test
    # print(testSinY)

    # square(2x)
    trainSqY = np.ones(len(trainSinY))
    testSqY = np.ones(len(testSinY))
    trainSqY[np.where(trainSinY >= 0)] = 1
    trainSqY[np.where(trainSinY < 0)] = -1
    testSqY[np.where(testSinY >= 0)] = 1
    testSqY[np.where(testSinY < 0)] = -1

    # RBF nodes
    rbfSigma = 1
    intervalOffset = math.pi / math.exp(math.pi)  # Just having fun with math. Don't know of this offset is necessary whatsoever.
    rbfMu = np.linspace(interval[0] + intervalOffset, interval[1] - intervalOffset, rbfNodes)
    return rbfMu,rbfSigma,trainX,testX,trainSinY,testSinY,trainSqY,testSqY

def calc_phi(X, rbfMu, rbfSigma):
    Phi = np.zeros((X.shape[0], len(rbfMu)))
    for i in range(X.shape[0]):
        for j in range(len(rbfMu)):
            Phi[i][j] = gaussianTransfer(X[i], rbfMu[j], rbfSigma)
        # Phi[i][j] = 1/(1+ np.exp(Phi[i][j]))
    # print(Phi[i][j])
    return Phi

def predict_square(testX,Phi,w):
    predicted = np.ones(len(testX))
    predicted[np.where(np.dot(Phi, w) >=0)] = 1
    predicted[np.where(np.dot(Phi, w) <0)] = -1
    return predicted

def predict_sin(Phi,w):
    return np.dot(Phi, w)

# Sequential learning of RBF
def online_delta_rule(epochs, train_ip, train_op, eta,phi):
    MSE = []
    W = np.random.rand(1,phi.shape[1]) * 0.5

    for iter in range(epochs):
        for ind,input in enumerate(phi):
            input = np.reshape(input,(1, len(input)))
            error = train_op[ind] - np.dot(input, W.T)
            delta_w = eta * np.dot(input.T, error)
            W += delta_w.T
        # print(W)
        predicted = np.dot(phi,W.T)

        mse = np.mean(np.abs((predicted - train_op)))

        if iter>1:
            MSE.append(mse)

    plt.plot(range(2, epochs), MSE, 'g-')
    plt.xlabel('Epochs')
    plt.ylabel('Residual error')
    plt.title('Plot of Residual error Vs epochs')
    plt.show()
    return MSE, W


def gaussianTransfer(x, mu, sigma):
    phi_i = np.exp(np.divide(-np.power((x-mu), 2), 2*np.power(sigma, 2)))
    return phi_i

for i in range(10,11):
    rbfMu, rbfSigma, trainX, testX, trainSinY, testSinY, trainSqY, testSqY = gen_data(i)
    Phi_train = calc_phi(trainX, rbfMu, rbfSigma)
    Phi_train = np.concatenate((Phi_train, np.ones((Phi_train.shape[0], 1))), axis=1)
    Phi_test = calc_phi(testX, rbfMu, rbfSigma)
    Phi_test = np.concatenate((Phi_test, np.ones((Phi_test.shape[0], 1))), axis=1)
    MSE, W = online_delta_rule(30000,trainX,trainSinY,0.0001, Phi_train)
    prediction = np.dot(Phi_train,W.T)
    test_pred = np.dot(Phi_test,W.T)
    plt.plot(trainX, trainSinY, 'k--', label='actual function')
    plt.plot(trainX,prediction,'r.',label='train prediction')
    plt.plot(testX,test_pred,'bo', label='test prediction')
    plt.legend()
    plt.title('For 6 RBF nodes')
    plt.show()
    print(MSE[-1])
