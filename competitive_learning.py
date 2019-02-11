import math
import numpy as np
from scipy import signal
from matplotlib import pyplot as plt


def competetive(data, nodecount, eta, iterations):
    # np.random.shuffle(data)
    RBF = data[[0,2,4,5,9,60,61,62]]
    print(RBF)
    temp_data =[]
    for j in range(iterations):
        rand_id = np.random.randint(0, len(data))
        randvec = data[rand_id]  # Breaks down in multidimensional case
        data = np.delete(data,rand_id)
        temp_data.append(randvec)
        distances = []
        for center in RBF:
            distances = np.append(distances, (np.linalg.norm(center - randvec)))
        RBF = RBF[np.argsort(distances)]
        for i in range(len(RBF)):
            RBF[i] += (eta * (randvec - RBF[i]))/(np.square(i+1))
        if len(data)==0:
            data = temp_data
            temp_data = []
    plt.show()
    return RBF

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
    eta = 0.2
    iterations = 500000
    # iterations = 10
    data = trainX
    rbfMu = competetive(data, rbfNodes, eta, iterations)
    circ_y = np.sin(2 * rbfMu)
    circle_center = np.column_stack((rbfMu,circ_y))
    plt.plot(trainX, trainSinY, 'k.', label='actual function')
    for i in range(len(circle_center)):
        my_circ = plt.Circle(circle_center[i], 0.5, facecolor='None', edgecolor= 'lightsteelblue')
        plt.gcf().gca().add_artist(my_circ)
    plt.show()

    return rbfMu,rbfSigma,trainX,testX,trainSinY,testSinY,trainSqY,testSqY,eta

def calc_phi(X, rbfMu, rbfSigma):
    Phi = np.zeros((X.shape[0], rbfMu.shape[0]))
    for i in range(X.shape[0]):
        for j in range(rbfMu.shape[0]):
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

def online_delta_rule(epochs, train_ip, train_op, eta, Phi):
    residualError = []
    instantError = []
    W = np.random.rand(1,Phi.shape[1]) * 0.5
    N = Phi.shape[0]


    for iter in range(epochs):
        resError = []
        instError = []
        for k in range(N):
            guess = np.dot(Phi[k], W.T)
            err = train_op[k]-guess
            resError.append(np.abs(err))
            instErr = (1/2)*np.power(err, 2)
            instError.append(instErr)
            deltaW = eta*err*Phi[k]
            W = W + deltaW
        instantError.append(np.mean(np.abs(instError)))
        residualError.append(np.mean(resError))

    plt.plot(residualError[5:], 'g-', label='Absolute residual error')
    plt.title('Absolute residual error')
    plt.show()
    plt.plot(instantError[50:], 'g-', label='Average instantaneous error')
    plt.title('Average instantaneous error')
    plt.show()
    return residualError, instantError, W



def gaussianTransfer(x, mu, sigma):
    phi_i = np.exp(np.divide(-np.power((x-mu), 2), 2*np.power(sigma, 2)))
    return phi_i

for i in range(10,11):
    rbfMu, rbfSigma, trainX, testX, trainSinY, testSinY, trainSqY, testSqY,eta = gen_data(i)
    # plt.show()
    Phi_train = calc_phi(trainX, rbfMu, rbfSigma)
    Phi_train = np.concatenate((Phi_train, np.ones((Phi_train.shape[0], 1))), axis=1)
    Phi_test = calc_phi(testX, rbfMu, rbfSigma)
    Phi_test = np.concatenate((Phi_test, np.ones((Phi_test.shape[0], 1))), axis=1)
    residualError, instantError, W = online_delta_rule(50000,trainX,trainSinY,0.001, Phi_train)
    plt.plot(trainX, trainSinY, 'k.', label='actual function')
    plt.plot
    # plt.show()
    prediction = np.dot(Phi_train,W.T)
    test_pred = np.dot(Phi_test,W.T)

    plt.plot(trainX,prediction,'r.',label='train prediction')
    plt.plot(testX,test_pred,'bo', label='test prediction')
    plt.title('RBF function approximation')
    plt.legend()
    plt.show()
    print(residualError[-1])

