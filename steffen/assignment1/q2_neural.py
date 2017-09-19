#!/usr/bin/env python

import numpy as np
import random
import os

from q1_softmax import softmax
from q2_sigmoid import sigmoid, sigmoid_grad
from q2_gradcheck import gradcheck_naive

from q3_sgd import sgd

import matplotlib.pyplot as plt

def forward_test(data, labels, params, dimensions):
    ofs = 0
    Dx, H, Dy = (dimensions[0], dimensions[1], dimensions[2])
    # print(params)
    # print(params[ofs:ofs + Dx * H])
    # print((Dx, H))
    W1 = np.reshape(params[ofs:ofs + Dx * H], (Dx, H))
    ofs += Dx * H
    b1 = np.reshape(params[ofs:ofs + H], (1, H))
    ofs += H
    W2 = np.reshape(params[ofs:ofs + H * Dy], (H, Dy))
    ofs += H * Dy
    b2 = np.reshape(params[ofs:ofs + Dy], (1, Dy))

    ### YOUR CODE HERE: forward propagation
    h = sigmoid(np.dot(data, W1) + b1)
    yHat = softmax(np.dot(h, W2) + b2)
    cost = np.count_nonzero(np.argmax(yHat, axis=1) - np.argmax(labels, axis=1))
    return cost

def forward_backward_prop(data, labels, params, dimensions):
    """
    Forward and backward propagation for a two-layer sigmoidal network

    Compute the forward propagation and for the cross entropy cost,
    and backward propagation for the gradients for all parameters.

    Arguments:
    data -- M x Dx matrix, where each row is a training example.
    labels -- M x Dy matrix, where each row is a one-hot vector.
    params -- Model parameters, these are unpacked for you.
    dimensions -- A tuple of input dimension, number of hidden units
                  and output dimension
    """

    ### Unpack network parameters (do not modify)
    ofs = 0
    Dx, H, Dy = (dimensions[0], dimensions[1], dimensions[2])
    # print(params)
    # print(params[ofs:ofs + Dx * H])
    # print((Dx, H))
    W1 = np.reshape(params[ofs:ofs + Dx * H], (Dx, H))
    ofs += Dx * H
    b1 = np.reshape(params[ofs:ofs + H], (1, H))
    ofs += H
    W2 = np.reshape(params[ofs:ofs + H * Dy], (H, Dy))
    ofs += H * Dy
    b2 = np.reshape(params[ofs:ofs + Dy], (1, Dy))

    ### YOUR CODE HERE: forward propagation
    h = sigmoid(np.dot(data, W1) + b1)
    yHat = softmax(np.dot(h, W2) + b2)
    cost = -np.sum(np.multiply(labels, np.log(yHat)))
    # print("cost", cost)
    ### END YOUR CODE

    ### YOUR CODE HERE: backward propagation
    d1 = np.subtract(yHat, labels)
    d2 = np.dot(d1, W2.T)
    d3 = np.multiply(d2, sigmoid_grad(h))
    gradW2 = np.dot(h.T, d1)
    gradb2 = np.dot(np.ones((1, d1.shape[0])), d1)
    gradW1 = np.dot(data.T, d3)
    gradb1 = np.dot(np.ones((1, d3.shape[0])), d3)
    
    ### END YOUR CODE

    ### Stack gradients (do not modify)
    grad = np.concatenate((gradW1.flatten(), gradb1.flatten(),
        gradW2.flatten(), gradb2.flatten()))

    return cost, grad

def dynamicNN(data, labels, params, dimensions):
    ofs = 0
    Dx = dimensions[0]
    Dy = dimensions[len(dimensions)-1]
    Weights = []
    biases = []
    for i in range(len(dimensions)-1):
        dim = dimensions[i]
        dim1 = dimensions[i+1]
        Weights.append(np.reshape(params[ofs:ofs + dim * dim1], (dim, dim1)))
        ofs += dim * dim1
        biases.append(np.reshape(params[ofs:ofs + dim1], (1, dim1)))
        ofs += dim1
    
    # forwardprop
    hiddenLayers = []
    lastSigmoid = len(dimensions)-2
    for i in range(lastSigmoid):
        hiddenLayers.append(sigmoid(np.dot(data, Weights[i]) + biases[i]))
    prediction = softmax(np.dot(hiddenLayers[lastSigmoid], Weights[lastSigmoid]) + biases[lastSigmoid])

    cost = -np.sum(np.multiply(labels, np.log(prediction)))

    # backprop
    

    return cost, grad

def sanity_check():
    """
    Set up fake data and parameters for the neural network, and test using
    gradcheck.
    """
    print("Running sanity check...")

    N = 20
    dimensions = [10, 5, 10]
    data = np.random.randn(N, dimensions[0])   # each row will be a datum
    labels = np.zeros((N, dimensions[2]))
    for i in range(N):
        labels[i, random.randint(0,dimensions[2]-1)] = 1

    params = np.random.randn((dimensions[0] + 1) * dimensions[1] + (dimensions[1] + 1) * dimensions[2], )

    gradcheck_naive(lambda params:
        forward_backward_prop(data, labels, params, dimensions), params)


def your_sanity_checks():
    """
    Use this space add any additional sanity checks by running:
        python q2_neural.py
    This function will not be called by the autograder, nor will
    your additional tests be graded.
    """
    print("Running your sanity checks...")
    ### YOUR CODE HERE
    PATH = os.getcwd()
    trainFile = os.path.join(PATH, "optdigits_train.txt")
    testFile = os.path.join(PATH, "optdigits_test.txt")

    with open(trainFile, 'r') as train_data:
        data = [line.split(',') for line in train_data.readlines()]
        for i in range(len(data)):
            for j in range(len(data[i])):
                data[i][j] = int(data[i][j])
        trainX = np.asarray(data)[:, :-1]
        indexY = np.transpose(np.asarray(data)[:, -1:])[0]
        shp = np.arange(indexY.shape[0])
        trainY = np.zeros((trainX.shape[0], 10))
        trainY[shp, indexY] = 1
        # print(trainY)

    with open(testFile, 'r') as test_data:
        data = [line.split(',') for line in test_data.readlines()]
        for i in range(len(data)):
            for j in range(len(data[i])):
                data[i][j] = int(data[i][j])
        testX = np.asarray(data)[:,:-1]
        indexY = np.transpose(np.asarray(data)[:, -1:])[0]
        shp = np.arange(indexY.shape[0])
        testY = np.zeros((testX.shape[0], 10))
        testY[shp, indexY] = 1
    
    dimensions = [64, 128, 10]

    siz = sum((dimensions[i] + 1) * dimensions[i+1] for i in range(len(dimensions)-1))
    params = np.random.randn(siz)

    trainChart = []
    testChart = []
    rate = 0.001
    for tenEpoc in range(500):
        sgd(forward_backward_prop, [trainX, trainY, params, dimensions], rate, 10, PRINT_EVERY=100)
        trainChart.append(forward_test(trainX, trainY, params, dimensions))
        testChart.append(forward_test(testX, testY, params, dimensions))
        print("Train iter\t", 10*tenEpoc, "\t:", trainChart[tenEpoc])
        print("Test iter\t", 10*tenEpoc, "\t:", testChart[tenEpoc])
        if testChart[tenEpoc] < 130:
            rate = 0.0001

    fig = plt.figure()

    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.plot(np.arange(len(trainChart)), trainChart, np.arange(len(testChart)), testChart)
    plt.legend(['train', 'test'], loc='upper right')
    plt.show()
    ### END YOUR CODE


if __name__ == "__main__":
    sanity_check()
    your_sanity_checks()
