#!/usr/bin/env python

import numpy as np
import random

from q1_softmax import softmax
from q2_sigmoid import sigmoid, sigmoid_grad
from q2_gradcheck import gradcheck_naive

eps = 1e-7

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

    W1 = np.reshape(params[ofs:ofs+ Dx * H], (Dx, H))
    ofs += Dx * H
    b1 = np.reshape(params[ofs:ofs + H], (1, H))
    ofs += H
    W2 = np.reshape(params[ofs:ofs + H * Dy], (H, Dy))
    ofs += H * Dy
    b2 = np.reshape(params[ofs:ofs + Dy], (1, Dy))

    ### YOUR CODE HERE: forward propagation
    x1 = np.dot(data, W1) + b1
    sig1 = sigmoid(x1)
    x2 = np.dot(sig1, W2) + b2
    sig2 = sigmoid(x2)
    out = np.array([softmax(x) for x in sig2])
    cost = np.sum(labels*np.log(out))
    # cost = np.sum(-1*labels*np.log(cost) - (1 - labels)*np.log(1 - cost), axis=1) # full regression crossentropy
    # cost = -1*np.sum(labels*np.log(cost), axis=1)
    # print(cost)
    ### END YOUR CODE

    ### YOUR CODE HERE: backward propagation
    # dSig2 = (out-labels) #cost * dCost(out) * dSoftmax(x2)# * dSigmoid(x2)
    dSig2 = -(out - labels) * sigmoid_grad(sig2)
    # print(cost)
    # print(dCost(out)[0])
    # print(dSoftmax(x2)[0])
    # print(dSig2[0])
    # exit()
    # print(dSig2l[0])
    # exit()
    gradb2 = np.dot(np.ones((1, dSig2.shape[0])), dSig2)
    # print(dSig2.shape)
    # print(b2.shape)
    # print(gradb2.shape)
    # exit()
    gradW2 = np.dot(sig1.T, dSig2)
    # print(dSig2.shape)
    # print(W2.shape)
    # print(gradW2.shape)
    # exit()

    # print(dSigmoid(x1).shape)
    # print(gradW2.shape)
    dSig1 = np.dot(dSig2, W2.T) * sigmoid_grad(sig1)
    # print(dSig1.shape)
    # exit()
    gradb1 = np.dot(np.ones((1, dSig1.shape[0])), dSig1)
    # print(dSig1.shape)
    # print(b1.shape)
    # print(gradb1.shape)
    # exit()
    gradW1 = np.dot(data.T, dSig1)
    # exit()
    ### END YOUR CODE


    ### Stack gradients (do not modify)
    grad = np.concatenate((gradW1.flatten(), gradb1.flatten(),
        gradW2.flatten(), gradb2.flatten()))

    return cost, grad


def sanity_check():
    """
    Set up fake data and parameters for the neural network, and test using
    gradcheck.
    """
    print "Running sanity check..."

    N = 20
    dimensions = [10, 5, 10]
    data = np.random.randn(N, dimensions[0])   # each row will be a datum
    labels = np.zeros((N, dimensions[2]))
    for i in xrange(N):
        labels[i, random.randint(0,dimensions[2]-1)] = 1

    params = np.random.randn((dimensions[0] + 1) * dimensions[1] + (
        dimensions[1] + 1) * dimensions[2], )

    gradcheck_naive(lambda params:
        forward_backward_prop(data, labels, params, dimensions), params)


def your_sanity_checks():
    """
    Use this space add any additional sanity checks by running:
        python q2_neural.py
    This function will not be called by the autograder, nor will
    your additional tests be graded.
    """
    print "Running your sanity checks..."
    ### YOUR CODE HERE
    pass
    ### END YOUR CODE


if __name__ == "__main__":
    sanity_check()
    your_sanity_checks()
