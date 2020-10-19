import numpy as np
from Layer import *

class FCNets(object):
    def __init__(self, input_size, hidden_size, output_size, std=1e-4):
        self.parameter = {}
        self.parameter['W1'] = std * np.random.randn(input_size, hidden_size)
        self.parameter['b1'] = np.zeros(hidden_size)
        self.parameter['W2'] = std * np.random.randn(hidden_size, output_size)
        self.parameter['b2'] = np.zeros(output_size)

    def loss(self, X, y, reg = 0.0):
        W1 = self.parameter['W1']
        b1 = self.parameter['b1']
        W2 = self.parameter['W2']
        b2 = self.parameter['b2']
        grad = {}

        N = X.shape[0]
        D = X.shape[1]
        C = W2.shape[1]

        layer1, layer1_cache = affine_forward(X, W1, b1)
        relu1, relu1_cache = relu_forward(layer1)
        scores, scores_cache = affine_forward(relu1, W2, b2)

        loss, dscores = softmax_loss(scores, y)
        loss += reg / 2 * (np.sum(W1*W1) + np.sum(W2*W2))

        drelu1, grad['W2'], grad['b2'] = affine_backward(dscores, scores_cache)
        dlayer1 = relu_backward(drelu1, relu1_cache)
        dx, grad['W1'], grad['b1'] = affine_backward(dlayer1, layer1_cache)
        return loss, grad



