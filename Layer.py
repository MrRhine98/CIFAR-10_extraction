import numpy as np

################################# This file contains different layers to build the neural netwrok ########################
########### Fully Connected Layer ############
def affine_forward(x, W, b):
    out = x.dot(W) + b
    cache = (x, W)
    return out, cache


def affine_backward(dout, cache):
    x, W = cache
    N = x.shape[0]
    dx = dout.dot(W.T)
    dW = x.T.dot(dout)
    db = np.ones((1, N)).dot(dout).reshape(-1)
    return dx, dW, db


############# ReLU Layer ######################
def relu_forward(x):
    out = np.maximum(x, 0)
    cache = x
    return out, cache


def relu_backward(dout, cache):
    x = cache
    dout[x <= 0] = 0
    dx = dout
    return dx


############# Softmax Layer #####################
def softmax_loss(x, y):
    """
    Computes the loss and gradient for softmax classification.

    Inputs:
    - x: Input data, of shape (N, C) where x[i, j] is the score for the jth
      class for the ith input.
    - y: Vector of labels, of shape (N,) where y[i] is the label for x[i] and
      0 <= y[i] < C

    Returns a tuple of:
    - loss: Scalar giving the loss
    - dx: Gradient of the loss with respect to x
    N = y.shape[0]
    row_sum = np.sum(np.exp(x), axis=1)
    log_sum = np.log(row_sum)
    correct_scores = x[np.arange(N), y]
    loss = np.sum(log_sum - correct_scores) / N
    """
    shifted_logits = x - np.max(x, axis=1, keepdims=True)
    Z = np.sum(np.exp(shifted_logits), axis=1, keepdims=True)
    log_probs = shifted_logits - np.log(Z)
    probs = np.exp(log_probs)
    N = x.shape[0]
    loss = -np.sum(log_probs[np.arange(N), y]) / N
    dx = probs.copy()
    dx[np.arange(N), y] -= 1
    dx /= N
    return loss, dx

