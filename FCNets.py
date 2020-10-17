import numpy as np

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

        N = X.shape[0]
        D = X.shape[1]
        C = W2.shape[1]

        relu1 = np.maximum(X.dot(W1) + b1, 0)
        scores = relu1.dot(W2) + b2

        row_sum = np.sum(np.exp(scores), axis=1)
        log_sum = np.log(row_sum)
        correct_scores = scores[:, y]
        loss = np.sum(log_sum - correct_scores) / N
        loss += reg / 2 * (np.sum(W1*W1) + np.sum(W2*W2))
        return loss


