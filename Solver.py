import numpy as np
import matplotlib.pyplot as plt
class solver(object):
    def __init__(self, model, data, **kwargs):
        self.model = model
        self.X_train = data['X_train']
        self.X_val = data['X_val']
        self.X_test = data['X_test']
        self.y_train = data['y_train']
        self.y_val = data['y_val']
        self.y_test = data['y_test']

        self.learning_rate = kwargs.pop('learning_rate', 1e-3)
        self.batch_size = kwargs.pop('batch_size', 100)
        self.num_epoch = kwargs.pop('num_epoch', 10)

        self.grad = {}
        self.loss_history = []

    def train(self):
        num_train = self.X_train.shape[0]
        iter_per_epoch = np.maximum(num_train // self.batch_size, 1)
        for epoch in np.arange(self.num_epoch):
            for iter in np.arange(iter_per_epoch):
                batch_mask = np.random.choice(num_train, self.batch_size, replace=False)
                X_batch = self.X_train[batch_mask]
                y_batch = self.y_train[batch_mask]
                loss, self.grad = self.model.loss(X_batch, y_batch, reg=0.8)
                self.loss_history.append(loss)
                print('Epoch %d, iter %d loss is' %(epoch+1, iter+1), loss)
                ' Update the parameter'
                for para in self.model.parameter.keys():
                    self.model.parameter[para] += -self.learning_rate * self.grad[para]
        print('Final loss is', loss)
        plt.figure()
        plt.title('Loss history')
        plt.xlabel('Update')
        plt.ylabel('loss')
        plt.plot(self.loss_history)
        plt.show()
