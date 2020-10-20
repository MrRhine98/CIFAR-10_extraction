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
        self.print_every = kwargs.pop('print_every', 100)

        self.grad = {}
        self.loss_history = []
        self.acc_val_history = []
        self.acc_train_history = []

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
                if iter % self.print_every == 0:
                    print('Epoch %d, iter %d loss is' %(epoch+1, iter+1), loss)
                    acc_val = self.check_accuracy(self.X_val, self.y_val)
                    acc_train = self.check_accuracy(self.X_train, self.y_train)
                    self.acc_train_history.append(acc_train)
                    self.acc_val_history.append(acc_val)
                ' Update the parameter'
                for para in self.model.parameter.keys():
                    self.model.parameter[para] += -self.learning_rate * self.grad[para]
        print('Final loss is', loss)
        plt.figure()
        plt.title('Loss history')
        plt.xlabel('Update')
        plt.ylabel('loss')
        plt.plot(self.loss_history)
        plt.figure()
        plt.title('val vs train accuracy')
        plt.xlabel('every 100 iterations')
        plt.ylabel('accuracy')
        plt.plot(self.acc_val_history, 'b')
        plt.plot(self.acc_train_history, 'g')
        plt.show()

    def check_accuracy(self, X, y, batch=1000):
        num_X = X.shape[0]
        if num_X <= batch:
            num = num_X
        else:
            num = batch

        mask = np.random.choice(num_X, num, replace=False)
        X_check = X[mask]
        y_check = y[mask]
        y_pred = self.model.predict(X_check)
        acc = np.mean(y_check == y_pred)
        return acc

