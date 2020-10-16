# Get the data from the CIFAR-10 datasets
import numpy as np
import pickle

def load_data(dir):
    with open(dir, 'rb') as f:
        dict = pickle.load(f, encoding='latin1')
        return dict

def get_image(dir, mode):
    xs = []
    ys = []
    if mode == 'train':
        for i in np.arange(5):
            dict = load_data( dir + str(i+1))
            X_temp = dict['data']
            X_temp = X_temp.reshape(10000, 3, 32, 32)
            xs.append(X_temp)
            y_temp = dict['labels']
            ys.append(y_temp)
        X_train = np.concatenate(xs)
        y_train = np.concatenate(ys)
        return X_train, y_train
    elif mode == 'test':
        dict = load_data(dir)
        X_temp = dict['data']
        X_test = X_temp.reshape(10000, 3, 32, 32)
        y_temp = dict['labels']
        y_test = np.array(y_temp)
        return X_test, y_test

