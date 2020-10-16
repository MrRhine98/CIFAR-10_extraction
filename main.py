# This is a test project for my machine learning project
import numpy as np
import get_data
import matplotlib.pyplot as plt
dir = {}
X = {}
y = {}
dir['train'] = 'datasets/CIFAR10/data_batch_'
dir['test'] = 'datasets/CIFAR10/test_batch'

for i in dir.keys():
    X[i], y[i] = get_data.get_image(dir[i], mode=i)

X_train = X['train']
y_train = y['train']
X_test = X['test']
y_test = y['test']
del X, y
print(X_train.shape)
print(y_train.shape)
print(X_test.shape)
print(y_test.shape)

# Show some examples of CIFAR-10
row = 5
col = 5
plt.figure()
for i in np.arange(row):
    for j in np.arange(col):
        indx = i * row + j + 1
        example_x = X_train[indx - 1, :, :, :].transpose(1, 2, 0)
        plt.subplot(row, col, indx)
        plt.imshow(example_x)
plt.show()
