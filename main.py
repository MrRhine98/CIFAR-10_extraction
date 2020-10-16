# This is a test project for my machine learning project
import numpy as np
import get_data
import matplotlib.pyplot as plt
dir = 'datasets/CIFAR10/data_batch_'
dir_test = 'datasets/CIFAR10/test_batch'
xs = []
ys = []

for i in np.arange(5):
    dict = get_data.load_data( dir + str(i+1))
    X_temp = dict['data']
    X_temp = X_temp.reshape(10000, 3, 32, 32)
    xs.append(X_temp)
    y_temp = dict['labels']
    ys.append(y_temp)
X_train = np.concatenate(xs)
y_train = np.concatenate(ys)
dict = get_data.load_data(dir_test)
X_temp = dict['data']
X_test = X_temp.reshape(10000, 3, 32, 32)
y_t = dict['labels']
y_test = np.array(y_temp)
print(type(y_test))
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
