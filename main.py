# This is a test project for my machine learning project
# Starting date: 2020.9.25
# d6dd12a487474cde166bccc484bd5acbccbf1bb2  token

##################### Some setup####################################
import get_data
from Solver import *
dir = {}

################### extract data from CIFAR-10######################
# dir contain the path for CIFAR-10 data
# X, y are middle variables
# X_train contains 50000 * 3 * 32 * 32 RGB pictures
# X_test contains 10000 *3 *32 * 32 RGB pictures
X = {}
y = {}
dir = {}
dir['train'] = 'datasets/CIFAR10/data_batch_'
dir['test'] = 'datasets/CIFAR10/test_batch'

for i in dir.keys():
    X[i], y[i] = get_data.get_image(dir[i], mode=i)

X_train = X['train']
y_train = y['train']
X_test = X['test']
y_test = y['test']

del X, y, i

############# Show some examples of CIFAR-10 #####################
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
        indx = i * row + j +1
        example_x = X_train[indx - 1, :, :, :].transpose(1, 2, 0)
        plt.subplot(row, col, indx)
        plt.imshow(example_x)
# plt.show()

###################### Preprocess the data #########################
' -> Divide the data into train, validation and test sets'
num_train = 40000
num_val = 10000
num_class = 10
mask = list(np.arange(num_val))
X_val = X_train[mask]
y_val = y_train[mask]
mask = list(np.arange(num_val, num_train + num_val))
X_train = X_train[mask]
y_train = y_train[mask]

' -> Substract the mean from every feature'
mean = np.mean(X_train, axis=0)
X_train = X_train - mean
X_test = X_test - mean
X_val = X_val - mean
del mean

' -> Flatten the image from 4-D data to 2-D data as an input to the network'
X_train = X_train.reshape((num_train, -1))
X_val = X_val.reshape((num_val, -1))
X_test = X_test.reshape((X_test.shape[0], -1))

' -> Package the data in a dictionary data{}'
data = {}
data['X_train'] = X_train       # (40000, 3072)
data['X_test'] = X_test         # (10000, 3072)
data['X_val'] = X_val           # (10000, 3072)
data['y_train'] = y_train       # (40000, )
data['y_test'] = y_test         # (10000, )
data['y_val'] = y_val           # (10000, )


#################### load the model and overfit #########################
import FCNets
loss_history = []
H = 100
D = X_train.shape[1]
C = num_class
FCnetwork = FCNets.FCNets(D, H, C)

Solver = solver(FCnetwork, data, num_epoch=3)
Solver.train()
'''
mask = list(np.arange(10))
X_mini = X_train[mask]
y_mini = y_train[mask]
for i in np.arange(100):
    loss = FCnetwork.loss(X_mini, y_mini)
    print('loss is', loss)
    loss_history.append(loss)
    FCnetwork.update(learning_rate=1e-3)
loss = FCnetwork.loss(X_mini, y_mini)
print('Final loss is', loss)
plt.figureplt.title('Loss history')
plt.xlabel('Update')
plt.ylabel('loss')
plt.plot(loss_history)
plt.show()
'''
