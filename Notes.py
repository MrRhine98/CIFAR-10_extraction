import numpy as np
from matplotlib import pyplot as plt

# Show some examples of CIFAR-10
row = 5
col = 5
plt.figure()
for i in np.arange(row):
    for j in np.arange(col):
        indx = i * row + j + 1
        example_x = X[indx - 1, :, :, :].transpose(1, 2, 0)
        plt.subplot(row, col, indx)
        plt.imshow(example_x)
plt.show()