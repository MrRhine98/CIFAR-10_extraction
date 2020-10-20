import numpy as np
from matplotlib import pyplot as plt
A = np.array([[1, 2, 3, 0, 4, 7, 5],
             [1, 2, 3, 4, 5, 6, 7]])
print(np.argmax(A, axis=1))