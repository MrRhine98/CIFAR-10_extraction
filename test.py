import numpy as np
from matplotlib import pyplot as plt

dir = {}
dir['train'] = 'datasets/CIFAR10/data_batch_'
dir['test'] = 'datasets/CIFAR10/test_batch'
for i in dir.keys():
    print(i)
    print(type(i))