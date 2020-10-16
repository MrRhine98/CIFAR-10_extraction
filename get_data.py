# Get the data from the CIFAR-10 datasets
import numpy as np
import pickle

def load_data(dir):
    with open(dir, 'rb') as f:
        dict = pickle.load(f, encoding='latin1')
        return dict

