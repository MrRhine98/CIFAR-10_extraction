# Two_layer_FC_neural_network
'get_data.py'
-->Functions that called to extract image data from CIFAR-10 are included in get_data.py
CIFAR-10 contains totally 60000 RGB picture in the size of 32 by 32. 10000 of them are test data, and the other 50000 are training data.
Corresponding labels are also included.

'Layer.py'
-->Construct affine layer and relu layer in order to build the network

'FCNets.py'
-->Contains the two layer network model and corresponding method to train and predict

'Solver.py'
-->Contains the methods to do the training process

'test.py & NOtes.py'
-->For testing unsure functions and take some notes
