# -*- coding: utf-8 -*-

import torch
from toolbox_02450 import train_neural_net

def define_train_test(hidden_units, M, X_train, y_train, X_test, y_test):
    
    model = define(M, hidden_units)
    net, final_loss, learning_curve = train(model, X_train, y_train)
    error_rate = test(net, X_test, y_test)
    
    return net, error_rate
    
    
    
    
def define(M, hidden_units):
    # Define the model structure #
    # The lambda-syntax defines an anonymous function, which is used here to 
    # make it easy to make new networks within each cross validation fold
    model = lambda: torch.nn.Sequential(
        torch.nn.Linear(M, hidden_units), #M features to H hiden units
        # 1st transfer function, either Tanh or ReLU:
        torch.nn.Tanh(),        #torch.nn.ReLU(),
        torch.nn.Linear(hidden_units, 1),  # H hidden units to 1 output neuron
        torch.nn.Sigmoid()      # final tranfer function
        )
        
    return model
    
def train(model, X_train, y_train):
    
    # Convert training set to PyTorch tensors
    X_train = torch.Tensor(X_train)
    y_train = torch.Tensor(y_train)
       
    # Since we're training a neural network for binary classification, we use a 
    # binary cross entropy loss (see the help(train_neural_net) for more on
    # the loss_fn input to the function)   
    loss_fn = torch.nn.BCELoss()
    
    # Train for a maximum of 10000 steps, or until convergence (see help for the 
    # function train_neural_net() for more on the tolerance/convergence))
    max_iter = 10000
    
    # Train model #
    # Go to the file 'toolbox_02450.py' in the Tools sub-folder of the toolbox
    # and see how the network is trained (search for 'def train_neural_net',
    # which is the place the function below is defined)
    net, final_loss, learning_curve = train_neural_net(model,
                                                       loss_fn,
                                                       X=X_train,
                                                       y=y_train,
                                                       n_replicates=3,
                                                       max_iter=max_iter)
    return net, final_loss, learning_curve


def test(net, X_test, y_test):
    
    # Convert test set to PyTorch tensors
    X_test = torch.Tensor(X_test)
    y_test = torch.Tensor(y_test)
    
    # Test model #
    # Determine estimated class labels for test set
    y_sigmoid = net(X_test) # activation of final note, i.e. prediction of network
    y_test_est = (y_sigmoid > .5).type(dtype=torch.uint8) # threshold output of sigmoidal function
    y_test = y_test.type(dtype=torch.uint8)
    
    # Determine errors and error rate
    e = (y_test_est != y_test)
    error_rate = (sum(e).type(torch.float)/len(y_test)).data.numpy()
    
    return error_rate