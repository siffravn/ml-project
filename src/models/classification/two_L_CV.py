# -*- coding: utf-8 -*-

import numpy as np
from sklearn import model_selection

import ann

import sys
import os
dirname = os.path.dirname(__file__)

foldername = os.path.join(dirname, '../../data')
sys.path.append(foldername)
import myData as d

md = d.myData()
N, M = md.X.shape

attributeNames = md.attributeNames[M-1]
classNames = ['class 1', 'class 2']

X = md.X[:,0:M-1].astype(float)
y = md.X[:,M-1].astype(float)

N, M = X.shape


# K1 = 5
# K2 = 5
# hidden_units = [1,2,3,4,5,6,7,8,9,10]
K1 = 2
K2 = 2

# Define values for ANN
hidden_units = [1,2,3]
val_errors = np.zeros(shape=(K1, K2, len(hidden_units)))
gen_errors = [None] * len(hidden_units)

# Define table
test_errors = np.zeros(shape=(K1, 2))


CV = model_selection.KFold(K1,shuffle=True)

for i, (train_index1, test_index1) in enumerate(CV.split(X,y)):
    
    # extract training and test set for current CV fold
    X_train1 = X[train_index1,:]
    y_train1 = y[train_index1]
    X_test1 = X[test_index1,:]
    y_test1 = y[test_index1]
    
    # test_index2 = 0;
    
    CV = model_selection.KFold(K2,shuffle=True)
    
    for j, (train_index2, test_index2) in enumerate(CV.split(X_train1,y_train1)):
        
        # Extract training and test set for current CV fold
        X_train2 = X[train_index2,:]
        y_train2 = y[train_index2]
        X_test2 = X[test_index2,:]
        y_test2 = y[test_index2]
        
        
        for h in hidden_units:
            
            print('Fold i = ', i)
            print('Fold j = ', j)
            print('Model hidden units = ', h)
            
            # Define the model structure #
            model = ann.define(M, h)
            
            # Train model #
            net, final_loss, learning_curve = ann.train(model, X_train2, y_train2)
            
            # Test model #
            val_error_rate = ann.test(net, X_test2, y_test2)
            
            # Save error_rate
            val_errors[i][j][hidden_units.index(h)] = val_error_rate # store error rate for current CV fold
        
    
    # Compute model generalization error for each model s
    print('Compute gen_error')
    
    for index in range(len(hidden_units)):
        h = hidden_units[index]
        
        # print('index =', index)
        # print('hidden =', h)
        
        errors = val_errors[i,:,index]
        
        gen_error = (len(test_index2)/len(train_index1))*errors
        gen_errors[index] = gen_error.sum()
        
        
    # Find and select optimal model 
    index = gen_errors.index(min(gen_errors))
    optimal_hidden_units = hidden_units[index]
    
    # Train and test optimal model #
    model = ann.define(M, optimal_hidden_units)
    net, final_loss, learning_curve = ann.train(model, X_train1, y_train1)
    test_error_rate = ann.test(net, X_test1, y_test1)
    
    # Save error_rate and hidden_unit
    test_errors[i,0] = optimal_hidden_units
    test_errors[i,1] = test_error_rate
    
    
# Compute the estimate of the generalization error
est_gen_error = sum((len(test_index1)/N)*test_errors[:,1])

print('Estimated generalization error is', est_gen_error)


