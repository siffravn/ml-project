# -*- coding: utf-8 -*-

import numpy as np
import torch
from sklearn import model_selection
from collections import Counter

import log_reg
import ann
import baseline
import tl_cv_util

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

K1 = 10
K2 = 10

# Define values for log_reg
lambdas = np.logspace(-2, 2, 20).tolist()
lr_val_errors = np.zeros(shape=(K1, K2, len(lambdas)))

# Define values for ANN
hidden_units = range(1,11,1)
ann_val_errors = np.zeros(shape=(K1, K2, len(hidden_units)))

# Define table
table = np.zeros(shape=(K1, 5))

train_indexs = []
test_indexs = []

CV = model_selection.KFold(K1,shuffle=True)

for i, (train_index1, test_index1) in enumerate(CV.split(X,y)):
    
    # extract training and test set for current CV fold
    X_train1 = X[train_index1,:]
    y_train1 = y[train_index1]
    X_test1 = X[test_index1,:]
    y_test1 = y[test_index1]
    
    CV = model_selection.KFold(K2,shuffle=True)
    
    for j, (train_index2, test_index2) in enumerate(CV.split(X_train1,y_train1)):
        
        # Extract training and test set for current CV fold
        X_train2 = X[train_index2,:]
        y_train2 = y[train_index2]
        X_test2 = X[test_index2,:]
        y_test2 = y[test_index2]
        
        # Logistic regression 
        for l in lambdas :
            
            print('Fold i = ', i)
            print('Fold j = ', j)
            print('Model lambda = ', l)
            
            # Define the model structure #
            # Train model #
            # Test model #            
            y_test_est2 = log_reg.define_train_test(l,
                                                    X_train2, 
                                                    y_train2, 
                                                    X_test2, 
                                                    y_test2)
            
            # store error rate for current CV fold
            lr_val_errors[i][j][lambdas.index(l)] = log_reg.error_rate(y_test_est2, y_test2)
        
        
        # Artificial neural net
        X_tensor_train2 = torch.Tensor(X_train2)
        y_tensor_train2 = torch.Tensor(np.expand_dims(y_train2, axis=1).astype(np.uint8))
        X_tensor_test2 = torch.Tensor(X_test2)
        
        for h in hidden_units :
            
            print('Fold i = ', i)
            print('Fold j = ', j)
            print('Model hidden units = ', h)
            
            # Define the model structure #
            # Train model #
            # Test model #   
            y_test_est2 = ann.define_train_test(h, 
                                                M, 
                                                X_tensor_train2, 
                                                y_tensor_train2, 
                                                X_tensor_test2)
            
            # Store error rate for current CV fold
            ann_val_errors[i][j][hidden_units.index(h)] = ann.error_rate(y_test_est2, y_test2)
    
    
    weight = len(test_index2)/len(train_index1)
    
    # Log-reg ##
    # Find_optimal_model
    optimal_lambda = tl_cv_util.find_optimal_model(lambdas, 
                                                    lr_val_errors[i,:,:], 
                                                    weight)
    
    # Re-train and test optimal model    
    y_test_est1 = log_reg.define_train_test(optimal_lambda,
                                            X_train1, 
                                            y_train1, 
                                            X_test1, 
                                            y_test1)
    
    test_error_rate = log_reg.error_rate(y_test_est1, y_test1)

    
    # Save error_rate and lambda
    # table[i,0] = np.round(np.log10(optimal_lambda),2)
    table[i,0] = np.round(np.log10(optimal_lambda),2)
    table[i,1] = test_error_rate
    
    
    
    # ANN ##
    X_tensor_train1 = torch.Tensor(X_train1)
    y_tensor_train1 = torch.Tensor(np.expand_dims(y_train1, axis=1).astype(np.uint8))
    X_tensor_test1 = torch.Tensor(X_test1)
    
    
    # Compute model generalization error for each model s
    optimal_hidden_units = tl_cv_util.find_optimal_model(hidden_units, 
                                                          ann_val_errors[i,:,:], 
                                                          weight)
    
    # Re-train and test optimal model
    y_test_est1 = ann.define_train_test(optimal_hidden_units,
                                        M, 
                                        X_tensor_train1, 
                                        y_tensor_train1, 
                                        X_tensor_test1)
    
    test_error_rate = ann.error_rate(y_test_est1, y_test1)
    
    # Save error_rate and hidden_unit
    table[i,2] = optimal_hidden_units
    table[i,3] = test_error_rate
    
    
    # Baseline ##
    # Re-train and test baseline
    y_test_est1 = baseline.define_train_test(y_train1, X_test1)
    
    # Save error_rate
    table[i][4] = baseline.error_rate(y_test_est1, y_test1)
    
    ## Preperation for performance testing
    train_indexs.append(train_index1)
    test_indexs.append(test_index1)
    
    
# Compute the estimate of the generalization error

est_gen_errors = []

weight = len(test_index1)/N

est_gen_error = sum(weight*table[:,1])
est_gen_errors.append(est_gen_error)
print('Log_reg:  Estimated generalization error is', est_gen_error)

est_gen_error = sum(weight*table[:,3])
est_gen_errors.append(est_gen_error)
print('Ann:      Estimated generalization error is', est_gen_error)

est_gen_error = sum(weight*table[:,4])
est_gen_errors.append(est_gen_error)
print('Baseline: Estimated generalization error is', est_gen_error)


## Preperation for performance testing
counter = Counter(table[:,0])
opt_lambda = counter.most_common(1)[0][0]
print('most common: ', opt_lambda)

counter = Counter(table[:,2])
opt_hidden_unit = int(counter.most_common(1)[0][0])
print('most common: ', opt_hidden_unit)
