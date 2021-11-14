import numpy as np
from toolbox_02450 import mcnemar

from two_L_CV import *
import log_reg
import ann
import baseline

# store predictions.
yhat = []
y_true = []

i=0
for train_index, test_index in zip(train_indexs, test_indexs):
    print('i: ', i)
    print('train_index len: ', len(train_index))
    print('test_index len: ', len(test_index))
    
    i=i+1
    
    # Extract training and test set for current CV fold
    X_train = X[train_index,:]
    y_train = y[train_index]
    X_test = X[test_index,:]
    y_test = y[test_index]
    
    dy = []
    
    
    ## Begin "inner loop" ##
    
    # Log-reg ##
    # Fit classifier and classify the test points
    y_est = log_reg.define_train_test(opt_lambda, X_train, y_train, X_test, y_test)
    
    dy.append( y_est )
    
    # ANN ##
    # Fit classifier and classify the test points
    y_est = ann.define_train_test(opt_hidden_unit, M, X_train, y_train, X_test, y_test)
    
    dy.append( y_est )
    
    # Baseline ##
    # Fit classifier and classify the test points
    y_est = baseline.define_train_test(y_train, X_test)
    
    dy.append( y_est )
    
    ## End "inner loop" ##
    
    
    dy = np.stack(dy, axis=1)
    yhat.append(dy)
    y_true.append(y_test)
    
    
yhat = np.concatenate(yhat)
y_true = np.concatenate(y_true)



# Compute the mcnemar interval ##
test_result = np.zeros(shape=(3, 4))
alpha = 0.05

# Log_reg vs ANN
print('Log_reg vs ANN')
thetahat, CI, p = mcnemar(y_true, yhat[:,0], yhat[:,1], alpha=alpha)
test_result[0,0] = thetahat
test_result[0,1] = CI[0]
test_result[0,2] = CI[1]
test_result[0,3] = p
print()

# Log_reg vs baseline
print('Log_reg vs baseline')
[thetahat, CI, p] = mcnemar(y_true, yhat[:,0], yhat[:,2], alpha=alpha)
test_result[1,0] = thetahat
test_result[1,1] = CI[0]
test_result[1,2] = CI[1]
test_result[1,3] = p
print()

# ANN vs baseline
print('Ann vs baseline')
[thetahat, CI, p] = mcnemar(y_true, yhat[:,1], yhat[:,2], alpha=alpha)
test_result[2,0] = thetahat
test_result[2,1] = CI[0]
test_result[2,2] = CI[1]
test_result[2,3] = p
print()
