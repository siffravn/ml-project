# -*- coding: utf-8 -*-
"""
Created on Fri Nov  5 15:55:56 2021

@author: marie
"""

import sys 
import os
dirname = os.path.dirname(__file__)

from matplotlib.pylab import (figure, semilogx, loglog, xlabel, ylabel, legend, 
                           title, subplot, show, grid)
import numpy as np
from scipy.io import loadmat
import sklearn.linear_model as lm
from sklearn import model_selection
from toolbox_02450 import rlr_validate

foldername = os.path.join(dirname, '../data')
sys.path.append(foldername)
import myData as d

foldername = os.path.join(dirname, '../features')
sys.path.append(foldername)
import standardize as stan
import numberedAttributes as no

md = d.myData()
X = md.X
attr = md.attributeNames
N, M = X.shape

X=np.array(X,dtype=np.float64)
#X_stand = stan.standardize(X, md.N)
#X_stand=np.array(X_stand,dtype=np.float64)
X = stan.standardize(X, N)
X = np.array(X,dtype=np.float64)

noAttr=no.addNumbers(attr)

#Definerer age kolonnen som egen array, vi vil forudse og 
#fjerner age kolonnen fra dataen:
#y=X_stand[:,8]
#X_del=np.delete(X_stand,8,axis=1)

y=X[:,8]
X_del=np.delete(X,8,axis=1)

N, M = X_del.shape

# Add offset attribute
X_del = np.concatenate((np.ones((X_del.shape[0],1)),X_del),1)
attributeNames = np.hstack((np.array([u'Offset']),np.delete(noAttr,8)))
M = M+1

## Crossvalidation
# Create crossvalidation partition for evaluation
K = 10
CV = model_selection.KFold(K, shuffle=True)
#CV = model_selection.KFold(K, shuffle=False)

# Values of lambda
lambdas=np.zeros(10)
for i in range(10):
    power=-1.2+0.3*(i+1)
    lambdas[i]=np.power(10.,power)
#lambdas = np.power(10.,range(-2,6))

# Initialize variables for regularization linear regression:
LR_Error_train = np.empty((K,1))
LR_Error_test = np.empty((K,1))
LR_Error_train_rlr = np.empty((K,1))
LR_Error_test_rlr = np.empty((K,1))
LR_w_rlr = np.empty((M,K))

#Initialize variables for ANN model:
ANN_Error_train = np.empty((K,1))
ANN_Error_test = np.empty((K,1))
ANN_Error_train_rlr = np.empty((K,1))
ANN_Error_test_rlr = np.empty((K,1))
ANN_w_rlr = np.empty((M,K))

n_hidden_units = 2      # number of hidden units
n_replicates = 1        # number of networks trained in each k-fold
max_iter = 100000

#Variables for baseline:
mu= np.empty((K, K))
opt_mus=np.empty((K,1))
B_Error_train = np.empty((K,1))
B_Error_test = np.empty((K,1))
B_Error_train2 = np.empty((K,1))
B_Error_test2 = np.empty((K,1))
B_gen_errors = np.empty((K,1))
B_gen_errors2 = np.empty((K,1))

k=0
k2=0
opt_lambdas=np.zeros(K)
for train_index, test_index in CV.split(X_del,y):
    ########## Baseline outer level ###############
    # Train baseline model (inner level): 
    k2=0
    for train_index2, test_index2 in CV.split(X_del[train_index],y[train_index]):
        
        mu[k2,k]=y[train_index2].mean()
        B_Error_train2[k2,0]=np.square(y[train_index2]-mu[k2,k]).sum(axis=0)/len(y[train_index2])
        B_Error_test2[k2,0]=np.square(y[test_index2]-mu[k2,k]).sum(axis=0)/len(y[test_index2])
        B_gen_errors2[k2,0]=len(y[test_index2])/len(y[train_index])*B_Error_test2[k,0]
        
        k2+=1
    # Find the mean with minimum error:    
    opt_mu=mu[B_gen_errors2[:,0].argmin(),k]
    opt_mus[k,0]=opt_mu
    # Outer fold train and test error on baseline model:
    B_Error_train[k,0]=np.square(y[train_index]-opt_mu).sum(axis=0)/len(y[train_index])
    B_Error_test[k,0]=np.square(y[test_index]-opt_mu).sum(axis=0)/len(y[test_index])
    B_gen_errors[k,0]=(len(y[test_index])/len(y))*B_Error_test[k,0]
    
    ####### End of baseline model outer level ###################
    
    
    ####### Reguralized linear regression outer level ###########
    # extract training and test set for current CV fold
    X_train = X_del[train_index]
    y_train = y[train_index]
    X_test = X_del[test_index]
    y_test = y[test_index]
    internal_cross_validation = 10    
    
    opt_val_err, opt_lambda, mean_w_vs_lambda, train_err_vs_lambda, test_err_vs_lambda = rlr_validate(X_train, y_train, lambdas, internal_cross_validation)
    
    opt_lambdas[k]=opt_lambda
 
    Xty = X_train.T @ y_train
    XtX = X_train.T @ X_train
   
    # Estimate weights for the optimal value of lambda, on entire training set
    lambdaI = opt_lambda * np.eye(M)
    lambdaI[0,0] = 0 # Do no regularize the bias term
    LR_w_rlr[:,k] = np.linalg.solve(XtX+lambdaI,Xty).squeeze()
    # Compute mean squared error with regularization with optimal lambda
    LR_Error_train_rlr[k] = np.square(y_train-X_train @ LR_w_rlr[:,k]).sum(axis=0)/y_train.shape[0]
    LR_Error_test_rlr[k] = np.square(y_test-X_test @ LR_w_rlr[:,k]).sum(axis=0)/y_test.shape[0]
    ####### End of reguralized linear regression outer level ###########
    
    # Display the results for the last cross-validation fold
    if k == K-1:
        figure(k, figsize=(14,6))
        subplot(1,2,1)
        semilogx(lambdas,mean_w_vs_lambda.T[:,1:],'.-') # Don't plot the bias term
        xlabel('Regularization factor')
        ylabel('Mean Coefficient Values')
        grid()
        # You can choose to display the legend, but it's omitted for a cleaner 
        # plot, since there are many attributes
        #legend(attributeNames[1:], loc='best')
        
        subplot(1,2,2)
        title('Optimal lambda: 1e{0}'.format(np.log10(opt_lambda)))
        loglog(lambdas,train_err_vs_lambda.T,'b.-',lambdas,test_err_vs_lambda.T,'r.-')
        xlabel('Regularization factor')
        ylabel('Squared error (crossvalidation)')
        legend(['Train error','Validation error'])
        grid()
    
    ####### End of reguralized linear regression plotting ###########


    k+=1

show()

optimalLambda=opt_lambdas.mean()

# Compute generalization error on baseline model:
B_gen_error=B_gen_errors[:,0].sum()
    
# Display results
print("Baseline outer level errors:")
print(B_Error_test)

print("Reguralized linear regression outer level errors:")
print(LR_Error_test)

print("Neural network outer level errors:")
print(ANN_Error_test)

print('Finished script')