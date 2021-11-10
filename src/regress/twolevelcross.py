# -*- coding: utf-8 -*-
"""
Created on Fri Nov  5 15:55:56 2021

@author: marie
"""

import sys 
import os
dirname = os.path.dirname(__file__)

from matplotlib.pylab import (figure, semilogx, loglog, xlabel, ylabel, legend, 
                           title, subplot, show, grid, subplots, rcParams, plot,
                           ylim, xlim)
import numpy as np
import torch
#from scipy.io import loadmat
#import sklearn.linear_model as lm
from sklearn import model_selection
from toolbox_02450 import rlr_validate,train_neural_net
#from scipy import stats

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
K = 5
CV = model_selection.KFold(K, shuffle=True)
#CV = model_selection.KFold(K, shuffle=False)

# Values of lambda
lambdas=np.zeros(10)
for i in range(10):
    power=-1.2+0.3*(i+1)
    lambdas[i]=np.power(10.,power)
#lambdas = np.power(10.,range(-2,6))

#Variables for baseline:
mu= np.empty((K, K))
opt_mus=np.empty((K,1))
B_Error_train = np.empty((K,1))
B_Error_test = np.empty((K,1))
B_Error_train2 = np.empty((K,1))
B_Error_test2 = np.empty((K,1))
B_gen_errors = np.empty((K,1))
B_gen_errors2 = np.empty((K,1))

# Initialize variables for regularization linear regression:
LR_Error_train = np.empty((K,1))
LR_Error_test = np.empty((K,1))
LR_Error_train_rlr = np.empty((K,1))
LR_Error_test_rlr = np.empty((K,1))
LR_w_rlr = np.empty((M,K))

#Setup figure for linear regression regularization factor:
regu, regu_axes = subplots(1,2, figsize=(14,6))
# Setup figure for display of learning curves and error rates in fold
summaries, summaries_axes = subplots(1,2, figsize=(10,5))
# Make a list for storing assigned color of learning curve for up to K=10
color_list = ['tab:orange', 'tab:green', 'tab:purple', 'tab:brown', 'tab:pink',
              'tab:gray', 'tab:olive', 'tab:cyan', 'tab:red', 'tab:blue']
rc = {"font.family" : "serif", 
      "mathtext.fontset" : "stix"}
rcParams.update(rc)
rcParams["font.serif"] = ["Times New Roman"] + rcParams["font.serif"]


# Parameters for neural network classifier
n_hidden_units = range(5,15)     # range of hidden units
H=len(n_hidden_units) #number of different hidden units tested
n_replicates = 1       # number of networks trained in each k-fold
max_iter = 5000

# Initialize empty arrays:
ANN_Error_test = np.empty((K,1))
ANN_Error_test2 = np.empty((H,1))
ANN_all_Error_test2 = np.empty((H,K))
ANN_best_models=[]


k=0
opt_lambdas=np.zeros(K)
for train_index, test_index in CV.split(X_del,y):
    print('\nCrossvalidation outer fold: {0}/{1}'.format(k+1,K))
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
    #B_gen_errors[k,0]=(len(y[test_index])/len(y))*B_Error_test[k,0]
    
    ####### End of baseline model outer level ###################
    
    
    ####### Reguralized linear regression outer level ###########
    # extract training and test set for current CV fold
    X_train = X_del[train_index]
    y_train = y[train_index]
    X_test = X_del[test_index]
    y_test = y[test_index]
    internal_cross_validation = K   
    
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
        regu.set_label('CV fold {0}'.format(k+1))
        regu_axes[0].semilogx(lambdas,mean_w_vs_lambda.T[:,1:],'.-') # Don't plot the bias term
        regu_axes[0].set_xlabel('Regularization factor')
        regu_axes[0].set_ylabel('Mean Coefficient Values')
        regu_axes[0].grid()
        # You can choose to display the legend, but it's omitted for a cleaner 
        # plot, since there are many attributes
        #legend(attributeNames[1:], loc='best')
        
        regu_axes[1].set_title('Optimal lambda: 1e{0}'.format(np.log10(opt_lambda)))
        regu_axes[1].loglog(lambdas,train_err_vs_lambda.T,'b.-',lambdas,test_err_vs_lambda.T,'r.-')
        regu_axes[1].set_xlabel('Regularization factor')
        regu_axes[1].set_ylabel('Squared error (crossvalidation)')
        regu_axes[1].legend(['Train error','Validation error'])
        regu_axes[1].grid()
        
        regu.show()
    
    ####### End of reguralized linear regression plotting ###########
    
    ####### Start of Neural Network outer fold ######################
    
    CV2 = model_selection.KFold(K, shuffle=True)
    k2=0
    for train_index, test_index in CV2.split(X_train,y_train):
        print('\nCrossvalidation inner fold: {0}/{1}'.format(k2+1,K)) 
        X_train2 = torch.Tensor(X[train_index,:])
        y_train2 = torch.Tensor(y[train_index])
        y_train2 = y_train2.reshape(len(y_train2),1)
        X_test2 = torch.Tensor(X[test_index,:])
        y_test2 = torch.Tensor(y[test_index])
        y_test2 = y_test2.reshape(len(y_test2),1)
        
        k3=0
        # Train the net on training data
        for h in n_hidden_units:
            # Define the model
            model = lambda: torch.nn.Sequential(
                                torch.nn.Linear(M,h), #M features to n_hidden_units
                                torch.nn.ReLU(), #torch.nn.Tanh(),   # 1st transfer function,
                                torch.nn.Linear(h, 1),# n_hidden_units to 1 output neuron
                                # torch.nn.Sigmoid()      # final tranfer function
                                # no final tranfer function, i.e. "linear output"
                                )
            loss_fn = torch.nn.MSELoss() # notice how this is now a mean-squared-error loss
            
            print('Training model of type:\n\n{}\n'.format(str(model())))

            net, final_loss, learning_curve = train_neural_net(model,
                                                               loss_fn,
                                                               X=X_train2,
                                                               y=y_train2,
                                                               n_replicates=n_replicates,
                                                               max_iter=max_iter, tolerance=1e-40)
            
            # print('\n\tBest loss for {} hidden units: {:.6f}\n'.format(h,final_loss))
            
            # Determine estimated class labels for test set
            y_test2_est = net(X_test2)
        
            # Determine errors and errors
            se = np.square(np.squeeze(y_test2_est.data.numpy())-np.squeeze(y_test2.data.numpy())) # squared error
            # mse = (sum(se).type(torch.float)/len(y_test)).data.numpy() #mean
            mse = se.sum()/len(y_test2)
            #errors2.append(mse)
            ANN_Error_test2[k3,0]=mse
            k3+=1
            
        ANN_all_Error_test2[:,k2]=ANN_Error_test2[:,0]
    
    # Convert data to torch for neural network:
    X_train = torch.Tensor(X_train)
    y_train = torch.Tensor(y_train)
    y_train = y_train.reshape(len(y_train),1)
    X_test = torch.Tensor(X_test)
    y_test = torch.Tensor(y_test)
    y_test = y_test.reshape(len(y_test),1)
    
    # Determine best model:
    model_gen_errors_est=ANN_all_Error_test2.sum(1)
    best_index=model_gen_errors_est.argmin()
    best_n_hidden_units=n_hidden_units[best_index]
    ANN_best_models.append(best_n_hidden_units)
    print('Best model is {} hidden units with test error: {:.4f}'.format(best_n_hidden_units,model_gen_errors_est.min()))

    #Train model with best hidden units on entire training set:
    model = lambda: torch.nn.Sequential(
                           torch.nn.Linear(M,best_n_hidden_units), #M features to n_hidden_units
                           torch.nn.ReLU(), #torch.nn.Tanh(),   # 1st transfer function,
                           torch.nn.Linear(best_n_hidden_units, 1),# n_hidden_units to 1 output neuron
                           # torch.nn.Sigmoid()      # final tranfer function
                           # no final tranfer function, i.e. "linear output"
                           )
       
    loss_fn = torch.nn.MSELoss() # notice how this is now a mean-squared-error loss
       
    print('Training best model in fold {} of type:\n\n{}\n'.format(k+1,str(model())))

    net, final_loss, learning_curve = train_neural_net(model,
                                                        loss_fn,
                                                        X=X_train,
                                                        y=y_train,
                                                        n_replicates=n_replicates,
                                                        max_iter=max_iter, tolerance=1e-40)
   
    

    # Determine estimated class labels for test set
    y_test_est = net(X_test)

    # Determine errors and errors
    se = np.square(np.squeeze(y_test_est.data.numpy())-np.squeeze(y_test.data.numpy())) # squared error
    # mse = (sum(se).type(torch.float)/len(y_test)).data.numpy() #mean
    mse = se.sum()/len(y_test)
    ANN_Error_test[k,0]=mse # store error rate for current CV fold 
    
    print('Best model in outer fold {} is {} hidden units with test error: {:.4f}'.format(k+1,best_n_hidden_units,mse))
    
    # Display the learning curve for the best net in the current fold
    h, = summaries_axes[0].plot(learning_curve, color=color_list[k])
    h.set_label('CV fold {0}, h={1}'.format(k+1,best_n_hidden_units))
    summaries_axes[0].set_xlabel('Iterations',size=15)
    summaries_axes[0].set_xlim((0, max_iter))
    summaries_axes[0].set_ylabel('Loss',size=15)
    summaries_axes[0].set_title('Learning curves',size=20)

    ####### End of Neural Network outer fold ########################

    k+=1
############ Plot neural network ###################################
# Display the MSE across folds
summaries_axes[1].bar(np.arange(1, K+1), np.squeeze(np.asarray(ANN_Error_test)), color=color_list)
summaries_axes[1].set_xlabel('Fold',size=15)
summaries_axes[1].set_xticks(np.arange(1, K+1))
summaries_axes[1].set_ylabel('MSE',size=15)
summaries_axes[1].set_title('Test mean-squared-error',size=20)

show()


figure(figsize=(6,6))
y_est = np.squeeze(y_test_est.data.numpy()); y_true = np.squeeze(y_test.data.numpy())
axis_range = [np.min([y_est, y_true])-0.5,np.max([y_est, y_true])+0.5]
plot(axis_range,axis_range,'k--')
plot(y_true, y_est,'ob',alpha=.25)
legend(['Perfect estimation','Model estimations'])
title('Age: estimation vs true \n (last CV-fold with {} hidden units)'.format(best_n_hidden_units),size=20)
ylim(axis_range); xlim(axis_range)
xlabel('True value',size=15)
ylabel('Estimated value',size=15)
grid()

show()
############ End of plot of neural network ##########################

# Compute generalization error on baseline model:
B_gen_error=np.mean(B_Error_test)

# Compute gen error on linear regression:
LR_gen_error=np.mean(LR_Error_test_rlr)
    
# Compute gen error on ANN:
ANN_gen_error=np.mean(ANN_Error_test)


    
# Display results
print("Baseline outer level errors:")
print(B_Error_test)
print('\nEstimated generalization error, RMSE, on baseline model: {0}'.format(round(B_gen_error, 4)))

print("Reguralized linear regression outer level errors:")
print(LR_Error_test_rlr)
print('\nEstimated generalization error, RMSE, on linear regression model: {0}'.format(round(LR_gen_error, 4)))

print("Neural network outer level errors:")
print(ANN_Error_test)
print('\nEstimated generalization error, RMSE, on neural network model: {0}'.format(round(ANN_gen_error, 4)))

print('Finished script')