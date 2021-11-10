# exercise 8.2.6
# Code for training on a neural network. It is now combined in 
# twolevelcross, however it has been modified quite a bit and some
# major bugs have been fixed. This code in itself is more or less useless.
import sys
import os
dirname = os.path.dirname(__file__)


import matplotlib.pyplot as plt
import numpy as np
from scipy.io import loadmat
import torch
from sklearn import model_selection
from toolbox_02450 import train_neural_net, draw_neural_net
from scipy import stats

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
X = stan.standardize(X, N)
X = np.array(X,dtype=np.float64)

noAttr=no.addNumbers(attr)

y=X[:,8]
X=np.delete(X,8,axis=1)


N, M = X.shape

# Normalize data
#X = stats.zscore(X)
                



# K-fold crossvalidation
K = 2                  # only three folds to speed up this example
CV = model_selection.KFold(K, shuffle=True)

# Setup figure for display of learning curves and error rates in fold
summaries, summaries_axes = plt.subplots(1,2, figsize=(10,5))
# Make a list for storing assigned color of learning curve for up to K=10
color_list = ['tab:orange', 'tab:green', 'tab:purple', 'tab:brown', 'tab:pink',
              'tab:gray', 'tab:olive', 'tab:cyan', 'tab:red', 'tab:blue']


# Parameters for neural network classifier
n_hidden_units = range(5,8)     # range of hidden units
H=len(n_hidden_units) #number of different hidden units tested
n_replicates = 1       # number of networks trained in each k-fold
max_iter = 5000

# Initialize empty arrays:
ANN_Error_test = np.empty((K,1))
ANN_Error_test2 = np.empty((H,1))
ANN_all_Error_test2 = np.empty((H,K))
ANN_best_models=[]

k1=0
for (k, (train_index, test_index)) in enumerate(CV.split(X,y)): 
    print('\nCrossvalidation outer fold: {0}/{1}'.format(k+1,K))    
    
    # Extract training and test set for current CV fold, convert to tensors
    X_train = torch.Tensor(X[train_index,:])
    y_train = torch.Tensor(y[train_index])
    X_test = torch.Tensor(X[test_index,:])
    y_test = torch.Tensor(y[test_index])
    
    
    CV2 = model_selection.KFold(K, shuffle=True)
    k2=0
    for (k, (train_index, test_index)) in enumerate(CV2.split(X_train,y_train)):
        print('\nCrossvalidation inner fold: {0}/{1}'.format(k2+1,K)) 
        X_train2 = torch.Tensor(X[train_index,:])
        y_train2 = torch.Tensor(y[train_index])
        X_test2 = torch.Tensor(X[test_index,:])
        y_test2 = torch.Tensor(y[test_index])
        
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
            se = (y_test2_est.float()-y_test2.float())**2 # squared error
            # mse = (sum(se).type(torch.float)/len(y_test)).data.numpy() #mean
            mse = (torch.sum(se).type(torch.float)/len(y_test)).data.numpy()
            #errors2.append(mse)
            ANN_Error_test2[k3,0]=mse
            k3+=1
            
        ANN_all_Error_test2[:,k2]=ANN_Error_test2[:,0]
        
        
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
    se = (y_test_est.float()-y_test.float())**2 # squared error
    # mse = (sum(se).type(torch.float)/len(y_test)).data.numpy() #mean
    mse = (torch.sum(se).type(torch.float)/len(y_test)).data.numpy()
    ANN_Error_test[k1,0]=mse # store error rate for current CV fold 
    
    print('Best model in outer fold {} is {} hidden units with test error: {:.4f}'.format(k+1,best_n_hidden_units,mse))
    
    # Display the learning curve for the best net in the current fold
    h, = summaries_axes[0].plot(learning_curve, color=color_list[k])
    h.set_label('CV fold {0}, h={1}'.format(k+1,best_n_hidden_units))
    summaries_axes[0].set_xlabel('Iterations',size=15)
    summaries_axes[0].set_xlim((0, max_iter))
    summaries_axes[0].set_ylabel('Loss',size=15)
    summaries_axes[0].set_title('Learning curves',size=20)
    
    k1+=1

# Display the MSE across folds
summaries_axes[1].bar(np.arange(1, K+1), np.squeeze(np.asarray(ANN_Error_test)), color=color_list)
summaries_axes[1].set_xlabel('Fold',size=15)
summaries_axes[1].set_xticks(np.arange(1, K+1))
summaries_axes[1].set_ylabel('MSE',size=15)
summaries_axes[1].set_title('Test mean-squared-error',size=20)
    
print('Diagram of best neural net in last fold:')
weights = [net[i].weight.data.numpy().T for i in [0,2]]
biases = [net[i].bias.data.numpy() for i in [0,2]]
tf =  [str(net[i]) for i in [1,2]]
#draw_neural_net(weights, biases, tf, attribute_names=noAttr)

# Print the average classification error rate
print('\nEstimated generalization error, RMSE: {0}'.format(round(np.mean(ANN_Error_test), 4)))

# When dealing with regression outputs, a simple way of looking at the quality
# of predictions visually is by plotting the estimated value as a function of 
# the true/known value - these values should all be along a straight line "y=x", 
# and if the points are above the line, the model overestimates, whereas if the
# points are below the y=x line, then the model underestimates the value
plt.figure(figsize=(6,6))
y_est = np.squeeze(y_test_est.data.numpy()); y_true = y_test.data.numpy()
axis_range = [np.min([y_est, y_true])-0.5,np.max([y_est, y_true])+0.5]
plt.plot(axis_range,axis_range,'k--')
plt.plot(y_true, y_est,'ob',alpha=.25)
plt.legend(['Perfect estimation','Model estimations'])
plt.title('Age: estimation vs true \n (last CV-fold with {} hidden units)'.format(best_n_hidden_units),size=20)
plt.ylim(axis_range); plt.xlim(axis_range)
plt.xlabel('True value',size=15)
plt.ylabel('Estimated value',size=15)
plt.grid()

plt.show()

print('Finished script')