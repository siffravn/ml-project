# Somewhat based on exercise 8.1.1
import sys 
import os
dirname = os.path.dirname(__file__)

from matplotlib.pylab import (figure, semilogx, loglog, xlabel, ylabel, legend, 
                           title, subplot, show, grid, rcParams)
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

# Data is imported:
md = d.myData()
X_ori = md.X
attr = md.attributeNames

# Data is standardized:
X=np.array(X_ori,dtype=np.float64)
X = stan.standardize(X, md.N)
X=np.array(X,dtype=np.float64)

#The attribute names are added numbers:
noAttr=no.addNumbers(attr)

#Definerer age kolonnen som egen array, vi vil forudse og 
#fjerner age kolonnen fra dataen:
y=X[:,8]
X_del=np.delete(X,8,axis=1)
N, M = X_del.shape
X_ori=np.delete(X_ori,8,axis=1)

# Add offset attribute
X_del = np.concatenate((np.ones((X_del.shape[0],1)),X_del),1)
attributeNames = np.hstack((np.array([u'Offset']),np.delete(noAttr,8)))
M = M+1

# Values of lambda
lambdas=np.zeros(10)
for i in range(10):
    power=-1.2+0.3*(i+1)
    lambdas[i]=np.power(10.,power)
#lambdas = np.power(10.,range(-2,6))

# Since only cross-validation (not two level) the whole data set is used and
# the rlr_validate function uses 10-K-Fold cross validation. 
X_train = X_del
y_train = y
internal_cross_validation = 10 

# Crossvalidation on the 10 lambda models is executed:
gen_error, opt_lambda, mean_w_vs_lambda, train_err_vs_lambda, test_err_vs_lambda = rlr_validate(X_train, y_train, lambdas, internal_cross_validation)

# Then the weights are calculated with the best lambda:
Xty = X_train.T @ y_train
XtX = X_train.T @ X_train
lambdaI = opt_lambda * np.eye(M)
lambdaI[0,0] = 0 # Do no regularize the bias term
w_rlr= np.linalg.solve(XtX+lambdaI,Xty).squeeze()
w_rlr_deStand=stan.deStandardize(X_ori,w_rlr,N)


# Figures are plotted:
figure(1, figsize=(14,6))
#Formatting math style text to times
rc = {"font.family" : "serif", 
      "mathtext.fontset" : "stix"}
rcParams.update(rc)
rcParams["font.serif"] = ["Times New Roman"] + rcParams["font.serif"]

subplot(1,2,1)
semilogx(lambdas,mean_w_vs_lambda.T[:,1:],'.-') # Don't plot the bias term
xlabel('Regularization factor', size=15)
ylabel('Mean Coefficient Values',size=15)
grid()
# You can choose to display the legend, but it's omitted for a cleaner 
# plot, since there are many attributes
legend(attributeNames[1:], loc='upper left')

subplot(1,2,2)
title('Optimal lambda: 1e{:.2f}'.format(np.log10(opt_lambda)),size=20)
loglog(lambdas,train_err_vs_lambda.T,'b.-',lambdas,test_err_vs_lambda.T,'r.-')
xlabel('Regularization factor',size=15)
ylabel('Squared error (crossvalidation)',size=15)
legend(['Train error','Test error'])
grid()
    
show()

    
# Display results
print('Weights with optimal lambda:')
for m in range(M):
    print('{:>15} {:>15}'.format(attributeNames[m], np.round(w_rlr[m],2)))
print('Weights with optimal lambda de-standardized:')
for m in range(M):
    print('{:>15} {:>15}'.format(attributeNames[m], np.round(w_rlr_deStand[m],2)))
print('Generalization error:')
print(gen_error)

print('Finished regularized linear regression cross validation')