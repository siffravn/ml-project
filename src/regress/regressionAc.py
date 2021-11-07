import sys
import os
dirname = os.path.dirname(__file__)


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

X=np.array(X,dtype=np.float64)

noAttr=no.addNumbers(attr)

#Definerer age kolonnen som egen array, vi vil forudse og 
#fjerner age kolonnen fra dataen:
y=X[:,8]
X_del=np.delete(X,8,axis=1)

N, M = X_del.shape

# Add offset attribute
X_del = np.concatenate((np.ones((X_del.shape[0],1)),X_del),1)
attributeNames = np.hstack((np.array([u'Offset']),np.delete(noAttr,8)))
M = M+1

optimalLambda=8.682703795368411
## Crossvalidation
# Create crossvalidation partition for evaluation
K = 10
CV = model_selection.KFold(K, shuffle=True)

for train_index, test_index in CV.split(X_del,y):
    
    # extract training and test set for current CV fold
    X_train = X_del[train_index]
    y_train = y[train_index]
    X_test = X_del[test_index]
    y_test = y[test_index]
    internal_cross_validation = 10    
    
    opt_val_err, opt_lambda, mean_w_vs_lambda, train_err_vs_lambda, test_err_vs_lambda = rlr_validate(X_train, y_train, optimalLambda, internal_cross_validation)