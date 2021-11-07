# -*- coding: utf-8 -*-
"""
Created on Fri Nov  5 16:27:59 2021

@author: marie
"""
# Dette script laver en two-level cross validation af en baseline model, som
# blot består i at tage gennemsnittet af træningssættet. Scriptet er inkorpereret
# i et samlet script "twolevelcross.py" som validerer tre modeller på samme
# split af data.

import sys 
import os
dirname = os.path.dirname(__file__)

import numpy as np
from sklearn import model_selection

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
opt_lambdas=np.zeros(K)
for train_index, test_index in CV.split(X_del,y):
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
    
    k+=1

# Compute generalization error on baseline model:
B_gen_error=B_gen_errors[:,0].sum()