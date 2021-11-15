# -*- coding: utf-8 -*-

import numpy as np
from toolbox_02450 import mcnemar

# from two_L_CV import yhat, y_true
from two_L_CV import *
    

    
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