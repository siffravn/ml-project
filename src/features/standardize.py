# -*- coding: utf-8 -*-
"""
Created on Fri Sep 24 22:24:49 2021

@author: Siff
"""

import numpy as np

def standardize(X, N):
    X_stand = X - np.ones((N, 1))*X.mean(0)
    X_stand = X_stand*(1/X_stand.astype(float).std(0,ddof=1))
    return X_stand