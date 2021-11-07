# -*- coding: utf-8 -*-
"""
Created on Fri Sep 24 22:24:49 2021

@author: Siff
"""

import numpy as np

def standardize(X, N):
    X_st = X - np.ones((N, 1))*X.mean(0)
    X_stand = X_st*(1/X.astype(float).std(0,ddof=1))
    return X_stand

def deStandardize(X,w,N):
    X_mean=np.hstack((np.zeros(1),X.mean(0)))
    X_std=np.hstack((np.ones(1),X.astype(float).std(0)))
    w_deStand=w*X_std+X_mean
    return w_deStand