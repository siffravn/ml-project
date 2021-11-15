# -*- coding: utf-8 -*-

import numpy as np
from sklearn.linear_model import LogisticRegression

def define_train_test(y_train, X_test):
    
    decision = train(y_train)
    y_est = predict(X_test, decision)
    
    return y_est


def train(y_train):
    decision = round(y_train.mean())
    
    return decision


def predict(X_test, decision):
    y_est = np.full(shape=(len(X_test)), fill_value=decision)
    
    return y_est
    

def error_rate(y_test_est, y_test):
    
    error_rate = np.sum(y_test_est != y_test) / len(y_test)
    
    return error_rate