# -*- coding: utf-8 -*-

# -*- coding: utf-8 -*-

import numpy as np
from sklearn.linear_model import LogisticRegression

def define_train_test(X_train, y_train, X_test, y_test):
    
    # Calculate mean and std for the training set
    mu = np.mean(X_train, 0)
    sigma = np.std(X_train, 0)
    
    # Define, train and test model
    model = define()
    model = train(model, X_train, y_train, mu, sigma)
    error_rate = test(model, X_test, y_test, mu, sigma)
    
    return error_rate


def define():
    
    model = LogisticRegression(penalty="none", fit_intercept=True)
    
    return model

def train(model, X_train, y_train, mu, sigma):
    
    # Standardize the training based on its mean and std
    mu = np.mean(X_train, 0)
    sigma = np.std(X_train, 0)
    
    X_train = (X_train - mu) / sigma
    
    model.fit(X_train, y_train)
    
    return model

def test(model, X_test, y_test, mu, sigma):
    
    # Standardize the test set based on training set mean and std
    X_test = (X_test - mu) / sigma
    
    y_test_est = model.predict(X_test).T 
    error_rate = np.sum(y_test_est != y_test) / len(y_test)
    
    return error_rate