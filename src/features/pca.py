# -*- coding: utf-8 -*-
"""
Created on Fri Sep 24 22:35:22 2021

@author: Siff
"""

import sys
import os
dirname = os.path.dirname(__file__)

foldername = os.path.join(dirname, '../data')
sys.path.append(foldername)
import myData as d

import standardize as stan

import numberedAttributes as nu

from scipy.linalg import svd
import matplotlib.pyplot as plt
import numpy as np


md = d.myData()
X = md.X.astype(float)
N,M = X.shape
attributeNames = md.attributeNames

X_stan = stan.standardize(X, N)

# PCA by computing SVD of X_stand (svd method does not take array of object)
U,S,Vh = svd(X_stan,full_matrices=False)

# scipy.linalg.svd returns "Vh", which is the Hermitian (transpose)
# of the vector V. So, for us to obtain the correct V, we transpose:
V = Vh.T

# Compute variance explained by principal components
rho = (S*S) / (S*S).sum() 

# Project the centered data onto principal component space
Z = X_stan @ V

# Plot variance explained
foldername = os.path.join(dirname, '../visualization')
sys.path.append(foldername)
import var_explained as ve

ve.makeplot(rho, 0.9)

# Plot PCA of the data
import scatter_plot as scatter

pc = list()
for e in range(0,7):
    pc.append('PC{0}'.format(e+1))

scatter.makeplot(Z, 7, pc)

noAttr=nu.addNumbers(attributeNames)

# We saw that the first 8 components in V explaiend more than 90
# percent of the variance. Let's look at their coefficients:
import pca_coefficients as pca_c

pca_c.makeplot(V, M, 8,noAttr)

import pca_3dplot as pca_3d
pca_3d.plot3d(Z,noAttr)
    
