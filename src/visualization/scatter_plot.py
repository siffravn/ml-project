# -*- coding: utf-8 -*-
"""
Created on Fri Sep 24 22:07:59 2021

@author: Siff
"""

from matplotlib.pyplot import (figure, subplot, plot, xlabel, ylabel, 
                                xticks, yticks,legend,show)
# import matplotlib.pyplot as plt
import numpy as np


# Scatter plot script is based om ex4_2_5

def makeplot(X,M,attributeNames):
    figure(figsize=(12,10))
    for m1 in range(M):
        for m2 in range(M):
            subplot(M, M, m1*M + m2 + 1)
            plot(np.array(X[:,m2]), np.array(X[:,m1]), '.')
            if m1==M-1:
                xlabel(attributeNames[m2])
            else:
                xticks([])
            if m2==0:
                ylabel(attributeNames[m1])
            else:
                yticks([])
    show()
    
import sys
import os
dirname = os.path.dirname(__file__)

foldername = os.path.join(dirname, '../data')
sys.path.append(foldername)
import myData as d

foldername = os.path.join(dirname, '../features')
sys.path.append(foldername)
import standardize as stan

md = d.myData()
X = md.X
M = md.M
attr = md.attributeNames

X_stand = stan.standardize(X, md.N)

makeplot(X_stand, M, attr)
