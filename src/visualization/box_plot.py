# -*- coding: utf-8 -*-
"""
Created on Fri Sep 24 22:34:17 2021

@author: Siff
"""

import matplotlib.pyplot as plt

def makeplot(X, attributeNames):
    fig1, ax1 = plt.subplots()
    ax1.set_title('Box plot of the 10 attributes')
    ax1.boxplot(X,labels=attributeNames,vert=False)
    
    plt.show()
    
    
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

makeplot(X_stand, attr)