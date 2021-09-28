# -*- coding: utf-8 -*-
"""
Created on Sat Sep 25 21:59:32 2021

@author: Siff
"""

import data.myData as d
import visualization.scatter_plot as scatter
import visualization.box_plot as box
import features.standardize as stan

md = d.myData()
X = md.X
M = md.M
attr = md.attributeNames

X_stand = stan.standardize(X, md.N)

# scatter.makeplot(X, M , attr)
scatter.makeplot(X_stand, M , attr)

box.makeplot(X_stand, attr)