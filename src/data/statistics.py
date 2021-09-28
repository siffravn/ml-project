# -*- coding: utf-8 -*-
"""
Created on Fri Sep 24 22:36:53 2021

@author: Siff
"""

import numpy as np
import myData as d

md = d.myData()

stat=np.vstack((md.attributeNames,md.X.mean(0)))
stat=np.vstack((stat,md.X.astype(float).std(0,ddof=1)))
stat=np.vstack((stat,np.median(md.X.astype(float),axis=0)))
stat=np.vstack((stat,md.X.max(0)))
stat=np.vstack((stat,md.X.min(0))).T
stat[0,:]=np.array(['Attributes','Mean','Standard Deviation','Median','Max','Min'])