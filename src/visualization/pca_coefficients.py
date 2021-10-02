# -*- coding: utf-8 -*-
"""
Created on Thu Sep 30 10:14:53 2021

@author: Siff
"""

import matplotlib.pyplot as plt
import numpy as np

def makeplot(V, M, n, attributeNames):
    plt.figure(figsize=(8,4))
    pcs = list(range(0,n))
    legendStrs = ['PC'+str(e+1) for e in pcs]
    # c = ['r','g','b']
    bw = .1
    r = np.arange(1,M+1)
    for i in pcs:    
        plt.bar(r+i*bw, V[:,i], width=bw)
        plt.xticks(r+bw, attributeNames, rotation='vertical')
        plt.xlabel('Attributes')
        plt.ylabel('Component coefficients')
        plt.legend(legendStrs, bbox_to_anchor=(1.05, 1.0), loc='best')
        plt.grid()
        plt.title('SAHD: PCA Component Coefficients')
        plt.tight_layout()
    
    plt.show()