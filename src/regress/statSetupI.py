# -*- coding: utf-8 -*-
"""
Created on Fri Nov 12 09:42:32 2021

@author: marie
"""

from scipy import stats

def setupI(zA,zB,alpha):
    # Calculate statistical analysis from two models test errors.
    # H0: Difference is 0 (Same performance)
    # H1: Difference is different from 0 (different performance)
    # Computes confidence interval for difference and a p-value to
    # evaluate whether H0 hypothesis is correct.
    N=len(zA)
    diff=zA-zB
    diff_mean=diff.mean()
    diff_std=diff.std()
    
    # Inverse of cdf is used => ppf so calculate confidence interval:
    z_lower=stats.t.ppf(alpha/2,df=N-1,loc=diff_mean,scale=diff_std)
    z_upper=stats.t.ppf(1-alpha/2,df=N-1,loc=diff_mean,scale=diff_std)
    conf=[z_lower,z_upper]
    # Calculate p-value:
    p=2*stats.t.cdf(-abs(diff_mean),df=N-1,loc=0,scale=diff_std)
    
    return conf, p