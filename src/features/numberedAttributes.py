# -*- coding: utf-8 -*-
"""
Created on Tue Sep 28 16:22:49 2021

@author: marie
"""
import numpy as np
# Attributes are linked with number for labels in figures:
def addNumbers(attributeNames):
    noAttributes=np.array(['1 - ','2 - ','3 - ','4 - ','5 - ','6 - ','7 - ','8 - ','9 - ','10 - '])+attributeNames
    return noAttributes