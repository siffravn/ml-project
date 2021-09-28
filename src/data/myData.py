# -*- coding: utf-8 -*-
"""
Created on Wed Sep 22 13:46:08 2021

@author: Siff
"""

import numpy as np
import pandas as pd
import os

class myData:
    
    def __init__(self):
        # Load the Iris csv data using the Pandas library
        dirname = os.path.dirname(__file__)
        filename = os.path.join(dirname, '../../data/raw/sahd-data.csv')
        df = pd.read_csv(filename)
        
        # Pandas returns a dataframe, (df) which could be used for handling the data.
        # We will however convert the dataframe to numpy arrays for this course as 
        # is also described in the table in the exercise
        raw_data = df.values  

        # Notice that raw_data both contains the information we want to store in an array
        # X (the sepal and petal dimensions) and the information that we wish to store 
        # in y (the class labels, that is the iris species).

        # We start by making the data matrix X by indexing into data.
        # We know that the attributes are stored in the eleven columns from inspecting 
        # the file.
        cols = range(0,11) 
        X_raw = raw_data[:, cols]
        #X_raw = np.delete(X_raw,0,1)
        
        
        # The column with famhist is chosen and defined in variable:
        famhist = X_raw[:,5] # -1 takes the last column
        # Then the unique number of classes is determined, should be two (Absent/Present):
        histNames = np.unique(famhist)
        # A dictionary for Absent/Present is made with coresponding numbers:
        histDict = dict(zip(histNames,range(len(histNames))))

        # With the dictionary, each datapoint for famhist is converted to a number in a vector:
        hist_binary = np.array([histDict[cl] for cl in famhist])

        # A new data matrix is made with the numerical famhist
        self.X=X_raw
        self.X[:,5]=hist_binary
        self.X = np.delete(self.X,0,1)
        

        # We can extract the attribute names that came from the header of the csv
        attributeNames = np.asarray(df.columns[cols])
        self.attributeNames = np.delete(attributeNames,0)
        
        self.N = len(famhist)
        self.M = len(self.attributeNames)
       
        
md = myData()
X = md.X
M = md.M
attr = md.attributeNames