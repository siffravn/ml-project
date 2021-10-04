# exercise 1.5.1
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats


def makeHistograms(Xstand,noAttributes):
    # Next, we plot histograms of all attributes.
    plt.figure(figsize=(14,10))
    u = np.floor(np.sqrt(M)); v = np.ceil(float(M)/u)
    for i in range(M):
        #Selecting location in figure:
        plt.subplot(u,v,i+1)
        #plotting histogram with attributename attached:
        plt.hist(Xstand[:,i],bins=15)
        plt.xlabel(noAttributes[i])
        # Make the y-axes equal for improved readability
        plt.ylim(0, N*2/3) 
        
        # Plotting the expected normal distribution on top
        # Fit a normal distribution to
        # the data:
        # mean and standard deviation
        mu, std = np.mean(Xstand[:,i]),np.std(Xstand[:,i])
        
        x = np.linspace(Xstand[:,i].min(), Xstand[:,i].max(), 1000)
        pdf = stats.norm.pdf(x,mu,std)*300
        plt.plot(x,pdf,'.',color='red') 
        
        
        
        if i%v!=0: plt.yticks([])
        #if i==0: plt.title('African heart disease: Histogram of standardized data')
            
    plt.suptitle('African heart disease: Histogram of  standardized data', y=1.05, size=16)
    plt.tight_layout();


