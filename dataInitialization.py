# exercise 1.5.1
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Load the Iris csv data using the Pandas library
filename = 'data/raw/sahd-data.csv'
df = pd.read_csv(filename)

# Pandas returns a dataframe, (df) which could be used for handling the data.
# We will however convert the dataframe to numpy arrays for this course as 
# is also described in the table in the exercise
raw_data = df.values  

# Notice that raw_data both contains the information we want to store in an array
# X (the sepal and petal dimensions) and the information that we wish to store 
# in y (the class labels, that is the iris species).

# We start by making the data matrix X by indexing into data.
# We know that the attributes are stored in the four columns from inspecting 
# the file.
cols = range(0,11) 
X = raw_data[:, cols]

# We can extract the attribute names that came from the header of the csv
attributeNames = np.asarray(df.columns[cols])

# The column with famhist is chosen and defined in variable:
classLabels = raw_data[:,5] # -1 takes the last column
# Then the unique number of classes is determined, should be two (Absent/Present):
classNames = np.unique(classLabels)
# A dictionary for Absent/Present is made with coresponding numbers:
classDict = dict(zip(classNames,range(len(classNames))))

# With the dictionary, each datapoint for famhist is converted to a number in a vector:
y = np.array([classDict[cl] for cl in classLabels])

# A new data matrix is made with the numerical famhist
Xnew=X
Xnew[:,5]=y

stat=np.vstack((attributeNames,Xnew.mean(0)))
stat=np.vstack((stat,Xnew.astype(float).std(0,ddof=1)))
stat=np.vstack((stat,np.median(Xnew.astype(float),axis=0)))
stat=np.vstack((stat,Xnew.max(0)))
stat=np.vstack((stat,Xnew.min(0))).T
stat[0,:]=np.array(['Attributes','Mean','Standard Deviation','Median','Max','Min'])

# The data is standardized:
#First indexing column is removed:
Xshort=Xnew[:,1:]
N,M=np.shape(Xshort)
Xstand=[]
for i in range(M):
    stand=Xshort[:,i]
    stand=(stand-stat[i+1,1])/stat[i+1,2]
    Xstand=Xstand+[stand]

# The data is plottet in a boxplot:
fig1, ax1 = plt.subplots()
ax1.set_title('Box plot of the 10 attributes')
ax1.boxplot(Xstand,labels=attributeNames[1:11],vert=False)

# # Next, we plot histograms of all attributes.
# plt.figure(figsize=(14,9))
# u = np.floor(np.sqrt(M)); v = np.ceil(float(M)/u)
# for i in range(M-1):
#     plt.subplot(u,v,i+1)
#     plt.hist(Xstand[:,i+1])
#     plt.xlabel(attributeNames[i+1])
#     plt.ylim(0, N) # Make the y-axes equal for improved readability
#     if i%v!=0: plt.yticks([])
#     if i==0: plt.title('African heart disease: Histogram')


