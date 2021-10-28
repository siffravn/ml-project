# exercise 5.2.6
from matplotlib.pylab import figure, plot, xlabel, ylabel, legend, ylim, show
import sklearn.linear_model as lm
import numpy as np

import sys
import os
dirname = os.path.dirname(__file__)

foldername = os.path.join(dirname, '../../data')
sys.path.append(foldername)
import myData as d

foldername = os.path.join(dirname, '../../features')
sys.path.append(foldername)
import standardize as stan

md = d.myData()
N, M = md.X.shape

X = md.X[:,0:M-1].astype(float)
y = md.X[:,M-1].astype(float)

# # Fit logistic regression model

model = lm.LogisticRegression()
model = model.fit(X,y)

# Classify presence of coronary heart disease (chd) as absent/present (0/1) and assess probabilities
y_est = model.predict(X)
y_est_absent_prob = model.predict_proba(X)[:, 0] 


# # Define a new data object (new type of wine), as in exercise 5.1.7
# First touple of data is used
x = np.array([160,12.00, 5.73,23.11,1,49,25.30, 97.20,52]).reshape(1,-1)

# Evaluate the probability of x being a white wine (class=0) 
x_class = model.predict_proba(x)[0,0]

# Evaluate classifier's misclassification rate over entire training data
misclass_rate = np.sum(y_est != y) / float(len(y_est))

# Display classification results
print('\nProbability of given sample absent of chd: {0:.4f}'.format(x_class))
print('\nOverall misclassification rate: {0:.3f}'.format(misclass_rate))

f = figure();
class0_ids = np.nonzero(y==0)[0].tolist()
plot(class0_ids, y_est_absent_prob[class0_ids], '.y')
class1_ids = np.nonzero(y==1)[0].tolist()
plot(class1_ids, y_est_absent_prob[class1_ids], '.r')
xlabel('Data object (people sample)'); ylabel('Predicted prob. of class absent');
legend(['absent', 'present'])
ylim(-0.01,1.5)

show()