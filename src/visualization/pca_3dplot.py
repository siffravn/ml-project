from mpl_toolkits import mplot3d

from matplotlib.pyplot import figure, plot, title, xlabel, ylabel, show, legend, axes
from scipy.linalg import svd

# Indices of the principal components to be plotted:
i = 0
j = 1
k = 2

# Plot PCA of the data
f = figure()
ax = axes(projection='3d')
title('Data projected onto PC 1, 2 and 3')
#Z = array(Z)
for c in range(C):
    # select indices belonging to class c:
    class_mask = y==c
    plot(Z[class_mask,i], Z[class_mask,j], Z[class_mask,k], 'o', alpha=.5)
legend(classNames)
xlabel('PC{0}'.format(i+1))
ylabel('PC{0}'.format(j+1))
zlabel('PC{0}'.format(k+1))

# Output result to screen
show()
