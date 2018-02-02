#!/usr/bin/python
import numpy as np
from scipy.stats import multivariate_normal as mvn
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal as mvn
from mpl_toolkits.mplot3d import Axes3D


sample = np.random.random((2, 50))*100

cov = np.cov(sample)
mean = np.mean(sample, 1)
x, y = np.mgrid[(mean[0]-10):(mean[0]+10):1.0, (mean[1]-10):(mean[1]+10):1.0]


x = x.reshape(400, 1)[:, 0]
y = y.reshape(400, 1)[:, 0]

test = [[x[i], y[i]] for i in range(400)]

h = mvn.pdf(test, mean, cov)

fig = plt.figure()
ax = Axes3D(fig)
ax.scatter3D(x, y, h)
# ax.plot_surface(x, y, h)
# ax.plot_surface(x, y, h, color='b')

plt.show()
