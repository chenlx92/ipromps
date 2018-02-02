#!/usr/bin/python
import numpy as np
import pandas as pd
import glob
import os
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

datasets_path = '../datasets/raw/rect/2018-01-18-10-12-41/multiModal_states.csv'
data_csv = pd.read_csv(datasets_path)
data = data_csv.values[:, 207:210].astype(float)

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

ax.scatter(data[:,0], data[:,1], data[:,2], c='r', marker='o')
plt.xlim([-0.5, 1.5])
plt.ylim([-0.5, 1.5])
ax.set_zlim(-0.5, 1.5)
ax.set_xlabel('X Label')
ax.set_ylabel('Y Label')
ax.set_zlabel('Z Label')

plt.show()

