#!/usr/bin/python
import numpy as np
import os
import ConfigParser
import matplotlib.pyplot as plt
from sklearn.externals import joblib

# the current file path
file_path = os.path.dirname(__file__)

# read models cfg file
cp_models = ConfigParser.SafeConfigParser()
cp_models.read(os.path.join(file_path, '../cfg/models.cfg'))
# read models params
datasets_path = os.path.join(file_path, cp_models.get('datasets', 'path'))
len_norm = cp_models.getint('datasets', 'len_norm')
num_basis = cp_models.getint('basisFunc', 'num_basisFunc')
sigma_basis = cp_models.getfloat('basisFunc', 'sigma_basisFunc')
datasets_norm_preproc = joblib.load(os.path.join(datasets_path, 'pkl/datasets_norm_preproc.pkl'))

num_basis = 12

ex = datasets_norm_preproc[0][0]['left_joints'][:,1]

x_grid = np.linspace(0, 1, len_norm)
c_grid = np.arange(0,num_basis)/(num_basis-1.0)

h = np.exp(-.5*(np.array(map(lambda x: x-c_grid, np.tile(x_grid, (num_basis, 1)).T)).T**2 / (sigma_basis**2)))

w = np.dot(np.linalg.inv(np.dot(h, h.T)), np.dot(h, ex.T)).T

y = np.dot(w, h)

fig = plt.figure(0)
plt.plot(x_grid, ex, 'o', markersize=5, label='target data', color='r')
plt.bar(c_grid, w, width=0.05)
plt.plot(x_grid, h.T, linewidth=4, alpha=0.5)
plt.plot(x_grid, y, linewidth=8, alpha=0.5, label='regression curve', color='g')

plt.ylim((0, 1.5))
plt.xlim((0, 1.1))
plt.xlabel('t')
plt.ylabel('y')

plt.legend()
plt.show()