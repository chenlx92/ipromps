#!/usr/bin/python
# coding:utf-8
from sklearn.externals import joblib
import numpy as np
import matplotlib.pyplot as plt
import pylab as pl
import scipy.stats as stats
import os
import ConfigParser
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.font_manager import FontProperties
from sklearn import mixture

# read conf file
file_path = os.path.dirname(__file__)
cp_models = ConfigParser.SafeConfigParser()
cp_models.read(os.path.join(file_path, '../cfg/models.cfg'))
# the datasets path
datasets_path = os.path.join(file_path, cp_models.get('datasets', 'path'))

# read datasets cfg file
cp_datasets = ConfigParser.SafeConfigParser()
cp_datasets.read(os.path.join(datasets_path, 'info/cfg/datasets.cfg'))
# read datasets params
data_index_sec = cp_datasets.items('index_17')
data_index = [map(int, task[1].split(',')) for task in data_index_sec]

# load datasets
datasets_raw = joblib.load(os.path.join(datasets_path, 'pkl/datasets_raw.pkl'))

temp = []
for task_idx, demo_idx in enumerate(data_index):
    b = [datasets_raw[task_idx][x]['left_hand'][0,0:3] for x in demo_idx]
    b= np.array(b)
    temp.append(b)

for task_idx in range(len(data_index)):
    fig = plt.figure(0)
    ax = fig.gca(projection='3d')
    ax.plot(temp[task_idx][:, 0], temp[task_idx][:, 1], temp[task_idx][:, 2], 'o',
            # label='training sets about human '+str(demo_idx), alpha=0.3)
            alpha=1.0)
    ax.set_xlabel('X/m')
    ax.set_ylabel('Y/m')
    ax.set_zlabel('Z/m')

    ax.legend()

obs = temp[0][:, 0:2].T
clf = mixture.GMM(n_components=2)
print obs[:10]
clf.fit(obs)
#预测
print clf.predict([[0], [2], [9], [10]])
plt.show()

