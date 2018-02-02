#!/usr/bin/python
import numpy as np
import matplotlib.pyplot as plt
import os
from sklearn.externals import joblib

joint_num = 3
datasets_path = '../datasets/tape/'
datasets_raw = joblib.load(os.path.join(datasets_path, 'pkl/datasets_raw.pkl'))
datasets_filtered = joblib.load(os.path.join(datasets_path, 'pkl/datasets_filtered.pkl'))
num_demo = 15
publish_rate = 50.0
info = 'left_hand'

fig = plt.figure(0)
fig.suptitle('the raw data of ' + info)
for demo_idx in range(num_demo):
    for joint_idx in range(joint_num):
        ax = fig.add_subplot(joint_num, 1, 1 + joint_idx)
        data = datasets_raw[0][demo_idx][info][:, joint_idx]
        plt.plot(np.array(range(len(data)))/publish_rate, data)

# plot the filtered data
fig = plt.figure(2)
fig.suptitle('the filtered data of ' + info)
for demo_idx in range(num_demo):
    for joint_idx in range(joint_num):
        ax = fig.add_subplot(joint_num, 1, 1 + joint_idx)
        data = datasets_filtered[0][demo_idx][info][:, joint_idx]
        plt.plot(np.array(range(len(data)))/publish_rate, data)

plt.show()
