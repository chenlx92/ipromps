#!/usr/bin/python

# Note!!! Changable codes for suitable use are listed as below.
# len_norm: the normal length
# dir_prefix, dir_str: the dir of datasets
# dataset_idx.append({...}): the info of interest
# ##
# the data structure to save the raw and norm datasets.
# data structure is: list(task1,2...)-->list(sample1,2...)-->dict(emg,imu,tf...)


import numpy as np
import pandas as pd
from scipy.interpolate import griddata
from sklearn.externals import joblib
import glob
import os
import matplotlib.pyplot as plt


len_norm = 101
dir_prefix = '../datasets/raw/handover_20171128/'
dir_str = ['pincer', 'wrench', 'screwdriver']


# load raw datasets
datasets = []
for dir_str_idx in dir_str:
    full_dir_idx = dir_prefix + dir_str_idx + '/csv/'
    print('loading data from ' + full_dir_idx)
    dir_list = glob.glob(os.path.join(full_dir_idx, "2017-11-28-*"))
    datasets_idx = []    # to save task samples
    for dir_idx in dir_list:
        data = pd.read_csv(dir_idx + '/multiModal_states.csv')
        datasets_idx.append({'stamp': (data.values[:,2]-data.values[0,2])*1e-9,
                            'emg': data.values[:,7:15],
                            'left_hand': data.values[:,207:210],
                            # 'left_elbow': data.values[:,196:199],
                            # 'left_shoulder': data.values[:,186:189],
                            'robot_joints': data.values[:,97:114]})
    datasets.append(datasets_idx)


# resampling the datasets
datasets_norm = []
for idx, datasets_idx in enumerate(datasets):
    print('resampling data from ' + dir_str[idx])
    datasets_norm_idx = []
    for datasets_idx_idx in datasets_idx:
        time_stamp = datasets_idx_idx['stamp']
        grid = np.linspace(0, time_stamp[-1], len_norm)
        # emg, left_hand, robot_joints
        emg_norm = griddata(time_stamp, datasets_idx_idx['emg'], grid, method='linear')
        left_hand_norm = griddata(time_stamp, datasets_idx_idx['left_hand'], grid, method='linear')
        robot_joints_norm = griddata(time_stamp, datasets_idx_idx['robot_joints'], grid, method='linear')
        # save them
        datasets_norm_idx.append({'emg': emg_norm,
                                  'left_hand': left_hand_norm,
                                  'robot_joints': robot_joints_norm,
                                  # 'stamp': grid,
                                  'alpha': time_stamp[-1]})
        # datasets_norm_idx.append({'data': np.hstack([emg_norm, left_hand_norm, robot_joints_norm]),
        #                           'alpha': time_stamp[-1]})
    datasets_norm.append(datasets_norm_idx)


# preprocessing for the norm data
y_full = np.array([]).reshape(0, num_joints)
for datasets4train_idx in datasets4train:
    for x in datasets4train_idx:
        h = np.hstack([x['emg'], x['left_hand'], x['robot_joints']])
        y_full = np.vstack([y_full, h])
min_max_scaler = preprocessing.MinMaxScaler()
dataset_norm_full = min_max_scaler.fit_transform(y_full)
# construct a data structure to train the model
datasets4train_post = []
for datasets4train_idx in datasets4train:
    task_demo_temp = []
    for x in datasets4train_idx:
        h = np.hstack([x['emg'], x['left_hand'], x['robot_joints']])
        h_post = min_max_scaler.fit_transform(h)


# the dataset as pkl
joblib.dump(datasets_norm, '../datasets/pkl/handover_20171128/datasets_norm.pkl')
joblib.dump(datasets, '../datasets/pkl/handover_20171128/datasets.pkl')

# plt.figure(0)
# for i in range(len(datasets[0])):
#    plt.plot(range(len(datasets[0][i]['left_hand'])), datasets[0][i]['left_hand'])
#
# plt.figure(1)
# for i in range(len(datasets_norm[0])):
#    plt.plot(range(len(datasets_norm[0][i]['left_hand'])), datasets_norm[0][i]['left_hand'])

# for i in range(len(datasets_norm[0])):
#    print datasets_norm[0][i]['alpha']
#
# plt.show()

print('Everyone is happy!!! You loaded and resampled the dataset successfully!')
    

