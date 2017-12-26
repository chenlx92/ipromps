#!/usr/bin/python

# data structure is: list(task1,2...)-->list(demo1,2...)-->dict(emg,imu,tf...)

import numpy as np
import pandas as pd
from scipy.interpolate import griddata
from sklearn.externals import joblib
import glob
import os

len_norm = 101
datasets_path = '../datasets/handover_20171128/raw'

##########################

# load raw datasets
datasets_raw = []
task_dir_list = glob.glob(os.path.join(datasets_path, "*"))
for task_dir in task_dir_list:
    full_task_dir = task_dir + '/csv/'
    print('Loading data from ' + full_task_dir)
    demo_dir_list = glob.glob(os.path.join(full_task_dir, "201*"))   # the prefix of data
    demo_temp = []
    for demo_dir_idx in demo_dir_list:
        data_csv = pd.read_csv(demo_dir_idx + '/multiModal_states.csv')     # the file name of csv
        # the info of interest
        demo_temp.append({
                          'stamp': (data_csv.values[:, 2]-data_csv.values[0, 2])*1e-9,
                          'emg': data_csv.values[:, 7:15],
                          'left_hand': data_csv.values[:, 207:210],
                          'left_joints': data_csv.values[:, 99:106]
                          })
    datasets_raw.append(demo_temp)

# resample the datasets
datasets_norm = []
for idx, task_data in enumerate(datasets_raw):
    print('Resampling data from ' + task_dir_list[idx])
    demo_norm_temp = []
    for demo_data in task_data:
        time_stamp = demo_data['stamp']
        grid = np.linspace(0, time_stamp[-1], len_norm)
        # resample
        emg_norm = griddata(time_stamp, demo_data['emg'], grid, method='linear')
        left_hand_norm = griddata(time_stamp, demo_data['left_hand'], grid, method='linear')
        left_joints_norm = griddata(time_stamp, demo_data['left_joints'], grid, method='linear')
        # append them to list
        demo_norm_temp.append({
                               'alpha': time_stamp[-1],
                               'emg': emg_norm,
                               'left_hand': left_hand_norm,
                               'left_joints': left_joints_norm
                              })
    datasets_norm.append(demo_norm_temp)

# save all the datasets
print('Saving the datasets ...')
joblib.dump(datasets_raw, datasets_path + '/../pkl/datasets_raw.pkl')
joblib.dump(datasets_norm, datasets_path + '/../pkl/datasets_len' + str(len_norm) + '.pkl')

print('Loaded, resampled and normalize the datasets successfully!!!')
