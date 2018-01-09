#!/usr/bin/python
import numpy as np
import pandas as pd
from scipy.interpolate import griddata
from sklearn.externals import joblib
import glob
import os
import ConfigParser
import scipy.signal as signal

# read conf file
file_path = os.path.dirname(__file__)
cp = ConfigParser.SafeConfigParser()
cp.read(os.path.join(file_path, '../config/model.conf'))

# load config param
datasets_path = os.path.join(file_path, cp.get('datasets', 'path'))
len_norm = cp.getint('datasets', 'len_norm')
filter_kernel = np.fromstring(cp.get('filter', 'filter_kernel'), dtype=int, sep=',')

# # the information and corresponding index in csv file
# info_n_idx_csv = {
#                 'emg': [7, 15],
#                 'left_hand': [207, 210],
#                 'left_joints': [99, 106]
#                 }

##
# data structure is: list(task1,2...)-->list(demo1,2...)-->dict(emg,imu,tf...)

# load raw datasets
datasets_raw = []
task_path_list = glob.glob(os.path.join(datasets_path, 'raw/*'))
task_name_list = [x.split('/')[-1] for x in task_path_list]
for task_path in task_path_list:
    task_csv_path = os.path.join(task_path, 'csv')
    print('Loading data from ' + task_csv_path)
    demo_path_list = glob.glob(os.path.join(task_csv_path, '201*'))   # the prefix of data
    demo_temp = []
    for demo_path in demo_path_list:
        data_csv = pd.read_csv(os.path.join(demo_path, 'multiModal_states.csv'))    # the file name of csv
        # the info of interest: convert the object to int / float
        demo_temp.append({
                          'stamp': (data_csv.values[:, 2].astype(int)-data_csv.values[0, 2])*1e-9,
                          'emg': data_csv.values[:, 7:15].astype(float),
                          'left_hand': data_csv.values[:, 207:210].astype(float),
                          'left_joints': data_csv.values[:, 99:106].astype(float)
                          })
    datasets_raw.append(demo_temp)

# resample the datasets
datasets_norm = []
for task_idx, task_data in enumerate(datasets_raw):
    print('Resampling data from ' + task_path_list[task_idx])
    demo_norm_temp = []
    for demo_data in task_data:
        time_stamp = demo_data['stamp']
        grid = np.linspace(0, time_stamp[-1], len_norm)
        # # pure resampling
        # emg_norm = griddata(time_stamp, demo_data['emg'], grid, method='linear')
        # left_hand_norm = griddata(time_stamp, demo_data['left_hand'], grid, method='linear')
        # left_joints_norm = griddata(time_stamp, demo_data['left_joints'], grid, method='linear')
        # filter the datasets
        emg_raw_filtered = signal.medfilt(demo_data['emg'], filter_kernel)
        left_hand_raw_filtered = signal.medfilt(demo_data['left_hand'], filter_kernel)
        left_joints_raw_filtered = signal.medfilt(demo_data['left_joints'], filter_kernel)
        # normalize the datasets
        emg_norm = griddata(time_stamp, emg_raw_filtered, grid, method='linear')
        left_hand_norm = griddata(time_stamp, left_hand_raw_filtered, grid, method='linear')
        left_joints_norm = griddata(time_stamp, left_joints_raw_filtered, grid, method='linear')
        # append them to list
        demo_norm_temp.append({
                                'alpha': time_stamp[-1],
                                'emg': emg_norm,
                                'left_hand': left_hand_norm,
                                'left_joints': left_joints_norm
                                })
    datasets_norm.append(demo_norm_temp)

# save all the datasets
print('Saving the datasets as pkl ...')
joblib.dump(task_name_list, os.path.join(datasets_path, 'pkl/task_name_list.pkl'))
joblib.dump(datasets_raw, os.path.join(datasets_path, 'pkl/datasets_raw.pkl'))
joblib.dump(datasets_norm, os.path.join(datasets_path, 'pkl/datasets_norm.pkl'))

# the finished reminder
print('Loaded, filtered, normalized and saved the datasets successfully!!!')
