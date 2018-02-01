#!/usr/bin/python
# data structure is: list(task1,2...)-->list(demo1,2...)-->dict(emg,imu,tf...)
import numpy as np
import pandas as pd
from scipy.interpolate import griddata
from sklearn.externals import joblib
import glob
import os
import ConfigParser
from sklearn import preprocessing
from scipy.ndimage.filters import gaussian_filter1d

# the current file path
file_path = os.path.dirname(__file__)

# read models cfg file
cp_models = ConfigParser.SafeConfigParser()
cp_models.read(os.path.join(file_path, '../cfg/models.cfg'))
# read models params
datasets_path = os.path.join(file_path, cp_models.get('datasets', 'path'))
len_norm = cp_models.getint('datasets', 'len_norm')
num_demos = cp_models.getint('datasets', 'num_demo')
num_joints = cp_models.getint('datasets', 'num_joints')
sigma = cp_models.get('filter', 'sigma')

# read datasets cfg file
cp_datasets = ConfigParser.SafeConfigParser()
cp_datasets.read(os.path.join(datasets_path, 'info/cfg/datasets.cfg'))
# read datasets params
# data_index0 = list(np.fromstring(cp_datasets.get('index_12', 'data_index0'), dtype=int, sep=','))
# data_index1 = list(np.fromstring(cp_datasets.get('index_12', 'data_index1'), dtype=int, sep=','))
# data_index2 = list(np.fromstring(cp_datasets.get('index_12', 'data_index2'), dtype=int, sep=','))
# data_index = [data_index0, data_index1, data_index2]
data_index0 = list(np.fromstring(cp_datasets.get('index_12', 'data_index0'), dtype=int, sep=','))
data_index = [data_index0]

# # the information and corresponding index in csv file
# info_n_idx_csv = {
#                 'stamp': [2]
#                 'emg': [7, 15],
#                 'left_hand': [207, 210],
#                 'left_joints': [99, 106]
#                 }


def main():
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
                              # 'emg': data_csv.values[:, 7:15].astype(float),
                              'left_hand': data_csv.values[:, 207:210].astype(float),
                              'left_joints': data_csv.values[:, 317:320].astype(float)  # robot ee actually
                              # 'left_joints': data_csv.values[:, 99:106].astype(float)
                              })
        datasets_raw.append(demo_temp)

    # filter the datasets: gaussian_filter1d
    datasets_filtered = []
    for task_idx, task_data in enumerate(datasets_raw):
        print('Filtering data of task: ' + task_name_list[task_idx])
        demo_norm_temp = []
        for demo_data in task_data:
            time_stamp = demo_data['stamp']
            # filter the datasets
            left_hand_filtered = gaussian_filter1d(demo_data['left_hand'].T, sigma=sigma).T
            left_joints_filtered = gaussian_filter1d(demo_data['left_joints'].T, sigma=sigma).T
            # append them to list
            demo_norm_temp.append({
                'alpha': time_stamp[-1],
                # 'emg': emg_filtered,
                'left_hand': left_hand_filtered,
                'left_joints': left_joints_filtered
            })
        datasets_filtered.append(demo_norm_temp)

    # resample the datasets
    datasets_norm = []
    for task_idx, task_data in enumerate(datasets_raw):
        print('Resampling data of task: ' + task_name_list[task_idx])
        demo_norm_temp = []
        for demo_data in task_data:
            time_stamp = demo_data['stamp']
            grid = np.linspace(0, time_stamp[-1], len_norm)
            # filter the datasets
            left_hand_filtered = gaussian_filter1d(demo_data['left_hand'].T, sigma=sigma).T
            left_joints_filtered = gaussian_filter1d(demo_data['left_joints'].T, sigma=sigma).T
            # normalize the datasets
            # emg_norm = griddata(time_stamp, emg_filtered, grid, method='linear')
            left_hand_norm = griddata(time_stamp, left_hand_filtered, grid, method='linear')
            left_joints_norm = griddata(time_stamp, left_joints_filtered, grid, method='linear')
            # append them to list
            demo_norm_temp.append({
                                    'alpha': time_stamp[-1],
                                    # 'emg': emg_norm,
                                    'left_hand': left_hand_norm,
                                    'left_joints': left_joints_norm
                                    })
        datasets_norm.append(demo_norm_temp)

    # preprocessing for the norm data
    # datasets4train = [x[0:num_demos] for x in datasets_norm]
    datasets4train = []
    for task_idx, demo_list in enumerate(data_index):
        data = [datasets_norm[task_idx][i] for i in demo_list]
        datasets4train.append(data)
    y_full = np.array([]).reshape(0, num_joints)
    for task_idx, task_data in enumerate(datasets4train):
        print('Preprocessing data of task: ' + task_name_list[task_idx])
        for demo_data in task_data:
            # h = np.hstack([demo_data['emg'], demo_data['left_hand'], demo_data['left_joints']])
            h = np.hstack([demo_data['left_hand'], demo_data['left_joints']])
            y_full = np.vstack([y_full, h])
    min_max_scaler = preprocessing.MinMaxScaler()
    datasets_norm_full = min_max_scaler.fit_transform(y_full)
    # construct a data structure to train the model
    datasets_norm_preproc = []
    for task_idx in range(len(datasets4train)):
        datasets_temp = []
        for demo_idx in range(num_demos):
            temp = datasets_norm_full[(task_idx * num_demos + demo_idx) * len_norm:
            (task_idx * num_demos + demo_idx) * len_norm + len_norm, :]
            datasets_temp.append({
                                  # 'emg': temp[:, 0:8],
                                  'left_hand': temp[:, 0:3],
                                  'left_joints': temp[:, 3:10],
                                  'alpha': datasets4train[task_idx][demo_idx]['alpha']})
        datasets_norm_preproc.append(datasets_temp)

    # save all the datasets
    print('Saving the datasets as pkl ...')
    joblib.dump(task_name_list, os.path.join(datasets_path, 'pkl/task_name_list.pkl'))
    joblib.dump(datasets_raw, os.path.join(datasets_path, 'pkl/datasets_raw.pkl'))
    joblib.dump(datasets_filtered, os.path.join(datasets_path, 'pkl/datasets_filtered.pkl'))
    joblib.dump(datasets_norm, os.path.join(datasets_path, 'pkl/datasets_norm.pkl'))
    joblib.dump(datasets_norm_preproc, os.path.join(datasets_path, 'pkl/datasets_norm_preproc.pkl'))
    joblib.dump(min_max_scaler, os.path.join(datasets_path, 'pkl/min_max_scaler.pkl'))

    # the finished reminder
    print('Loaded, filtered, normalized, preprocessed and saved the datasets successfully!!!')


if __name__ == '__main__':
    main()
