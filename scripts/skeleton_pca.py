#!/usr/bin/python
import numpy as np
import pandas as pd
from scipy.interpolate import griddata
from sklearn.externals import joblib
import glob
import os
import ConfigParser
from sklearn import preprocessing
from scipy.ndimage.filters import gaussian_filter1d
from sklearn.decomposition import PCA

# the current file path
file_path = os.path.dirname(__file__)

# read models cfg file
cp_models = ConfigParser.SafeConfigParser()
cp_models.read(os.path.join(file_path, '../cfg/models.cfg'))
# read models params
datasets_path = os.path.join(file_path, cp_models.get('datasets', 'path'))
len_norm = cp_models.getint('datasets', 'len_norm')
num_demo = cp_models.getint('datasets', 'num_demo')
num_joints = cp_models.getint('datasets', 'num_joints')
sigma = cp_models.get('filter', 'sigma')

# read datasets cfg file
cp_datasets = ConfigParser.SafeConfigParser()
cp_datasets.read(os.path.join(datasets_path, './info/cfg/datasets.cfg'))
# read datasets params
data_index_sec = cp_datasets.items('index_15')
data_index = [map(int, task[1].split(',')) for task in data_index_sec]

# tf of interest
tfoi = ['/head_1', '/neck_1', '/torso_1', '/left_shoulder_1', '/left_elbow_1',
        '/left_hand_1', '/right_shoulder_1', '/right_elbow_1', '/right_hand_1', '/left_hip_1',
        '/left_knee_1', '/left_foot_1', '/right_hip_1', '/right_knee_1', '/right_foot_1']

init_1st_col_csv = 152
init_2nd_col_csv = 155
col_interval_csv = 11


def main():

    # datasets-related info
    task_path_list = glob.glob(os.path.join(datasets_path, 'raw/*'))
    task_name_list = [task_path.split('/')[-1] for task_path in task_path_list]

    # load raw datasets
    datasets_raw = []
    for task_path in task_path_list:
        task_csv_path = os.path.join(task_path, 'csv')
        print('Loading data from ' + task_csv_path)
        demo_path_list = glob.glob(os.path.join(task_csv_path, '201*'))   # the prefix of data
        demo_temp = []
        for demo_path in demo_path_list:
            data_csv = pd.read_csv(os.path.join(demo_path, 'multiModal_states.csv'))    # the file name of csv
            # the info of interest: convert the object to int / float
            temp_dict = {}
            for idx, term in enumerate(tfoi):
                temp_dict[term] = data_csv.values[:,
                                                  (init_1st_col_csv+col_interval_csv*idx):
                                                  (init_2nd_col_csv+col_interval_csv*idx)].astype(float)
            demo_temp.append(temp_dict)
        datasets_raw.append(demo_temp)

    # filter the datasets: gaussian_filter1d
    datasets_filtered = []
    for task_idx, task_data in enumerate(datasets_raw):
        print('Filtering data of task: ' + task_name_list[task_idx])
        # filter the datasets
        demo_norm_temp = []
        for demo_data in task_data:
            temp_dict = {}
            for idx, term in enumerate(tfoi):
                temp_dict[term] = gaussian_filter1d(demo_data[term].T, sigma=sigma).T
            demo_norm_temp.append(temp_dict)
        datasets_filtered.append(demo_norm_temp)

    # resample the datasets
    datasets_norm = []
    for task_idx, task_data in enumerate(datasets_raw):
        print('Resampling data of task: ' + task_name_list[task_idx])
        demo_norm_temp = []
        for demo_data in task_data:
            time_len = demo_data[tfoi[0]].shape[0]
            grid = np.linspace(0, time_len-1, len_norm)
            temp_dict = {}
            for idx, term in enumerate(tfoi):
                filtered_data = gaussian_filter1d(demo_data[term].T, sigma=sigma).T
                temp_dict[term] = griddata(range(time_len), filtered_data, grid, method='linear')
            demo_norm_temp.append(temp_dict)
        datasets_norm.append(demo_norm_temp)

    # preprocessing for the norm data
    datasets4train = []
    for task_idx, demo_list in enumerate(data_index):
        data = [datasets_norm[task_idx][i] for i in demo_list]
        datasets4train.append(data)
    y_full = np.array([]).reshape(0, 3*len(tfoi))
    for task_idx, task_data in enumerate(datasets4train):
        print('Preprocessing data of task: ' + task_name_list[task_idx])
        for demo_data in task_data:
            h_temp = np.array([]).reshape(len_norm, 0)
            for term in tfoi:
                h_temp = np.hstack([h_temp, demo_data[term]])
            y_full = np.vstack([y_full, h_temp])

    pca = PCA()
    pca.fit(y_full)

    import matplotlib.pyplot as plt
    plt.figure()
    plt.plot(pca.explained_variance_, 'k', linewidth=2)
    plt.xlabel('n_components', fontsize=16)
    plt.ylabel('explained_variance_', fontsize=16)
    plt.show()

    # datasets_norm_full = min_max_scaler.fit_transform(y_full)
    # # construct a data structure to train the model
    # datasets_norm_preproc = []
    # for task_idx in range(len(datasets4train)):
    #     datasets_temp = []
    #     for demo_idx in range(num_demo):
    #         temp = datasets_norm_full[(task_idx * num_demo + demo_idx) * len_norm:
    #         (task_idx * num_demo + demo_idx) * len_norm + len_norm, :]
    #         datasets_temp.append({
    #                               # 'emg': temp[:, 0:8],
    #                               'left_hand': temp[:, 0:3],
    #                               'left_joints': temp[:, 3:6],
    #                               'alpha': datasets4train[task_idx][demo_idx]['alpha']})
    #     datasets_norm_preproc.append(datasets_temp)
    #
    # # save all the datasets
    # print('Saving the datasets as pkl ...')
    # # joblib.dump(task_name_list, os.path.join(datasets_path, 'pkl/task_name_list.pkl'))
    # # joblib.dump(datasets_raw, os.path.join(datasets_path, 'pkl/datasets_raw.pkl'))
    # # joblib.dump(datasets_filtered, os.path.join(datasets_path, 'pkl/datasets_filtered.pkl'))
    # # joblib.dump(datasets_norm, os.path.join(datasets_path, 'pkl/datasets_norm.pkl'))
    # # joblib.dump(datasets_norm_preproc, os.path.join(datasets_path, 'pkl/datasets_norm_preproc.pkl'))
    # # joblib.dump(min_max_scaler, os.path.join(datasets_path, 'pkl/min_max_scaler.pkl'))

    # # the finished reminder
    print('Loaded, filtered, normalized, preprocessed and saved the datasets successfully!!!')


if __name__ == '__main__':
    main()
