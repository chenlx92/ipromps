#!/usr/bin/python
# use removing the low variance feature to do the feature selection
import numpy as np
import pandas as pd
from scipy.interpolate import griddata
from sklearn.externals import joblib
import glob
import os
import ConfigParser
from sklearn import preprocessing
from scipy.ndimage.filters import gaussian_filter1d
from sklearn.feature_selection import VarianceThreshold
import matplotlib.pyplot as plt

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


def main():
    # datasets-related info
    task_path_list = glob.glob(os.path.join(datasets_path, 'raw/*'))
    task_name_list = [task_path.split('/')[-1] for task_path in task_path_list]

    # load raw datasets
    datasets_raw = []
    for task_path in task_path_list:
        task_csv_path = os.path.join(task_path, 'csv')
        print('Loading data from ' + task_csv_path)
        demo_path_list = glob.glob(os.path.join(task_csv_path, '201*'))   # the prefix of data, guarantee the right file
        demo_temp = []
        for demo_path in demo_path_list:
            data_csv = pd.read_csv(os.path.join(demo_path, 'multiModal_states.csv'))    # the file name of csv
            # the info of interest: convert the object to int / float
            demo_temp.append({
                              'stamp': (data_csv.values[:, 2].astype(int)-data_csv.values[0, 2])*1e-9,  # the time stamp
                              # 'human': np.array( [
                              'human': np.hstack( [
                                  data_csv.values[:, (152 + 11 * 0):(155 + 11 * 0)].astype(float),   # human left hand
                                  data_csv.values[:, (152 + 11 * 1):(155 + 11 * 1)].astype(float),
                                  data_csv.values[:, (152 + 11 * 2):(155 + 11 * 2)].astype(float),
                                  data_csv.values[:, (152 + 11 * 3):(155 + 11 * 3)].astype(float),
                                  data_csv.values[:, (152 + 11 * 4):(155 + 11 * 4)].astype(float),
                                  data_csv.values[:, (152 + 11 * 5):(155 + 11 * 5)].astype(float),
                                  data_csv.values[:, (152 + 11 * 6):(155 + 11 * 6)].astype(float),
                                  data_csv.values[:, (152 + 11 * 7):(155 + 11 * 7)].astype(float),
                                  data_csv.values[:, (152 + 11 * 8):(155 + 11 * 8)].astype(float),
                                  # data_csv.values[:, (152 + 11 * 9):(155 + 11 * 9)].astype(float),
                                  # data_csv.values[:, (152 + 11 * 10):(155 + 11 * 10)].astype(float),
                                  # data_csv.values[:, (152 + 11 * 11):(155 + 11 * 11)].astype(float),
                                  # data_csv.values[:, (152 + 11 * 12):(155 + 11 * 12)].astype(float),
                                  # data_csv.values[:, (152 + 11 * 13):(155 + 11 * 13)].astype(float),
                                  # data_csv.values[:, (152 + 11 * 14):(155 + 11 * 14)].astype(float),
                                                  ]),
                              'left_joints': data_csv.values[:, 317:320].astype(float)  # robot ee actually
                              # 'left_joints': data_csv.values[:, 99:106].astype(float)
                              })
        datasets_raw.append(demo_temp)

    human_dim = datasets_raw[0][0]['human'].shape[1]
    datasets4train = []
    for task_idx, demo_list in enumerate(data_index):
        data = [datasets_raw[task_idx][i] for i in demo_list]
        datasets4train.append(data)
    y_full = np.array([]).reshape(0, human_dim)
    for task_idx, task_data in enumerate(datasets4train):
        for demo_data in task_data:
            y_full = np.vstack([y_full, demo_data['human']])

    ## single channel
    sel = VarianceThreshold(threshold=0.011)
    print np.diag(np.cov(y_full.T), 0)

    print sel.fit_transform(y_full)

    fig = plt.figure(0)
    ax = fig.add_subplot(1,1,1)
    label1 = ('head_x', 'head_y', 'head_z', 'neck_x', 'neck_y', 'neck_z',
              'torso_x', 'torso_y', 'torso_z', 'left_shoulder_x', 'left_shoulder_y', 'left_shoulder_z',
              'left_elbow_x', 'left_elbow_y', 'left_elbow_z',
              'left_hand_x', 'left_hand_y', 'left_hand_z', 'right_shoulder_x', 'right_shoulder_y', 'right_shoulder_z',
              'right_elbow_x', 'right_elbow_y', 'right_elbow_z', 'right_hand_x', 'right_hand_y', 'right_hand_z')
    plt.errorbar(np.linspace(1,human_dim,human_dim), np.zeros([1, human_dim])[0], np.diag(np.cov(y_full.T), 0), fmt='o', markersize=8,capsize=8, elinewidth=5)
    plt.xticks(np.linspace(1, human_dim, human_dim), label1, rotation=20)
    ax.set_xlabel('feature channel')
    ax.set_ylabel('position variance/(m*m)')

    ## distance
    mean_full = np.mean(y_full,0)
    euclidean_dis = np.array([]).reshape([y_full.shape[0],0])
    for i in range(np.mean(y_full,0).shape[0]/3):
        x = y_full[:, (i*3):(i+1)*3]
        y = mean_full[(i*3):(i+1)*3]
        dis = np.sqrt(np.sum(np.square(x - y),1)).reshape([y_full.shape[0],1])
        euclidean_dis = np.hstack([euclidean_dis, dis])
    print np.diag(np.cov(euclidean_dis.T), 0)
    sel2 = VarianceThreshold(threshold=0.005)
    print sel2.fit_transform(euclidean_dis)
    fig2 = plt.figure(3)
    ax2 = fig2.add_subplot(1,1,1)
    plt.errorbar(np.linspace(1,human_dim/3,human_dim/3), np.zeros([1, human_dim/3])[0], np.diag(np.cov(euclidean_dis.T), 0), fmt='o', markersize=8,capsize=8, elinewidth=5)
    plt.xlim([0,human_dim/3+1])
    label2 = ('head', 'neck', 'torso', 'left_shoulder', 'left_elbow', 'left_hand', 'right_shoulder', 'right_elbow',
             'right_hand')
    plt.xticks(np.linspace(1,human_dim/3,human_dim/3), label2, rotation=15)
    ax2.set_xlabel('feature channel')
    ax2.set_ylabel('position variance/(m*m)')

    plt.show()

if __name__ == '__main__':
    main()
