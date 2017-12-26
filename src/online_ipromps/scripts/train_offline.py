#!/usr/bin/python

import numpy as np
import ipromps_lib
import scipy.linalg
from sklearn.externals import joblib
from sklearn import preprocessing
import scipy.signal as signal
import scipy.stats as stats
import matplotlib.pyplot as plt
import pylab as pl

# datasets dir
datasets_path = '../datasets/handover_20171128/pkl'
datasets_norm_dir = '../datasets/handover_20171128/pkl/datasets_len101.pkl'

# datasets param
num_joints = 8+3+7
num_obs_joints = 8+3
num_demos = 20

# optional ipromps model param
len_norm = 101
num_basis = 31
sigma_basis = 0.05
num_alpha_candidate = 10

# measurement noise
emg_noise = 0.3
hand_noise = 0.3
pose_noise = 0.3
measure_noise_cov_full = scipy.linalg.block_diag(np.eye(8) * emg_noise,
                                                 np.eye(3) * hand_noise,
                                                 np.eye(7) * pose_noise)
# the med filter kernel
filt_kernel = [9, 1]

###########################

# load norm datasets
datasets_norm = joblib.load(datasets_norm_dir)
datasets4train = [x[0:num_demos] for x in datasets_norm]

# preprocessing for the norm data
print('Preprocessing the data...')
y_full = np.array([]).reshape(0, num_joints)
for datasets4train_idx in datasets4train:
    for demo_idx in datasets4train_idx:
        h = np.hstack([demo_idx['emg'], demo_idx['left_hand'], demo_idx['left_joints']])
        y_full = np.vstack([y_full, h])
min_max_scaler = preprocessing.MinMaxScaler()
datasets_norm_full = min_max_scaler.fit_transform(y_full)

# a filter
print('Filtering the data...')
datasets_norm_full = signal.medfilt(datasets_norm_full, filt_kernel)

# construct a data structure to train the model
datasets4train_post = []
for task_idx in range(len(datasets4train)):
    datasets_temp = []
    for demo_idx in range(num_demos):
        temp = datasets_norm_full[(task_idx*num_demos+demo_idx)*len_norm:
                                  (task_idx * num_demos + demo_idx)*len_norm + len_norm, :]
        datasets_temp.append({'emg': temp[:, 0:8],
                              'left_hand': temp[:, 8:11],
                              'left_joints': temp[:, 11:18],
                              'alpha': datasets4train[task_idx][demo_idx]['alpha']})
    datasets4train_post.append(datasets_temp)

# create iProMPs sets
ipromps_set = [ipromps_lib.IProMP(num_joints=num_joints, num_basis=num_basis, sigma_basis=sigma_basis,
                                  num_samples=len_norm, num_obs_joints=num_obs_joints,
                                  sigmay=measure_noise_cov_full) for x in datasets4train]

# add demo and alpha var for each IProMPs
for idx, task_idx in enumerate(ipromps_set):
    print('Training the task %d IProMP...'%(idx))
    # for demo_idx in datasets4train[idx]:
    for demo_idx in datasets4train_post[idx]:
        demo_temp = np.hstack([demo_idx['emg'], demo_idx['left_hand'], demo_idx['left_joints']])
        task_idx.add_demonstration(demo_temp)   # spatial variance demo
        task_idx.add_alpha(demo_idx['alpha'])   # temporal variance demo

# save the trained models
print('Saving the trained models...')
joblib.dump([ipromps_set, datasets4train_post, min_max_scaler, filt_kernel], datasets_path + '/ipromps_set.pkl')

print('Trained the IProMPs successfully!!!')


