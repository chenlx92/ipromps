#!/usr/bin/python
# Filename: train_offline.py

from __future__ import print_function

import numpy as np
import matplotlib.pyplot as plt
import ipromps_lib
import scipy.linalg
from sklearn.externals import joblib
from sklearn import preprocessing

#-- parameter of model --#
num_demos = 20
num_joints=19
num_basis=31
sigma_basis=0.05
num_samples=101
num_obs_joints=12
# measurement noise
imu_noise = 1.0
emg_noise = 2.0
pose_noise = 1.0
# phase estimation para
num_alpha_candidate = 10
nominal_duration = 1.0
nominal_interval = nominal_duration / (num_samples-1)
states_rate = 50.0

#################################
# load raw date sets
#################################
dataset_aluminum_hold = joblib.load('./datasets/pkl/dataset_aluminum_hold.pkl')
dataset_spanner_handover = joblib.load('./datasets/pkl/dataset_spanner_handover.pkl')
dataset_tape_hold = joblib.load('./datasets/pkl/dataset_tape_hold.pkl')
#################################
# load norm date sets
#################################
dataset_aluminum_hold_norm = joblib.load('./datasets/pkl/dataset_aluminum_hold_norm.pkl')
dataset_spanner_handover_norm = joblib.load('./datasets/pkl/dataset_spanner_handover_norm.pkl')
dataset_tape_hold_norm = joblib.load('./datasets/pkl/dataset_tape_hold_norm.pkl')

dataset_norm = [dataset_aluminum_hold_norm[0:num_demos],
                dataset_spanner_handover_norm[0:num_demos],
                dataset_tape_hold_norm[0:num_demos]]

#################################
# preprocessing
#################################
y_full=np.array([]).reshape(0,num_joints)
for data_norm in dataset_norm:
    for x in data_norm:
        h = np.hstack([x['imu'], x['emg'], x['pose']])
        y_full = np.vstack([y_full, h])
min_max_scaler = preprocessing.MinMaxScaler()
dataset_norm_full = min_max_scaler.fit_transform(y_full)

#################################
# Interaction ProMPs train
#################################
# the measurement noise cov matrix
meansurement_noise_cov_full = scipy.linalg.block_diag(np.eye(4) * imu_noise,
                                                      np.eye(8) * emg_noise,
                                                      np.eye(7) * pose_noise)
# create a 3 tasks iProMP
ipromp_aluminum_hold = ipromps_lib.IProMP(num_joints=num_joints, num_basis=num_basis, sigma_basis=sigma_basis,
                                          num_samples=num_samples, num_obs_joints=num_obs_joints,
                                          sigmay=meansurement_noise_cov_full)
ipromp_spanner_handover = ipromps_lib.IProMP(num_joints=num_joints, num_basis=num_basis, sigma_basis=sigma_basis,
                                             num_samples=num_samples, num_obs_joints=num_obs_joints,
                                             sigmay=meansurement_noise_cov_full)
ipromp_tape_hold = ipromps_lib.IProMP(num_joints=num_joints, num_basis=num_basis, sigma_basis=sigma_basis,
                                      num_samples=num_samples, num_obs_joints=num_obs_joints,
                                      sigmay=meansurement_noise_cov_full)

# add demostration
dataset_norm_aluminum_hold = dataset_norm_full[0:num_demos*num_basis,:]
dataset_norm_spanner_handover = dataset_norm_full[num_demos*num_basis:2*num_demos*num_basis,:]
dataset_norm_tape_hold = dataset_norm_full[2*num_demos*num_basis:3*num_demos*num_basis,:]
for idx in range(num_demos):
    # aluminum_hold
    ipromp_aluminum_hold.add_demonstration(dataset_norm_aluminum_hold[idx*num_basis:(idx+1)*num_basis, :])
    # spanner_handover
    ipromp_spanner_handover.add_demonstration(dataset_norm_spanner_handover[idx*num_basis:(idx+1)*num_basis, :])
    # tape_hold
    ipromp_tape_hold.add_demonstration(dataset_norm_tape_hold[idx*num_basis:(idx+1)*num_basis, :])

# model the phase distribution
for i in range(num_demos):
    # aluminum_hold
    alpha = (len(dataset_aluminum_hold[i]['imu']) - 1) / states_rate / nominal_duration
    ipromp_aluminum_hold.add_alpha(alpha)
    # spanner_handover
    alpha = (len(dataset_spanner_handover[i]['imu']) - 1) / states_rate / nominal_duration
    ipromp_spanner_handover.add_alpha(alpha)
    # tape_hold
    alpha = (len(dataset_tape_hold[i]['imu']) - 1) / states_rate / nominal_duration
    ipromp_tape_hold.add_alpha(alpha)

# save the trained models as pkl
joblib.dump(ipromp_aluminum_hold, "./trained_models/ipromp_aluminum_hold.pkl")
joblib.dump(ipromp_spanner_handover, "./trained_models/ipromp_spanner_handover.pkl")
joblib.dump(ipromp_tape_hold, "./trained_models/ipromp_tape_hold.pkl")
