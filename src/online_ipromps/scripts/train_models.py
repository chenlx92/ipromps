#!/usr/bin/python
import numpy as np
import ipromps_lib
import scipy.linalg
from sklearn.externals import joblib
from sklearn import preprocessing
import os
import ConfigParser


# read conf file
file_path = os.path.dirname(__file__)
cp = ConfigParser.SafeConfigParser()
cp.read(os.path.join(file_path, '../config/model.conf'))

# datasets path
datasets_path = os.path.join(file_path, cp.get('datasets', 'path'))
datasets_pkl_path = os.path.join(datasets_path, 'pkl')
datasets_norm_path = os.path.join(datasets_pkl_path, 'datasets_norm.pkl')
task_name_path = os.path.join(datasets_pkl_path, 'task_name_list.pkl')

# datasets param
num_joints = cp.getint('datasets', 'num_joints')
num_obs_joints = cp.getint('datasets', 'num_obs_joints')
num_demos = cp.getint('datasets', 'num_demo')
len_norm = cp.getint('datasets', 'len_norm')

# optional ipromps model param
num_basis = cp.getint('basisFunc', 'num_basisFunc')
sigma_basis = cp.getfloat('basisFunc', 'sigma_basisFunc')
num_alpha_candidate = cp.getint('phase', 'num_phaseCandidate')

# the med filter kernel
filter_kernel = np.fromstring(cp.get('filter', 'filter_kernel'), dtype=int, sep=',')

# measurement noise
emg_noise = cp.getfloat('noise', 'emg')
hand_noise = cp.getfloat('noise', 'left_hand')
joints_noise = cp.getfloat('noise', 'left_joints')
noise_cov_full = scipy.linalg.block_diag(np.eye(8) * emg_noise,
                                         np.eye(3) * hand_noise,
                                         np.eye(7) * joints_noise)


def main():
    # load norm datasets
    datasets_norm = joblib.load(datasets_norm_path)
    datasets4train = [x[0:num_demos] for x in datasets_norm]
    task_name_list = joblib.load(task_name_path)

    # preprocessing for the norm data
    print('Preprocessing the data...')
    y_full = np.array([]).reshape(0, num_joints)
    for datasets4train_idx in datasets4train:
        for demo_idx in datasets4train_idx:
            h = np.hstack([demo_idx['emg'], demo_idx['left_hand'], demo_idx['left_joints']])
            y_full = np.vstack([y_full, h])
    min_max_scaler = preprocessing.MinMaxScaler()
    datasets_norm_full = min_max_scaler.fit_transform(y_full)

    # construct a data structure to train the model
    datasets_norm_preproc = []
    for task_idx in range(len(datasets4train)):
        datasets_temp = []
        for demo_idx in range(num_demos):
            temp = datasets_norm_full[(task_idx*num_demos+demo_idx)*len_norm:
                                      (task_idx * num_demos + demo_idx)*len_norm + len_norm, :]
            datasets_temp.append({'emg': temp[:, 0:8],
                                  'left_hand': temp[:, 8:11],
                                  'left_joints': temp[:, 11:18],
                                  'alpha': datasets4train[task_idx][demo_idx]['alpha']})
        datasets_norm_preproc.append(datasets_temp)

    # create iProMPs sets
    ipromps_set = [ipromps_lib.IProMP(num_joints=num_joints, num_obs_joints=num_obs_joints, num_basis=num_basis,
                                      sigma_basis=sigma_basis, num_samples=len_norm, sigmay=noise_cov_full,
                                      min_max_scaler=min_max_scaler, num_alpha_candidate=num_alpha_candidate)
                   for x in datasets4train]

    # add demo and alpha var for each IProMPs
    for idx, task_idx in enumerate(ipromps_set):
        print('Training the IProMP for task %s...' % task_name_list[idx])
        # for demo_idx in datasets4train[idx]:
        for demo_idx in datasets_norm_preproc[idx]:
            demo_temp = np.hstack([demo_idx['emg'], demo_idx['left_hand'], demo_idx['left_joints']])
            task_idx.add_demonstration(demo_temp)   # spatial variance demo
            task_idx.add_alpha(demo_idx['alpha'])   # temporal variance demo

    # save the trained models
    print('Saving the trained models...')
    joblib.dump(ipromps_set, os.path.join(datasets_pkl_path, 'ipromps_set.pkl'))
    joblib.dump(datasets_norm_preproc, os.path.join(datasets_pkl_path, 'datasets_norm_preproc.pkl'))

    print('Trained the IProMPs successfully!!!')


if __name__ == '__main__':
    main()
