#!/usr/bin/python
import numpy as np
import ipromps_lib
import scipy.linalg
from sklearn.externals import joblib
import os
import ConfigParser


# read conf file
file_path = os.path.dirname(__file__)
cp = ConfigParser.SafeConfigParser()
cp.read(os.path.join(file_path, '../config/model.conf'))

# datasets path
datasets_path = os.path.join(file_path, cp.get('datasets', 'path'))
datasets_pkl_path = os.path.join(datasets_path, 'pkl')

# the pkl data
task_name_path = os.path.join(datasets_pkl_path, 'task_name_list.pkl')
datasets_norm_preproc_path = os.path.join(datasets_pkl_path, 'datasets_norm_preproc.pkl')
min_max_scaler_path = os.path.join(datasets_pkl_path, 'min_max_scaler.pkl')
noise_cov_path = os.path.join(datasets_pkl_path, 'noise_cov.pkl')

# datasets param
num_joints = cp.getint('datasets', 'num_joints')
num_obs_joints = cp.getint('datasets', 'num_obs_joints')
len_norm = cp.getint('datasets', 'len_norm')

# optional ipromps model param
num_basis = cp.getint('basisFunc', 'num_basisFunc')
sigma_basis = cp.getfloat('basisFunc', 'sigma_basisFunc')
num_alpha_candidate = cp.getint('phase', 'num_phaseCandidate')


def main():
    # load the data from pkl
    task_name = joblib.load(task_name_path)
    datasets_norm_preproc = joblib.load(datasets_norm_preproc_path)
    min_max_scaler = joblib.load(min_max_scaler_path)
    noise_cov = joblib.load(noise_cov_path)

    # create iProMPs sets
    ipromps_set = [ipromps_lib.IProMP(num_joints=num_joints, num_obs_joints=num_obs_joints, num_basis=num_basis,
                                      sigma_basis=sigma_basis, num_samples=len_norm, sigmay=noise_cov,
                                      min_max_scaler=min_max_scaler, num_alpha_candidate=num_alpha_candidate)
                   for x in datasets_norm_preproc]

    # add demo and alpha var for each IProMPs
    for idx, ipromp in enumerate(ipromps_set):
        print('Training the IProMP for task: %s...' % task_name[idx])
        # for demo_idx in datasets4train[idx]:
        for demo_idx in datasets_norm_preproc[idx]:
            demo_temp = np.hstack([demo_idx['emg'], demo_idx['left_hand'], demo_idx['left_joints']])
            ipromp.add_demonstration(demo_temp)   # spatial variance demo
            ipromp.add_alpha(demo_idx['alpha'])   # temporal variance demo

    # save the trained models
    print('Saving the trained models...')
    joblib.dump(ipromps_set, os.path.join(datasets_pkl_path, 'ipromps_set.pkl'))

    print('Trained the IProMPs successfully!!!')


if __name__ == '__main__':
    main()
