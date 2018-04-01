#!/usr/bin/python
import train_model_func
from sklearn.model_selection import LeaveOneOut
import numpy as np
import ipromps_lib
from sklearn.externals import joblib
import os
import ConfigParser
import error_compute

# the current file path
file_path = os.path.dirname(__file__)

# read models cfg file
cp_models = ConfigParser.SafeConfigParser()
cp_models.read(os.path.join(file_path, '../cfg/models.cfg'))
# read models params
datasets_path = os.path.join(file_path, cp_models.get('datasets', 'path'))
num_joints = cp_models.getint('datasets', 'num_joints')
num_obs_joints = cp_models.getint('datasets', 'num_obs_joints')
len_norm = cp_models.getint('datasets', 'len_norm')
num_basis = cp_models.getint('basisFunc', 'num_basisFunc')
sigma_basis = cp_models.getfloat('basisFunc', 'sigma_basisFunc')
num_alpha_candidate = cp_models.getint('phase', 'num_phaseCandidate')

# the pkl data
datasets_pkl_path = os.path.join(datasets_path, 'pkl')
task_name_path = os.path.join(datasets_pkl_path, 'task_name_list.pkl')
datasets_norm_preproc_path = os.path.join(datasets_pkl_path, 'datasets_norm_preproc.pkl')
min_max_scaler_path = os.path.join(datasets_pkl_path, 'min_max_scaler.pkl')
noise_cov_path = os.path.join(datasets_pkl_path, 'noise_cov.pkl')


def main(obs_ratio, task_id):
    # load the data from pkl
    datasets_norm_preproc = joblib.load(datasets_norm_preproc_path)

    X = datasets_norm_preproc[0]
    loo = LeaveOneOut()
    loo.get_n_splits(X)

    error_full = []
    i = 0.0
    for train_index, test_index in loo.split(X):
        ipromps_set = train_model_func.main(train_index)
        [idx_max_prob, error] = error_compute.main(ipromps_set, test_index, obs_ratio, task_id)
        error_full.append(error)
        if idx_max_prob == task_id:
            i=i+1.0
    acc = i/len(datasets_norm_preproc[0])
    return [error_full, acc]


if __name__ == '__main__':
    main()
