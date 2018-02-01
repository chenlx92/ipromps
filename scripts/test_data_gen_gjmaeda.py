#!/usr/bin/python
import rospy
from states_manager.msg import multiModal
import numpy as np
from scipy.ndimage.filters import gaussian_filter1d
import os
import ConfigParser
from sklearn.externals import joblib
import pandas as pd
import visualization
import ipromps_lib
import scipy.io as sio

# read the current file path
file_path = os.path.dirname(__file__)
# read model cfg file
cp_models = ConfigParser.SafeConfigParser()
cp_models.read(os.path.join(file_path, '../cfg/models.cfg'))

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
min_max_scaler = joblib.load(os.path.join(min_max_scaler_path, 'pkl/ipromps_set.pkl'))

# load param
datasets_path = os.path.join(file_path, cp_models.get('datasets', 'path'))
num_alpha_candidate = cp_models.getint('phase', 'num_phaseCandidate')
task_name_path = os.path.join(datasets_path, 'pkl/task_name_list.pkl')
task_name = joblib.load(task_name_path)
sigma = cp_models.get('filter', 'sigma')
num_joints = cp_models.getint('datasets', 'num_joints')


def main():

    data_mat = sio.loadmat('../datasets/guilherme_datasets/taskProMP.mat')
    data = data_mat['taskProMP'][0]

    datasets_raw = []
    for task_idx in range(len(data)):
        demo_temp = []
        for demo_idx in range(len(data[task_idx])):
            # the info of interest: convert the object to int / float
            demo_temp.append({
                'stamp': range(len(data[task_idx][demo_idx][0][:, 0:3].astype(float))),
                'left_hand': data[task_idx][demo_idx][0][:, 0:3].astype(float),
                'left_joints': data[task_idx][demo_idx][0][:, 10:13].astype(float)
            })
        datasets_raw.append(demo_temp)

    # read test data
    obs_data_dict = datasets_raw[1][14]

    left_hand = obs_data_dict['left_hand']
    left_joints = obs_data_dict['left_joints']
    obs_data = np.hstack([left_hand, left_joints])
    timestamp = obs_data_dict['stamp']


    # create iProMPs sets
    ipromps_set = [ipromps_lib.IProMP(num_joints=num_joints, num_obs_joints=num_obs_joints, num_basis=num_basis,
                                      sigma_basis=sigma_basis, num_samples=len_norm, sigmay=noise_cov,
                                      min_max_scaler=min_max_scaler, num_alpha_candidate=num_alpha_candidate)
                   for x in range(3)]

    # add demo for each IProMPs
    for idx, ipromp in enumerate(ipromps_set):
        print('Training the IProMP for task: %s...' % task_name[idx])
        # for demo_idx in datasets4train[idx]:
        for demo_idx in datasets_norm_preproc[idx]:
            # demo_temp = np.hstack([demo_idx['emg'], demo_idx['left_hand'], demo_idx['left_joints']])
            demo_temp = np.hstack([demo_idx['left_hand'], demo_idx['left_joints']])
            ipromp.add_demonstration(demo_temp)  # spatial variance demo
            ipromp.add_alpha(demo_idx['alpha'])  # temporal variance demo

    # phase estimation
    print('Phase estimating...')
    alpha_max_list = []
    for ipromp in ipromps_set:
        alpha_temp = ipromp.alpha_candidate()
        idx_max = ipromp.estimate_alpha(alpha_temp, obs_data, timestamp)
        alpha_max_list.append(alpha_temp[idx_max]['candidate'])
        ipromp.set_alpha(alpha_temp[idx_max]['candidate'])

    # task recognition
    print('Adding via points in each trained model...')
    for task_idx, ipromp in enumerate(ipromps_set):
        for idx in range(len(timestamp)):
            ipromp.add_viapoint(timestamp[idx] / alpha_max_list[task_idx], obs_data[idx, :])
        ipromp.param_update(unit_update=True)
    print('Computing the likelihood for each model under observations...')

    prob_task = []
    for ipromp in ipromps_set:
        prob_task_temp = ipromp.prob_obs()
        prob_task.append(prob_task_temp)
    idx_max_prob = np.argmax(prob_task)
    # idx_max_prob = 0 # a trick for testing
    print('The max fit model index is task %s' % task_name[idx_max_prob])

    # robot motion generation
    [traj_time, traj] = ipromps_set[idx_max_prob].gen_real_traj(alpha_max_list[idx_max_prob])
    traj = ipromps_set[idx_max_prob].min_max_scaler.inverse_transform(traj)
    robot_traj = traj[:, 3:6]

    # save the robot traj
    print('Saving the robot traj...')
    joblib.dump([robot_traj, obs_data_dict], os.path.join(datasets_path, 'pkl/robot_traj_offline_guilherme.pkl'))


if __name__ == '__main__':
    main()
    # visualization.main()