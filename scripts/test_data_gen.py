#!/usr/bin/python
import rospy
from states_manager.msg import multiModal
import numpy as np
from scipy.ndimage.filters import gaussian_filter1d
import os
import ConfigParser
from sklearn.externals import joblib
import visualization

# read the current file path
file_path = os.path.dirname(__file__)
# read model cfg file
cp_models = ConfigParser.SafeConfigParser()
cp_models.read(os.path.join(file_path, '../cfg/models.cfg'))

# load param
datasets_path = os.path.join(file_path, cp_models.get('datasets', 'path'))
num_alpha_candidate = cp_models.getint('phase', 'num_phaseCandidate')
task_name_path = os.path.join(datasets_path, 'pkl/task_name_list.pkl')
task_name = joblib.load(task_name_path)
sigma = cp_models.get('filter', 'sigma')
num_joints = cp_models.getint('datasets', 'num_joints')
ipromps_set = joblib.load(os.path.join(datasets_path, 'pkl/ipromps_set.pkl'))
datasets_raw = joblib.load(os.path.join(datasets_path, 'pkl/datasets_raw.pkl'))


def main():

    # read test data
    obs_data_dict = datasets_raw[1][14]

    left_hand = obs_data_dict['left_hand']
    left_joints = obs_data_dict['left_joints']
    obs_data = np.hstack([left_hand, left_joints])
    timestamp = obs_data_dict['stamp']

    # filter the data
    obs_data = gaussian_filter1d(obs_data.T, sigma=sigma).T
    # preprocessing for the data
    obs_data_post_arr = ipromps_set[0].min_max_scaler.transform(obs_data)
    # consider the unobserved info
    obs_data_post_arr[:, 3:6] = 0.0

    # choose the data
    obs_data_post_arr = obs_data_post_arr[0:40, :]
    timestamp = timestamp[0:40]

    # phase estimation
    print('Phase estimating...')
    alpha_max_list = []
    for ipromp in ipromps_set:
        alpha_temp = ipromp.alpha_candidate()
        idx_max = ipromp.estimate_alpha(alpha_temp, obs_data_post_arr, timestamp)
        alpha_max_list.append(alpha_temp[idx_max]['candidate'])
        ipromp.set_alpha(alpha_temp[idx_max]['candidate'])

    # task recognition
    print('Adding via points in each trained model...')
    for task_idx, ipromp in enumerate(ipromps_set):
        for idx in range(len(timestamp)):
            ipromp.add_viapoint(timestamp[idx] / alpha_max_list[task_idx], obs_data_post_arr[idx, :])
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

    # save the conditional result
    print('Saving the post IProMPs...')
    joblib.dump(ipromps_set, os.path.join(datasets_path, 'pkl/ipromps_set_post_offline.pkl'))
    # save the robot traj
    print('Saving the robot traj...')
    joblib.dump([robot_traj, obs_data_dict], os.path.join(datasets_path, 'pkl/robot_traj_offline.pkl'))


if __name__ == '__main__':
    main()
    # visualization.main()