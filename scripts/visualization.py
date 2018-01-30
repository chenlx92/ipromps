#!/usr/bin/python
from sklearn.externals import joblib
import numpy as np
import matplotlib.pyplot as plt
import pylab as pl
import scipy.stats as stats
import os
import ConfigParser
from mpl_toolkits.mplot3d import Axes3D


# read conf file
file_path = os.path.dirname(__file__)
cp_models = ConfigParser.SafeConfigParser()
cp_models.read(os.path.join(file_path, '../cfg/models.cfg'))
# the datasets path
datasets_path = os.path.join(file_path, cp_models.get('datasets', 'path'))

# load datasets
ipromps_set = joblib.load(os.path.join(datasets_path, 'pkl/ipromps_set.pkl'))
ipromps_set_post = joblib.load(os.path.join(datasets_path, 'pkl/ipromps_set_post.pkl'))
robot_traj = joblib.load(os.path.join(datasets_path, 'pkl/robot_traj.pkl'))
datasets_norm_preproc = joblib.load(os.path.join(datasets_path, 'pkl/datasets_norm_preproc.pkl'))
datasets_raw = joblib.load(os.path.join(datasets_path, 'pkl/datasets_raw.pkl'))
datasets_filtered = joblib.load(os.path.join(datasets_path, 'pkl/datasets_filtered.pkl'))
task_name = joblib.load(os.path.join(datasets_path, 'pkl/task_name_list.pkl'))
[robot_traj_offline, ground_truth] = joblib.load(os.path.join(datasets_path, 'pkl/robot_traj_offline.pkl'))

# read datasets cfg file
datasets_path = os.path.join(file_path, cp_models.get('datasets', 'path'))
cp_datasets = ConfigParser.SafeConfigParser()
cp_datasets.read(os.path.join(datasets_path, 'info/cfg/datasets.cfg'))
# read datasets params
data_index0 = list(np.fromstring(cp_datasets.get('index', 'data_index0'), dtype=int, sep=','))
data_index1 = list(np.fromstring(cp_datasets.get('index', 'data_index1'), dtype=int, sep=','))
data_index2 = list(np.fromstring(cp_datasets.get('index', 'data_index2'), dtype=int, sep=','))
data_index = [data_index0, data_index1, data_index2]

# the idx of interest info in data structure
info_n_idx = {
            'left_hand': [0, 3],
            'left_joints': [3, 6]
            }
# the info to be plotted
info = cp_models.get('visualization', 'info')
joint_num = info_n_idx[info][1] - info_n_idx[info][0]


# plot the raw data
def plot_raw_data(num=0):
    for task_idx, ipromps_idx in enumerate(ipromps_set):
        fig = plt.figure(task_idx + num)
        fig.suptitle('the raw data of ' + info)
        for demo_idx in range(ipromps_idx.num_demos):
            for joint_idx in range(joint_num):
                ax = fig.add_subplot(joint_num, 1, 1 + joint_idx)
                data = datasets_raw[task_idx][demo_idx][info][:, joint_idx]
                plt.plot(range(len(data)), data)


# plot the filtered data
def plot_filtered_data(num=0):
    for task_idx, ipromps_idx in enumerate(ipromps_set):
        fig = plt.figure(task_idx + num)
        fig.suptitle('the filtered data of ' + info)
        for demo_idx in range(ipromps_idx.num_demos):
            for joint_idx in range(joint_num):
                ax = fig.add_subplot(joint_num, 1, 1 + joint_idx)
                data = datasets_filtered[task_idx][demo_idx][info][:, joint_idx]
                plt.plot(range(len(data)), data)


# plot the prior distribution
def plot_prior(num=0):
    for task_idx, ipromps_idx in enumerate(ipromps_set):
        fig = plt.figure(task_idx+num)
        fig.suptitle('the prior of ' + info + ' for ' + task_name[task_idx] + ' model')
        for joint_idx in range(joint_num):
            ax = fig.add_subplot(joint_num, 1, 1+joint_idx)
            ipromps_idx.promps[joint_idx + info_n_idx[info][0]].plot_prior(b_regression=True)


# plot alpha distribute
def plot_alpha(num=0):
    fig = plt.figure(num)
    for idx, ipromp in enumerate(ipromps_set):
        ax = fig.add_subplot(len(ipromps_set), 1, 1+idx)
        h = ipromps_set[idx].alpha
        h.sort()
        h_mean = np.mean(h)
        h_std = np.std(h)
        pdf = stats.norm.pdf(h, h_mean, h_std)
        pl.hist(h, normed=True, color='b')
        plt.plot(h, pdf, linewidth=5, color='r', marker='o', markersize=10)
        candidate = ipromp.alpha_candidate()
        candidate_x = [x['candidate'] for x in candidate]
        prob = [x['prob'] for x in candidate]
        plt.plot(candidate_x, prob, linewidth=0, color='g', marker='o', markersize=14)


# plot the post distribution
def plot_post(num=0):
    for task_idx, ipromps_idx in enumerate(ipromps_set_post):
        fig = plt.figure(task_idx+num)
        for joint_idx in range(joint_num):
            ax = fig.add_subplot(joint_num, 1, 1 + joint_idx)
            ipromps_idx.promps[joint_idx + info_n_idx[info][0]].plot_nUpdated(color='r', via_show=True)


# plot the generated robot motion trajectory
def plot_robot_traj(num=0):
    fig = plt.figure(num)
    fig.suptitle('predict robot motion')
    for joint_idx in range(7):
        ax = fig.add_subplot(7, 1, 1 + joint_idx)
        plt.plot(np.linspace(0, 1.0, 101), robot_traj[:, joint_idx])


# plot the raw data index
def plot_raw_data_index(num=0):
    for task_idx, demo_list in enumerate(data_index):
        for demo_idx in demo_list:
            fig = plt.figure(num + task_idx)
            fig.suptitle('the raw data of ' + info)
            for joint_idx in range(joint_num):
                ax = fig.add_subplot(joint_num, 1, 1 + joint_idx)
                data = datasets_raw[task_idx][demo_idx][info][:, joint_idx]
                plt.plot(range(len(data)), data, label=str(demo_idx))
                plt.legend()


# plot the filter data index
def plot_filter_data_index(num=0):
    for task_idx, demo_list in enumerate(data_index):
        for demo_idx in demo_list:
            fig = plt.figure(num + task_idx)
            fig.suptitle('the raw data of ' + info)
            for joint_idx in range(joint_num):
                ax = fig.add_subplot(joint_num, 1, 1 + joint_idx)
                data = datasets_filtered[task_idx][demo_idx][info][:, joint_idx]
                plt.plot(range(len(data)), data, label=str(demo_idx))
                ax.legend()


# plot the 3d raw traj
def plot_3d_raw_traj(num=0):
    for task_idx, demo_list in enumerate(data_index):
        fig = plt.figure(task_idx + num)
        ax = fig.gca(projection='3d')
        for demo_idx in demo_list:
            for joint_idx in range(joint_num):
                data = datasets_raw[task_idx][demo_idx][info]
                ax.plot(data[:, 0], data[:, 1], data[:, 2], label=str(demo_idx))
                ax.set_xlabel('X Label')
                ax.set_ylabel('Y Label')
                ax.set_zlabel('Z Label')
                ax.legend()


# plot the 3d filtered traj
def plot_3d_filtered_traj(num=0):
    for task_idx, demo_list in enumerate(data_index):
        fig = plt.figure(task_idx+num)
        ax = fig.gca(projection='3d')
        for demo_idx in demo_list:
            data = datasets_filtered[task_idx][demo_idx][info]
            ax.plot(data[:, 0], data[:, 1], data[:, 2], linewidth=3, label='human'+str(demo_idx), alpha=0.5)
            ax.set_xlabel('X Label')
            ax.set_ylabel('Y Label')
            ax.set_zlabel('Z Label')
            ax.legend()


# plot the 3d filtered robot traj
def plot_3d_filtered_r_traj(num=0):
    for task_idx, demo_list in enumerate(data_index):
        fig = plt.figure(task_idx+num)
        ax = fig.gca(projection='3d')
        for demo_idx in demo_list:
            data = datasets_filtered[task_idx][demo_idx]['left_joints']
            ax.plot(data[:, 0], data[:, 1], data[:, 2],
                    linewidth=3, linestyle=':', label='robot'+str(demo_idx), alpha=0.3)
            ax.set_xlabel('X Label')
            ax.set_ylabel('Y Label')
            ax.set_zlabel('Z Label')
            ax.legend()


# plot the 3d generated robot traj
def plot_3d_filtered_r_traj_offline(num=0):
    for task_idx, demo_list in enumerate(data_index):
        fig = plt.figure(task_idx+num)
        ax = fig.gca(projection='3d')
        data = robot_traj_offline
        ax.plot(data[:, 0], data[:, 1], data[:, 2],
                linewidth=8, linestyle='-', label='generated robot traj')
        data = ground_truth['left_joints']
        ax.plot(data[:, 0], data[:, 1], data[:, 2],
                linewidth=8, linestyle='-', label='ground truth robot traj', alpha=0.3)
        ax.set_xlabel('X Label')
        ax.set_ylabel('Y Label')
        ax.set_zlabel('Z Label')
        ax.legend()


def main():
    # plt.close('all')
    # plot_raw_data()
    # plot_filtered_data()
    # plot_prior()
    # plot_post()
    # plot_alpha()
    # plot_robot_traj()
    # plot_raw_data_index()
    # plot_filter_data_index(20)
    # plot_3d_raw_traj()
    plot_3d_filtered_traj(10)
    plot_3d_filtered_r_traj(10)
    plot_3d_filtered_r_traj_offline(10)
    plt.show()


if __name__ == '__main__':
    main()
