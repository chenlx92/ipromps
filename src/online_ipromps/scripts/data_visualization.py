#!/usr/bin/python
from sklearn.externals import joblib
import numpy as np
import matplotlib.pyplot as plt
import pylab as pl
import scipy.stats as stats
import os
import ConfigParser

# read conf file
file_path = os.path.dirname(__file__)
cp = ConfigParser.SafeConfigParser()
cp.read(os.path.join(file_path, '../config/model.conf'))
# the datasets path
datasets_path = os.path.join(file_path, cp.get('datasets', 'path'))
# the interest info and corresponding index in csv file
info_n_idx = {
            'emg': [0, 8],
            'left_hand': [8, 11],
            'left_joints': [11, 18]
            }

# the info to be plotted
info = cp.get('visualization', 'info')
joint_num = info_n_idx[info][1] - info_n_idx[info][0]

# load datasets
ipromps_set = joblib.load(os.path.join(datasets_path, 'pkl/ipromps_set.pkl'))
ipromps_set_post = joblib.load(os.path.join(datasets_path, 'pkl/ipromps_set_post.pkl'))
robot_traj = joblib.load(os.path.join(datasets_path, 'pkl/robot_traj.pkl'))
datasets_norm_preproc = joblib.load(os.path.join(datasets_path, 'pkl/datasets_norm_preproc.pkl'))
datasets_raw = joblib.load(os.path.join(datasets_path, 'pkl/datasets_raw.pkl'))


# plot the raw data
def plot_raw_data(num=0):
    for task_idx, ipromps_idx in enumerate(ipromps_set_post):
        fig = plt.figure(task_idx+num)
        fig.suptitle('the raw data of ' + info)
        for demo_idx in range(ipromps_idx.num_demos):
            for joint_idx in range(joint_num):
                ax = fig.add_subplot(joint_num, 1, 1 + joint_idx)
                data = datasets_raw[task_idx][demo_idx][info][:, joint_idx]
                plt.plot(range(len(data)), data)


# plot the prior distribution
def plot_prior(num=10):
    for task_idx, ipromps_idx in enumerate(ipromps_set_post):
        fig = plt.figure(task_idx+num)
        fig.suptitle('the raw data of ' + info)
        for joint_idx in range(joint_num):
            ax = fig.add_subplot(joint_num, 1, 1+joint_idx)
            ipromps_idx.promps[joint_idx + info_n_idx[info][0]].plot_prior()


# plot alpha distribute
def plot_alpha(num=20):
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
def plot_post(num=30):
    for task_idx, ipromps_idx in enumerate(ipromps_set_post):
        fig = plt.figure(task_idx+num)
        for joint_idx in range(joint_num):
            ax = fig.add_subplot(joint_num, 1, 1 + joint_idx)
            ipromps_idx.promps[joint_idx + info_n_idx[info][0]].plot_nUpdated(color='r', via_show=True)


# plot the generated robot motion trajectory
def plot_robot_traj(num=40):
    fig = plt.figure(num)
    fig.suptitle('predict robot motion')
    for joint_idx in range(7):
        ax = fig.add_subplot(7, 1, 1 + joint_idx)
        plt.plot(np.linspace(0, 1.0, 101), robot_traj[:, joint_idx])


def main():
    # plot_raw_data()
    plot_prior(0)
    # plot_post(0)
    # plot_alpha()
    # plot_robot_traj()
    plt.show()


if __name__ == '__main__':
    main()
