#!/usr/bin/python

from sklearn.externals import joblib
import numpy as np
import matplotlib.pyplot as plt
import pylab as pl
import scipy.stats as stats

path = '../datasets/handover_20171128/pkl'
info_n_idx = {
            'emg': [0, 8],
            'left_hand': [8, 11],
            'left_joints': [11, 18]
            }

###########################
# the info to be plotted
info = 'emg'
joint_num = info_n_idx[info][1] - info_n_idx[info][0]
plt.close('all')

# load datasets
print('Loading the models...')
[ipromps_set, datasets4train_post, filt_kernel] = joblib.load(path + '/ipromps_set.pkl')
ipromps_set_post = joblib.load(path + '/ipromps_set_post.pkl')
datasets_len101 = joblib.load(path + '/datasets_len101.pkl')
robot_traj = joblib.load(path + '/robot_traj.pkl')


# # plot the norm data
# for task_idx, ipromps_idx in enumerate(ipromps_set_post):
#     fig = plt.figure(task_idx+10)   # from fig. 10
#     fig.suptitle('the norm ' + info)
#     for demo_idx in range(ipromps_idx.num_demos):
#         for joint_idx in range(joint_num):
#             ax = fig.add_subplot(joint_num, 1, 1 + joint_idx)
#             plt.plot(ipromps_idx.x, datasets_len101[task_idx][demo_idx][info][:, joint_idx])
# # plot datasets4train_post data
# for task_idx, ipromps_idx in enumerate(ipromps_set_post):
#     fig = plt.figure(task_idx) # from fig. 0
#     fig.suptitle(info)
#     for demo_idx in range(ipromps_idx.num_demos):
#         for joint_idx in range(joint_num):
#             ax = fig.add_subplot(joint_num, 1, 1 + joint_idx)
#             plt.plot(ipromps_idx.x, datasets4train_post[task_idx][demo_idx][info][:, joint_idx])
# # plot the prior
# for task_idx, ipromps_idx in enumerate(ipromps_set_post):
#     fig = plt.figure(task_idx)
#     for joint_idx in range(joint_num):
#         ax = fig.add_subplot(joint_num, 1, 1+joint_idx)
#         ipromps_idx.promps[joint_idx + info_n_idx[info][0]].plot_prior()
# # plot post distribution
# for task_idx, ipromps_idx in enumerate(ipromps_set_post):
#     print(ipromps_idx.promps[0].alpha_fit)
#     fig = plt.figure(task_idx)
#     for joint_idx in range(joint_num):
#         ax = fig.add_subplot(joint_num, 1, 1 + joint_idx)
#         ipromps_idx.promps[joint_idx + info_n_idx[info][0]].plot_nUpdated(color='r', via_show=True)

# # plot the predict robot motion
# fig = plt.figure(50)
# fig.suptitle('predict robot motion')
# for joint_idx in range(7):
#     ax = fig.add_subplot(7, 1, 1 + joint_idx)
#     plt.plot(np.linspace(0, 1.0, 101), robot_traj[:, joint_idx])

# plot alpha distribute
fig = plt.figure(100)
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
plt.show()
