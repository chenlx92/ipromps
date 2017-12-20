#!/usr/bin/python

from sklearn.externals import joblib
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats
import pylab as pl

path = '../datasets/handover_20171128/pkl'

###########################

# load datasets ipromps model min_max_scaler
print('Loading the models...')
[ipromps_set, datasets4train_post, min_max_scaler, filt_kernel] = joblib.load(path + '/ipromps_set.pkl')
ipromps_set_post = joblib.load(path + '/ipromps_set_post.pkl')
datasets_len101 = joblib.load(path + '/datasets_len101.pkl')

# # plot prior
# for task_idx, ipromps_idx in enumerate(ipromps_set):
#     plt.figure(task_idx)
#     for joint_idx in range(7):
#         plt.subplot(711 + joint_idx)
#         ipromps_idx.promps[11 + joint_idx].plot_prior()
#
# # plot datasets4train_post data
# for task_idx, ipromps_idx in enumerate(datasets4train_post):
#     plt.figure(task_idx)
#     for demo_idx in range(20):
#         for joint_idx in range(7):
#             plt.subplot(711 + joint_idx)
#             plt.plot(np.linspace(0, 1.0, 101), datasets4train_post[task_idx][demo_idx]['left_joints'][:, joint_idx])
#
# # plot the norm data
# for task_idx, ipromps_idx in enumerate(datasets_len101):
#     plt.figure(20+task_idx)
#     for demo_idx in range(20):
#         for joint_idx in range(7):
#             plt.subplot(711 + joint_idx)
#             plt.plot(np.linspace(0, 1.0, 101), datasets_len101[task_idx][demo_idx]['left_joints'][:, joint_idx])

# plot prior
for task_idx, ipromps_idx in enumerate(ipromps_set):
    plt.figure(task_idx+100)
    for joint_idx in range(7):
        plt.subplot(711 + joint_idx)
        ipromps_idx.promps[11 + joint_idx].plot_prior()

# plot post
for task_idx, ipromps_idx in enumerate(ipromps_set_post):
    plt.figure(task_idx+100)
    for joint_idx in range(7):
        plt.subplot(711 + joint_idx)
        ipromps_idx.promps[11 + joint_idx].plot_nUpdated(color='r', via_show=True)

# # plot alpha distribute
# plt.figure(10)
# for idx, ipromp in enumerate(ipromps_set):
#     plt.subplot(310 + idx)
#     h = ipromps_set[idx].alpha
#     h.sort()
#     h_mean = np.mean(h)
#     h_std = np.std(h)
#     pdf = stats.norm.pdf(h, h_mean, h_std)
#     pl.hist(h, normed=True, color='b')
#     plt.plot(h, pdf, linewidth=5, color='r', marker='o', markersize=10)
#     # candidate = ipromps_set[0].alpha_candidate(num_alpha_candidate)
#     # candidate_x = [x['candidate'] for x in candidate]
#     # prob = [x['prob'] for x in candidate]
#     # plt.plot(candidate_x, prob, linewidth=0, color='g', marker='o', markersize=14);
#     # print("the aluminum_hold alpha mean is ", hmean)
#     # print("the aluminum_hold alpha std is hmean", hstd)

plt.show()
