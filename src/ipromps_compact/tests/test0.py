#!/usr/bin/python
# Filename: imu_emg_pose_test_compact.py

from __future__ import print_function

import numpy as np
import matplotlib.pyplot as plt
import ipromps_lib
import scipy.linalg
# from scipy.stats import entropy
# import rospy
from sklearn.externals import joblib
import scipy.stats as stats
import pylab as pl

from optparse import OptionParser
parser = OptionParser()
parser.add_option("-t", "--tdbk", action="store",
                  dest="task_id",
                  default=0,
                  help="the index of demo in dataset")
parser.add_option("-i", "--idbk", action="store",
                  dest="idx_demo",
                  default=0,
                  help="the index of demo in dataset")
(options, args) = parser.parse_args()
task_id = np.int(options.task_id)
idx_demo = np.int(options.idx_demo)

plt.close('all')    # close all windows

# parameter of model
num_demos = 20         # number of trajectoreis for training
obs_ratio = 10
num_joints=19
num_basis=31
sigma_basis=0.05
num_samples=101
num_obs_joints=4
# measurement noise
imu_noise = 1.0
emg_noise = 2.0
pose_noise = 1.0
# phase estimation para
num_alpha_candidate = 10
nominal_duration = 1.0
nominal_interval = nominal_duration / (num_samples-1)
states_refresh_rate = 50.0
# preprocessing: scaling factor for data
sf_imu = 1000.0
sf_emg = 100.0
sf_pose = 0.1

# plot options
b_plot_raw_dateset = False
b_plot_norm_dateset = False
b_plot_prior_distribution = False
b_plot_update_distribution = False
b_plot_phase_distribution = True



#################################
# load raw date sets
#################################
dataset_aluminum_hold = joblib.load('./pkl/dataset_aluminum_hold.pkl')
dataset_spanner_handover = joblib.load('./pkl/dataset_spanner_handover.pkl')
dataset_tape_hold = joblib.load('./pkl/dataset_tape_hold.pkl')
#################################
# load norm date sets
#################################
dataset_aluminum_hold_norm = joblib.load('./pkl/dataset_aluminum_hold_norm.pkl')
dataset_spanner_handover_norm = joblib.load('./pkl/dataset_spanner_handover_norm.pkl')
dataset_tape_hold_norm = joblib.load('./pkl/dataset_tape_hold_norm.pkl')


#################################
# Interaction ProMPs train
#################################
# the measurement noise cov matrix
imu_meansurement_noise_cov = np.eye(4) * imu_noise
emg_meansurement_noise_cov = np.eye(8) * emg_noise
pose_meansurement_noise_cov = np.eye(7) * pose_noise
meansurement_noise_cov_full = scipy.linalg.block_diag(imu_meansurement_noise_cov,
                                                      emg_meansurement_noise_cov,
                                                      pose_meansurement_noise_cov)
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
for idx in range(num_demos):
    # aluminum_hold
    demo_temp = np.hstack([dataset_aluminum_hold_norm[idx]['imu']/sf_imu, dataset_aluminum_hold_norm[idx]['emg']/sf_emg])
    demo_temp = np.hstack([demo_temp, dataset_aluminum_hold_norm[idx]['pose']/sf_pose])
    ipromp_aluminum_hold.add_demonstration(demo_temp)
    # spanner_handover
    demo_temp = np.hstack([dataset_spanner_handover_norm[idx]['imu']/sf_imu, dataset_spanner_handover_norm[idx]['emg']/sf_emg])
    demo_temp = np.hstack([demo_temp, dataset_spanner_handover_norm[idx]['pose']/sf_pose])
    ipromp_spanner_handover.add_demonstration(demo_temp)
    # tape_hold
    demo_temp = np.hstack([dataset_tape_hold_norm[idx]['imu']/sf_imu, dataset_tape_hold_norm[idx]['emg']/sf_emg])
    demo_temp = np.hstack([demo_temp, dataset_tape_hold_norm[idx]['pose']/sf_pose])
    ipromp_tape_hold.add_demonstration(demo_temp)

# model the phase distribution
for i in range(num_demos):
    alpha = (len(dataset_aluminum_hold[i]['imu']) - 1) / states_refresh_rate / num_samples
    ipromp_aluminum_hold.add_alpha(alpha)
    ##
    alpha = (len(dataset_spanner_handover[i]['imu']) - 1) / states_refresh_rate / num_samples
    ipromp_spanner_handover.add_alpha(alpha)
    ##
    alpha = (len(dataset_tape_hold[i]['imu']) - 1) / states_refresh_rate / num_samples
    ipromp_tape_hold.add_alpha(alpha)


################################
# Interaction ProMPs test
################################
# select the testset
selected_testset = None
selected_testset = dataset_aluminum_hold_norm[idx_demo] if task_id==0 else selected_testset
selected_testset = dataset_spanner_handover_norm[idx_demo] if task_id==1 else selected_testset
selected_testset = dataset_tape_hold_norm[idx_demo] if task_id==2 else selected_testset
# construct the test set
test_set = np.hstack((selected_testset['imu']/sf_imu, selected_testset['emg']/sf_emg, np.zeros([num_samples, 7])))
robot_response = selected_testset['pose']/sf_pose

# add via points to update the distribution
for idx in range(obs_ratio):
    ipromp_aluminum_hold.add_viapoint(nominal_interval*idx, test_set[idx, :])
    ipromp_spanner_handover.add_viapoint(nominal_interval*idx, test_set[idx, :])
    ipromp_tape_hold.add_viapoint(nominal_interval*idx, test_set[idx, :])

# the model info
# print('the number of demonstration is ',num_demos)
# print('the number of observation is ', obs_ratio*nominal_interval)

# likelihood of observation
prob_aluminum_hold = ipromp_aluminum_hold.prob_obs()
print('from obs, the log pro of aluminum_hold is', prob_aluminum_hold)
##
prob_spanner_handover = ipromp_spanner_handover.prob_obs()
print('from obs, the log pro of spanner_handover is', prob_spanner_handover)
##
prob_tape_hold = ipromp_tape_hold.prob_obs()
print('from obs, the log pro of tape_hold is', prob_tape_hold)

idx_max_pro = np.argmax([prob_aluminum_hold, prob_spanner_handover, prob_tape_hold])
if idx_max_pro == task_id:
    print("OK, you are right!!!")
else:
    print("Sorry, you are wrong!!!, for %d", idx_max_pro)
# if idx_max_pro == 0:
#     print('the obs comes from aluminum_hold')
# elif idx_max_pro == 1:
#     print('the obs comes from spanner_handover')
# elif idx_max_pro == 2:
#     print('the obs comes from tape_hold')


#################################
# compute the position error
#################################
# position_error = None
# # if idx_max_pro == 0:
# predict_robot_response = ipromp_aluminum_hold.generate_trajectory()
# position_error = np.linalg.norm(predict_robot_response[-1,12:15]-robot_response[-1,0:3])
# print('if aluminum_hold, the obs position error is', position_error)
# # elif idx_max_pro == 1:
# predict_robot_response = ipromp_spanner_handover.generate_trajectory()
# position_error = np.linalg.norm(predict_robot_response[-1, 12:15] - robot_response[-1,0:3])
# print('if spanner_handover, the obs position error is', position_error)
# # elif idx_max_pro == 2:
# predict_robot_response = ipromp_tape_hold.generate_trajectory()
# position_error = np.linalg.norm(predict_robot_response[-1, 12:15] - robot_response[-1,0:3])
# print('if tape_hold, the obs position error is', position_error)


# #################################
# # the KL divergence of IMU
# #################################
# mean_a_imu = ipromp_aluminum_hold.mean_W_full[0:44]
# cov_a_imu = ipromp_aluminum_hold.cov_W_full[0:44,0:44]
# mean_s_imu = ipromp_spanner_handover.mean_W_full[0:44]
# cov_s_imu = ipromp_spanner_handover.cov_W_full[0:44,0:44]
# kl_divergence_imu_a_s = math.log(np.linalg.det(cov_s_imu)/np.linalg.det(cov_a_imu)) - 44 \
#                         + np.trace(np.dot(np.linalg.inv(cov_s_imu), cov_a_imu)) + \
#                         np.dot((mean_s_imu-mean_a_imu).T, np.dot(np.linalg.inv(cov_s_imu), (mean_s_imu-mean_a_imu)))
#
# mean_a_imu_emg = ipromp_aluminum_hold.mean_W_full[0:132]
# cov_a_imu_emg = ipromp_aluminum_hold.cov_W_full[0:132,0:132]
# mean_s_imu_emg = ipromp_spanner_handover.mean_W_full[0:132]
# cov_s_imu_emg = ipromp_spanner_handover.cov_W_full[0:132,0:132]
# kl_divergence_imu_emg_a_s = math.log(np.linalg.det(cov_s_imu_emg)/np.linalg.det(cov_a_imu_emg)) - 132\
#                         + np.trace(np.dot(np.linalg.inv(cov_s_imu_emg), cov_a_imu_emg)) + \
#                         np.dot((mean_s_imu_emg-mean_a_imu_emg).T, np.dot(np.linalg.inv(cov_s_imu_emg), (mean_s_imu_emg - mean_a_imu_emg)))


#################################
# plot raw data
#################################
if b_plot_raw_dateset == True:
    ## plot the aluminum hold task raw data
    plt.figure(0)
    for ch_ex in range(4):
       plt.subplot(411+ch_ex)
       for idx in range(num_demos):
           plt.plot(range(len(dataset_aluminum_hold[idx]['imu'][:, ch_ex])), dataset_aluminum_hold[idx]['imu'][:, ch_ex]); plt.axis('off')
    pl.savefig('./fig/aluminum_hold_imu_raw.eps', format='eps');pl.savefig('./fig/aluminum_hold_imu_raw.pdf', format='pdf')
    plt.figure(1)
    for ch_ex in range(8):
       plt.subplot(421+ch_ex)
       for idx in range(num_demos):
           plt.plot(range(len(dataset_aluminum_hold[idx]['emg'][:, ch_ex])), dataset_aluminum_hold[idx]['emg'][:, ch_ex]); plt.axis('off')
    pl.savefig('./fig/aluminum_hold_emg_raw.eps', format='eps');pl.savefig('./fig/aluminum_hold_emg_raw.pdf', format='pdf')
    plt.figure(2)
    for ch_ex in range(7):
       plt.subplot(711+ch_ex)
       for idx in range(num_demos):
           plt.plot(range(len(dataset_aluminum_hold[idx]['pose'][:, ch_ex])), dataset_aluminum_hold[idx]['pose'][:, ch_ex]); plt.axis('off')
    pl.savefig('./fig/aluminum_hold_pose_raw.eps', format='eps');pl.savefig('./fig/aluminum_hold_pose_raw.pdf', format='pdf')
    ## plot the spanner handover task raw data
    plt.figure(10)
    for ch_ex in range(4):
       plt.subplot(411 + ch_ex)
       for idx in range(num_demos):
           plt.plot(range(len(dataset_spanner_handover[idx]['imu'][:, ch_ex])), dataset_spanner_handover[idx]['imu'][:, ch_ex]); plt.axis('off')
    pl.savefig('./fig/spanner_handover_imu_raw.eps', format='eps');pl.savefig('./fig/spanner_handover_imu_raw.pdf', format='pdf')
    plt.figure(11)
    for ch_ex in range(8):
       plt.subplot(421 + ch_ex)
       for idx in range(num_demos):
           plt.plot(range(len(dataset_spanner_handover[idx]['emg'][:, ch_ex])), dataset_spanner_handover[idx]['emg'][:, ch_ex]); plt.axis('off')
    pl.savefig('./fig/spanner_handover_emg_raw.eps', format='eps');pl.savefig('./fig/spanner_handover_emg_raw.pdf', format='pdf')
    plt.figure(12)
    for ch_ex in range(7):
       plt.subplot(711 + ch_ex)
       for idx in range(num_demos):
           plt.plot(range(len(dataset_spanner_handover[idx]['pose'][:, ch_ex])), dataset_spanner_handover[idx]['pose'][:, ch_ex]); plt.axis('off')
    pl.savefig('./fig/spanner_handover_pose_raw.eps', format='eps');pl.savefig('./fig/spanner_handover_pose_raw.pdf', format='pdf')
    ## plot the tape hold task raw data
    plt.figure(20)
    for ch_ex in range(4):
       plt.subplot(411 + ch_ex)
       for idx in range(num_demos):
           plt.plot(range(len(dataset_tape_hold[idx]['imu'][:, ch_ex])), dataset_tape_hold[idx]['imu'][:, ch_ex]); plt.axis('off')
    pl.savefig('./fig/tape_hold_imu_raw.eps', format='eps');pl.savefig('./fig/tape_hold_imu_raw.pdf', format='pdf')
    plt.figure(21)
    for ch_ex in range(8):
       plt.subplot(421 + ch_ex)
       for idx in range(num_demos):
           plt.plot(range(len(dataset_tape_hold[idx]['emg'][:, ch_ex])), dataset_tape_hold[idx]['emg'][:, ch_ex]); plt.axis('off')
    pl.savefig('./fig/tape_hold_emg_raw.eps', format='eps');pl.savefig('./fig/tape_hold_emg_raw.pdf', format='pdf')
    plt.figure(22)
    for ch_ex in range(7):
       plt.subplot(711 + ch_ex)
       for idx in range(num_demos):
           plt.plot(range(len(dataset_tape_hold[idx]['pose'][:, ch_ex])), dataset_tape_hold[idx]['pose'][:, ch_ex]); plt.axis('off')
    pl.savefig('./fig/tape_hold_pose_raw.eps', format='eps');pl.savefig('./fig/tape_hold_pose_raw.pdf', format='pdf')


#################################
# plot normalized data
#################################
if b_plot_norm_dateset == True:
    ## plot the aluminum hold task raw data
    plt.figure(50)
    for ch_ex in range(4):
        plt.subplot(411 + ch_ex)
        for idx in range(num_demos):
            plt.plot(ipromp_aluminum_hold.x, dataset_aluminum_hold_norm[idx]['imu'][:, ch_ex]/sf_imu, linewidth=1);
            plt.axis('off')
    pl.savefig('./fig/aluminum_hold_imu_norm.eps', format='eps');pl.savefig('./fig/aluminum_hold_imu_norm.pdf', format='pdf')
    plt.figure(51)
    for ch_ex in range(8):
        plt.subplot(421 + ch_ex)
        for idx in range(num_demos):
            plt.plot(ipromp_aluminum_hold.x,
                     dataset_aluminum_hold_norm[idx]['emg'][:, ch_ex]/sf_emg,   linewidth=1);
            plt.axis('off')
    pl.savefig('./fig/aluminum_hold_emg_norm.eps', format='eps');pl.savefig('./fig/aluminum_hold_emg_norm.pdf', format='pdf')
    plt.figure(52)
    for ch_ex in range(7):
        plt.subplot(711 + ch_ex)
        for idx in range(num_demos):
            plt.plot(ipromp_aluminum_hold.x,
                     dataset_aluminum_hold_norm[idx]['pose'][:, ch_ex]/sf_pose,   linewidth=1);
            plt.axis('off')
    pl.savefig('./fig/aluminum_hold_pose_norm.eps', format='eps');pl.savefig('./fig/aluminum_hold_pose_norm.pdf', format='pdf')
    ## plot the spanner handover task raw data
    plt.figure(60)
    for ch_ex in range(4):
        plt.subplot(411 + ch_ex)
        for idx in range(num_demos):
            plt.plot(ipromp_spanner_handover.x,
                     dataset_spanner_handover_norm[idx]['imu'][:, ch_ex]/sf_imu,   linewidth=1);
            plt.axis('off')
    pl.savefig('./fig/spanner_handover_imu_norm.eps', format='eps');pl.savefig('./fig/spanner_handover_imu_norm.pdf', format='pdf')
    plt.figure(61)
    for ch_ex in range(8):
        plt.subplot(421 + ch_ex)
        for idx in range(num_demos):
            plt.plot(ipromp_spanner_handover.x,
                     dataset_spanner_handover_norm[idx]['emg'][:, ch_ex]/sf_emg,   linewidth=1);
            plt.axis('off')
    pl.savefig('./fig/spanner_handover_emg_norm.eps', format='eps');pl.savefig('./fig/spanner_handover_emg_norm.pdf', format='pdf')
    plt.figure(62)
    for ch_ex in range(7):
        plt.subplot(711 + ch_ex)
        for idx in range(num_demos):
            plt.plot(ipromp_spanner_handover.x,
                     dataset_spanner_handover_norm[idx]['pose'][:, ch_ex]/sf_pose,   linewidth=1);
            plt.axis('off')
    pl.savefig('./fig/spanner_handover_pose_norm.eps', format='eps');pl.savefig('./fig/spanner_handover_pose_norm.pdf', format='pdf')
    ## plot the tape hold task raw data
    plt.figure(70)
    for ch_ex in range(4):
        plt.subplot(411 + ch_ex)
        for idx in range(num_demos):
            plt.plot(ipromp_tape_hold.x, dataset_tape_hold_norm[idx]['imu'][:, ch_ex]/sf_imu,   linewidth=1);
            plt.axis('off')
    pl.savefig('./fig/tape_hold_imu_norm.eps', format='eps');pl.savefig('./fig/tape_hold_imu_norm.pdf', format='pdf')
    plt.figure(71)
    for ch_ex in range(8):
        plt.subplot(421 + ch_ex)
        for idx in range(num_demos):
            plt.plot(ipromp_tape_hold.x, dataset_tape_hold_norm[idx]['emg'][:, ch_ex]/sf_emg,   linewidth=1);
            plt.axis('off')
    pl.savefig('./fig/tape_hold_emg_norm.eps', format='eps');pl.savefig('./fig/tape_hold_emg_norm.pdf', format='pdf')
    plt.figure(72)
    for ch_ex in range(7):
        plt.subplot(711 + ch_ex)
        for idx in range(num_demos):
            plt.plot(ipromp_tape_hold.x,
                     dataset_tape_hold_norm[idx]['pose'][:, ch_ex]/sf_pose,   linewidth=1);
            plt.axis('off')
            pl.savefig('./fig/tape_hold_pose_norm.eps', format='eps');pl.savefig('./fig/tape_hold_pose_norm.pdf', format='pdf')


#################################
# plot the prior distributioin
#################################
if b_plot_prior_distribution == True:
    # plot ipromp_aluminum_hold
    plt.figure(50)
    for i in range(4):
        plt.subplot(411+i)
        ipromp_aluminum_hold.promps[i].plot_prior(color='b', legend='alumnium hold model, imu'); plt.axis('off')
    pl.savefig('./fig/aluminum_hold_imu_prior.eps', format='eps'); pl.savefig('./fig/aluminum_hold_imu_prior.pdf', format='pdf')
    plt.figure(51)
    for i in range(8):
        plt.subplot(421+i)
        ipromp_aluminum_hold.promps[4+i].plot_prior(color='y', legend='alumnium hold model, emg');plt.axis('off')
    pl.savefig('./fig/aluminum_hold_emg_prior.eps', format='eps'); pl.savefig('./fig/aluminum_hold_emg_prior.pdf', format='pdf')
    plt.figure(52)
    for i in range(7):
        plt.subplot(711+i)
        ipromp_aluminum_hold.promps[4+8+i].plot_prior(color='r', legend='alumnium hold model, pose');plt.axis('off')
    pl.savefig('./fig/aluminum_hold_pose_prior.eps', format='eps'); pl.savefig('./fig/aluminum_hold_pose_prior.pdf', format='pdf')
    # plot ipromp_spanner_handover
    plt.figure(60)
    for i in range(4):
        plt.subplot(411+i)
        ipromp_spanner_handover.promps[i].plot_prior(color='b', legend='spanner handover model, imu');plt.axis('off')
    pl.savefig('./fig/spanner_handover_imu_prior.eps', format='eps'); pl.savefig('./fig/spanner_handover_imu_prior.pdf', format='pdf')
    plt.figure(61)
    for i in range(8):
        plt.subplot(421+i)
        ipromp_spanner_handover.promps[4+i].plot_prior(color='y', legend='spanner handover model, emg');plt.axis('off')
    pl.savefig('./fig/spanner_handover_emg_prior.eps', format='eps'); pl.savefig('./fig/spanner_handover_emg_prior.pdf', format='pdf')
    plt.figure(62)
    for i in range(7):
        plt.subplot(711+i)
        ipromp_spanner_handover.promps[4+8+i].plot_prior(color='r', legend='spanner handover model, pose');plt.axis('off')
    pl.savefig('./fig/spanner_handover_pose_prior.eps', format='eps'); pl.savefig('./fig/spanner_handover_pose_prior.pdf', format='pdf')
    # plot ipromp_tape_hold
    plt.figure(70)
    for i in range(4):
        plt.subplot(411+i)
        ipromp_tape_hold.promps[i].plot_prior(color='b', legend='tape hold model, imu');plt.axis('off')
    pl.savefig('./fig/tape_hold_imu_prior.eps', format='eps'); pl.savefig('./fig/tape_hold_imu_prior.pdf', format='pdf')
    plt.figure(71)
    for i in range(8):
        plt.subplot(421+i)
        ipromp_tape_hold.promps[4+i].plot_prior(color='y', legend='tape hold model, emg');plt.axis('off')
    pl.savefig('./fig/tape_hold_emg_prior.eps', format='eps'); pl.savefig('./fig/tape_hold_emg_prior.pdf', format='pdf')
    plt.figure(72)
    for i in range(7):
        plt.subplot(711+i)
        ipromp_tape_hold.promps[4+8+i].plot(color='r', legend='tape hold model, pose');plt.axis('off')
    pl.savefig('./fig/tape_hold_pose_prior.eps', format='eps'); pl.savefig('./fig/tape_hold_pose_prior.pdf', format='pdf')


#################################
# plot the updated distributioin
#################################
if b_plot_update_distribution == True:
    # plot ipromp_aluminum_hold
    plt.figure(50)
    for i in range(4):
        plt.subplot(411+i)
        # plt.plot(ipromp_aluminum_hold.x, test_set[:, i], color='r', linewidth=3, label='ground truth'); plt.legend();
        ipromp_aluminum_hold.promps[i].plot_nUpdated(color='g', legend='updated distribution', via_show=True); plt.legend();
    pl.savefig('./fig/aluminum_hold_imu_post.eps', format='eps');pl.savefig('./fig/aluminum_hold_imu_post.pdf', format='pdf')
    plt.figure(51)
    for i in range(8):
        plt.subplot(421+i)
        # plt.plot(ipromp_aluminum_hold.x, test_set[:, 4+i], color='r', linewidth=3, label='ground truth'); plt.legend();
        ipromp_aluminum_hold.promps[4+i].plot_nUpdated(color='g', legend='updated distribution', via_show=True); plt.legend();
    pl.savefig('./fig/aluminum_hold_emg_post.eps', format='eps');pl.savefig('./fig/aluminum_hold_emg_post.pdf', format='pdf')
    plt.figure(52)
    for i in range(7):
        plt.subplot(711+i)
        # plt.plot(ipromp_aluminum_hold.x, robot_response[:, i], color='r', linewidth=3, label='ground truth'); plt.legend();
        ipromp_aluminum_hold.promps[4+8+i].plot_nUpdated(color='g', legend='updated distribution', via_show=False); plt.legend();
    pl.savefig('./fig/aluminum_hold_pose_post.eps', format='eps');pl.savefig('./fig/aluminum_hold_pose_post.pdf', format='pdf')
    # plot ipromp_spanner_handover
    plt.figure(60)
    for i in range(4):
        plt.subplot(411+i)
        # plt.plot(ipromp_aluminum_hold.x, test_set[:, i], color='r', linewidth=3, label='ground truth'); plt.legend()
        ipromp_spanner_handover.promps[i].plot_nUpdated(color='g', legend='updated distribution', via_show=True); plt.legend();
    pl.savefig('./fig/spanner_handover_imu_post.eps', format='eps');pl.savefig('./fig/spanner_handover_imu_post.pdf', format='pdf')
    plt.figure(61)
    for i in range(8):
        plt.subplot(421+i)
        # plt.plot(ipromp_aluminum_hold.x, test_set[:, 4+i], color='r', linewidth=3, label='ground truth'); plt.legend()
        ipromp_spanner_handover.promps[4+i].plot_nUpdated(color='g', legend='updated distribution', via_show=True); plt.legend();
    pl.savefig('./fig/spanner_handover_emg_post.eps', format='eps');pl.savefig('./fig/spanner_handover_emg_post.pdf', format='pdf')
    plt.figure(62)
    for i in range(7):
        plt.subplot(711+i)
        # plt.plot(ipromp_aluminum_hold.x, robot_response[:, i], color='r', linewidth=3, label='ground truth'); plt.legend()
        ipromp_spanner_handover.promps[4+8+i].plot_nUpdated(color='g', legend='updated distribution', via_show=False); plt.legend();
    pl.savefig('./fig/spanner_handover_pose_post.eps', format='eps');pl.savefig('./fig/spanner_handover_pose_post.pdf', format='pdf')
    # plot ipromp_tape_hold
    plt.figure(70)
    for i in range(4):
        plt.subplot(411+i)
        # plt.plot(ipromp_aluminum_hold.x, test_set[:, i], color='r', linewidth=3, label='ground truth'); plt.legend()
        ipromp_tape_hold.promps[i].plot_nUpdated(color='g', legend='updated distribution', via_show=True); plt.legend();
    pl.savefig('./fig/tape_hold_imu_post.eps', format='eps');pl.savefig('./fig/tape_hold_imu_post.pdf', format='pdf')
    plt.figure(71)
    for i in range(8):
        plt.subplot(421+i)
        # plt.plot(ipromp_tape_hold.x, test_set[:, 4+i], color='r', linewidth=3, label='ground truth'); plt.legend()
        ipromp_tape_hold.promps[4+i].plot_nUpdated(color='g', legend='updated distribution', via_show=True); plt.legend();
    pl.savefig('./fig/tape_hold_emg_post.eps', format='eps');pl.savefig('./fig/tape_hold_emg_post.pdf', format='pdf')
    plt.figure(72)
    for i in range(7):
        plt.subplot(711+i)
        # plt.plot(ipromp_tape_hold.x, robot_response[:, i], color='r', linewidth=3, label='ground truth'); plt.legend()
        ipromp_tape_hold.promps[4+8+i].plot_nUpdated(color='g', legend='updated distribution', via_show=False); plt.legend();
    pl.savefig('./fig/tape_hold_pose_post.eps', format='eps');pl.savefig('./fig/tape_hold_pose_post.pdf', format='pdf')


#################################
# plot the phase distributioin
#################################
if b_plot_phase_distribution == True:
    fig = plt.figure(100)
    ##
    plt.subplot(311)
    h = ipromp_aluminum_hold.alpha
    h.sort()
    hmean = np.mean(h)
    hstd = np.std(h)
    pdf = stats.norm.pdf(h, hmean, hstd)
    pl.hist(h,normed=True,color='b')
    plt.plot(h, pdf, linewidth=5, color='r', marker='o',markersize=10) # including h here is crucial
    candidate = ipromp_aluminum_hold.alpha_candidate(num_alpha_candidate)
    candidate_x = [x['candidate'] for x in candidate]
    prob = [x['prob'] for x in candidate]
    plt.plot(candidate_x, prob, linewidth=0, color='g', marker='o', markersize=14);
    print("the aluminum_hold alpha mean is ", hmean)
    print("the aluminum_hold alpha std is hmean", hstd)
    ##
    plt.subplot(312)
    h = ipromp_spanner_handover.alpha
    h.sort()
    hmean = np.mean(h)
    hstd = np.std(h)
    pdf = stats.norm.pdf(h, hmean, hstd)
    pl.hist(h,normed=True,color='b')
    plt.plot(h, pdf, linewidth=5, color='r', marker='o',markersize=10) # including h here is crucial
    candidate = ipromp_spanner_handover.alpha_candidate(num_alpha_candidate)
    candidate_x = [x['candidate'] for x in candidate]
    prob = [x['prob'] for x in candidate]
    plt.plot(candidate_x, prob, linewidth=0, color='g', marker='o', markersize=14);
    print("the spanner_handover alpha mean is ", hmean)
    print("the spanner_handover alpha std is hmean", hstd)
    ##
    plt.subplot(313)
    h = ipromp_tape_hold.alpha
    h.sort()
    hmean = np.mean(h)
    hstd = np.std(h)
    pdf = stats.norm.pdf(h, hmean, hstd)
    pl.hist(h,normed=True,color='b')
    plt.plot(h, pdf, linewidth=5, color='r', marker='o',markersize=10) # including h here is crucial
    candidate = ipromp_tape_hold.alpha_candidate(num_alpha_candidate)
    candidate_x = [x['candidate'] for x in candidate]
    prob = [x['prob'] for x in candidate]
    plt.plot(candidate_x, prob, linewidth=0, color='g', marker='o', markersize=14);
    print("the tape_hold alpha mean is ", hmean)
    print("the tape_hold alpha std is hmean", hstd)
    pl.savefig('./fig/phase_distribution.eps', format='eps');pl.savefig('./fig/phase_distribution.pdf', format='pdf')

plt.show()