#!/usr/bin/python
# Filename: imu_pose_test.py

from __future__ import print_function

import numpy as np
import matplotlib.pyplot as plt
import ipromps_lib
import scipy.linalg
# from scipy.stats import entropy
# import rospy
import math
from sklearn.externals import joblib
import sys

plt.close('all')    # close all windows

# model parameter
len_normal = 101    # the len of normalized traj, don't change it
num_demos = 10         # number of trajectoreis for training
obs_ratio = 10
num_joints=11
num_basis=31
sigma_basis=0.05
num_samples=101
num_obs_joints=4
# measurement noise
imu_noise = 1
emg_noise = 2
pose_noise = 1
# phase estimation para
num_alpha_candidate = 10
states_refresh_rate = 50.0
nominal_duration = 1.0
# scaling factor for data
sf_imu = 1000.0
sf_emg = 100.0
sf_pose = 0.1

# plot options
b_plot_raw_dateset = False
b_plot_prior_distribution = True
b_plot_update_distribution = False
b_plot_phase_distribution = False



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
pose_meansurement_noise_cov = np.eye(7) * pose_noise
meansurement_noise_cov_full = scipy.linalg.block_diag(imu_meansurement_noise_cov, pose_meansurement_noise_cov)
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
for idx in range(0, num_demos):
    # add demonstration of aluminum_hold
    demo_temp = np.hstack([dataset_aluminum_hold_norm[idx]["imu"]/sf_imu, dataset_aluminum_hold_norm[idx]["pose"]/sf_pose])
    ipromp_aluminum_hold.add_demonstration(demo_temp)
    # add demonstration of spanner_handover
    demo_temp = np.hstack([dataset_spanner_handover_norm[idx]["imu"]/sf_imu, dataset_spanner_handover_norm[idx]["pose"]/sf_pose])
    ipromp_spanner_handover.add_demonstration(demo_temp)
    # add demonstration of tape_hold
    demo_temp = np.hstack([dataset_tape_hold_norm[idx]["imu"]/sf_imu, dataset_tape_hold_norm[idx]["pose"]/sf_pose])
    ipromp_tape_hold.add_demonstration(demo_temp)


################################
# Interaction ProMPs test
################################
# select the testset
# right_id = 0; selected_testset = dataset_aluminum_hold_norm[np.int(sys.argv[1])]
# right_id = 1; selected_testset = dataset_spanner_handover_norm[np.int(sys.argv[1])]
right_id = 2; selected_testset = dataset_tape_hold_norm[1]
# construct the test set
test_set = np.hstack((selected_testset["imu"]/sf_imu, np.zeros([len_normal, 7])))
robot_response = selected_testset["pose"]/sf_pose

# add via/obsys points to update the distribution
for idx in range(obs_ratio):
    ipromp_aluminum_hold.add_viapoint(0.01*idx, test_set[idx, :])
    ipromp_spanner_handover.add_viapoint(0.01*idx, test_set[idx,:])
    ipromp_tape_hold.add_viapoint(0.01*idx, test_set[idx,:])


################################
# the test result
################################
# the model info
print('the number of demonstration is ',num_demos)
# print('the number of observation is ', obs_ratio/100.0)

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
if idx_max_pro == right_id:
    print("OK, you are right!!!")
else:
    print("Sorry, you are wrong!!!, for %d", idx_max_pro)

# if idx_max_pro == 0:
#     print('the obs comes from aluminum_hold')
# elif idx_max_pro == 1:
#     print('the obs comes from spanner_handover')
# elif idx_max_pro == 2:
#     print('the obs comes from tape_hold')


# #################################
# # compute the position error
# #################################
# position_error = None
# predict_robot_response = ipromp_aluminum_hold.generate_trajectory()
# position_error = np.linalg.norm(predict_robot_response[-1,4:7] - robot_response[-1,0:3])
# print('if aluminum_hold, the obs position error is', position_error)
# # elif idx_max_pro == 1:
# predict_robot_response = ipromp_spanner_handover.generate_trajectory()
# position_error = np.linalg.norm(predict_robot_response[-1, 4:7] - robot_response[-1,0:3])
# print('if spanner_handover, the obs position error is', position_error)
# # elif idx_max_pro == 2:
# predict_robot_response = ipromp_tape_hold.generate_trajectory()
# position_error = np.linalg.norm(predict_robot_response[-1, 4:7] - robot_response[-1,0:3])
# print('if tape_hold, the obs position error is', position_error)


#################################
# plot raw data
#################################
if b_plot_raw_dateset == True:
    ## plot the aluminum hold task raw data
    plt.figure(0)
    for ch_ex in range(4):
       plt.subplot(411+ch_ex)
       for idx in range(num_demos):
           plt.plot(range(len(dataset_aluminum_hold[idx]["imu"][:, ch_ex])), dataset_aluminum_hold[idx]["imu"][:, ch_ex]); pltaxis = 0
    plt.figure(2)
    for ch_ex in range(7):
       plt.subplot(711+ch_ex)
       for idx in range(num_demos):
           plt.plot(range(len(dataset_aluminum_hold[idx]["pose"][:, ch_ex])), dataset_aluminum_hold[idx]["pose"][:, ch_ex]); pltaxis = 0
    ## plot the spanner handover task raw data
    plt.figure(10)
    for ch_ex in range(4):
       plt.subplot(411 + ch_ex)
       for idx in range(num_demos):
           plt.plot(range(len(dataset_spanner_handover[idx]["imu"][:, ch_ex])), dataset_spanner_handover[idx]["imu"][:, ch_ex]); pltaxis = 0
    plt.figure(12)
    for ch_ex in range(7):
       plt.subplot(711 + ch_ex)
       for idx in range(num_demos):
           plt.plot(range(len(dataset_spanner_handover[idx]["pose"][:, ch_ex])), dataset_spanner_handover[idx]["pose"][:, ch_ex]); pltaxis = 0
    ## plot the tape hold task raw data
    plt.figure(20)
    for ch_ex in range(4):
       plt.subplot(411 + ch_ex)
       for idx in range(num_demos):
           plt.plot(range(len(dataset_tape_hold[idx]["imu"][:, ch_ex])), dataset_tape_hold[idx]["imu"][:, ch_ex]); pltaxis = 0
    plt.figure(22)
    for ch_ex in range(7):
       plt.subplot(711 + ch_ex)
       for idx in range(num_demos):
           plt.plot(range(len(dataset_tape_hold[idx]["pose"][:, ch_ex])), dataset_tape_hold[idx]["pose"][:, ch_ex]); pltaxis = 0


#################################
# plot the prior distributioin
#################################
if b_plot_prior_distribution == True:
    # plot ipromp_aluminum_hold
    plt.figure(50)
    for i in range(4):
        plt.subplot(411+i)
        ipromp_aluminum_hold.promps[i].plot_prior(color='b', legend='alumnium hold model, imu'); pltaxis = 0
    plt.figure(52)
    for i in range(7):
        plt.subplot(711+i)
        ipromp_aluminum_hold.promps[4+i].plot_prior(color='b', legend='alumnium hold model, pose'); pltaxis = 0
    # plot ipromp_spanner_handover
    plt.figure(60)
    for i in range(4):
        plt.subplot(411+i)
        ipromp_spanner_handover.promps[i].plot_prior(color='b', legend='spanner handover model, imu'); pltaxis = 0
    plt.figure(62)
    for i in range(7):
        plt.subplot(711+i)
        ipromp_spanner_handover.promps[4+i].plot_prior(color='b', legend='spanner handover model, pose'); pltaxis = 0
    # plot ipromp_tape_hold
    plt.figure(70)
    for i in range(4):
        plt.subplot(411+i)
        ipromp_tape_hold.promps[i].plot_prior(color='b', legend='tape hold model, imu'); pltaxis = 0
    plt.figure(72)
    for i in range(7):
        plt.subplot(711+i)
        ipromp_tape_hold.promps[4+i].plot_prior(color='b', legend='tape hold model, pose'); pltaxis = 0


#################################
# plot the updated distributioin
#################################
if b_plot_update_distribution == True:
    # plot ipromp_aluminum_hold
    plt.figure(50)
    for i in range(4):
        plt.subplot(411+i)
        plt.plot(ipromp_aluminum_hold.x, test_set[:, i], color='r', linewidth=3, label='ground truth'); plt.legend();
        ipromp_aluminum_hold.promps[i].plot_nUpdated(color='g', legend='updated distribution', via_show=True); plt.legend();
    plt.figure(52)
    for i in range(7):
        plt.subplot(711+i)
        plt.plot(ipromp_aluminum_hold.x, robot_response[:, i], color='r', linewidth=3, label='ground truth'); plt.legend();
        ipromp_aluminum_hold.promps[4+i].plot_nUpdated(color='g', legend='updated distribution', via_show=True); plt.legend();
    # plot ipromp_spanner_handover
    plt.figure(60)
    for i in range(4):
        plt.subplot(411+i)
        plt.plot(ipromp_aluminum_hold.x, test_set[:, i], color='r', linewidth=3, label='ground truth'); plt.legend()
        ipromp_spanner_handover.promps[i].plot_nUpdated(color='g', legend='updated distribution', via_show=True); plt.legend();
    plt.figure(62)
    for i in range(7):
        plt.subplot(711+i)
        plt.plot(ipromp_aluminum_hold.x, robot_response[:, i], color='r', linewidth=3, label='ground truth'); plt.legend()
        ipromp_spanner_handover.promps[4+i].plot_nUpdated(color='g', legend='updated distribution', via_show=True); plt.legend();
    # plot ipromp_tape_hold
    plt.figure(70)
    for i in range(4):
        plt.subplot(411+i)
        plt.plot(ipromp_tape_hold.x, test_set[:, i], color='r', linewidth=3, label='ground truth'); plt.legend()
        ipromp_tape_hold.promps[i].plot_nUpdated(color='g', legend='updated distribution', via_show=True); plt.legend();
    plt.figure(72)
    for i in range(7):
        plt.subplot(711+i)
        plt.plot(ipromp_tape_hold.x, robot_response[:, i], color='r', linewidth=3, label='ground truth'); plt.legend()
        ipromp_tape_hold.promps[4+i].plot_nUpdated(color='g', legend='updated distribution', via_show=True); plt.legend();

plt.show()