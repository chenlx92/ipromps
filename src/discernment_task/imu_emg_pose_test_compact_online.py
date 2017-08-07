#!/usr/bin/python
# Filename: imu_emg_pose_test_compact_online.py

from __future__ import print_function

import numpy as np
import matplotlib.pyplot as plt
import iprompslib_imu_emg_pose
import scipy.linalg
# from scipy.stats import entropy
# import rospy
from sklearn.externals import joblib
import scipy.stats as stats
import pylab as pl
from scipy.interpolate import griddata

# init
plt.close('all')    # close all windows
len_normal = 101    # the len of normalized traj, don't change it

# model parameter
nrDemo = 20         # number of trajectoreis for training
num_obs = 10
num_alpha_candidate = 10
nominal_duration = 1.0
# measurement noise
imu_noise = 1
emg_noise = 2
pose_noise = 1
num_alpha_candidate = 10
states_refresh_rate = 50.0
# scaling factor for data
sf_imu = 1000.0
sf_emg = 100.0
sf_pose = 0.1

# plot options
b_plot_raw_dateset = False
b_plot_prior_distribution = False
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
# Interaction ProMPs init
#################################
# create a 3 tasks iProMP
ipromp_aluminum_hold = iprompslib_imu_emg_pose.IProMP(num_joints=19, nrBasis=11, sigma=0.05, num_samples=101)
ipromp_spanner_handover = iprompslib_imu_emg_pose.IProMP(num_joints=19, nrBasis=11, sigma=0.05, num_samples=101)
ipromp_tape_hold = iprompslib_imu_emg_pose.IProMP(num_joints=19, nrBasis=11, sigma=0.05, num_samples=101)

# add demostration
for idx in range(nrDemo):
    # train aluminum_hold
    demo_temp = np.hstack([dataset_aluminum_hold_norm[idx]["imu"]/sf_imu, dataset_aluminum_hold_norm[idx]["emg"]/sf_emg])
    demo_temp = np.hstack([demo_temp, dataset_aluminum_hold_norm[idx]["pose"]/sf_pose])
    ipromp_aluminum_hold.add_demonstration(demo_temp)
    # train spanner_handover
    demo_temp = np.hstack([dataset_spanner_handover_norm[idx]["imu"]/sf_imu, dataset_spanner_handover_norm[idx]["emg"]/sf_emg])
    demo_temp = np.hstack([demo_temp, dataset_spanner_handover_norm[idx]["pose"]/sf_pose])
    ipromp_spanner_handover.add_demonstration(demo_temp)
    # tain tape_hold
    demo_temp = np.hstack([dataset_tape_hold_norm[idx]["imu"]/sf_imu, dataset_tape_hold_norm[idx]["emg"]/sf_emg])
    demo_temp = np.hstack([demo_temp, dataset_tape_hold_norm[idx]["pose"]/sf_pose])
    ipromp_tape_hold.add_demonstration(demo_temp)

# model the phase distribution
for i in range(nrDemo):
    # compute the scaling factor alpha for each demonstration
    alpha = (len(dataset_aluminum_hold[i]["imu"]) - 1.0) / states_refresh_rate / nominal_duration
    ipromp_aluminum_hold.add_alpha(alpha)
    ##
    alpha = (len(dataset_spanner_handover[i]["imu"]) - 1.0) / states_refresh_rate / nominal_duration
    ipromp_spanner_handover.add_alpha(alpha)
    ##
    alpha = (len(dataset_tape_hold[i]["imu"]) - 1.0) / states_refresh_rate / nominal_duration
    ipromp_tape_hold.add_alpha(alpha)


################################
# Interaction ProMPs testset
################################
# select the testset
selected_testset = dataset_aluminum_hold[22]

# construct the testset
test_set_temp = np.hstack((selected_testset["imu"][0:num_obs, :]/sf_imu, selected_testset["emg"][0:num_obs, :]/sf_emg))
test_set = np.hstack((test_set_temp, np.zeros([num_obs, 7])))
# robot_response = selected_testset["pose"]


# the measurement noise
imu_meansurement_noise_cov = np.eye(4) * imu_noise
emg_meansurement_noise_cov = np.eye(8) * emg_noise
pose_meansurement_noise_cov = np.eye(7) * pose_noise
meansurement_noise_cov_full = scipy.linalg.block_diag(imu_meansurement_noise_cov, emg_meansurement_noise_cov, pose_meansurement_noise_cov)


################################
# phase estimation
################################
alpha_aluminum_hold = ipromp_aluminum_hold.alpha_candidate(num_alpha_candidate)
id_max_aluminum_hold = ipromp_aluminum_hold.alpha_estimate(alpha_aluminum_hold, test_set, states_refresh_rate, meansurement_noise_cov_full)
alpha_max_aluminum_hold = alpha_aluminum_hold["candidate"][id_max_aluminum_hold]
##
alpha_spanner_handover = ipromp_spanner_handover.alpha_candidate(num_alpha_candidate)
id_max_spanner_handover = ipromp_spanner_handover.alpha_estimate(alpha_spanner_handover, test_set, states_refresh_rate, meansurement_noise_cov_full)
alpha_max_spanner_handover = alpha_spanner_handover["candidate"][id_max_spanner_handover]
##
alpha_tape_hold = ipromp_tape_hold.alpha_candidate(num_alpha_candidate)
id_max_tape_hold = ipromp_tape_hold.alpha_estimate(alpha_tape_hold, test_set, states_refresh_rate, meansurement_noise_cov_full)
alpha_max_tape_hold = alpha_tape_hold["candidate"][id_max_tape_hold]


################################
# add via points
################################
for idx in range(num_obs):
    ipromp_aluminum_hold.add_viapoint(idx/states_refresh_rate/alpha_max_aluminum_hold, test_set[idx, :], meansurement_noise_cov_full)
    ipromp_spanner_handover.add_viapoint(idx/states_refresh_rate/alpha_max_spanner_handover, test_set[idx, :], meansurement_noise_cov_full)
    ipromp_tape_hold.add_viapoint(idx/states_refresh_rate/alpha_max_tape_hold, test_set[idx, :], meansurement_noise_cov_full)


################################
# resume the real traj
################################
id_traj, traj = ipromp_aluminum_hold.gen_predict_traj(alpha_max_tape_hold, states_refresh_rate)
##
plt.figure(200)
for i in range(7):
    plt.subplot(711+i)
    plt.plot(id_traj, traj[:, 12+i])


#################################
# plot raw data
#################################
if b_plot_raw_dateset == True:
    ## plot the aluminum hold task raw data
    plt.figure(0)
    for ch_ex in range(4):
       plt.subplot(411+ch_ex)
       for idx in range(nrDemo):
           plt.plot(range(len(dataset_aluminum_hold[idx]["imu"][:, ch_ex])), dataset_aluminum_hold[idx]["imu"][:, ch_ex])
    plt.figure(1)
    for ch_ex in range(8):
       plt.subplot(421+ch_ex)
       for idx in range(nrDemo):
           plt.plot(range(len(dataset_aluminum_hold[idx]["emg"][:, ch_ex])), dataset_aluminum_hold[idx]["emg"][:, ch_ex])
    plt.figure(2)
    for ch_ex in range(7):
       plt.subplot(711+ch_ex)
       for idx in range(nrDemo):
           plt.plot(range(len(dataset_aluminum_hold[idx]["pose"][:, ch_ex])), dataset_aluminum_hold[idx]["pose"][:, ch_ex])
    ## plot the spanner handover task raw data
    plt.figure(10)
    for ch_ex in range(4):
       plt.subplot(411 + ch_ex)
       for idx in range(nrDemo):
           plt.plot(range(len(dataset_spanner_handover[idx]["imu"][:, ch_ex])), dataset_spanner_handover[idx]["imu"][:, ch_ex])
    plt.figure(11)
    for ch_ex in range(8):
       plt.subplot(421 + ch_ex)
       for idx in range(nrDemo):
           plt.plot(range(len(dataset_spanner_handover[idx]["emg"][:, ch_ex])), dataset_spanner_handover[idx]["emg"][:, ch_ex])
    plt.figure(12)
    for ch_ex in range(7):
       plt.subplot(711 + ch_ex)
       for idx in range(nrDemo):
           plt.plot(range(len(dataset_spanner_handover[idx]["pose"][:, ch_ex])), dataset_spanner_handover[idx]["pose"][:, ch_ex])
    ## plot the tape hold task raw data
    plt.figure(20)
    for ch_ex in range(4):
       plt.subplot(411 + ch_ex)
       for idx in range(nrDemo):
           plt.plot(range(len(dataset_tape_hold[idx]["imu"][:, ch_ex])), dataset_tape_hold[idx]["imu"][:, ch_ex])
    plt.figure(21)
    for ch_ex in range(8):
       plt.subplot(421 + ch_ex)
       for idx in range(nrDemo):
           plt.plot(range(len(dataset_tape_hold[idx]["emg"][:, ch_ex])), dataset_tape_hold[idx]["emg"][:, ch_ex])
    plt.figure(22)
    for ch_ex in range(7):
       plt.subplot(711 + ch_ex)
       for idx in range(nrDemo):
           plt.plot(range(len(dataset_tape_hold[idx]["pose"][:, ch_ex])), dataset_tape_hold[idx]["pose"][:, ch_ex])


#################################
# plot the prior distributioin
#################################
if b_plot_prior_distribution == True:
    # plot ipromp_aluminum_hold
    plt.figure(50)
    for i in range(4):
        plt.subplot(411+i)
        ipromp_aluminum_hold.promps[i].plot(color='g', legend='alumnium hold model, imu');plt.legend()
    plt.figure(51)
    for i in range(8):
        plt.subplot(421+i)
        ipromp_aluminum_hold.promps[4+i].plot(color='g', legend='alumnium hold model, emg');plt.legend()
    plt.figure(52)
    for i in range(7):
        plt.subplot(711+i)
        ipromp_aluminum_hold.promps[4+8+i].plot(color='g', legend='alumnium hold model, pose');plt.legend()
    # plot ipromp_spanner_handover
    plt.figure(60)
    for i in range(4):
        plt.subplot(411+i)
        ipromp_spanner_handover.promps[i].plot(color='g', legend='spanner handover model, imu');plt.legend()
    plt.figure(61)
    for i in range(8):
        plt.subplot(421+i)
        ipromp_spanner_handover.promps[4+i].plot(color='g', legend='spanner handover model, emg');plt.legend()
    plt.figure(62)
    for i in range(7):
        plt.subplot(711+i)
        ipromp_spanner_handover.promps[4+8+i].plot(color='g', legend='spanner handover model, pose');plt.legend()
    # plot ipromp_tape_hold
    plt.figure(70)
    for i in range(4):
        plt.subplot(411+i)
        ipromp_tape_hold.promps[i].plot(color='g', legend='tape hold model, imu');plt.legend()
    plt.figure(71)
    for i in range(8):
        plt.subplot(421+i)
        ipromp_tape_hold.promps[4+i].plot(color='g', legend='tape hold model, emg');plt.legend()
    plt.figure(72)
    for i in range(7):
        plt.subplot(711+i)
        ipromp_tape_hold.promps[4+8+i].plot(color='g', legend='tape hold model, pose');plt.legend()


#################################
# plot the updated distributioin
#################################
if b_plot_update_distribution == True:
    # plot ipromp_aluminum_hold
    plt.figure(50)
    for i in range(4):
        plt.subplot(411+i)
        # plt.plot(ipromp_aluminum_hold.x, test_set[:, i], color='r', linewidth=3, label='ground truth'); plt.legend();
        ipromp_aluminum_hold.promps[i].plot_updated(color='b', legend='updated distribution', via_show=True); plt.legend();
    plt.figure(51)
    for i in range(8):
        plt.subplot(421+i)
        # plt.plot(ipromp_aluminum_hold.x, test_set[:, 4+i], color='r', linewidth=3, label='ground truth'); plt.legend();
        ipromp_aluminum_hold.promps[4+i].plot_updated(color='b', legend='updated distribution', via_show=True); plt.legend();
    plt.figure(52)
    for i in range(7):
        plt.subplot(711+i)
        # plt.plot(ipromp_aluminum_hold.x, robot_response[:, i], color='r', linewidth=3, label='ground truth'); plt.legend();
        ipromp_aluminum_hold.promps[4+8+i].plot_updated(color='b', legend='updated distribution', via_show=False); plt.legend();
    # plot ipromp_spanner_handover
    plt.figure(60)
    for i in range(4):
        plt.subplot(411+i)
        # plt.plot(ipromp_aluminum_hold.x, test_set[:, i], color='r', linewidth=3, label='ground truth'); plt.legend()
        ipromp_spanner_handover.promps[i].plot_updated(color='b', legend='updated distribution', via_show=True); plt.legend();
    plt.figure(61)
    for i in range(8):
        plt.subplot(421+i)
        # plt.plot(ipromp_aluminum_hold.x, test_set[:, 4+i], color='r', linewidth=3, label='ground truth'); plt.legend()
        ipromp_spanner_handover.promps[4+i].plot_updated(color='b', legend='updated distribution', via_show=True); plt.legend();
    plt.figure(62)
    for i in range(7):
        plt.subplot(711+i)
        # plt.plot(ipromp_aluminum_hold.x, robot_response[:, i], color='r', linewidth=3, label='ground truth'); plt.legend()
        ipromp_spanner_handover.promps[4+8+i].plot_updated(color='b', legend='updated distribution', via_show=False); plt.legend();
    # plot ipromp_tape_hold
    plt.figure(70)
    for i in range(4):
        plt.subplot(411+i)
        # plt.plot(ipromp_aluminum_hold.x, test_set[:, i], color='r', linewidth=3, label='ground truth'); plt.legend()
        ipromp_tape_hold.promps[i].plot_updated(color='b', legend='updated distribution', via_show=True); plt.legend();
    plt.figure(71)
    for i in range(8):
        plt.subplot(421+i)
        # plt.plot(ipromp_aluminum_hold.x, test_set[:, 4+i], color='r', linewidth=3, label='ground truth'); plt.legend()
        ipromp_tape_hold.promps[4+i].plot_updated(color='b', legend='updated distribution', via_show=True); plt.legend();
    plt.figure(72)
    for i in range(7):
        plt.subplot(711+i)
        # plt.plot(ipromp_aluminum_hold.x, robot_response[:, i], color='r', linewidth=3, label='ground truth'); plt.legend()
        ipromp_tape_hold.promps[4+8+i].plot_updated(color='b', legend='updated distribution', via_show=False); plt.legend();


#################################
# plot the phase distributioin
#################################
if b_plot_phase_distribution == True:
    plt.figure(100)
    ##
    plt.subplot(311)
    h = ipromp_aluminum_hold.alpha_demo
    h.sort()
    hmean = np.mean(h)
    hstd = np.std(h)
    pdf = stats.norm.pdf(h, hmean, hstd)
    pl.hist(h,normed=True,color='b')
    plt.plot(h, pdf, linewidth=5, color='r', marker='o',markersize=10) # including h here is crucial
    candidate = ipromp_aluminum_hold.alpha_candidate(num_alpha_candidate)
    plt.plot(candidate["candidate"], candidate["prob"], linewidth=0, color='g', marker='o', markersize=14)
    print("the aluminum_hold alpha mean is ", hmean)
    print("the aluminum_hold alpha std is hmean", hstd)
    ##
    plt.subplot(312)
    h = ipromp_spanner_handover.alpha_demo
    h.sort()
    hmean = np.mean(h)
    hstd = np.std(h)
    pdf = stats.norm.pdf(h, hmean, hstd)
    pl.hist(h,normed=True,color='b')
    plt.plot(h, pdf, linewidth=5, color='r', marker='o',markersize=10) # including h here is crucial
    candidate = ipromp_spanner_handover.alpha_candidate(num_alpha_candidate)
    plt.plot(candidate["candidate"], candidate["prob"], linewidth=0, color='g', marker='o', markersize=14)
    print("the spanner_handover alpha mean is ", hmean)
    print("the spanner_handover alpha std is hmean", hstd)
    ##
    plt.subplot(313)
    h = ipromp_tape_hold.alpha_demo
    h.sort()
    hmean = np.mean(h)
    hstd = np.std(h)
    pdf = stats.norm.pdf(h, hmean, hstd)
    pl.hist(h,normed=True,color='b')
    plt.plot(h, pdf, linewidth=5, color='r', marker='o',markersize=10) # including h here is crucial
    candidate = ipromp_tape_hold.alpha_candidate(num_alpha_candidate)
    plt.plot(candidate["candidate"], candidate["prob"], linewidth=0, color='g', marker='o', markersize=14)
    print("the tape_hold alpha mean is ", hmean)
    print("the tape_hold alpha std is hmean", hstd)

plt.show()