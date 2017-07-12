#!/usr/bin/python
# Filename: imu_emg_joint_test.py

from __future__ import print_function

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
# import scipy.signal as signal
import iprompslib_imu_emg_joint
import scipy.linalg
# import rospy

plt.close('all')    # close all windows
len_normal = 101    # the len of normalized traj
nrDemo = 10         # number of trajectoreis for training


##################################################################################
# load EMG date sets
##################################################################################
print('loading EMG data sets of aluminum hold task')
# read emg csv files of aluminum_hold
dir_prefix = '../../recorder/datasets/imu_emg_joint_3_task/aluminum_hold/csv/'
train_set_aluminum_hold_emg_00_pd = pd.read_csv(dir_prefix + '2017-07-09-09-57-38-myo_raw_emg_pub.csv')
train_set_aluminum_hold_emg_01_pd = pd.read_csv(dir_prefix + '2017-07-09-09-57-55-myo_raw_emg_pub.csv')
train_set_aluminum_hold_emg_02_pd = pd.read_csv(dir_prefix + '2017-07-09-09-58-24-myo_raw_emg_pub.csv')
train_set_aluminum_hold_emg_03_pd = pd.read_csv(dir_prefix + '2017-07-09-09-58-51-myo_raw_emg_pub.csv')
train_set_aluminum_hold_emg_04_pd = pd.read_csv(dir_prefix + '2017-07-09-09-59-13-myo_raw_emg_pub.csv')
train_set_aluminum_hold_emg_05_pd = pd.read_csv(dir_prefix + '2017-07-09-10-00-01-myo_raw_emg_pub.csv')
train_set_aluminum_hold_emg_06_pd = pd.read_csv(dir_prefix + '2017-07-09-10-01-08-myo_raw_emg_pub.csv')
train_set_aluminum_hold_emg_07_pd = pd.read_csv(dir_prefix + '2017-07-09-10-01-38-myo_raw_emg_pub.csv')
train_set_aluminum_hold_emg_08_pd = pd.read_csv(dir_prefix + '2017-07-09-10-02-08-myo_raw_emg_pub.csv')
train_set_aluminum_hold_emg_09_pd = pd.read_csv(dir_prefix + '2017-07-09-10-02-29-myo_raw_emg_pub.csv')
train_set_aluminum_hold_emg_10_pd = pd.read_csv(dir_prefix + '2017-07-09-10-02-52-myo_raw_emg_pub.csv')
train_set_aluminum_hold_emg_11_pd = pd.read_csv(dir_prefix + '2017-07-09-10-03-26-myo_raw_emg_pub.csv')
train_set_aluminum_hold_emg_12_pd = pd.read_csv(dir_prefix + '2017-07-09-10-03-43-myo_raw_emg_pub.csv')
train_set_aluminum_hold_emg_13_pd = pd.read_csv(dir_prefix + '2017-07-09-10-04-17-myo_raw_emg_pub.csv')
train_set_aluminum_hold_emg_14_pd = pd.read_csv(dir_prefix + '2017-07-09-10-04-36-myo_raw_emg_pub.csv')
train_set_aluminum_hold_emg_15_pd = pd.read_csv(dir_prefix + '2017-07-09-10-04-57-myo_raw_emg_pub.csv')
train_set_aluminum_hold_emg_16_pd = pd.read_csv(dir_prefix + '2017-07-09-10-05-20-myo_raw_emg_pub.csv')
train_set_aluminum_hold_emg_17_pd = pd.read_csv(dir_prefix + '2017-07-09-10-05-46-myo_raw_emg_pub.csv')
train_set_aluminum_hold_emg_18_pd = pd.read_csv(dir_prefix + '2017-07-09-10-06-07-myo_raw_emg_pub.csv')
train_set_aluminum_hold_emg_19_pd = pd.read_csv(dir_prefix + '2017-07-09-10-06-36-myo_raw_emg_pub.csv')
train_set_aluminum_hold_emg_20_pd = pd.read_csv(dir_prefix + '2017-07-09-10-07-04-myo_raw_emg_pub.csv')
train_set_aluminum_hold_emg_21_pd = pd.read_csv(dir_prefix + '2017-07-09-10-07-23-myo_raw_emg_pub.csv')
train_set_aluminum_hold_emg_22_pd = pd.read_csv(dir_prefix + '2017-07-09-10-07-48-myo_raw_emg_pub.csv')
train_set_aluminum_hold_emg_23_pd = pd.read_csv(dir_prefix + '2017-07-09-10-08-13-myo_raw_emg_pub.csv')
train_set_aluminum_hold_emg_24_pd = pd.read_csv(dir_prefix + '2017-07-09-10-08-39-myo_raw_emg_pub.csv')
train_set_aluminum_hold_emg_25_pd = pd.read_csv(dir_prefix + '2017-07-09-10-08-58-myo_raw_emg_pub.csv')
train_set_aluminum_hold_emg_26_pd = pd.read_csv(dir_prefix + '2017-07-09-10-09-22-myo_raw_emg_pub.csv')
train_set_aluminum_hold_emg_27_pd = pd.read_csv(dir_prefix + '2017-07-09-10-09-54-myo_raw_emg_pub.csv')
train_set_aluminum_hold_emg_28_pd = pd.read_csv(dir_prefix + '2017-07-09-10-10-15-myo_raw_emg_pub.csv')
train_set_aluminum_hold_emg_29_pd = pd.read_csv(dir_prefix + '2017-07-09-10-10-39-myo_raw_emg_pub.csv')
test_set_aluminum_hold_emg_pd = pd.read_csv(dir_prefix + '2017-07-09-10-07-04-myo_raw_emg_pub.csv')
# invert the object to float32 for easy computing
train_set_aluminum_hold_emg_00 = np.float32(train_set_aluminum_hold_emg_00_pd.values[:,5:13])
train_set_aluminum_hold_emg_01 = np.float32(train_set_aluminum_hold_emg_01_pd.values[:,5:13])
train_set_aluminum_hold_emg_02 = np.float32(train_set_aluminum_hold_emg_02_pd.values[:,5:13])
train_set_aluminum_hold_emg_03 = np.float32(train_set_aluminum_hold_emg_03_pd.values[:,5:13])
train_set_aluminum_hold_emg_04 = np.float32(train_set_aluminum_hold_emg_04_pd.values[:,5:13])
train_set_aluminum_hold_emg_05 = np.float32(train_set_aluminum_hold_emg_05_pd.values[:,5:13])
train_set_aluminum_hold_emg_06 = np.float32(train_set_aluminum_hold_emg_06_pd.values[:,5:13])
train_set_aluminum_hold_emg_07 = np.float32(train_set_aluminum_hold_emg_07_pd.values[:,5:13])
train_set_aluminum_hold_emg_08 = np.float32(train_set_aluminum_hold_emg_08_pd.values[:,5:13])
train_set_aluminum_hold_emg_09 = np.float32(train_set_aluminum_hold_emg_09_pd.values[:,5:13])
train_set_aluminum_hold_emg_10 = np.float32(train_set_aluminum_hold_emg_10_pd.values[:,5:13])
train_set_aluminum_hold_emg_11 = np.float32(train_set_aluminum_hold_emg_11_pd.values[:,5:13])
train_set_aluminum_hold_emg_12 = np.float32(train_set_aluminum_hold_emg_12_pd.values[:,5:13])
train_set_aluminum_hold_emg_13 = np.float32(train_set_aluminum_hold_emg_13_pd.values[:,5:13])
train_set_aluminum_hold_emg_14 = np.float32(train_set_aluminum_hold_emg_14_pd.values[:,5:13])
train_set_aluminum_hold_emg_15 = np.float32(train_set_aluminum_hold_emg_15_pd.values[:,5:13])
train_set_aluminum_hold_emg_16 = np.float32(train_set_aluminum_hold_emg_16_pd.values[:,5:13])
train_set_aluminum_hold_emg_17 = np.float32(train_set_aluminum_hold_emg_17_pd.values[:,5:13])
train_set_aluminum_hold_emg_18 = np.float32(train_set_aluminum_hold_emg_18_pd.values[:,5:13])
train_set_aluminum_hold_emg_19 = np.float32(train_set_aluminum_hold_emg_19_pd.values[:,5:13])
train_set_aluminum_hold_emg_20 = np.float32(train_set_aluminum_hold_emg_20_pd.values[:,5:13])
train_set_aluminum_hold_emg_21 = np.float32(train_set_aluminum_hold_emg_21_pd.values[:,5:13])
train_set_aluminum_hold_emg_22 = np.float32(train_set_aluminum_hold_emg_22_pd.values[:,5:13])
train_set_aluminum_hold_emg_23 = np.float32(train_set_aluminum_hold_emg_23_pd.values[:,5:13])
train_set_aluminum_hold_emg_24 = np.float32(train_set_aluminum_hold_emg_24_pd.values[:,5:13])
train_set_aluminum_hold_emg_25 = np.float32(train_set_aluminum_hold_emg_25_pd.values[:,5:13])
train_set_aluminum_hold_emg_26 = np.float32(train_set_aluminum_hold_emg_26_pd.values[:,5:13])
train_set_aluminum_hold_emg_27 = np.float32(train_set_aluminum_hold_emg_27_pd.values[:,5:13])
train_set_aluminum_hold_emg_28 = np.float32(train_set_aluminum_hold_emg_28_pd.values[:,5:13])
train_set_aluminum_hold_emg_29 = np.float32(train_set_aluminum_hold_emg_29_pd.values[:,5:13])
test_set_aluminum_hold_emg = np.float32(test_set_aluminum_hold_emg_pd.values[:,5:13])
#########################################################################################################
print('loading EMG data sets of spanner handover task')
# read emg csv files of spanner_handover
dir_prefix = '../../recorder/datasets/imu_emg_joint_3_task/spanner_handover/csv/'
train_set_spanner_handover_emg_00_pd = pd.read_csv(dir_prefix + '2017-07-09-09-29-42-myo_raw_emg_pub.csv')
train_set_spanner_handover_emg_01_pd = pd.read_csv(dir_prefix + '2017-07-09-09-30-17-myo_raw_emg_pub.csv')
train_set_spanner_handover_emg_02_pd = pd.read_csv(dir_prefix + '2017-07-09-09-30-41-myo_raw_emg_pub.csv')
train_set_spanner_handover_emg_03_pd = pd.read_csv(dir_prefix + '2017-07-09-09-31-03-myo_raw_emg_pub.csv')
train_set_spanner_handover_emg_04_pd = pd.read_csv(dir_prefix + '2017-07-09-09-31-23-myo_raw_emg_pub.csv')
train_set_spanner_handover_emg_05_pd = pd.read_csv(dir_prefix + '2017-07-09-09-31-40-myo_raw_emg_pub.csv')
train_set_spanner_handover_emg_06_pd = pd.read_csv(dir_prefix + '2017-07-09-09-32-24-myo_raw_emg_pub.csv')
train_set_spanner_handover_emg_07_pd = pd.read_csv(dir_prefix + '2017-07-09-09-32-43-myo_raw_emg_pub.csv')
train_set_spanner_handover_emg_08_pd = pd.read_csv(dir_prefix + '2017-07-09-09-33-04-myo_raw_emg_pub.csv')
train_set_spanner_handover_emg_09_pd = pd.read_csv(dir_prefix + '2017-07-09-09-33-23-myo_raw_emg_pub.csv')
train_set_spanner_handover_emg_10_pd = pd.read_csv(dir_prefix + '2017-07-09-09-33-40-myo_raw_emg_pub.csv')
train_set_spanner_handover_emg_11_pd = pd.read_csv(dir_prefix + '2017-07-09-09-34-00-myo_raw_emg_pub.csv')
train_set_spanner_handover_emg_12_pd = pd.read_csv(dir_prefix + '2017-07-09-09-34-32-myo_raw_emg_pub.csv')
train_set_spanner_handover_emg_13_pd = pd.read_csv(dir_prefix + '2017-07-09-09-34-49-myo_raw_emg_pub.csv')
train_set_spanner_handover_emg_14_pd = pd.read_csv(dir_prefix + '2017-07-09-09-35-11-myo_raw_emg_pub.csv')
train_set_spanner_handover_emg_15_pd = pd.read_csv(dir_prefix + '2017-07-09-09-35-49-myo_raw_emg_pub.csv')
train_set_spanner_handover_emg_16_pd = pd.read_csv(dir_prefix + '2017-07-09-09-36-19-myo_raw_emg_pub.csv')
train_set_spanner_handover_emg_17_pd = pd.read_csv(dir_prefix + '2017-07-09-09-36-38-myo_raw_emg_pub.csv')
train_set_spanner_handover_emg_18_pd = pd.read_csv(dir_prefix + '2017-07-09-09-36-58-myo_raw_emg_pub.csv')
train_set_spanner_handover_emg_19_pd = pd.read_csv(dir_prefix + '2017-07-09-09-37-16-myo_raw_emg_pub.csv')
train_set_spanner_handover_emg_20_pd = pd.read_csv(dir_prefix + '2017-07-09-09-37-41-myo_raw_emg_pub.csv')
train_set_spanner_handover_emg_21_pd = pd.read_csv(dir_prefix + '2017-07-09-09-38-02-myo_raw_emg_pub.csv')
train_set_spanner_handover_emg_22_pd = pd.read_csv(dir_prefix + '2017-07-09-09-38-19-myo_raw_emg_pub.csv')
train_set_spanner_handover_emg_23_pd = pd.read_csv(dir_prefix + '2017-07-09-09-38-54-myo_raw_emg_pub.csv')
train_set_spanner_handover_emg_24_pd = pd.read_csv(dir_prefix + '2017-07-09-09-41-39-myo_raw_emg_pub.csv')
train_set_spanner_handover_emg_25_pd = pd.read_csv(dir_prefix + '2017-07-09-09-42-03-myo_raw_emg_pub.csv')
train_set_spanner_handover_emg_26_pd = pd.read_csv(dir_prefix + '2017-07-09-09-42-21-myo_raw_emg_pub.csv')
train_set_spanner_handover_emg_27_pd = pd.read_csv(dir_prefix + '2017-07-09-09-43-00-myo_raw_emg_pub.csv')
train_set_spanner_handover_emg_28_pd = pd.read_csv(dir_prefix + '2017-07-09-09-43-33-myo_raw_emg_pub.csv')
train_set_spanner_handover_emg_29_pd = pd.read_csv(dir_prefix + '2017-07-09-09-43-52-myo_raw_emg_pub.csv')
test_set_spanner_handover_emg_pd = pd.read_csv(dir_prefix + '2017-07-09-09-37-41-myo_raw_emg_pub.csv')
# invert the object to float32 for easy computing
train_set_spanner_handover_emg_00 = np.float32(train_set_spanner_handover_emg_00_pd.values[:,5:13])
train_set_spanner_handover_emg_01 = np.float32(train_set_spanner_handover_emg_01_pd.values[:,5:13])
train_set_spanner_handover_emg_02 = np.float32(train_set_spanner_handover_emg_02_pd.values[:,5:13])
train_set_spanner_handover_emg_03 = np.float32(train_set_spanner_handover_emg_03_pd.values[:,5:13])
train_set_spanner_handover_emg_04 = np.float32(train_set_spanner_handover_emg_04_pd.values[:,5:13])
train_set_spanner_handover_emg_05 = np.float32(train_set_spanner_handover_emg_05_pd.values[:,5:13])
train_set_spanner_handover_emg_06 = np.float32(train_set_spanner_handover_emg_06_pd.values[:,5:13])
train_set_spanner_handover_emg_07 = np.float32(train_set_spanner_handover_emg_07_pd.values[:,5:13])
train_set_spanner_handover_emg_08 = np.float32(train_set_spanner_handover_emg_08_pd.values[:,5:13])
train_set_spanner_handover_emg_09 = np.float32(train_set_spanner_handover_emg_09_pd.values[:,5:13])
train_set_spanner_handover_emg_10 = np.float32(train_set_spanner_handover_emg_10_pd.values[:,5:13])
train_set_spanner_handover_emg_11 = np.float32(train_set_spanner_handover_emg_11_pd.values[:,5:13])
train_set_spanner_handover_emg_12 = np.float32(train_set_spanner_handover_emg_12_pd.values[:,5:13])
train_set_spanner_handover_emg_13 = np.float32(train_set_spanner_handover_emg_13_pd.values[:,5:13])
train_set_spanner_handover_emg_14 = np.float32(train_set_spanner_handover_emg_14_pd.values[:,5:13])
train_set_spanner_handover_emg_15 = np.float32(train_set_spanner_handover_emg_15_pd.values[:,5:13])
train_set_spanner_handover_emg_16 = np.float32(train_set_spanner_handover_emg_16_pd.values[:,5:13])
train_set_spanner_handover_emg_17 = np.float32(train_set_spanner_handover_emg_17_pd.values[:,5:13])
train_set_spanner_handover_emg_18 = np.float32(train_set_spanner_handover_emg_18_pd.values[:,5:13])
train_set_spanner_handover_emg_19 = np.float32(train_set_spanner_handover_emg_19_pd.values[:,5:13])
train_set_spanner_handover_emg_20 = np.float32(train_set_spanner_handover_emg_20_pd.values[:,5:13])
train_set_spanner_handover_emg_21 = np.float32(train_set_spanner_handover_emg_21_pd.values[:,5:13])
train_set_spanner_handover_emg_22 = np.float32(train_set_spanner_handover_emg_22_pd.values[:,5:13])
train_set_spanner_handover_emg_23 = np.float32(train_set_spanner_handover_emg_23_pd.values[:,5:13])
train_set_spanner_handover_emg_24 = np.float32(train_set_spanner_handover_emg_24_pd.values[:,5:13])
train_set_spanner_handover_emg_25 = np.float32(train_set_spanner_handover_emg_25_pd.values[:,5:13])
train_set_spanner_handover_emg_26 = np.float32(train_set_spanner_handover_emg_26_pd.values[:,5:13])
train_set_spanner_handover_emg_27 = np.float32(train_set_spanner_handover_emg_27_pd.values[:,5:13])
train_set_spanner_handover_emg_28 = np.float32(train_set_spanner_handover_emg_28_pd.values[:,5:13])
train_set_spanner_handover_emg_29 = np.float32(train_set_spanner_handover_emg_29_pd.values[:,5:13])
test_set_spanner_handover_emg = np.float32(test_set_spanner_handover_emg_pd.values[:,5:13])
#############################################################################################################
print('loading EMG data sets of tape hold task')
# read emg csv files of tape_hold
dir_prefix = '../../recorder/datasets/imu_emg_joint_3_task/tape_hold/csv/'
train_set_tape_hold_emg_00_pd = pd.read_csv(dir_prefix + '2017-07-09-10-14-01-myo_raw_emg_pub.csv')
train_set_tape_hold_emg_01_pd = pd.read_csv(dir_prefix + '2017-07-09-10-14-22-myo_raw_emg_pub.csv')
train_set_tape_hold_emg_02_pd = pd.read_csv(dir_prefix + '2017-07-09-10-14-43-myo_raw_emg_pub.csv')
train_set_tape_hold_emg_03_pd = pd.read_csv(dir_prefix + '2017-07-09-10-15-47-myo_raw_emg_pub.csv')
train_set_tape_hold_emg_04_pd = pd.read_csv(dir_prefix + '2017-07-09-10-16-04-myo_raw_emg_pub.csv')
train_set_tape_hold_emg_05_pd = pd.read_csv(dir_prefix + '2017-07-09-10-16-22-myo_raw_emg_pub.csv')
train_set_tape_hold_emg_06_pd = pd.read_csv(dir_prefix + '2017-07-09-10-16-40-myo_raw_emg_pub.csv')
train_set_tape_hold_emg_07_pd = pd.read_csv(dir_prefix + '2017-07-09-10-16-56-myo_raw_emg_pub.csv')
train_set_tape_hold_emg_08_pd = pd.read_csv(dir_prefix + '2017-07-09-10-17-13-myo_raw_emg_pub.csv')
train_set_tape_hold_emg_09_pd = pd.read_csv(dir_prefix + '2017-07-09-10-17-48-myo_raw_emg_pub.csv')
train_set_tape_hold_emg_10_pd = pd.read_csv(dir_prefix + '2017-07-09-10-18-04-myo_raw_emg_pub.csv')
train_set_tape_hold_emg_11_pd = pd.read_csv(dir_prefix + '2017-07-09-10-18-33-myo_raw_emg_pub.csv')
train_set_tape_hold_emg_12_pd = pd.read_csv(dir_prefix + '2017-07-09-10-18-49-myo_raw_emg_pub.csv')
train_set_tape_hold_emg_13_pd = pd.read_csv(dir_prefix + '2017-07-09-10-19-05-myo_raw_emg_pub.csv')
train_set_tape_hold_emg_14_pd = pd.read_csv(dir_prefix + '2017-07-09-10-19-22-myo_raw_emg_pub.csv')
train_set_tape_hold_emg_15_pd = pd.read_csv(dir_prefix + '2017-07-09-10-19-51-myo_raw_emg_pub.csv')
train_set_tape_hold_emg_16_pd = pd.read_csv(dir_prefix + '2017-07-09-10-20-08-myo_raw_emg_pub.csv')
train_set_tape_hold_emg_17_pd = pd.read_csv(dir_prefix + '2017-07-09-10-20-26-myo_raw_emg_pub.csv')
train_set_tape_hold_emg_18_pd = pd.read_csv(dir_prefix + '2017-07-09-10-20-44-myo_raw_emg_pub.csv')
train_set_tape_hold_emg_19_pd = pd.read_csv(dir_prefix + '2017-07-09-10-21-03-myo_raw_emg_pub.csv')
train_set_tape_hold_emg_20_pd = pd.read_csv(dir_prefix + '2017-07-09-10-21-22-myo_raw_emg_pub.csv')
train_set_tape_hold_emg_21_pd = pd.read_csv(dir_prefix + '2017-07-09-10-21-44-myo_raw_emg_pub.csv')
train_set_tape_hold_emg_22_pd = pd.read_csv(dir_prefix + '2017-07-09-10-22-00-myo_raw_emg_pub.csv')
train_set_tape_hold_emg_23_pd = pd.read_csv(dir_prefix + '2017-07-09-10-22-17-myo_raw_emg_pub.csv')
train_set_tape_hold_emg_24_pd = pd.read_csv(dir_prefix + '2017-07-09-10-22-37-myo_raw_emg_pub.csv')
train_set_tape_hold_emg_25_pd = pd.read_csv(dir_prefix + '2017-07-09-10-23-07-myo_raw_emg_pub.csv')
train_set_tape_hold_emg_26_pd = pd.read_csv(dir_prefix + '2017-07-09-10-23-29-myo_raw_emg_pub.csv')
train_set_tape_hold_emg_27_pd = pd.read_csv(dir_prefix + '2017-07-09-10-23-48-myo_raw_emg_pub.csv')
train_set_tape_hold_emg_28_pd = pd.read_csv(dir_prefix + '2017-07-09-10-24-21-myo_raw_emg_pub.csv')
train_set_tape_hold_emg_29_pd = pd.read_csv(dir_prefix + '2017-07-09-10-24-46-myo_raw_emg_pub.csv')
test_set_tape_hold_emg_pd = pd.read_csv(dir_prefix + '2017-07-09-10-21-22-myo_raw_emg_pub.csv')
# invert the object to float32 for easy computing
train_set_tape_hold_emg_00 = np.float32(train_set_tape_hold_emg_00_pd.values[:,5:13])
train_set_tape_hold_emg_01 = np.float32(train_set_tape_hold_emg_01_pd.values[:,5:13])
train_set_tape_hold_emg_02 = np.float32(train_set_tape_hold_emg_02_pd.values[:,5:13])
train_set_tape_hold_emg_03 = np.float32(train_set_tape_hold_emg_03_pd.values[:,5:13])
train_set_tape_hold_emg_04 = np.float32(train_set_tape_hold_emg_04_pd.values[:,5:13])
train_set_tape_hold_emg_05 = np.float32(train_set_tape_hold_emg_05_pd.values[:,5:13])
train_set_tape_hold_emg_06 = np.float32(train_set_tape_hold_emg_06_pd.values[:,5:13])
train_set_tape_hold_emg_07 = np.float32(train_set_tape_hold_emg_07_pd.values[:,5:13])
train_set_tape_hold_emg_08 = np.float32(train_set_tape_hold_emg_08_pd.values[:,5:13])
train_set_tape_hold_emg_09 = np.float32(train_set_tape_hold_emg_09_pd.values[:,5:13])
train_set_tape_hold_emg_10 = np.float32(train_set_tape_hold_emg_10_pd.values[:,5:13])
train_set_tape_hold_emg_11 = np.float32(train_set_tape_hold_emg_11_pd.values[:,5:13])
train_set_tape_hold_emg_12 = np.float32(train_set_tape_hold_emg_12_pd.values[:,5:13])
train_set_tape_hold_emg_13 = np.float32(train_set_tape_hold_emg_13_pd.values[:,5:13])
train_set_tape_hold_emg_14 = np.float32(train_set_tape_hold_emg_14_pd.values[:,5:13])
train_set_tape_hold_emg_15 = np.float32(train_set_tape_hold_emg_15_pd.values[:,5:13])
train_set_tape_hold_emg_16 = np.float32(train_set_tape_hold_emg_16_pd.values[:,5:13])
train_set_tape_hold_emg_17 = np.float32(train_set_tape_hold_emg_17_pd.values[:,5:13])
train_set_tape_hold_emg_18 = np.float32(train_set_tape_hold_emg_18_pd.values[:,5:13])
train_set_tape_hold_emg_19 = np.float32(train_set_tape_hold_emg_19_pd.values[:,5:13])
train_set_tape_hold_emg_20 = np.float32(train_set_tape_hold_emg_20_pd.values[:,5:13])
train_set_tape_hold_emg_21 = np.float32(train_set_tape_hold_emg_21_pd.values[:,5:13])
train_set_tape_hold_emg_22 = np.float32(train_set_tape_hold_emg_22_pd.values[:,5:13])
train_set_tape_hold_emg_23 = np.float32(train_set_tape_hold_emg_23_pd.values[:,5:13])
train_set_tape_hold_emg_24 = np.float32(train_set_tape_hold_emg_24_pd.values[:,5:13])
train_set_tape_hold_emg_25 = np.float32(train_set_tape_hold_emg_25_pd.values[:,5:13])
train_set_tape_hold_emg_26 = np.float32(train_set_tape_hold_emg_26_pd.values[:,5:13])
train_set_tape_hold_emg_27 = np.float32(train_set_tape_hold_emg_27_pd.values[:,5:13])
train_set_tape_hold_emg_28 = np.float32(train_set_tape_hold_emg_28_pd.values[:,5:13])
train_set_tape_hold_emg_29 = np.float32(train_set_tape_hold_emg_29_pd.values[:,5:13])
test_set_tape_hold_emg = np.float32(test_set_tape_hold_emg_pd.values[:,5:13])


##################################################################################
# load IMU date sets
##################################################################################
print('loading IMU data sets of aluminum hold task')
# read imu csv files of aluminum_hold
dir_prefix = '../../recorder/datasets/imu_emg_joint_3_task/aluminum_hold/csv/'
train_set_aluminum_hold_imu_00_pd = pd.read_csv(dir_prefix + '2017-07-09-09-57-38-myo_raw_imu_pub (copy).csv')
train_set_aluminum_hold_imu_01_pd = pd.read_csv(dir_prefix + '2017-07-09-09-57-55-myo_raw_imu_pub (copy).csv')
train_set_aluminum_hold_imu_02_pd = pd.read_csv(dir_prefix + '2017-07-09-09-58-24-myo_raw_imu_pub (copy).csv')
train_set_aluminum_hold_imu_03_pd = pd.read_csv(dir_prefix + '2017-07-09-09-58-51-myo_raw_imu_pub (copy).csv')
train_set_aluminum_hold_imu_04_pd = pd.read_csv(dir_prefix + '2017-07-09-09-59-13-myo_raw_imu_pub (copy).csv')
train_set_aluminum_hold_imu_05_pd = pd.read_csv(dir_prefix + '2017-07-09-10-00-01-myo_raw_imu_pub (copy).csv')
train_set_aluminum_hold_imu_06_pd = pd.read_csv(dir_prefix + '2017-07-09-10-01-08-myo_raw_imu_pub (copy).csv')
train_set_aluminum_hold_imu_07_pd = pd.read_csv(dir_prefix + '2017-07-09-10-01-38-myo_raw_imu_pub (copy).csv')
train_set_aluminum_hold_imu_08_pd = pd.read_csv(dir_prefix + '2017-07-09-10-02-08-myo_raw_imu_pub (copy).csv')
train_set_aluminum_hold_imu_09_pd = pd.read_csv(dir_prefix + '2017-07-09-10-02-29-myo_raw_imu_pub (copy).csv')
train_set_aluminum_hold_imu_10_pd = pd.read_csv(dir_prefix + '2017-07-09-10-02-52-myo_raw_imu_pub (copy).csv')
train_set_aluminum_hold_imu_11_pd = pd.read_csv(dir_prefix + '2017-07-09-10-03-26-myo_raw_imu_pub (copy).csv')
train_set_aluminum_hold_imu_12_pd = pd.read_csv(dir_prefix + '2017-07-09-10-03-43-myo_raw_imu_pub (copy).csv')
train_set_aluminum_hold_imu_13_pd = pd.read_csv(dir_prefix + '2017-07-09-10-04-17-myo_raw_imu_pub (copy).csv')
train_set_aluminum_hold_imu_14_pd = pd.read_csv(dir_prefix + '2017-07-09-10-04-36-myo_raw_imu_pub (copy).csv')
train_set_aluminum_hold_imu_15_pd = pd.read_csv(dir_prefix + '2017-07-09-10-04-57-myo_raw_imu_pub (copy).csv')
train_set_aluminum_hold_imu_16_pd = pd.read_csv(dir_prefix + '2017-07-09-10-05-20-myo_raw_imu_pub (copy).csv')
train_set_aluminum_hold_imu_17_pd = pd.read_csv(dir_prefix + '2017-07-09-10-05-46-myo_raw_imu_pub (copy).csv')
train_set_aluminum_hold_imu_18_pd = pd.read_csv(dir_prefix + '2017-07-09-10-06-07-myo_raw_imu_pub (copy).csv')
train_set_aluminum_hold_imu_19_pd = pd.read_csv(dir_prefix + '2017-07-09-10-06-36-myo_raw_imu_pub (copy).csv')
train_set_aluminum_hold_imu_20_pd = pd.read_csv(dir_prefix + '2017-07-09-10-07-04-myo_raw_imu_pub (copy).csv')
train_set_aluminum_hold_imu_21_pd = pd.read_csv(dir_prefix + '2017-07-09-10-07-23-myo_raw_imu_pub (copy).csv')
train_set_aluminum_hold_imu_22_pd = pd.read_csv(dir_prefix + '2017-07-09-10-07-48-myo_raw_imu_pub (copy).csv')
train_set_aluminum_hold_imu_23_pd = pd.read_csv(dir_prefix + '2017-07-09-10-08-13-myo_raw_imu_pub (copy).csv')
train_set_aluminum_hold_imu_24_pd = pd.read_csv(dir_prefix + '2017-07-09-10-08-39-myo_raw_imu_pub (copy).csv')
train_set_aluminum_hold_imu_25_pd = pd.read_csv(dir_prefix + '2017-07-09-10-08-58-myo_raw_imu_pub (copy).csv')
train_set_aluminum_hold_imu_26_pd = pd.read_csv(dir_prefix + '2017-07-09-10-09-22-myo_raw_imu_pub (copy).csv')
train_set_aluminum_hold_imu_27_pd = pd.read_csv(dir_prefix + '2017-07-09-10-09-54-myo_raw_imu_pub (copy).csv')
train_set_aluminum_hold_imu_28_pd = pd.read_csv(dir_prefix + '2017-07-09-10-10-15-myo_raw_imu_pub (copy).csv')
train_set_aluminum_hold_imu_29_pd = pd.read_csv(dir_prefix + '2017-07-09-10-10-39-myo_raw_imu_pub (copy).csv')
test_set_aluminum_hold_imu_pd = pd.read_csv(dir_prefix + '2017-07-09-10-07-04-myo_raw_imu_pub (copy).csv')
# invert the object to float32 for easy computing
train_set_aluminum_hold_imu_00 = np.float32(train_set_aluminum_hold_imu_00_pd.values[:,:])
train_set_aluminum_hold_imu_01 = np.float32(train_set_aluminum_hold_imu_01_pd.values[:,:])
train_set_aluminum_hold_imu_02 = np.float32(train_set_aluminum_hold_imu_02_pd.values[:,:])
train_set_aluminum_hold_imu_03 = np.float32(train_set_aluminum_hold_imu_03_pd.values[:,:])
train_set_aluminum_hold_imu_04 = np.float32(train_set_aluminum_hold_imu_04_pd.values[:,:])
train_set_aluminum_hold_imu_05 = np.float32(train_set_aluminum_hold_imu_05_pd.values[:,:])
train_set_aluminum_hold_imu_06 = np.float32(train_set_aluminum_hold_imu_06_pd.values[:,:])
train_set_aluminum_hold_imu_07 = np.float32(train_set_aluminum_hold_imu_07_pd.values[:,:])
train_set_aluminum_hold_imu_08 = np.float32(train_set_aluminum_hold_imu_08_pd.values[:,:])
train_set_aluminum_hold_imu_09 = np.float32(train_set_aluminum_hold_imu_09_pd.values[:,:])
train_set_aluminum_hold_imu_10 = np.float32(train_set_aluminum_hold_imu_10_pd.values[:,:])
train_set_aluminum_hold_imu_11 = np.float32(train_set_aluminum_hold_imu_11_pd.values[:,:])
train_set_aluminum_hold_imu_12 = np.float32(train_set_aluminum_hold_imu_12_pd.values[:,:])
train_set_aluminum_hold_imu_13 = np.float32(train_set_aluminum_hold_imu_13_pd.values[:,:])
train_set_aluminum_hold_imu_14 = np.float32(train_set_aluminum_hold_imu_14_pd.values[:,:])
train_set_aluminum_hold_imu_15 = np.float32(train_set_aluminum_hold_imu_15_pd.values[:,:])
train_set_aluminum_hold_imu_16 = np.float32(train_set_aluminum_hold_imu_16_pd.values[:,:])
train_set_aluminum_hold_imu_17 = np.float32(train_set_aluminum_hold_imu_17_pd.values[:,:])
train_set_aluminum_hold_imu_18 = np.float32(train_set_aluminum_hold_imu_18_pd.values[:,:])
train_set_aluminum_hold_imu_19 = np.float32(train_set_aluminum_hold_imu_19_pd.values[:,:])
train_set_aluminum_hold_imu_20 = np.float32(train_set_aluminum_hold_imu_20_pd.values[:,:])
train_set_aluminum_hold_imu_21 = np.float32(train_set_aluminum_hold_imu_21_pd.values[:,:])
train_set_aluminum_hold_imu_22 = np.float32(train_set_aluminum_hold_imu_22_pd.values[:,:])
train_set_aluminum_hold_imu_23 = np.float32(train_set_aluminum_hold_imu_23_pd.values[:,:])
train_set_aluminum_hold_imu_24 = np.float32(train_set_aluminum_hold_imu_24_pd.values[:,:])
train_set_aluminum_hold_imu_25 = np.float32(train_set_aluminum_hold_imu_25_pd.values[:,:])
train_set_aluminum_hold_imu_26 = np.float32(train_set_aluminum_hold_imu_26_pd.values[:,:])
train_set_aluminum_hold_imu_27 = np.float32(train_set_aluminum_hold_imu_27_pd.values[:,:])
train_set_aluminum_hold_imu_28 = np.float32(train_set_aluminum_hold_imu_28_pd.values[:,:])
train_set_aluminum_hold_imu_29 = np.float32(train_set_aluminum_hold_imu_29_pd.values[:,:])
test_set_aluminum_hold_imu = np.float32(test_set_aluminum_hold_imu_pd.values[:,:])
#########################################################################################################
print('loading IMU data sets of spanner handover task')
# read imu csv files of spanner_handover
dir_prefix = '../../recorder/datasets/imu_emg_joint_3_task/spanner_handover/csv/'
train_set_spanner_handover_imu_00_pd = pd.read_csv(dir_prefix + '2017-07-09-09-29-42-myo_raw_imu_pub (copy).csv')
train_set_spanner_handover_imu_01_pd = pd.read_csv(dir_prefix + '2017-07-09-09-30-17-myo_raw_imu_pub (copy).csv')
train_set_spanner_handover_imu_02_pd = pd.read_csv(dir_prefix + '2017-07-09-09-30-41-myo_raw_imu_pub (copy).csv')
train_set_spanner_handover_imu_03_pd = pd.read_csv(dir_prefix + '2017-07-09-09-31-03-myo_raw_imu_pub (copy).csv')
train_set_spanner_handover_imu_04_pd = pd.read_csv(dir_prefix + '2017-07-09-09-31-23-myo_raw_imu_pub (copy).csv')
train_set_spanner_handover_imu_05_pd = pd.read_csv(dir_prefix + '2017-07-09-09-31-40-myo_raw_imu_pub (copy).csv')
train_set_spanner_handover_imu_06_pd = pd.read_csv(dir_prefix + '2017-07-09-09-32-24-myo_raw_imu_pub (copy).csv')
train_set_spanner_handover_imu_07_pd = pd.read_csv(dir_prefix + '2017-07-09-09-32-43-myo_raw_imu_pub (copy).csv')
train_set_spanner_handover_imu_08_pd = pd.read_csv(dir_prefix + '2017-07-09-09-33-04-myo_raw_imu_pub (copy).csv')
train_set_spanner_handover_imu_09_pd = pd.read_csv(dir_prefix + '2017-07-09-09-33-23-myo_raw_imu_pub (copy).csv')
train_set_spanner_handover_imu_10_pd = pd.read_csv(dir_prefix + '2017-07-09-09-33-40-myo_raw_imu_pub (copy).csv')
train_set_spanner_handover_imu_11_pd = pd.read_csv(dir_prefix + '2017-07-09-09-34-00-myo_raw_imu_pub (copy).csv')
train_set_spanner_handover_imu_12_pd = pd.read_csv(dir_prefix + '2017-07-09-09-34-32-myo_raw_imu_pub (copy).csv')
train_set_spanner_handover_imu_13_pd = pd.read_csv(dir_prefix + '2017-07-09-09-34-49-myo_raw_imu_pub (copy).csv')
train_set_spanner_handover_imu_14_pd = pd.read_csv(dir_prefix + '2017-07-09-09-35-11-myo_raw_imu_pub (copy).csv')
train_set_spanner_handover_imu_15_pd = pd.read_csv(dir_prefix + '2017-07-09-09-35-49-myo_raw_imu_pub (copy).csv')
train_set_spanner_handover_imu_16_pd = pd.read_csv(dir_prefix + '2017-07-09-09-36-19-myo_raw_imu_pub (copy).csv')
train_set_spanner_handover_imu_17_pd = pd.read_csv(dir_prefix + '2017-07-09-09-36-38-myo_raw_imu_pub (copy).csv')
train_set_spanner_handover_imu_18_pd = pd.read_csv(dir_prefix + '2017-07-09-09-36-58-myo_raw_imu_pub (copy).csv')
train_set_spanner_handover_imu_19_pd = pd.read_csv(dir_prefix + '2017-07-09-09-37-16-myo_raw_imu_pub (copy).csv')
train_set_spanner_handover_imu_20_pd = pd.read_csv(dir_prefix + '2017-07-09-09-37-41-myo_raw_imu_pub (copy).csv')
train_set_spanner_handover_imu_21_pd = pd.read_csv(dir_prefix + '2017-07-09-09-38-02-myo_raw_imu_pub (copy).csv')
train_set_spanner_handover_imu_22_pd = pd.read_csv(dir_prefix + '2017-07-09-09-38-19-myo_raw_imu_pub (copy).csv')
train_set_spanner_handover_imu_23_pd = pd.read_csv(dir_prefix + '2017-07-09-09-38-54-myo_raw_imu_pub (copy).csv')
train_set_spanner_handover_imu_24_pd = pd.read_csv(dir_prefix + '2017-07-09-09-41-39-myo_raw_imu_pub (copy).csv')
train_set_spanner_handover_imu_25_pd = pd.read_csv(dir_prefix + '2017-07-09-09-42-03-myo_raw_imu_pub (copy).csv')
train_set_spanner_handover_imu_26_pd = pd.read_csv(dir_prefix + '2017-07-09-09-42-21-myo_raw_imu_pub (copy).csv')
train_set_spanner_handover_imu_27_pd = pd.read_csv(dir_prefix + '2017-07-09-09-43-00-myo_raw_imu_pub (copy).csv')
train_set_spanner_handover_imu_28_pd = pd.read_csv(dir_prefix + '2017-07-09-09-43-33-myo_raw_imu_pub (copy).csv')
train_set_spanner_handover_imu_29_pd = pd.read_csv(dir_prefix + '2017-07-09-09-43-52-myo_raw_imu_pub (copy).csv')
test_set_spanner_handover_imu_pd = pd.read_csv(dir_prefix + '2017-07-09-09-37-41-myo_raw_imu_pub (copy).csv')
# invert the object to float32 for easy computing
train_set_spanner_handover_imu_00 = np.float32(train_set_spanner_handover_imu_00_pd.values[:,:])
train_set_spanner_handover_imu_01 = np.float32(train_set_spanner_handover_imu_01_pd.values[:,:])
train_set_spanner_handover_imu_02 = np.float32(train_set_spanner_handover_imu_02_pd.values[:,:])
train_set_spanner_handover_imu_03 = np.float32(train_set_spanner_handover_imu_03_pd.values[:,:])
train_set_spanner_handover_imu_04 = np.float32(train_set_spanner_handover_imu_04_pd.values[:,:])
train_set_spanner_handover_imu_05 = np.float32(train_set_spanner_handover_imu_05_pd.values[:,:])
train_set_spanner_handover_imu_06 = np.float32(train_set_spanner_handover_imu_06_pd.values[:,:])
train_set_spanner_handover_imu_07 = np.float32(train_set_spanner_handover_imu_07_pd.values[:,:])
train_set_spanner_handover_imu_08 = np.float32(train_set_spanner_handover_imu_08_pd.values[:,:])
train_set_spanner_handover_imu_09 = np.float32(train_set_spanner_handover_imu_09_pd.values[:,:])
train_set_spanner_handover_imu_10 = np.float32(train_set_spanner_handover_imu_10_pd.values[:,:])
train_set_spanner_handover_imu_11 = np.float32(train_set_spanner_handover_imu_11_pd.values[:,:])
train_set_spanner_handover_imu_12 = np.float32(train_set_spanner_handover_imu_12_pd.values[:,:])
train_set_spanner_handover_imu_13 = np.float32(train_set_spanner_handover_imu_13_pd.values[:,:])
train_set_spanner_handover_imu_14 = np.float32(train_set_spanner_handover_imu_14_pd.values[:,:])
train_set_spanner_handover_imu_15 = np.float32(train_set_spanner_handover_imu_15_pd.values[:,:])
train_set_spanner_handover_imu_16 = np.float32(train_set_spanner_handover_imu_16_pd.values[:,:])
train_set_spanner_handover_imu_17 = np.float32(train_set_spanner_handover_imu_17_pd.values[:,:])
train_set_spanner_handover_imu_18 = np.float32(train_set_spanner_handover_imu_18_pd.values[:,:])
train_set_spanner_handover_imu_19 = np.float32(train_set_spanner_handover_imu_19_pd.values[:,:])
train_set_spanner_handover_imu_20 = np.float32(train_set_spanner_handover_imu_20_pd.values[:,:])
train_set_spanner_handover_imu_21 = np.float32(train_set_spanner_handover_imu_21_pd.values[:,:])
train_set_spanner_handover_imu_22 = np.float32(train_set_spanner_handover_imu_22_pd.values[:,:])
train_set_spanner_handover_imu_23 = np.float32(train_set_spanner_handover_imu_23_pd.values[:,:])
train_set_spanner_handover_imu_24 = np.float32(train_set_spanner_handover_imu_24_pd.values[:,:])
train_set_spanner_handover_imu_25 = np.float32(train_set_spanner_handover_imu_25_pd.values[:,:])
train_set_spanner_handover_imu_26 = np.float32(train_set_spanner_handover_imu_26_pd.values[:,:])
train_set_spanner_handover_imu_27 = np.float32(train_set_spanner_handover_imu_27_pd.values[:,:])
train_set_spanner_handover_imu_28 = np.float32(train_set_spanner_handover_imu_28_pd.values[:,:])
train_set_spanner_handover_imu_29 = np.float32(train_set_spanner_handover_imu_29_pd.values[:,:])
test_set_spanner_handover_imu = np.float32(test_set_spanner_handover_imu_pd.values[:,:])
#############################################################################################################
print('loading IMU data sets of tape hold task')
# read imu csv files of tape_hold
dir_prefix = '../../recorder/datasets/imu_emg_joint_3_task/tape_hold/csv/'
train_set_tape_hold_imu_00_pd = pd.read_csv(dir_prefix + '2017-07-09-10-14-01-myo_raw_imu_pub (copy).csv')
train_set_tape_hold_imu_01_pd = pd.read_csv(dir_prefix + '2017-07-09-10-14-22-myo_raw_imu_pub (copy).csv')
train_set_tape_hold_imu_02_pd = pd.read_csv(dir_prefix + '2017-07-09-10-14-43-myo_raw_imu_pub (copy).csv')
train_set_tape_hold_imu_03_pd = pd.read_csv(dir_prefix + '2017-07-09-10-15-47-myo_raw_imu_pub (copy).csv')
train_set_tape_hold_imu_04_pd = pd.read_csv(dir_prefix + '2017-07-09-10-16-04-myo_raw_imu_pub (copy).csv')
train_set_tape_hold_imu_05_pd = pd.read_csv(dir_prefix + '2017-07-09-10-16-22-myo_raw_imu_pub (copy).csv')
train_set_tape_hold_imu_06_pd = pd.read_csv(dir_prefix + '2017-07-09-10-16-40-myo_raw_imu_pub (copy).csv')
train_set_tape_hold_imu_07_pd = pd.read_csv(dir_prefix + '2017-07-09-10-16-56-myo_raw_imu_pub (copy).csv')
train_set_tape_hold_imu_08_pd = pd.read_csv(dir_prefix + '2017-07-09-10-17-13-myo_raw_imu_pub (copy).csv')
train_set_tape_hold_imu_09_pd = pd.read_csv(dir_prefix + '2017-07-09-10-17-48-myo_raw_imu_pub (copy).csv')
train_set_tape_hold_imu_10_pd = pd.read_csv(dir_prefix + '2017-07-09-10-18-04-myo_raw_imu_pub (copy).csv')
train_set_tape_hold_imu_11_pd = pd.read_csv(dir_prefix + '2017-07-09-10-18-33-myo_raw_imu_pub (copy).csv')
train_set_tape_hold_imu_12_pd = pd.read_csv(dir_prefix + '2017-07-09-10-18-49-myo_raw_imu_pub (copy).csv')
train_set_tape_hold_imu_13_pd = pd.read_csv(dir_prefix + '2017-07-09-10-19-05-myo_raw_imu_pub (copy).csv')
train_set_tape_hold_imu_14_pd = pd.read_csv(dir_prefix + '2017-07-09-10-19-22-myo_raw_imu_pub (copy).csv')
train_set_tape_hold_imu_15_pd = pd.read_csv(dir_prefix + '2017-07-09-10-19-51-myo_raw_imu_pub (copy).csv')
train_set_tape_hold_imu_16_pd = pd.read_csv(dir_prefix + '2017-07-09-10-20-08-myo_raw_imu_pub (copy).csv')
train_set_tape_hold_imu_17_pd = pd.read_csv(dir_prefix + '2017-07-09-10-20-26-myo_raw_imu_pub (copy).csv')
train_set_tape_hold_imu_18_pd = pd.read_csv(dir_prefix + '2017-07-09-10-20-44-myo_raw_imu_pub (copy).csv')
train_set_tape_hold_imu_19_pd = pd.read_csv(dir_prefix + '2017-07-09-10-21-03-myo_raw_imu_pub (copy).csv')
train_set_tape_hold_imu_20_pd = pd.read_csv(dir_prefix + '2017-07-09-10-21-22-myo_raw_imu_pub (copy).csv')
train_set_tape_hold_imu_21_pd = pd.read_csv(dir_prefix + '2017-07-09-10-21-44-myo_raw_imu_pub (copy).csv')
train_set_tape_hold_imu_22_pd = pd.read_csv(dir_prefix + '2017-07-09-10-22-00-myo_raw_imu_pub (copy).csv')
train_set_tape_hold_imu_23_pd = pd.read_csv(dir_prefix + '2017-07-09-10-22-17-myo_raw_imu_pub (copy).csv')
train_set_tape_hold_imu_24_pd = pd.read_csv(dir_prefix + '2017-07-09-10-22-37-myo_raw_imu_pub (copy).csv')
train_set_tape_hold_imu_25_pd = pd.read_csv(dir_prefix + '2017-07-09-10-23-07-myo_raw_imu_pub (copy).csv')
train_set_tape_hold_imu_26_pd = pd.read_csv(dir_prefix + '2017-07-09-10-23-29-myo_raw_imu_pub (copy).csv')
train_set_tape_hold_imu_27_pd = pd.read_csv(dir_prefix + '2017-07-09-10-23-48-myo_raw_imu_pub (copy).csv')
train_set_tape_hold_imu_28_pd = pd.read_csv(dir_prefix + '2017-07-09-10-24-21-myo_raw_imu_pub (copy).csv')
train_set_tape_hold_imu_29_pd = pd.read_csv(dir_prefix + '2017-07-09-10-24-46-myo_raw_imu_pub (copy).csv')
test_set_tape_hold_imu_pd = pd.read_csv(dir_prefix + '2017-07-09-10-21-22-myo_raw_imu_pub (copy).csv')
# invert the object to float32 for easy computing
train_set_tape_hold_imu_00 = np.float32(train_set_tape_hold_imu_00_pd.values[:,:])
train_set_tape_hold_imu_01 = np.float32(train_set_tape_hold_imu_01_pd.values[:,:])
train_set_tape_hold_imu_02 = np.float32(train_set_tape_hold_imu_02_pd.values[:,:])
train_set_tape_hold_imu_03 = np.float32(train_set_tape_hold_imu_03_pd.values[:,:])
train_set_tape_hold_imu_04 = np.float32(train_set_tape_hold_imu_04_pd.values[:,:])
train_set_tape_hold_imu_05 = np.float32(train_set_tape_hold_imu_05_pd.values[:,:])
train_set_tape_hold_imu_06 = np.float32(train_set_tape_hold_imu_06_pd.values[:,:])
train_set_tape_hold_imu_07 = np.float32(train_set_tape_hold_imu_07_pd.values[:,:])
train_set_tape_hold_imu_08 = np.float32(train_set_tape_hold_imu_08_pd.values[:,:])
train_set_tape_hold_imu_09 = np.float32(train_set_tape_hold_imu_09_pd.values[:,:])
train_set_tape_hold_imu_10 = np.float32(train_set_tape_hold_imu_10_pd.values[:,:])
train_set_tape_hold_imu_11 = np.float32(train_set_tape_hold_imu_11_pd.values[:,:])
train_set_tape_hold_imu_12 = np.float32(train_set_tape_hold_imu_12_pd.values[:,:])
train_set_tape_hold_imu_13 = np.float32(train_set_tape_hold_imu_13_pd.values[:,:])
train_set_tape_hold_imu_14 = np.float32(train_set_tape_hold_imu_14_pd.values[:,:])
train_set_tape_hold_imu_15 = np.float32(train_set_tape_hold_imu_15_pd.values[:,:])
train_set_tape_hold_imu_16 = np.float32(train_set_tape_hold_imu_16_pd.values[:,:])
train_set_tape_hold_imu_17 = np.float32(train_set_tape_hold_imu_17_pd.values[:,:])
train_set_tape_hold_imu_18 = np.float32(train_set_tape_hold_imu_18_pd.values[:,:])
train_set_tape_hold_imu_19 = np.float32(train_set_tape_hold_imu_19_pd.values[:,:])
train_set_tape_hold_imu_20 = np.float32(train_set_tape_hold_imu_20_pd.values[:,:])
train_set_tape_hold_imu_21 = np.float32(train_set_tape_hold_imu_21_pd.values[:,:])
train_set_tape_hold_imu_22 = np.float32(train_set_tape_hold_imu_22_pd.values[:,:])
train_set_tape_hold_imu_23 = np.float32(train_set_tape_hold_imu_23_pd.values[:,:])
train_set_tape_hold_imu_24 = np.float32(train_set_tape_hold_imu_24_pd.values[:,:])
train_set_tape_hold_imu_25 = np.float32(train_set_tape_hold_imu_25_pd.values[:,:])
train_set_tape_hold_imu_26 = np.float32(train_set_tape_hold_imu_26_pd.values[:,:])
train_set_tape_hold_imu_27 = np.float32(train_set_tape_hold_imu_27_pd.values[:,:])
train_set_tape_hold_imu_28 = np.float32(train_set_tape_hold_imu_28_pd.values[:,:])
train_set_tape_hold_imu_29 = np.float32(train_set_tape_hold_imu_29_pd.values[:,:])
test_set_tape_hold_imu = np.float32(test_set_tape_hold_imu_pd.values[:,:])


##################################################################################
# load Pose date sets
##################################################################################
print('loading pose data sets of aluminum hold task')
# read pose csv files of aluminum_hold
dir_prefix = '../../recorder/datasets/imu_emg_joint_3_task/aluminum_hold/csv/'
train_set_aluminum_hold_pose_00_pd = pd.read_csv(dir_prefix + '2017-07-09-09-57-38-robot-limb-left-endpoint_state.csv')
train_set_aluminum_hold_pose_01_pd = pd.read_csv(dir_prefix + '2017-07-09-09-57-55-robot-limb-left-endpoint_state.csv')
train_set_aluminum_hold_pose_02_pd = pd.read_csv(dir_prefix + '2017-07-09-09-58-24-robot-limb-left-endpoint_state.csv')
train_set_aluminum_hold_pose_03_pd = pd.read_csv(dir_prefix + '2017-07-09-09-58-51-robot-limb-left-endpoint_state.csv')
train_set_aluminum_hold_pose_04_pd = pd.read_csv(dir_prefix + '2017-07-09-09-59-13-robot-limb-left-endpoint_state.csv')
train_set_aluminum_hold_pose_05_pd = pd.read_csv(dir_prefix + '2017-07-09-10-00-01-robot-limb-left-endpoint_state.csv')
train_set_aluminum_hold_pose_06_pd = pd.read_csv(dir_prefix + '2017-07-09-10-01-08-robot-limb-left-endpoint_state.csv')
train_set_aluminum_hold_pose_07_pd = pd.read_csv(dir_prefix + '2017-07-09-10-01-38-robot-limb-left-endpoint_state.csv')
train_set_aluminum_hold_pose_08_pd = pd.read_csv(dir_prefix + '2017-07-09-10-02-08-robot-limb-left-endpoint_state.csv')
train_set_aluminum_hold_pose_09_pd = pd.read_csv(dir_prefix + '2017-07-09-10-02-29-robot-limb-left-endpoint_state.csv')
train_set_aluminum_hold_pose_10_pd = pd.read_csv(dir_prefix + '2017-07-09-10-02-52-robot-limb-left-endpoint_state.csv')
train_set_aluminum_hold_pose_11_pd = pd.read_csv(dir_prefix + '2017-07-09-10-03-26-robot-limb-left-endpoint_state.csv')
train_set_aluminum_hold_pose_12_pd = pd.read_csv(dir_prefix + '2017-07-09-10-03-43-robot-limb-left-endpoint_state.csv')
train_set_aluminum_hold_pose_13_pd = pd.read_csv(dir_prefix + '2017-07-09-10-04-17-robot-limb-left-endpoint_state.csv')
train_set_aluminum_hold_pose_14_pd = pd.read_csv(dir_prefix + '2017-07-09-10-04-36-robot-limb-left-endpoint_state.csv')
train_set_aluminum_hold_pose_15_pd = pd.read_csv(dir_prefix + '2017-07-09-10-04-57-robot-limb-left-endpoint_state.csv')
train_set_aluminum_hold_pose_16_pd = pd.read_csv(dir_prefix + '2017-07-09-10-05-20-robot-limb-left-endpoint_state.csv')
train_set_aluminum_hold_pose_17_pd = pd.read_csv(dir_prefix + '2017-07-09-10-05-46-robot-limb-left-endpoint_state.csv')
train_set_aluminum_hold_pose_18_pd = pd.read_csv(dir_prefix + '2017-07-09-10-06-07-robot-limb-left-endpoint_state.csv')
train_set_aluminum_hold_pose_19_pd = pd.read_csv(dir_prefix + '2017-07-09-10-06-36-robot-limb-left-endpoint_state.csv')
train_set_aluminum_hold_pose_20_pd = pd.read_csv(dir_prefix + '2017-07-09-10-07-04-robot-limb-left-endpoint_state.csv')
train_set_aluminum_hold_pose_21_pd = pd.read_csv(dir_prefix + '2017-07-09-10-07-23-robot-limb-left-endpoint_state.csv')
train_set_aluminum_hold_pose_22_pd = pd.read_csv(dir_prefix + '2017-07-09-10-07-48-robot-limb-left-endpoint_state.csv')
train_set_aluminum_hold_pose_23_pd = pd.read_csv(dir_prefix + '2017-07-09-10-08-13-robot-limb-left-endpoint_state.csv')
train_set_aluminum_hold_pose_24_pd = pd.read_csv(dir_prefix + '2017-07-09-10-08-39-robot-limb-left-endpoint_state.csv')
train_set_aluminum_hold_pose_25_pd = pd.read_csv(dir_prefix + '2017-07-09-10-08-58-robot-limb-left-endpoint_state.csv')
train_set_aluminum_hold_pose_26_pd = pd.read_csv(dir_prefix + '2017-07-09-10-09-22-robot-limb-left-endpoint_state.csv')
train_set_aluminum_hold_pose_27_pd = pd.read_csv(dir_prefix + '2017-07-09-10-09-54-robot-limb-left-endpoint_state.csv')
train_set_aluminum_hold_pose_28_pd = pd.read_csv(dir_prefix + '2017-07-09-10-10-15-robot-limb-left-endpoint_state.csv')
train_set_aluminum_hold_pose_29_pd = pd.read_csv(dir_prefix + '2017-07-09-10-10-39-robot-limb-left-endpoint_state.csv')
test_set_aluminum_hold_pose_pd = pd.read_csv(dir_prefix + '2017-07-09-10-07-04-robot-limb-left-endpoint_state.csv')
# invert the object to float32 for easy computing
train_set_aluminum_hold_pose_00 = np.float32(train_set_aluminum_hold_pose_00_pd.values[:,5:12])
train_set_aluminum_hold_pose_01 = np.float32(train_set_aluminum_hold_pose_01_pd.values[:,5:12])
train_set_aluminum_hold_pose_02 = np.float32(train_set_aluminum_hold_pose_02_pd.values[:,5:12])
train_set_aluminum_hold_pose_03 = np.float32(train_set_aluminum_hold_pose_03_pd.values[:,5:12])
train_set_aluminum_hold_pose_04 = np.float32(train_set_aluminum_hold_pose_04_pd.values[:,5:12])
train_set_aluminum_hold_pose_05 = np.float32(train_set_aluminum_hold_pose_05_pd.values[:,5:12])
train_set_aluminum_hold_pose_06 = np.float32(train_set_aluminum_hold_pose_06_pd.values[:,5:12])
train_set_aluminum_hold_pose_07 = np.float32(train_set_aluminum_hold_pose_07_pd.values[:,5:12])
train_set_aluminum_hold_pose_08 = np.float32(train_set_aluminum_hold_pose_08_pd.values[:,5:12])
train_set_aluminum_hold_pose_09 = np.float32(train_set_aluminum_hold_pose_09_pd.values[:,5:12])
train_set_aluminum_hold_pose_10 = np.float32(train_set_aluminum_hold_pose_10_pd.values[:,5:12])
train_set_aluminum_hold_pose_11 = np.float32(train_set_aluminum_hold_pose_11_pd.values[:,5:12])
train_set_aluminum_hold_pose_12 = np.float32(train_set_aluminum_hold_pose_12_pd.values[:,5:12])
train_set_aluminum_hold_pose_13 = np.float32(train_set_aluminum_hold_pose_13_pd.values[:,5:12])
train_set_aluminum_hold_pose_14 = np.float32(train_set_aluminum_hold_pose_14_pd.values[:,5:12])
train_set_aluminum_hold_pose_15 = np.float32(train_set_aluminum_hold_pose_15_pd.values[:,5:12])
train_set_aluminum_hold_pose_16 = np.float32(train_set_aluminum_hold_pose_16_pd.values[:,5:12])
train_set_aluminum_hold_pose_17 = np.float32(train_set_aluminum_hold_pose_17_pd.values[:,5:12])
train_set_aluminum_hold_pose_18 = np.float32(train_set_aluminum_hold_pose_18_pd.values[:,5:12])
train_set_aluminum_hold_pose_19 = np.float32(train_set_aluminum_hold_pose_19_pd.values[:,5:12])
train_set_aluminum_hold_pose_20 = np.float32(train_set_aluminum_hold_pose_20_pd.values[:,5:12])
train_set_aluminum_hold_pose_21 = np.float32(train_set_aluminum_hold_pose_21_pd.values[:,5:12])
train_set_aluminum_hold_pose_22 = np.float32(train_set_aluminum_hold_pose_22_pd.values[:,5:12])
train_set_aluminum_hold_pose_23 = np.float32(train_set_aluminum_hold_pose_23_pd.values[:,5:12])
train_set_aluminum_hold_pose_24 = np.float32(train_set_aluminum_hold_pose_24_pd.values[:,5:12])
train_set_aluminum_hold_pose_25 = np.float32(train_set_aluminum_hold_pose_25_pd.values[:,5:12])
train_set_aluminum_hold_pose_26 = np.float32(train_set_aluminum_hold_pose_26_pd.values[:,5:12])
train_set_aluminum_hold_pose_27 = np.float32(train_set_aluminum_hold_pose_27_pd.values[:,5:12])
train_set_aluminum_hold_pose_28 = np.float32(train_set_aluminum_hold_pose_28_pd.values[:,5:12])
train_set_aluminum_hold_pose_29 = np.float32(train_set_aluminum_hold_pose_29_pd.values[:,5:12])
test_set_aluminum_hold_pose = np.float32(test_set_aluminum_hold_pose_pd.values[:,5:12])
#########################################################################################################
print('loading Pose data sets of spanner handover task')
# read pose csv files of spanner_handover
dir_prefix = '../../recorder/datasets/imu_emg_joint_3_task/spanner_handover/csv/'
train_set_spanner_handover_pose_00_pd = pd.read_csv(dir_prefix + '2017-07-09-09-29-42-robot-limb-left-endpoint_state.csv')
train_set_spanner_handover_pose_01_pd = pd.read_csv(dir_prefix + '2017-07-09-09-30-17-robot-limb-left-endpoint_state.csv')
train_set_spanner_handover_pose_02_pd = pd.read_csv(dir_prefix + '2017-07-09-09-30-41-robot-limb-left-endpoint_state.csv')
train_set_spanner_handover_pose_03_pd = pd.read_csv(dir_prefix + '2017-07-09-09-31-03-robot-limb-left-endpoint_state.csv')
train_set_spanner_handover_pose_04_pd = pd.read_csv(dir_prefix + '2017-07-09-09-31-23-robot-limb-left-endpoint_state.csv')
train_set_spanner_handover_pose_05_pd = pd.read_csv(dir_prefix + '2017-07-09-09-31-40-robot-limb-left-endpoint_state.csv')
train_set_spanner_handover_pose_06_pd = pd.read_csv(dir_prefix + '2017-07-09-09-32-24-robot-limb-left-endpoint_state.csv')
train_set_spanner_handover_pose_07_pd = pd.read_csv(dir_prefix + '2017-07-09-09-32-43-robot-limb-left-endpoint_state.csv')
train_set_spanner_handover_pose_08_pd = pd.read_csv(dir_prefix + '2017-07-09-09-33-04-robot-limb-left-endpoint_state.csv')
train_set_spanner_handover_pose_09_pd = pd.read_csv(dir_prefix + '2017-07-09-09-33-23-robot-limb-left-endpoint_state.csv')
train_set_spanner_handover_pose_10_pd = pd.read_csv(dir_prefix + '2017-07-09-09-33-40-robot-limb-left-endpoint_state.csv')
train_set_spanner_handover_pose_11_pd = pd.read_csv(dir_prefix + '2017-07-09-09-34-00-robot-limb-left-endpoint_state.csv')
train_set_spanner_handover_pose_12_pd = pd.read_csv(dir_prefix + '2017-07-09-09-34-32-robot-limb-left-endpoint_state.csv')
train_set_spanner_handover_pose_13_pd = pd.read_csv(dir_prefix + '2017-07-09-09-34-49-robot-limb-left-endpoint_state.csv')
train_set_spanner_handover_pose_14_pd = pd.read_csv(dir_prefix + '2017-07-09-09-35-11-robot-limb-left-endpoint_state.csv')
train_set_spanner_handover_pose_15_pd = pd.read_csv(dir_prefix + '2017-07-09-09-35-49-robot-limb-left-endpoint_state.csv')
train_set_spanner_handover_pose_16_pd = pd.read_csv(dir_prefix + '2017-07-09-09-36-19-robot-limb-left-endpoint_state.csv')
train_set_spanner_handover_pose_17_pd = pd.read_csv(dir_prefix + '2017-07-09-09-36-38-robot-limb-left-endpoint_state.csv')
train_set_spanner_handover_pose_18_pd = pd.read_csv(dir_prefix + '2017-07-09-09-36-58-robot-limb-left-endpoint_state.csv')
train_set_spanner_handover_pose_19_pd = pd.read_csv(dir_prefix + '2017-07-09-09-37-16-robot-limb-left-endpoint_state.csv')
train_set_spanner_handover_pose_20_pd = pd.read_csv(dir_prefix + '2017-07-09-09-37-41-robot-limb-left-endpoint_state.csv')
train_set_spanner_handover_pose_21_pd = pd.read_csv(dir_prefix + '2017-07-09-09-38-02-robot-limb-left-endpoint_state.csv')
train_set_spanner_handover_pose_22_pd = pd.read_csv(dir_prefix + '2017-07-09-09-38-19-robot-limb-left-endpoint_state.csv')
train_set_spanner_handover_pose_23_pd = pd.read_csv(dir_prefix + '2017-07-09-09-38-54-robot-limb-left-endpoint_state.csv')
train_set_spanner_handover_pose_24_pd = pd.read_csv(dir_prefix + '2017-07-09-09-41-39-robot-limb-left-endpoint_state.csv')
train_set_spanner_handover_pose_25_pd = pd.read_csv(dir_prefix + '2017-07-09-09-42-03-robot-limb-left-endpoint_state.csv')
train_set_spanner_handover_pose_26_pd = pd.read_csv(dir_prefix + '2017-07-09-09-42-21-robot-limb-left-endpoint_state.csv')
train_set_spanner_handover_pose_27_pd = pd.read_csv(dir_prefix + '2017-07-09-09-43-00-robot-limb-left-endpoint_state.csv')
train_set_spanner_handover_pose_28_pd = pd.read_csv(dir_prefix + '2017-07-09-09-43-33-robot-limb-left-endpoint_state.csv')
train_set_spanner_handover_pose_29_pd = pd.read_csv(dir_prefix + '2017-07-09-09-43-52-robot-limb-left-endpoint_state.csv')
test_set_spanner_handover_pose_pd = pd.read_csv(dir_prefix + '2017-07-09-09-37-41-robot-limb-left-endpoint_state.csv')
# invert the object to float32 for easy computing
train_set_spanner_handover_pose_00 = np.float32(train_set_spanner_handover_pose_00_pd.values[:,5:12])
train_set_spanner_handover_pose_01 = np.float32(train_set_spanner_handover_pose_01_pd.values[:,5:12])
train_set_spanner_handover_pose_02 = np.float32(train_set_spanner_handover_pose_02_pd.values[:,5:12])
train_set_spanner_handover_pose_03 = np.float32(train_set_spanner_handover_pose_03_pd.values[:,5:12])
train_set_spanner_handover_pose_04 = np.float32(train_set_spanner_handover_pose_04_pd.values[:,5:12])
train_set_spanner_handover_pose_05 = np.float32(train_set_spanner_handover_pose_05_pd.values[:,5:12])
train_set_spanner_handover_pose_06 = np.float32(train_set_spanner_handover_pose_06_pd.values[:,5:12])
train_set_spanner_handover_pose_07 = np.float32(train_set_spanner_handover_pose_07_pd.values[:,5:12])
train_set_spanner_handover_pose_08 = np.float32(train_set_spanner_handover_pose_08_pd.values[:,5:12])
train_set_spanner_handover_pose_09 = np.float32(train_set_spanner_handover_pose_09_pd.values[:,5:12])
train_set_spanner_handover_pose_10 = np.float32(train_set_spanner_handover_pose_10_pd.values[:,5:12])
train_set_spanner_handover_pose_11 = np.float32(train_set_spanner_handover_pose_11_pd.values[:,5:12])
train_set_spanner_handover_pose_12 = np.float32(train_set_spanner_handover_pose_12_pd.values[:,5:12])
train_set_spanner_handover_pose_13 = np.float32(train_set_spanner_handover_pose_13_pd.values[:,5:12])
train_set_spanner_handover_pose_14 = np.float32(train_set_spanner_handover_pose_14_pd.values[:,5:12])
train_set_spanner_handover_pose_15 = np.float32(train_set_spanner_handover_pose_15_pd.values[:,5:12])
train_set_spanner_handover_pose_16 = np.float32(train_set_spanner_handover_pose_16_pd.values[:,5:12])
train_set_spanner_handover_pose_17 = np.float32(train_set_spanner_handover_pose_17_pd.values[:,5:12])
train_set_spanner_handover_pose_18 = np.float32(train_set_spanner_handover_pose_18_pd.values[:,5:12])
train_set_spanner_handover_pose_19 = np.float32(train_set_spanner_handover_pose_19_pd.values[:,5:12])
train_set_spanner_handover_pose_20 = np.float32(train_set_spanner_handover_pose_20_pd.values[:,5:12])
train_set_spanner_handover_pose_21 = np.float32(train_set_spanner_handover_pose_21_pd.values[:,5:12])
train_set_spanner_handover_pose_22 = np.float32(train_set_spanner_handover_pose_22_pd.values[:,5:12])
train_set_spanner_handover_pose_23 = np.float32(train_set_spanner_handover_pose_23_pd.values[:,5:12])
train_set_spanner_handover_pose_24 = np.float32(train_set_spanner_handover_pose_24_pd.values[:,5:12])
train_set_spanner_handover_pose_25 = np.float32(train_set_spanner_handover_pose_25_pd.values[:,5:12])
train_set_spanner_handover_pose_26 = np.float32(train_set_spanner_handover_pose_26_pd.values[:,5:12])
train_set_spanner_handover_pose_27 = np.float32(train_set_spanner_handover_pose_27_pd.values[:,5:12])
train_set_spanner_handover_pose_28 = np.float32(train_set_spanner_handover_pose_28_pd.values[:,5:12])
train_set_spanner_handover_pose_29 = np.float32(train_set_spanner_handover_pose_29_pd.values[:,5:12])
test_set_spanner_handover_pose = np.float32(test_set_spanner_handover_pose_pd.values[:,5:12])
#############################################################################################################
print('loading Pose data sets of tape hold task')
# read pose csv files of tape_hold
dir_prefix = '../../recorder/datasets/imu_emg_joint_3_task/tape_hold/csv/'
train_set_tape_hold_pose_00_pd = pd.read_csv(dir_prefix + '2017-07-09-10-14-01-robot-limb-left-endpoint_state.csv')
train_set_tape_hold_pose_01_pd = pd.read_csv(dir_prefix + '2017-07-09-10-14-22-robot-limb-left-endpoint_state.csv')
train_set_tape_hold_pose_02_pd = pd.read_csv(dir_prefix + '2017-07-09-10-14-43-robot-limb-left-endpoint_state.csv')
train_set_tape_hold_pose_03_pd = pd.read_csv(dir_prefix + '2017-07-09-10-15-47-robot-limb-left-endpoint_state.csv')
train_set_tape_hold_pose_04_pd = pd.read_csv(dir_prefix + '2017-07-09-10-16-04-robot-limb-left-endpoint_state.csv')
train_set_tape_hold_pose_05_pd = pd.read_csv(dir_prefix + '2017-07-09-10-16-22-robot-limb-left-endpoint_state.csv')
train_set_tape_hold_pose_06_pd = pd.read_csv(dir_prefix + '2017-07-09-10-16-40-robot-limb-left-endpoint_state.csv')
train_set_tape_hold_pose_07_pd = pd.read_csv(dir_prefix + '2017-07-09-10-16-56-robot-limb-left-endpoint_state.csv')
train_set_tape_hold_pose_08_pd = pd.read_csv(dir_prefix + '2017-07-09-10-17-13-robot-limb-left-endpoint_state.csv')
train_set_tape_hold_pose_09_pd = pd.read_csv(dir_prefix + '2017-07-09-10-17-48-robot-limb-left-endpoint_state.csv')
train_set_tape_hold_pose_10_pd = pd.read_csv(dir_prefix + '2017-07-09-10-18-04-robot-limb-left-endpoint_state.csv')
train_set_tape_hold_pose_11_pd = pd.read_csv(dir_prefix + '2017-07-09-10-18-33-robot-limb-left-endpoint_state.csv')
train_set_tape_hold_pose_12_pd = pd.read_csv(dir_prefix + '2017-07-09-10-18-49-robot-limb-left-endpoint_state.csv')
train_set_tape_hold_pose_13_pd = pd.read_csv(dir_prefix + '2017-07-09-10-19-05-robot-limb-left-endpoint_state.csv')
train_set_tape_hold_pose_14_pd = pd.read_csv(dir_prefix + '2017-07-09-10-19-22-robot-limb-left-endpoint_state.csv')
train_set_tape_hold_pose_15_pd = pd.read_csv(dir_prefix + '2017-07-09-10-19-51-robot-limb-left-endpoint_state.csv')
train_set_tape_hold_pose_16_pd = pd.read_csv(dir_prefix + '2017-07-09-10-20-08-robot-limb-left-endpoint_state.csv')
train_set_tape_hold_pose_17_pd = pd.read_csv(dir_prefix + '2017-07-09-10-20-26-robot-limb-left-endpoint_state.csv')
train_set_tape_hold_pose_18_pd = pd.read_csv(dir_prefix + '2017-07-09-10-20-44-robot-limb-left-endpoint_state.csv')
train_set_tape_hold_pose_19_pd = pd.read_csv(dir_prefix + '2017-07-09-10-21-03-robot-limb-left-endpoint_state.csv')
train_set_tape_hold_pose_20_pd = pd.read_csv(dir_prefix + '2017-07-09-10-21-22-robot-limb-left-endpoint_state.csv')
train_set_tape_hold_pose_21_pd = pd.read_csv(dir_prefix + '2017-07-09-10-21-44-robot-limb-left-endpoint_state.csv')
train_set_tape_hold_pose_22_pd = pd.read_csv(dir_prefix + '2017-07-09-10-22-00-robot-limb-left-endpoint_state.csv')
train_set_tape_hold_pose_23_pd = pd.read_csv(dir_prefix + '2017-07-09-10-22-17-robot-limb-left-endpoint_state.csv')
train_set_tape_hold_pose_24_pd = pd.read_csv(dir_prefix + '2017-07-09-10-22-37-robot-limb-left-endpoint_state.csv')
train_set_tape_hold_pose_25_pd = pd.read_csv(dir_prefix + '2017-07-09-10-23-07-robot-limb-left-endpoint_state.csv')
train_set_tape_hold_pose_26_pd = pd.read_csv(dir_prefix + '2017-07-09-10-23-29-robot-limb-left-endpoint_state.csv')
train_set_tape_hold_pose_27_pd = pd.read_csv(dir_prefix + '2017-07-09-10-23-48-robot-limb-left-endpoint_state.csv')
train_set_tape_hold_pose_28_pd = pd.read_csv(dir_prefix + '2017-07-09-10-24-21-robot-limb-left-endpoint_state.csv')
train_set_tape_hold_pose_29_pd = pd.read_csv(dir_prefix + '2017-07-09-10-24-46-robot-limb-left-endpoint_state.csv')
test_set_tape_hold_pose_pd = pd.read_csv(dir_prefix + '2017-07-09-10-21-22-robot-limb-left-endpoint_state.csv')
# invert the object to float32 for easy computing
train_set_tape_hold_pose_00 = np.float32(train_set_tape_hold_pose_00_pd.values[:,5:12])
train_set_tape_hold_pose_01 = np.float32(train_set_tape_hold_pose_01_pd.values[:,5:12])
train_set_tape_hold_pose_02 = np.float32(train_set_tape_hold_pose_02_pd.values[:,5:12])
train_set_tape_hold_pose_03 = np.float32(train_set_tape_hold_pose_03_pd.values[:,5:12])
train_set_tape_hold_pose_04 = np.float32(train_set_tape_hold_pose_04_pd.values[:,5:12])
train_set_tape_hold_pose_05 = np.float32(train_set_tape_hold_pose_05_pd.values[:,5:12])
train_set_tape_hold_pose_06 = np.float32(train_set_tape_hold_pose_06_pd.values[:,5:12])
train_set_tape_hold_pose_07 = np.float32(train_set_tape_hold_pose_07_pd.values[:,5:12])
train_set_tape_hold_pose_08 = np.float32(train_set_tape_hold_pose_08_pd.values[:,5:12])
train_set_tape_hold_pose_09 = np.float32(train_set_tape_hold_pose_09_pd.values[:,5:12])
train_set_tape_hold_pose_10 = np.float32(train_set_tape_hold_pose_10_pd.values[:,5:12])
train_set_tape_hold_pose_11 = np.float32(train_set_tape_hold_pose_11_pd.values[:,5:12])
train_set_tape_hold_pose_12 = np.float32(train_set_tape_hold_pose_12_pd.values[:,5:12])
train_set_tape_hold_pose_13 = np.float32(train_set_tape_hold_pose_13_pd.values[:,5:12])
train_set_tape_hold_pose_14 = np.float32(train_set_tape_hold_pose_14_pd.values[:,5:12])
train_set_tape_hold_pose_15 = np.float32(train_set_tape_hold_pose_15_pd.values[:,5:12])
train_set_tape_hold_pose_16 = np.float32(train_set_tape_hold_pose_16_pd.values[:,5:12])
train_set_tape_hold_pose_17 = np.float32(train_set_tape_hold_pose_17_pd.values[:,5:12])
train_set_tape_hold_pose_18 = np.float32(train_set_tape_hold_pose_18_pd.values[:,5:12])
train_set_tape_hold_pose_19 = np.float32(train_set_tape_hold_pose_19_pd.values[:,5:12])
train_set_tape_hold_pose_20 = np.float32(train_set_tape_hold_pose_20_pd.values[:,5:12])
train_set_tape_hold_pose_21 = np.float32(train_set_tape_hold_pose_21_pd.values[:,5:12])
train_set_tape_hold_pose_22 = np.float32(train_set_tape_hold_pose_22_pd.values[:,5:12])
train_set_tape_hold_pose_23 = np.float32(train_set_tape_hold_pose_23_pd.values[:,5:12])
train_set_tape_hold_pose_24 = np.float32(train_set_tape_hold_pose_24_pd.values[:,5:12])
train_set_tape_hold_pose_25 = np.float32(train_set_tape_hold_pose_25_pd.values[:,5:12])
train_set_tape_hold_pose_26 = np.float32(train_set_tape_hold_pose_26_pd.values[:,5:12])
train_set_tape_hold_pose_27 = np.float32(train_set_tape_hold_pose_27_pd.values[:,5:12])
train_set_tape_hold_pose_28 = np.float32(train_set_tape_hold_pose_28_pd.values[:,5:12])
train_set_tape_hold_pose_29 = np.float32(train_set_tape_hold_pose_29_pd.values[:,5:12])
test_set_tape_hold_pose = np.float32(test_set_tape_hold_pose_pd.values[:,5:12])

######################################################
######################################################
##### the all file loaded successfully as above ######
######################################################
######################################################


##################################################################################
# resampling the EMG data for experiencing the same duration
##################################################################################
# rospy.loginfo('normalizing data into same duration')
print('normalizing EMG data into same duration of aluminum_hold')
# resampling emg signals of aluminum_hold
train_set_aluminum_hold_emg_norm00=np.array([]);train_set_aluminum_hold_emg_norm01=np.array([]);train_set_aluminum_hold_emg_norm02=np.array([]);train_set_aluminum_hold_emg_norm03=np.array([]);train_set_aluminum_hold_emg_norm04=np.array([]);
train_set_aluminum_hold_emg_norm05=np.array([]);train_set_aluminum_hold_emg_norm06=np.array([]);train_set_aluminum_hold_emg_norm07=np.array([]);train_set_aluminum_hold_emg_norm08=np.array([]);train_set_aluminum_hold_emg_norm09=np.array([]);
train_set_aluminum_hold_emg_norm10=np.array([]);train_set_aluminum_hold_emg_norm11=np.array([]);train_set_aluminum_hold_emg_norm12=np.array([]);train_set_aluminum_hold_emg_norm13=np.array([]);train_set_aluminum_hold_emg_norm14=np.array([]);
train_set_aluminum_hold_emg_norm15=np.array([]);train_set_aluminum_hold_emg_norm16=np.array([]);train_set_aluminum_hold_emg_norm17=np.array([]);train_set_aluminum_hold_emg_norm18=np.array([]);train_set_aluminum_hold_emg_norm19=np.array([]);
train_set_aluminum_hold_emg_norm20=np.array([]);train_set_aluminum_hold_emg_norm21=np.array([]);train_set_aluminum_hold_emg_norm22=np.array([]);train_set_aluminum_hold_emg_norm23=np.array([]);train_set_aluminum_hold_emg_norm24=np.array([]);
train_set_aluminum_hold_emg_norm25=np.array([]);train_set_aluminum_hold_emg_norm26=np.array([]);train_set_aluminum_hold_emg_norm27=np.array([]);train_set_aluminum_hold_emg_norm28=np.array([]);train_set_aluminum_hold_emg_norm29=np.array([]);
test_set_aluminum_hold_emg_norm=np.array([]);
for ch_ex in range(8):
    train_set_aluminum_hold_emg_norm00 = np.hstack(( train_set_aluminum_hold_emg_norm00, np.interp(np.linspace(0, len(train_set_aluminum_hold_emg_00)-1, len_normal), np.arange(0,len(train_set_aluminum_hold_emg_00),1.), train_set_aluminum_hold_emg_00[:,ch_ex]) ))
    train_set_aluminum_hold_emg_norm01 = np.hstack(( train_set_aluminum_hold_emg_norm01, np.interp(np.linspace(0, len(train_set_aluminum_hold_emg_01)-1, len_normal), np.arange(0,len(train_set_aluminum_hold_emg_01),1.), train_set_aluminum_hold_emg_01[:,ch_ex]) ))
    train_set_aluminum_hold_emg_norm02 = np.hstack(( train_set_aluminum_hold_emg_norm02, np.interp(np.linspace(0, len(train_set_aluminum_hold_emg_02)-1, len_normal), np.arange(0,len(train_set_aluminum_hold_emg_02),1.), train_set_aluminum_hold_emg_02[:,ch_ex]) ))
    train_set_aluminum_hold_emg_norm03 = np.hstack(( train_set_aluminum_hold_emg_norm03, np.interp(np.linspace(0, len(train_set_aluminum_hold_emg_03)-1, len_normal), np.arange(0,len(train_set_aluminum_hold_emg_03),1.), train_set_aluminum_hold_emg_03[:,ch_ex]) ))
    train_set_aluminum_hold_emg_norm04 = np.hstack(( train_set_aluminum_hold_emg_norm04, np.interp(np.linspace(0, len(train_set_aluminum_hold_emg_04)-1, len_normal), np.arange(0,len(train_set_aluminum_hold_emg_04),1.), train_set_aluminum_hold_emg_04[:,ch_ex]) ))
    train_set_aluminum_hold_emg_norm05 = np.hstack(( train_set_aluminum_hold_emg_norm05, np.interp(np.linspace(0, len(train_set_aluminum_hold_emg_05)-1, len_normal), np.arange(0,len(train_set_aluminum_hold_emg_05),1.), train_set_aluminum_hold_emg_05[:,ch_ex]) ))
    train_set_aluminum_hold_emg_norm06 = np.hstack(( train_set_aluminum_hold_emg_norm06, np.interp(np.linspace(0, len(train_set_aluminum_hold_emg_06)-1, len_normal), np.arange(0,len(train_set_aluminum_hold_emg_06),1.), train_set_aluminum_hold_emg_06[:,ch_ex]) ))
    train_set_aluminum_hold_emg_norm07 = np.hstack(( train_set_aluminum_hold_emg_norm07, np.interp(np.linspace(0, len(train_set_aluminum_hold_emg_07)-1, len_normal), np.arange(0,len(train_set_aluminum_hold_emg_07),1.), train_set_aluminum_hold_emg_07[:,ch_ex]) ))
    train_set_aluminum_hold_emg_norm08 = np.hstack(( train_set_aluminum_hold_emg_norm08, np.interp(np.linspace(0, len(train_set_aluminum_hold_emg_08)-1, len_normal), np.arange(0,len(train_set_aluminum_hold_emg_08),1.), train_set_aluminum_hold_emg_08[:,ch_ex]) ))
    train_set_aluminum_hold_emg_norm09 = np.hstack(( train_set_aluminum_hold_emg_norm09, np.interp(np.linspace(0, len(train_set_aluminum_hold_emg_09)-1, len_normal), np.arange(0,len(train_set_aluminum_hold_emg_09),1.), train_set_aluminum_hold_emg_09[:,ch_ex]) ))
    train_set_aluminum_hold_emg_norm10 = np.hstack(( train_set_aluminum_hold_emg_norm10, np.interp(np.linspace(0, len(train_set_aluminum_hold_emg_10)-1, len_normal), np.arange(0,len(train_set_aluminum_hold_emg_10),1.), train_set_aluminum_hold_emg_10[:,ch_ex]) ))
    train_set_aluminum_hold_emg_norm11 = np.hstack(( train_set_aluminum_hold_emg_norm11, np.interp(np.linspace(0, len(train_set_aluminum_hold_emg_11)-1, len_normal), np.arange(0,len(train_set_aluminum_hold_emg_11),1.), train_set_aluminum_hold_emg_11[:,ch_ex]) ))
    train_set_aluminum_hold_emg_norm12 = np.hstack(( train_set_aluminum_hold_emg_norm12, np.interp(np.linspace(0, len(train_set_aluminum_hold_emg_12)-1, len_normal), np.arange(0,len(train_set_aluminum_hold_emg_12),1.), train_set_aluminum_hold_emg_12[:,ch_ex]) ))
    train_set_aluminum_hold_emg_norm13 = np.hstack(( train_set_aluminum_hold_emg_norm13, np.interp(np.linspace(0, len(train_set_aluminum_hold_emg_13)-1, len_normal), np.arange(0,len(train_set_aluminum_hold_emg_13),1.), train_set_aluminum_hold_emg_13[:,ch_ex]) ))
    train_set_aluminum_hold_emg_norm14 = np.hstack(( train_set_aluminum_hold_emg_norm14, np.interp(np.linspace(0, len(train_set_aluminum_hold_emg_14)-1, len_normal), np.arange(0,len(train_set_aluminum_hold_emg_14),1.), train_set_aluminum_hold_emg_14[:,ch_ex]) ))
    train_set_aluminum_hold_emg_norm15 = np.hstack(( train_set_aluminum_hold_emg_norm15, np.interp(np.linspace(0, len(train_set_aluminum_hold_emg_15)-1, len_normal), np.arange(0,len(train_set_aluminum_hold_emg_15),1.), train_set_aluminum_hold_emg_15[:,ch_ex]) ))
    train_set_aluminum_hold_emg_norm16 = np.hstack(( train_set_aluminum_hold_emg_norm16, np.interp(np.linspace(0, len(train_set_aluminum_hold_emg_16)-1, len_normal), np.arange(0,len(train_set_aluminum_hold_emg_16),1.), train_set_aluminum_hold_emg_16[:,ch_ex]) ))
    train_set_aluminum_hold_emg_norm17 = np.hstack(( train_set_aluminum_hold_emg_norm17, np.interp(np.linspace(0, len(train_set_aluminum_hold_emg_17)-1, len_normal), np.arange(0,len(train_set_aluminum_hold_emg_17),1.), train_set_aluminum_hold_emg_17[:,ch_ex]) ))
    train_set_aluminum_hold_emg_norm18 = np.hstack(( train_set_aluminum_hold_emg_norm18, np.interp(np.linspace(0, len(train_set_aluminum_hold_emg_18)-1, len_normal), np.arange(0,len(train_set_aluminum_hold_emg_18),1.), train_set_aluminum_hold_emg_18[:,ch_ex]) ))
    train_set_aluminum_hold_emg_norm19 = np.hstack(( train_set_aluminum_hold_emg_norm19, np.interp(np.linspace(0, len(train_set_aluminum_hold_emg_19)-1, len_normal), np.arange(0,len(train_set_aluminum_hold_emg_19),1.), train_set_aluminum_hold_emg_19[:,ch_ex]) ))
    train_set_aluminum_hold_emg_norm20 = np.hstack(( train_set_aluminum_hold_emg_norm20, np.interp(np.linspace(0, len(train_set_aluminum_hold_emg_20)-1, len_normal), np.arange(0,len(train_set_aluminum_hold_emg_20),1.), train_set_aluminum_hold_emg_20[:, ch_ex])))
    train_set_aluminum_hold_emg_norm21 = np.hstack(( train_set_aluminum_hold_emg_norm21, np.interp(np.linspace(0, len(train_set_aluminum_hold_emg_21)-1, len_normal), np.arange(0,len(train_set_aluminum_hold_emg_21),1.), train_set_aluminum_hold_emg_21[:, ch_ex])))
    train_set_aluminum_hold_emg_norm22 = np.hstack(( train_set_aluminum_hold_emg_norm22, np.interp(np.linspace(0, len(train_set_aluminum_hold_emg_22)-1, len_normal), np.arange(0,len(train_set_aluminum_hold_emg_22),1.), train_set_aluminum_hold_emg_22[:, ch_ex])))
    train_set_aluminum_hold_emg_norm23 = np.hstack(( train_set_aluminum_hold_emg_norm23, np.interp(np.linspace(0, len(train_set_aluminum_hold_emg_23)-1, len_normal), np.arange(0,len(train_set_aluminum_hold_emg_23),1.), train_set_aluminum_hold_emg_23[:, ch_ex])))
    train_set_aluminum_hold_emg_norm24 = np.hstack(( train_set_aluminum_hold_emg_norm24, np.interp(np.linspace(0, len(train_set_aluminum_hold_emg_24)-1, len_normal), np.arange(0,len(train_set_aluminum_hold_emg_24),1.), train_set_aluminum_hold_emg_24[:, ch_ex])))
    train_set_aluminum_hold_emg_norm25 = np.hstack(( train_set_aluminum_hold_emg_norm25, np.interp(np.linspace(0, len(train_set_aluminum_hold_emg_25)-1, len_normal), np.arange(0,len(train_set_aluminum_hold_emg_25),1.), train_set_aluminum_hold_emg_25[:, ch_ex])))
    train_set_aluminum_hold_emg_norm26 = np.hstack(( train_set_aluminum_hold_emg_norm26, np.interp(np.linspace(0, len(train_set_aluminum_hold_emg_26)-1, len_normal), np.arange(0,len(train_set_aluminum_hold_emg_26),1.), train_set_aluminum_hold_emg_26[:, ch_ex])))
    train_set_aluminum_hold_emg_norm27 = np.hstack(( train_set_aluminum_hold_emg_norm27, np.interp(np.linspace(0, len(train_set_aluminum_hold_emg_27)-1, len_normal), np.arange(0,len(train_set_aluminum_hold_emg_27),1.), train_set_aluminum_hold_emg_27[:, ch_ex])))
    train_set_aluminum_hold_emg_norm28 = np.hstack(( train_set_aluminum_hold_emg_norm28, np.interp(np.linspace(0, len(train_set_aluminum_hold_emg_28)-1, len_normal), np.arange(0,len(train_set_aluminum_hold_emg_28),1.), train_set_aluminum_hold_emg_28[:, ch_ex])))
    train_set_aluminum_hold_emg_norm29 = np.hstack(( train_set_aluminum_hold_emg_norm29, np.interp(np.linspace(0, len(train_set_aluminum_hold_emg_29)-1, len_normal), np.arange(0,len(train_set_aluminum_hold_emg_29),1.), train_set_aluminum_hold_emg_29[:, ch_ex])))
    test_set_aluminum_hold_emg_norm = np.hstack(( test_set_aluminum_hold_emg_norm, np.interp(np.linspace(0, len(test_set_aluminum_hold_emg)-1, len_normal), np.arange(0,len(test_set_aluminum_hold_emg),1.), test_set_aluminum_hold_emg[:,ch_ex]) ))
train_set_aluminum_hold_emg_norm00 = train_set_aluminum_hold_emg_norm00.reshape(8,len_normal).T
train_set_aluminum_hold_emg_norm01 = train_set_aluminum_hold_emg_norm01.reshape(8,len_normal).T
train_set_aluminum_hold_emg_norm02 = train_set_aluminum_hold_emg_norm02.reshape(8,len_normal).T
train_set_aluminum_hold_emg_norm03 = train_set_aluminum_hold_emg_norm03.reshape(8,len_normal).T
train_set_aluminum_hold_emg_norm04 = train_set_aluminum_hold_emg_norm04.reshape(8,len_normal).T
train_set_aluminum_hold_emg_norm05 = train_set_aluminum_hold_emg_norm05.reshape(8,len_normal).T
train_set_aluminum_hold_emg_norm06 = train_set_aluminum_hold_emg_norm06.reshape(8,len_normal).T
train_set_aluminum_hold_emg_norm07 = train_set_aluminum_hold_emg_norm07.reshape(8,len_normal).T
train_set_aluminum_hold_emg_norm08 = train_set_aluminum_hold_emg_norm08.reshape(8,len_normal).T
train_set_aluminum_hold_emg_norm09 = train_set_aluminum_hold_emg_norm09.reshape(8,len_normal).T
train_set_aluminum_hold_emg_norm10 = train_set_aluminum_hold_emg_norm10.reshape(8,len_normal).T
train_set_aluminum_hold_emg_norm11 = train_set_aluminum_hold_emg_norm11.reshape(8,len_normal).T
train_set_aluminum_hold_emg_norm12 = train_set_aluminum_hold_emg_norm12.reshape(8,len_normal).T
train_set_aluminum_hold_emg_norm13 = train_set_aluminum_hold_emg_norm13.reshape(8,len_normal).T
train_set_aluminum_hold_emg_norm14 = train_set_aluminum_hold_emg_norm14.reshape(8,len_normal).T
train_set_aluminum_hold_emg_norm15 = train_set_aluminum_hold_emg_norm15.reshape(8,len_normal).T
train_set_aluminum_hold_emg_norm16 = train_set_aluminum_hold_emg_norm16.reshape(8,len_normal).T
train_set_aluminum_hold_emg_norm17 = train_set_aluminum_hold_emg_norm17.reshape(8,len_normal).T
train_set_aluminum_hold_emg_norm18 = train_set_aluminum_hold_emg_norm18.reshape(8,len_normal).T
train_set_aluminum_hold_emg_norm19 = train_set_aluminum_hold_emg_norm19.reshape(8,len_normal).T
train_set_aluminum_hold_emg_norm20 = train_set_aluminum_hold_emg_norm20.reshape(8,len_normal).T
train_set_aluminum_hold_emg_norm21 = train_set_aluminum_hold_emg_norm21.reshape(8,len_normal).T
train_set_aluminum_hold_emg_norm22 = train_set_aluminum_hold_emg_norm22.reshape(8,len_normal).T
train_set_aluminum_hold_emg_norm23 = train_set_aluminum_hold_emg_norm23.reshape(8,len_normal).T
train_set_aluminum_hold_emg_norm24 = train_set_aluminum_hold_emg_norm24.reshape(8,len_normal).T
train_set_aluminum_hold_emg_norm25 = train_set_aluminum_hold_emg_norm25.reshape(8,len_normal).T
train_set_aluminum_hold_emg_norm26 = train_set_aluminum_hold_emg_norm26.reshape(8,len_normal).T
train_set_aluminum_hold_emg_norm27 = train_set_aluminum_hold_emg_norm27.reshape(8,len_normal).T
train_set_aluminum_hold_emg_norm28 = train_set_aluminum_hold_emg_norm28.reshape(8,len_normal).T
train_set_aluminum_hold_emg_norm29 = train_set_aluminum_hold_emg_norm29.reshape(8,len_normal).T
test_set_aluminum_hold_emg_norm = test_set_aluminum_hold_emg_norm.reshape(8,len_normal).T
train_set_aluminum_hold_emg_norm_full = np.array([train_set_aluminum_hold_emg_norm00,train_set_aluminum_hold_emg_norm01,train_set_aluminum_hold_emg_norm02,train_set_aluminum_hold_emg_norm03,train_set_aluminum_hold_emg_norm04,
                                    train_set_aluminum_hold_emg_norm05,train_set_aluminum_hold_emg_norm06,train_set_aluminum_hold_emg_norm07,train_set_aluminum_hold_emg_norm08,train_set_aluminum_hold_emg_norm09,
                                    train_set_aluminum_hold_emg_norm10,train_set_aluminum_hold_emg_norm11,train_set_aluminum_hold_emg_norm12,train_set_aluminum_hold_emg_norm13,train_set_aluminum_hold_emg_norm14,
                                    train_set_aluminum_hold_emg_norm15,train_set_aluminum_hold_emg_norm16,train_set_aluminum_hold_emg_norm17,train_set_aluminum_hold_emg_norm18,train_set_aluminum_hold_emg_norm19])
##########################################################################################
print('normalizing EMG data into same duration of spanner_handover')
# resampling emg signals of aluminum_hold
train_set_spanner_handover_emg_norm00=np.array([]);train_set_spanner_handover_emg_norm01=np.array([]);train_set_spanner_handover_emg_norm02=np.array([]);train_set_spanner_handover_emg_norm03=np.array([]);train_set_spanner_handover_emg_norm04=np.array([]);
train_set_spanner_handover_emg_norm05=np.array([]);train_set_spanner_handover_emg_norm06=np.array([]);train_set_spanner_handover_emg_norm07=np.array([]);train_set_spanner_handover_emg_norm08=np.array([]);train_set_spanner_handover_emg_norm09=np.array([]);
train_set_spanner_handover_emg_norm10=np.array([]);train_set_spanner_handover_emg_norm11=np.array([]);train_set_spanner_handover_emg_norm12=np.array([]);train_set_spanner_handover_emg_norm13=np.array([]);train_set_spanner_handover_emg_norm14=np.array([]);
train_set_spanner_handover_emg_norm15=np.array([]);train_set_spanner_handover_emg_norm16=np.array([]);train_set_spanner_handover_emg_norm17=np.array([]);train_set_spanner_handover_emg_norm18=np.array([]);train_set_spanner_handover_emg_norm19=np.array([]);
train_set_spanner_handover_emg_norm20=np.array([]);train_set_spanner_handover_emg_norm21=np.array([]);train_set_spanner_handover_emg_norm22=np.array([]);train_set_spanner_handover_emg_norm23=np.array([]);train_set_spanner_handover_emg_norm24=np.array([]);
train_set_spanner_handover_emg_norm25=np.array([]);train_set_spanner_handover_emg_norm26=np.array([]);train_set_spanner_handover_emg_norm27=np.array([]);train_set_spanner_handover_emg_norm28=np.array([]);train_set_spanner_handover_emg_norm29=np.array([]);
test_set_spanner_handover_emg_norm=np.array([]);
for ch_ex in range(8):
    train_set_spanner_handover_emg_norm00 = np.hstack(( train_set_spanner_handover_emg_norm00, np.interp(np.linspace(0, len(train_set_spanner_handover_emg_00)-1, len_normal), np.arange(0,len(train_set_spanner_handover_emg_00),1.), train_set_spanner_handover_emg_00[:,ch_ex]) ))
    train_set_spanner_handover_emg_norm01 = np.hstack(( train_set_spanner_handover_emg_norm01, np.interp(np.linspace(0, len(train_set_spanner_handover_emg_01)-1, len_normal), np.arange(0,len(train_set_spanner_handover_emg_01),1.), train_set_spanner_handover_emg_01[:,ch_ex]) ))
    train_set_spanner_handover_emg_norm02 = np.hstack(( train_set_spanner_handover_emg_norm02, np.interp(np.linspace(0, len(train_set_spanner_handover_emg_02)-1, len_normal), np.arange(0,len(train_set_spanner_handover_emg_02),1.), train_set_spanner_handover_emg_02[:,ch_ex]) ))
    train_set_spanner_handover_emg_norm03 = np.hstack(( train_set_spanner_handover_emg_norm03, np.interp(np.linspace(0, len(train_set_spanner_handover_emg_03)-1, len_normal), np.arange(0,len(train_set_spanner_handover_emg_03),1.), train_set_spanner_handover_emg_03[:,ch_ex]) ))
    train_set_spanner_handover_emg_norm04 = np.hstack(( train_set_spanner_handover_emg_norm04, np.interp(np.linspace(0, len(train_set_spanner_handover_emg_04)-1, len_normal), np.arange(0,len(train_set_spanner_handover_emg_04),1.), train_set_spanner_handover_emg_04[:,ch_ex]) ))
    train_set_spanner_handover_emg_norm05 = np.hstack(( train_set_spanner_handover_emg_norm05, np.interp(np.linspace(0, len(train_set_spanner_handover_emg_05)-1, len_normal), np.arange(0,len(train_set_spanner_handover_emg_05),1.), train_set_spanner_handover_emg_05[:,ch_ex]) ))
    train_set_spanner_handover_emg_norm06 = np.hstack(( train_set_spanner_handover_emg_norm06, np.interp(np.linspace(0, len(train_set_spanner_handover_emg_06)-1, len_normal), np.arange(0,len(train_set_spanner_handover_emg_06),1.), train_set_spanner_handover_emg_06[:,ch_ex]) ))
    train_set_spanner_handover_emg_norm07 = np.hstack(( train_set_spanner_handover_emg_norm07, np.interp(np.linspace(0, len(train_set_spanner_handover_emg_07)-1, len_normal), np.arange(0,len(train_set_spanner_handover_emg_07),1.), train_set_spanner_handover_emg_07[:,ch_ex]) ))
    train_set_spanner_handover_emg_norm08 = np.hstack(( train_set_spanner_handover_emg_norm08, np.interp(np.linspace(0, len(train_set_spanner_handover_emg_08)-1, len_normal), np.arange(0,len(train_set_spanner_handover_emg_08),1.), train_set_spanner_handover_emg_08[:,ch_ex]) ))
    train_set_spanner_handover_emg_norm09 = np.hstack(( train_set_spanner_handover_emg_norm09, np.interp(np.linspace(0, len(train_set_spanner_handover_emg_09)-1, len_normal), np.arange(0,len(train_set_spanner_handover_emg_09),1.), train_set_spanner_handover_emg_09[:,ch_ex]) ))
    train_set_spanner_handover_emg_norm10 = np.hstack(( train_set_spanner_handover_emg_norm10, np.interp(np.linspace(0, len(train_set_spanner_handover_emg_10)-1, len_normal), np.arange(0,len(train_set_spanner_handover_emg_10),1.), train_set_spanner_handover_emg_10[:,ch_ex]) ))
    train_set_spanner_handover_emg_norm11 = np.hstack(( train_set_spanner_handover_emg_norm11, np.interp(np.linspace(0, len(train_set_spanner_handover_emg_11)-1, len_normal), np.arange(0,len(train_set_spanner_handover_emg_11),1.), train_set_spanner_handover_emg_11[:,ch_ex]) ))
    train_set_spanner_handover_emg_norm12 = np.hstack(( train_set_spanner_handover_emg_norm12, np.interp(np.linspace(0, len(train_set_spanner_handover_emg_12)-1, len_normal), np.arange(0,len(train_set_spanner_handover_emg_12),1.), train_set_spanner_handover_emg_12[:,ch_ex]) ))
    train_set_spanner_handover_emg_norm13 = np.hstack(( train_set_spanner_handover_emg_norm13, np.interp(np.linspace(0, len(train_set_spanner_handover_emg_13)-1, len_normal), np.arange(0,len(train_set_spanner_handover_emg_13),1.), train_set_spanner_handover_emg_13[:,ch_ex]) ))
    train_set_spanner_handover_emg_norm14 = np.hstack(( train_set_spanner_handover_emg_norm14, np.interp(np.linspace(0, len(train_set_spanner_handover_emg_14)-1, len_normal), np.arange(0,len(train_set_spanner_handover_emg_14),1.), train_set_spanner_handover_emg_14[:,ch_ex]) ))
    train_set_spanner_handover_emg_norm15 = np.hstack(( train_set_spanner_handover_emg_norm15, np.interp(np.linspace(0, len(train_set_spanner_handover_emg_15)-1, len_normal), np.arange(0,len(train_set_spanner_handover_emg_15),1.), train_set_spanner_handover_emg_15[:,ch_ex]) ))
    train_set_spanner_handover_emg_norm16 = np.hstack(( train_set_spanner_handover_emg_norm16, np.interp(np.linspace(0, len(train_set_spanner_handover_emg_16)-1, len_normal), np.arange(0,len(train_set_spanner_handover_emg_16),1.), train_set_spanner_handover_emg_16[:,ch_ex]) ))
    train_set_spanner_handover_emg_norm17 = np.hstack(( train_set_spanner_handover_emg_norm17, np.interp(np.linspace(0, len(train_set_spanner_handover_emg_17)-1, len_normal), np.arange(0,len(train_set_spanner_handover_emg_17),1.), train_set_spanner_handover_emg_17[:,ch_ex]) ))
    train_set_spanner_handover_emg_norm18 = np.hstack(( train_set_spanner_handover_emg_norm18, np.interp(np.linspace(0, len(train_set_spanner_handover_emg_18)-1, len_normal), np.arange(0,len(train_set_spanner_handover_emg_18),1.), train_set_spanner_handover_emg_18[:,ch_ex]) ))
    train_set_spanner_handover_emg_norm19 = np.hstack(( train_set_spanner_handover_emg_norm19, np.interp(np.linspace(0, len(train_set_spanner_handover_emg_19)-1, len_normal), np.arange(0,len(train_set_spanner_handover_emg_19),1.), train_set_spanner_handover_emg_19[:,ch_ex]) ))
    train_set_spanner_handover_emg_norm20 = np.hstack((train_set_spanner_handover_emg_norm20, np.interp( np.linspace(0, len(train_set_spanner_handover_emg_20) - 1, len_normal), np.arange(0, len(train_set_spanner_handover_emg_20), 1.), train_set_spanner_handover_emg_20[:, ch_ex])))
    train_set_spanner_handover_emg_norm21 = np.hstack((train_set_spanner_handover_emg_norm21, np.interp( np.linspace(0, len(train_set_spanner_handover_emg_21) - 1, len_normal), np.arange(0, len(train_set_spanner_handover_emg_21), 1.), train_set_spanner_handover_emg_21[:, ch_ex])))
    train_set_spanner_handover_emg_norm22 = np.hstack((train_set_spanner_handover_emg_norm22, np.interp( np.linspace(0, len(train_set_spanner_handover_emg_22) - 1, len_normal), np.arange(0, len(train_set_spanner_handover_emg_22), 1.), train_set_spanner_handover_emg_22[:, ch_ex])))
    train_set_spanner_handover_emg_norm23 = np.hstack((train_set_spanner_handover_emg_norm23, np.interp( np.linspace(0, len(train_set_spanner_handover_emg_23) - 1, len_normal), np.arange(0, len(train_set_spanner_handover_emg_23), 1.), train_set_spanner_handover_emg_23[:, ch_ex])))
    train_set_spanner_handover_emg_norm24 = np.hstack((train_set_spanner_handover_emg_norm24, np.interp( np.linspace(0, len(train_set_spanner_handover_emg_24) - 1, len_normal), np.arange(0, len(train_set_spanner_handover_emg_24), 1.), train_set_spanner_handover_emg_24[:, ch_ex])))
    train_set_spanner_handover_emg_norm25 = np.hstack((train_set_spanner_handover_emg_norm25, np.interp( np.linspace(0, len(train_set_spanner_handover_emg_25) - 1, len_normal), np.arange(0, len(train_set_spanner_handover_emg_25), 1.), train_set_spanner_handover_emg_25[:, ch_ex])))
    train_set_spanner_handover_emg_norm26 = np.hstack((train_set_spanner_handover_emg_norm26, np.interp( np.linspace(0, len(train_set_spanner_handover_emg_26) - 1, len_normal), np.arange(0, len(train_set_spanner_handover_emg_26), 1.), train_set_spanner_handover_emg_26[:, ch_ex])))
    train_set_spanner_handover_emg_norm27 = np.hstack((train_set_spanner_handover_emg_norm27, np.interp( np.linspace(0, len(train_set_spanner_handover_emg_27) - 1, len_normal), np.arange(0, len(train_set_spanner_handover_emg_27), 1.), train_set_spanner_handover_emg_27[:, ch_ex])))
    train_set_spanner_handover_emg_norm28 = np.hstack((train_set_spanner_handover_emg_norm28, np.interp( np.linspace(0, len(train_set_spanner_handover_emg_28) - 1, len_normal), np.arange(0, len(train_set_spanner_handover_emg_28), 1.), train_set_spanner_handover_emg_28[:, ch_ex])))
    train_set_spanner_handover_emg_norm29 = np.hstack((train_set_spanner_handover_emg_norm29, np.interp( np.linspace(0, len(train_set_spanner_handover_emg_29) - 1, len_normal), np.arange(0, len(train_set_spanner_handover_emg_29), 1.), train_set_spanner_handover_emg_29[:, ch_ex])))
    test_set_spanner_handover_emg_norm = np.hstack(( test_set_spanner_handover_emg_norm, np.interp(np.linspace(0, len(test_set_spanner_handover_emg)-1, len_normal), np.arange(0,len(test_set_spanner_handover_emg),1.), test_set_spanner_handover_emg[:,ch_ex]) ))
train_set_spanner_handover_emg_norm00 = train_set_spanner_handover_emg_norm00.reshape(8,len_normal).T
train_set_spanner_handover_emg_norm01 = train_set_spanner_handover_emg_norm01.reshape(8,len_normal).T
train_set_spanner_handover_emg_norm02 = train_set_spanner_handover_emg_norm02.reshape(8,len_normal).T
train_set_spanner_handover_emg_norm03 = train_set_spanner_handover_emg_norm03.reshape(8,len_normal).T
train_set_spanner_handover_emg_norm04 = train_set_spanner_handover_emg_norm04.reshape(8,len_normal).T
train_set_spanner_handover_emg_norm05 = train_set_spanner_handover_emg_norm05.reshape(8,len_normal).T
train_set_spanner_handover_emg_norm06 = train_set_spanner_handover_emg_norm06.reshape(8,len_normal).T
train_set_spanner_handover_emg_norm07 = train_set_spanner_handover_emg_norm07.reshape(8,len_normal).T
train_set_spanner_handover_emg_norm08 = train_set_spanner_handover_emg_norm08.reshape(8,len_normal).T
train_set_spanner_handover_emg_norm09 = train_set_spanner_handover_emg_norm09.reshape(8,len_normal).T
train_set_spanner_handover_emg_norm10 = train_set_spanner_handover_emg_norm10.reshape(8,len_normal).T
train_set_spanner_handover_emg_norm11 = train_set_spanner_handover_emg_norm11.reshape(8,len_normal).T
train_set_spanner_handover_emg_norm12 = train_set_spanner_handover_emg_norm12.reshape(8,len_normal).T
train_set_spanner_handover_emg_norm13 = train_set_spanner_handover_emg_norm13.reshape(8,len_normal).T
train_set_spanner_handover_emg_norm14 = train_set_spanner_handover_emg_norm14.reshape(8,len_normal).T
train_set_spanner_handover_emg_norm15 = train_set_spanner_handover_emg_norm15.reshape(8,len_normal).T
train_set_spanner_handover_emg_norm16 = train_set_spanner_handover_emg_norm16.reshape(8,len_normal).T
train_set_spanner_handover_emg_norm17 = train_set_spanner_handover_emg_norm17.reshape(8,len_normal).T
train_set_spanner_handover_emg_norm18 = train_set_spanner_handover_emg_norm18.reshape(8,len_normal).T
train_set_spanner_handover_emg_norm19 = train_set_spanner_handover_emg_norm19.reshape(8,len_normal).T
train_set_spanner_handover_emg_norm20 = train_set_spanner_handover_emg_norm20.reshape(8,len_normal).T
train_set_spanner_handover_emg_norm21 = train_set_spanner_handover_emg_norm21.reshape(8,len_normal).T
train_set_spanner_handover_emg_norm22 = train_set_spanner_handover_emg_norm22.reshape(8,len_normal).T
train_set_spanner_handover_emg_norm23 = train_set_spanner_handover_emg_norm23.reshape(8,len_normal).T
train_set_spanner_handover_emg_norm24 = train_set_spanner_handover_emg_norm24.reshape(8,len_normal).T
train_set_spanner_handover_emg_norm25 = train_set_spanner_handover_emg_norm25.reshape(8,len_normal).T
train_set_spanner_handover_emg_norm26 = train_set_spanner_handover_emg_norm26.reshape(8,len_normal).T
train_set_spanner_handover_emg_norm27 = train_set_spanner_handover_emg_norm27.reshape(8,len_normal).T
train_set_spanner_handover_emg_norm28 = train_set_spanner_handover_emg_norm28.reshape(8,len_normal).T
train_set_spanner_handover_emg_norm29 = train_set_spanner_handover_emg_norm29.reshape(8,len_normal).T
test_set_spanner_handover_emg_norm = test_set_spanner_handover_emg_norm.reshape(8,len_normal).T
train_set_spanner_handover_emg_norm_full = np.array([train_set_spanner_handover_emg_norm00,train_set_spanner_handover_emg_norm01,train_set_spanner_handover_emg_norm02,train_set_spanner_handover_emg_norm03,train_set_spanner_handover_emg_norm04,
                                    train_set_spanner_handover_emg_norm05,train_set_spanner_handover_emg_norm06,train_set_spanner_handover_emg_norm07,train_set_spanner_handover_emg_norm08,train_set_spanner_handover_emg_norm09,
                                    train_set_spanner_handover_emg_norm10,train_set_spanner_handover_emg_norm11,train_set_spanner_handover_emg_norm12,train_set_spanner_handover_emg_norm13,train_set_spanner_handover_emg_norm14,
                                    train_set_spanner_handover_emg_norm15,train_set_spanner_handover_emg_norm16,train_set_spanner_handover_emg_norm17,train_set_spanner_handover_emg_norm18,train_set_spanner_handover_emg_norm19])
##########################################################################################
print('normalizing EMG data into same duration of tape_hold')
# resampling emg signals of aluminum_hold
train_set_tape_hold_emg_norm00=np.array([]);train_set_tape_hold_emg_norm01=np.array([]);train_set_tape_hold_emg_norm02=np.array([]);train_set_tape_hold_emg_norm03=np.array([]);train_set_tape_hold_emg_norm04=np.array([]);
train_set_tape_hold_emg_norm05=np.array([]);train_set_tape_hold_emg_norm06=np.array([]);train_set_tape_hold_emg_norm07=np.array([]);train_set_tape_hold_emg_norm08=np.array([]);train_set_tape_hold_emg_norm09=np.array([]);
train_set_tape_hold_emg_norm10=np.array([]);train_set_tape_hold_emg_norm11=np.array([]);train_set_tape_hold_emg_norm12=np.array([]);train_set_tape_hold_emg_norm13=np.array([]);train_set_tape_hold_emg_norm14=np.array([]);
train_set_tape_hold_emg_norm15=np.array([]);train_set_tape_hold_emg_norm16=np.array([]);train_set_tape_hold_emg_norm17=np.array([]);train_set_tape_hold_emg_norm18=np.array([]);train_set_tape_hold_emg_norm19=np.array([]);
train_set_tape_hold_emg_norm20=np.array([]);train_set_tape_hold_emg_norm21=np.array([]);train_set_tape_hold_emg_norm22=np.array([]);train_set_tape_hold_emg_norm23=np.array([]);train_set_tape_hold_emg_norm24=np.array([]);
train_set_tape_hold_emg_norm25=np.array([]);train_set_tape_hold_emg_norm26=np.array([]);train_set_tape_hold_emg_norm27=np.array([]);train_set_tape_hold_emg_norm28=np.array([]);train_set_tape_hold_emg_norm29=np.array([]);
test_set_tape_hold_emg_norm=np.array([]);
for ch_ex in range(8):
    train_set_tape_hold_emg_norm00 = np.hstack(( train_set_tape_hold_emg_norm00, np.interp(np.linspace(0, len(train_set_tape_hold_emg_00)-1, len_normal), np.arange(0,len(train_set_tape_hold_emg_00),1.), train_set_tape_hold_emg_00[:,ch_ex]) ))
    train_set_tape_hold_emg_norm01 = np.hstack(( train_set_tape_hold_emg_norm01, np.interp(np.linspace(0, len(train_set_tape_hold_emg_01)-1, len_normal), np.arange(0,len(train_set_tape_hold_emg_01),1.), train_set_tape_hold_emg_01[:,ch_ex]) ))
    train_set_tape_hold_emg_norm02 = np.hstack(( train_set_tape_hold_emg_norm02, np.interp(np.linspace(0, len(train_set_tape_hold_emg_02)-1, len_normal), np.arange(0,len(train_set_tape_hold_emg_02),1.), train_set_tape_hold_emg_02[:,ch_ex]) ))
    train_set_tape_hold_emg_norm03 = np.hstack(( train_set_tape_hold_emg_norm03, np.interp(np.linspace(0, len(train_set_tape_hold_emg_03)-1, len_normal), np.arange(0,len(train_set_tape_hold_emg_03),1.), train_set_tape_hold_emg_03[:,ch_ex]) ))
    train_set_tape_hold_emg_norm04 = np.hstack(( train_set_tape_hold_emg_norm04, np.interp(np.linspace(0, len(train_set_tape_hold_emg_04)-1, len_normal), np.arange(0,len(train_set_tape_hold_emg_04),1.), train_set_tape_hold_emg_04[:,ch_ex]) ))
    train_set_tape_hold_emg_norm05 = np.hstack(( train_set_tape_hold_emg_norm05, np.interp(np.linspace(0, len(train_set_tape_hold_emg_05)-1, len_normal), np.arange(0,len(train_set_tape_hold_emg_05),1.), train_set_tape_hold_emg_05[:,ch_ex]) ))
    train_set_tape_hold_emg_norm06 = np.hstack(( train_set_tape_hold_emg_norm06, np.interp(np.linspace(0, len(train_set_tape_hold_emg_06)-1, len_normal), np.arange(0,len(train_set_tape_hold_emg_06),1.), train_set_tape_hold_emg_06[:,ch_ex]) ))
    train_set_tape_hold_emg_norm07 = np.hstack(( train_set_tape_hold_emg_norm07, np.interp(np.linspace(0, len(train_set_tape_hold_emg_07)-1, len_normal), np.arange(0,len(train_set_tape_hold_emg_07),1.), train_set_tape_hold_emg_07[:,ch_ex]) ))
    train_set_tape_hold_emg_norm08 = np.hstack(( train_set_tape_hold_emg_norm08, np.interp(np.linspace(0, len(train_set_tape_hold_emg_08)-1, len_normal), np.arange(0,len(train_set_tape_hold_emg_08),1.), train_set_tape_hold_emg_08[:,ch_ex]) ))
    train_set_tape_hold_emg_norm09 = np.hstack(( train_set_tape_hold_emg_norm09, np.interp(np.linspace(0, len(train_set_tape_hold_emg_09)-1, len_normal), np.arange(0,len(train_set_tape_hold_emg_09),1.), train_set_tape_hold_emg_09[:,ch_ex]) ))
    train_set_tape_hold_emg_norm10 = np.hstack(( train_set_tape_hold_emg_norm10, np.interp(np.linspace(0, len(train_set_tape_hold_emg_10)-1, len_normal), np.arange(0,len(train_set_tape_hold_emg_10),1.), train_set_tape_hold_emg_10[:,ch_ex]) ))
    train_set_tape_hold_emg_norm11 = np.hstack(( train_set_tape_hold_emg_norm11, np.interp(np.linspace(0, len(train_set_tape_hold_emg_11)-1, len_normal), np.arange(0,len(train_set_tape_hold_emg_11),1.), train_set_tape_hold_emg_11[:,ch_ex]) ))
    train_set_tape_hold_emg_norm12 = np.hstack(( train_set_tape_hold_emg_norm12, np.interp(np.linspace(0, len(train_set_tape_hold_emg_12)-1, len_normal), np.arange(0,len(train_set_tape_hold_emg_12),1.), train_set_tape_hold_emg_12[:,ch_ex]) ))
    train_set_tape_hold_emg_norm13 = np.hstack(( train_set_tape_hold_emg_norm13, np.interp(np.linspace(0, len(train_set_tape_hold_emg_13)-1, len_normal), np.arange(0,len(train_set_tape_hold_emg_13),1.), train_set_tape_hold_emg_13[:,ch_ex]) ))
    train_set_tape_hold_emg_norm14 = np.hstack(( train_set_tape_hold_emg_norm14, np.interp(np.linspace(0, len(train_set_tape_hold_emg_14)-1, len_normal), np.arange(0,len(train_set_tape_hold_emg_14),1.), train_set_tape_hold_emg_14[:,ch_ex]) ))
    train_set_tape_hold_emg_norm15 = np.hstack(( train_set_tape_hold_emg_norm15, np.interp(np.linspace(0, len(train_set_tape_hold_emg_15)-1, len_normal), np.arange(0,len(train_set_tape_hold_emg_15),1.), train_set_tape_hold_emg_15[:,ch_ex]) ))
    train_set_tape_hold_emg_norm16 = np.hstack(( train_set_tape_hold_emg_norm16, np.interp(np.linspace(0, len(train_set_tape_hold_emg_16)-1, len_normal), np.arange(0,len(train_set_tape_hold_emg_16),1.), train_set_tape_hold_emg_16[:,ch_ex]) ))
    train_set_tape_hold_emg_norm17 = np.hstack(( train_set_tape_hold_emg_norm17, np.interp(np.linspace(0, len(train_set_tape_hold_emg_17)-1, len_normal), np.arange(0,len(train_set_tape_hold_emg_17),1.), train_set_tape_hold_emg_17[:,ch_ex]) ))
    train_set_tape_hold_emg_norm18 = np.hstack(( train_set_tape_hold_emg_norm18, np.interp(np.linspace(0, len(train_set_tape_hold_emg_18)-1, len_normal), np.arange(0,len(train_set_tape_hold_emg_18),1.), train_set_tape_hold_emg_18[:,ch_ex]) ))
    train_set_tape_hold_emg_norm19 = np.hstack(( train_set_tape_hold_emg_norm19, np.interp(np.linspace(0, len(train_set_tape_hold_emg_19)-1, len_normal), np.arange(0,len(train_set_tape_hold_emg_19),1.), train_set_tape_hold_emg_19[:,ch_ex]) ))
    train_set_tape_hold_emg_norm20 = np.hstack((train_set_tape_hold_emg_norm20, np.interp( np.linspace(0, len(train_set_tape_hold_emg_20) - 1, len_normal), np.arange(0, len(train_set_tape_hold_emg_20), 1.), train_set_tape_hold_emg_20[:, ch_ex])))
    train_set_tape_hold_emg_norm21 = np.hstack((train_set_tape_hold_emg_norm21, np.interp( np.linspace(0, len(train_set_tape_hold_emg_21) - 1, len_normal), np.arange(0, len(train_set_tape_hold_emg_21), 1.), train_set_tape_hold_emg_21[:, ch_ex])))
    train_set_tape_hold_emg_norm22 = np.hstack((train_set_tape_hold_emg_norm22, np.interp( np.linspace(0, len(train_set_tape_hold_emg_22) - 1, len_normal), np.arange(0, len(train_set_tape_hold_emg_22), 1.), train_set_tape_hold_emg_22[:, ch_ex])))
    train_set_tape_hold_emg_norm23 = np.hstack((train_set_tape_hold_emg_norm23, np.interp( np.linspace(0, len(train_set_tape_hold_emg_23) - 1, len_normal), np.arange(0, len(train_set_tape_hold_emg_23), 1.), train_set_tape_hold_emg_23[:, ch_ex])))
    train_set_tape_hold_emg_norm24 = np.hstack((train_set_tape_hold_emg_norm24, np.interp( np.linspace(0, len(train_set_tape_hold_emg_24) - 1, len_normal), np.arange(0, len(train_set_tape_hold_emg_24), 1.), train_set_tape_hold_emg_24[:, ch_ex])))
    train_set_tape_hold_emg_norm25 = np.hstack((train_set_tape_hold_emg_norm25, np.interp( np.linspace(0, len(train_set_tape_hold_emg_25) - 1, len_normal), np.arange(0, len(train_set_tape_hold_emg_25), 1.), train_set_tape_hold_emg_25[:, ch_ex])))
    train_set_tape_hold_emg_norm26 = np.hstack((train_set_tape_hold_emg_norm26, np.interp( np.linspace(0, len(train_set_tape_hold_emg_26) - 1, len_normal), np.arange(0, len(train_set_tape_hold_emg_26), 1.), train_set_tape_hold_emg_26[:, ch_ex])))
    train_set_tape_hold_emg_norm27 = np.hstack((train_set_tape_hold_emg_norm27, np.interp( np.linspace(0, len(train_set_tape_hold_emg_27) - 1, len_normal), np.arange(0, len(train_set_tape_hold_emg_27), 1.), train_set_tape_hold_emg_27[:, ch_ex])))
    train_set_tape_hold_emg_norm28 = np.hstack((train_set_tape_hold_emg_norm28, np.interp( np.linspace(0, len(train_set_tape_hold_emg_28) - 1, len_normal), np.arange(0, len(train_set_tape_hold_emg_28), 1.), train_set_tape_hold_emg_28[:, ch_ex])))
    train_set_tape_hold_emg_norm29 = np.hstack((train_set_tape_hold_emg_norm29, np.interp( np.linspace(0, len(train_set_tape_hold_emg_29) - 1, len_normal), np.arange(0, len(train_set_tape_hold_emg_29), 1.), train_set_tape_hold_emg_29[:, ch_ex])))
    test_set_tape_hold_emg_norm = np.hstack(( test_set_tape_hold_emg_norm, np.interp(np.linspace(0, len(test_set_tape_hold_emg)-1, len_normal), np.arange(0,len(test_set_tape_hold_emg),1.), test_set_tape_hold_emg[:,ch_ex]) ))
train_set_tape_hold_emg_norm00 = train_set_tape_hold_emg_norm00.reshape(8,len_normal).T
train_set_tape_hold_emg_norm01 = train_set_tape_hold_emg_norm01.reshape(8,len_normal).T
train_set_tape_hold_emg_norm02 = train_set_tape_hold_emg_norm02.reshape(8,len_normal).T
train_set_tape_hold_emg_norm03 = train_set_tape_hold_emg_norm03.reshape(8,len_normal).T
train_set_tape_hold_emg_norm04 = train_set_tape_hold_emg_norm04.reshape(8,len_normal).T
train_set_tape_hold_emg_norm05 = train_set_tape_hold_emg_norm05.reshape(8,len_normal).T
train_set_tape_hold_emg_norm06 = train_set_tape_hold_emg_norm06.reshape(8,len_normal).T
train_set_tape_hold_emg_norm07 = train_set_tape_hold_emg_norm07.reshape(8,len_normal).T
train_set_tape_hold_emg_norm08 = train_set_tape_hold_emg_norm08.reshape(8,len_normal).T
train_set_tape_hold_emg_norm09 = train_set_tape_hold_emg_norm09.reshape(8,len_normal).T
train_set_tape_hold_emg_norm10 = train_set_tape_hold_emg_norm10.reshape(8,len_normal).T
train_set_tape_hold_emg_norm11 = train_set_tape_hold_emg_norm11.reshape(8,len_normal).T
train_set_tape_hold_emg_norm12 = train_set_tape_hold_emg_norm12.reshape(8,len_normal).T
train_set_tape_hold_emg_norm13 = train_set_tape_hold_emg_norm13.reshape(8,len_normal).T
train_set_tape_hold_emg_norm14 = train_set_tape_hold_emg_norm14.reshape(8,len_normal).T
train_set_tape_hold_emg_norm15 = train_set_tape_hold_emg_norm15.reshape(8,len_normal).T
train_set_tape_hold_emg_norm16 = train_set_tape_hold_emg_norm16.reshape(8,len_normal).T
train_set_tape_hold_emg_norm17 = train_set_tape_hold_emg_norm17.reshape(8,len_normal).T
train_set_tape_hold_emg_norm18 = train_set_tape_hold_emg_norm18.reshape(8,len_normal).T
train_set_tape_hold_emg_norm19 = train_set_tape_hold_emg_norm19.reshape(8,len_normal).T
train_set_tape_hold_emg_norm20 = train_set_tape_hold_emg_norm20.reshape(8,len_normal).T
train_set_tape_hold_emg_norm21 = train_set_tape_hold_emg_norm21.reshape(8,len_normal).T
train_set_tape_hold_emg_norm22 = train_set_tape_hold_emg_norm22.reshape(8,len_normal).T
train_set_tape_hold_emg_norm23 = train_set_tape_hold_emg_norm23.reshape(8,len_normal).T
train_set_tape_hold_emg_norm24 = train_set_tape_hold_emg_norm24.reshape(8,len_normal).T
train_set_tape_hold_emg_norm25 = train_set_tape_hold_emg_norm25.reshape(8,len_normal).T
train_set_tape_hold_emg_norm26 = train_set_tape_hold_emg_norm26.reshape(8,len_normal).T
train_set_tape_hold_emg_norm27 = train_set_tape_hold_emg_norm27.reshape(8,len_normal).T
train_set_tape_hold_emg_norm28 = train_set_tape_hold_emg_norm28.reshape(8,len_normal).T
train_set_tape_hold_emg_norm29 = train_set_tape_hold_emg_norm29.reshape(8,len_normal).T
test_set_tape_hold_emg_norm = test_set_tape_hold_emg_norm.reshape(8,len_normal).T
train_set_tape_hold_emg_norm_full = np.array([train_set_tape_hold_emg_norm00,train_set_tape_hold_emg_norm01,train_set_tape_hold_emg_norm02,train_set_tape_hold_emg_norm03,train_set_tape_hold_emg_norm04,
                                    train_set_tape_hold_emg_norm05,train_set_tape_hold_emg_norm06,train_set_tape_hold_emg_norm07,train_set_tape_hold_emg_norm08,train_set_tape_hold_emg_norm09,
                                    train_set_tape_hold_emg_norm10,train_set_tape_hold_emg_norm11,train_set_tape_hold_emg_norm12,train_set_tape_hold_emg_norm13,train_set_tape_hold_emg_norm14,
                                    train_set_tape_hold_emg_norm15,train_set_tape_hold_emg_norm16,train_set_tape_hold_emg_norm17,train_set_tape_hold_emg_norm18,train_set_tape_hold_emg_norm19])


##################################################################################
# resampling the IMU data for experiencing the same duration
##################################################################################
# rospy.loginfo('normalizing data into same duration')
print('normalizing IMU data into same duration of aluminum_hold')
# resampling imu signals of aluminum_hold
train_set_aluminum_hold_imu_norm00=np.array([]);train_set_aluminum_hold_imu_norm01=np.array([]);train_set_aluminum_hold_imu_norm02=np.array([]);train_set_aluminum_hold_imu_norm03=np.array([]);train_set_aluminum_hold_imu_norm04=np.array([]);
train_set_aluminum_hold_imu_norm05=np.array([]);train_set_aluminum_hold_imu_norm06=np.array([]);train_set_aluminum_hold_imu_norm07=np.array([]);train_set_aluminum_hold_imu_norm08=np.array([]);train_set_aluminum_hold_imu_norm09=np.array([]);
train_set_aluminum_hold_imu_norm10=np.array([]);train_set_aluminum_hold_imu_norm11=np.array([]);train_set_aluminum_hold_imu_norm12=np.array([]);train_set_aluminum_hold_imu_norm13=np.array([]);train_set_aluminum_hold_imu_norm14=np.array([]);
train_set_aluminum_hold_imu_norm15=np.array([]);train_set_aluminum_hold_imu_norm16=np.array([]);train_set_aluminum_hold_imu_norm17=np.array([]);train_set_aluminum_hold_imu_norm18=np.array([]);train_set_aluminum_hold_imu_norm19=np.array([]);
train_set_aluminum_hold_imu_norm20=np.array([]);train_set_aluminum_hold_imu_norm21=np.array([]);train_set_aluminum_hold_imu_norm22=np.array([]);train_set_aluminum_hold_imu_norm23=np.array([]);train_set_aluminum_hold_imu_norm24=np.array([]);
train_set_aluminum_hold_imu_norm25=np.array([]);train_set_aluminum_hold_imu_norm26=np.array([]);train_set_aluminum_hold_imu_norm27=np.array([]);train_set_aluminum_hold_imu_norm28=np.array([]);train_set_aluminum_hold_imu_norm29=np.array([]);
test_set_aluminum_hold_imu_norm=np.array([]);
for ch_ex in range(4):
    train_set_aluminum_hold_imu_norm00 = np.hstack(( train_set_aluminum_hold_imu_norm00, np.interp(np.linspace(0, len(train_set_aluminum_hold_imu_00)-1, len_normal), np.arange(0,len(train_set_aluminum_hold_imu_00),1.), train_set_aluminum_hold_imu_00[:,ch_ex]) ))
    train_set_aluminum_hold_imu_norm01 = np.hstack(( train_set_aluminum_hold_imu_norm01, np.interp(np.linspace(0, len(train_set_aluminum_hold_imu_01)-1, len_normal), np.arange(0,len(train_set_aluminum_hold_imu_01),1.), train_set_aluminum_hold_imu_01[:,ch_ex]) ))
    train_set_aluminum_hold_imu_norm02 = np.hstack(( train_set_aluminum_hold_imu_norm02, np.interp(np.linspace(0, len(train_set_aluminum_hold_imu_02)-1, len_normal), np.arange(0,len(train_set_aluminum_hold_imu_02),1.), train_set_aluminum_hold_imu_02[:,ch_ex]) ))
    train_set_aluminum_hold_imu_norm03 = np.hstack(( train_set_aluminum_hold_imu_norm03, np.interp(np.linspace(0, len(train_set_aluminum_hold_imu_03)-1, len_normal), np.arange(0,len(train_set_aluminum_hold_imu_03),1.), train_set_aluminum_hold_imu_03[:,ch_ex]) ))
    train_set_aluminum_hold_imu_norm04 = np.hstack(( train_set_aluminum_hold_imu_norm04, np.interp(np.linspace(0, len(train_set_aluminum_hold_imu_04)-1, len_normal), np.arange(0,len(train_set_aluminum_hold_imu_04),1.), train_set_aluminum_hold_imu_04[:,ch_ex]) ))
    train_set_aluminum_hold_imu_norm05 = np.hstack(( train_set_aluminum_hold_imu_norm05, np.interp(np.linspace(0, len(train_set_aluminum_hold_imu_05)-1, len_normal), np.arange(0,len(train_set_aluminum_hold_imu_05),1.), train_set_aluminum_hold_imu_05[:,ch_ex]) ))
    train_set_aluminum_hold_imu_norm06 = np.hstack(( train_set_aluminum_hold_imu_norm06, np.interp(np.linspace(0, len(train_set_aluminum_hold_imu_06)-1, len_normal), np.arange(0,len(train_set_aluminum_hold_imu_06),1.), train_set_aluminum_hold_imu_06[:,ch_ex]) ))
    train_set_aluminum_hold_imu_norm07 = np.hstack(( train_set_aluminum_hold_imu_norm07, np.interp(np.linspace(0, len(train_set_aluminum_hold_imu_07)-1, len_normal), np.arange(0,len(train_set_aluminum_hold_imu_07),1.), train_set_aluminum_hold_imu_07[:,ch_ex]) ))
    train_set_aluminum_hold_imu_norm08 = np.hstack(( train_set_aluminum_hold_imu_norm08, np.interp(np.linspace(0, len(train_set_aluminum_hold_imu_08)-1, len_normal), np.arange(0,len(train_set_aluminum_hold_imu_08),1.), train_set_aluminum_hold_imu_08[:,ch_ex]) ))
    train_set_aluminum_hold_imu_norm09 = np.hstack(( train_set_aluminum_hold_imu_norm09, np.interp(np.linspace(0, len(train_set_aluminum_hold_imu_09)-1, len_normal), np.arange(0,len(train_set_aluminum_hold_imu_09),1.), train_set_aluminum_hold_imu_09[:,ch_ex]) ))
    train_set_aluminum_hold_imu_norm10 = np.hstack(( train_set_aluminum_hold_imu_norm10, np.interp(np.linspace(0, len(train_set_aluminum_hold_imu_10)-1, len_normal), np.arange(0,len(train_set_aluminum_hold_imu_10),1.), train_set_aluminum_hold_imu_10[:,ch_ex]) ))
    train_set_aluminum_hold_imu_norm11 = np.hstack(( train_set_aluminum_hold_imu_norm11, np.interp(np.linspace(0, len(train_set_aluminum_hold_imu_11)-1, len_normal), np.arange(0,len(train_set_aluminum_hold_imu_11),1.), train_set_aluminum_hold_imu_11[:,ch_ex]) ))
    train_set_aluminum_hold_imu_norm12 = np.hstack(( train_set_aluminum_hold_imu_norm12, np.interp(np.linspace(0, len(train_set_aluminum_hold_imu_12)-1, len_normal), np.arange(0,len(train_set_aluminum_hold_imu_12),1.), train_set_aluminum_hold_imu_12[:,ch_ex]) ))
    train_set_aluminum_hold_imu_norm13 = np.hstack(( train_set_aluminum_hold_imu_norm13, np.interp(np.linspace(0, len(train_set_aluminum_hold_imu_13)-1, len_normal), np.arange(0,len(train_set_aluminum_hold_imu_13),1.), train_set_aluminum_hold_imu_13[:,ch_ex]) ))
    train_set_aluminum_hold_imu_norm14 = np.hstack(( train_set_aluminum_hold_imu_norm14, np.interp(np.linspace(0, len(train_set_aluminum_hold_imu_14)-1, len_normal), np.arange(0,len(train_set_aluminum_hold_imu_14),1.), train_set_aluminum_hold_imu_14[:,ch_ex]) ))
    train_set_aluminum_hold_imu_norm15 = np.hstack(( train_set_aluminum_hold_imu_norm15, np.interp(np.linspace(0, len(train_set_aluminum_hold_imu_15)-1, len_normal), np.arange(0,len(train_set_aluminum_hold_imu_15),1.), train_set_aluminum_hold_imu_15[:,ch_ex]) ))
    train_set_aluminum_hold_imu_norm16 = np.hstack(( train_set_aluminum_hold_imu_norm16, np.interp(np.linspace(0, len(train_set_aluminum_hold_imu_16)-1, len_normal), np.arange(0,len(train_set_aluminum_hold_imu_16),1.), train_set_aluminum_hold_imu_16[:,ch_ex]) ))
    train_set_aluminum_hold_imu_norm17 = np.hstack(( train_set_aluminum_hold_imu_norm17, np.interp(np.linspace(0, len(train_set_aluminum_hold_imu_17)-1, len_normal), np.arange(0,len(train_set_aluminum_hold_imu_17),1.), train_set_aluminum_hold_imu_17[:,ch_ex]) ))
    train_set_aluminum_hold_imu_norm18 = np.hstack(( train_set_aluminum_hold_imu_norm18, np.interp(np.linspace(0, len(train_set_aluminum_hold_imu_18)-1, len_normal), np.arange(0,len(train_set_aluminum_hold_imu_18),1.), train_set_aluminum_hold_imu_18[:,ch_ex]) ))
    train_set_aluminum_hold_imu_norm19 = np.hstack(( train_set_aluminum_hold_imu_norm19, np.interp(np.linspace(0, len(train_set_aluminum_hold_imu_19)-1, len_normal), np.arange(0,len(train_set_aluminum_hold_imu_19),1.), train_set_aluminum_hold_imu_19[:,ch_ex]) ))
    train_set_aluminum_hold_imu_norm20 = np.hstack((train_set_aluminum_hold_imu_norm20, np.interp( np.linspace(0, len(train_set_aluminum_hold_imu_20) - 1, len_normal), np.arange(0, len(train_set_aluminum_hold_imu_20), 1.), train_set_aluminum_hold_imu_20[:, ch_ex])))
    train_set_aluminum_hold_imu_norm21 = np.hstack((train_set_aluminum_hold_imu_norm21, np.interp( np.linspace(0, len(train_set_aluminum_hold_imu_21) - 1, len_normal), np.arange(0, len(train_set_aluminum_hold_imu_21), 1.), train_set_aluminum_hold_imu_21[:, ch_ex])))
    train_set_aluminum_hold_imu_norm22 = np.hstack((train_set_aluminum_hold_imu_norm22, np.interp( np.linspace(0, len(train_set_aluminum_hold_imu_22) - 1, len_normal), np.arange(0, len(train_set_aluminum_hold_imu_22), 1.), train_set_aluminum_hold_imu_22[:, ch_ex])))
    train_set_aluminum_hold_imu_norm23 = np.hstack((train_set_aluminum_hold_imu_norm23, np.interp( np.linspace(0, len(train_set_aluminum_hold_imu_23) - 1, len_normal), np.arange(0, len(train_set_aluminum_hold_imu_23), 1.), train_set_aluminum_hold_imu_23[:, ch_ex])))
    train_set_aluminum_hold_imu_norm24 = np.hstack((train_set_aluminum_hold_imu_norm24, np.interp( np.linspace(0, len(train_set_aluminum_hold_imu_24) - 1, len_normal), np.arange(0, len(train_set_aluminum_hold_imu_24), 1.), train_set_aluminum_hold_imu_24[:, ch_ex])))
    train_set_aluminum_hold_imu_norm25 = np.hstack((train_set_aluminum_hold_imu_norm25, np.interp( np.linspace(0, len(train_set_aluminum_hold_imu_25) - 1, len_normal), np.arange(0, len(train_set_aluminum_hold_imu_25), 1.), train_set_aluminum_hold_imu_25[:, ch_ex])))
    train_set_aluminum_hold_imu_norm26 = np.hstack((train_set_aluminum_hold_imu_norm26, np.interp( np.linspace(0, len(train_set_aluminum_hold_imu_26) - 1, len_normal), np.arange(0, len(train_set_aluminum_hold_imu_26), 1.), train_set_aluminum_hold_imu_26[:, ch_ex])))
    train_set_aluminum_hold_imu_norm27 = np.hstack((train_set_aluminum_hold_imu_norm27, np.interp( np.linspace(0, len(train_set_aluminum_hold_imu_27) - 1, len_normal), np.arange(0, len(train_set_aluminum_hold_imu_27), 1.), train_set_aluminum_hold_imu_27[:, ch_ex])))
    train_set_aluminum_hold_imu_norm28 = np.hstack((train_set_aluminum_hold_imu_norm28, np.interp( np.linspace(0, len(train_set_aluminum_hold_imu_28) - 1, len_normal), np.arange(0, len(train_set_aluminum_hold_imu_28), 1.), train_set_aluminum_hold_imu_28[:, ch_ex])))
    train_set_aluminum_hold_imu_norm29 = np.hstack((train_set_aluminum_hold_imu_norm29, np.interp( np.linspace(0, len(train_set_aluminum_hold_imu_29) - 1, len_normal), np.arange(0, len(train_set_aluminum_hold_imu_29), 1.), train_set_aluminum_hold_imu_29[:, ch_ex])))
    test_set_aluminum_hold_imu_norm = np.hstack(( test_set_aluminum_hold_imu_norm, np.interp(np.linspace(0, len(test_set_aluminum_hold_imu)-1, len_normal), np.arange(0,len(test_set_aluminum_hold_imu),1.), test_set_aluminum_hold_imu[:,ch_ex]) ))
train_set_aluminum_hold_imu_norm00 = train_set_aluminum_hold_imu_norm00.reshape(4,len_normal).T
train_set_aluminum_hold_imu_norm01 = train_set_aluminum_hold_imu_norm01.reshape(4,len_normal).T
train_set_aluminum_hold_imu_norm02 = train_set_aluminum_hold_imu_norm02.reshape(4,len_normal).T
train_set_aluminum_hold_imu_norm03 = train_set_aluminum_hold_imu_norm03.reshape(4,len_normal).T
train_set_aluminum_hold_imu_norm04 = train_set_aluminum_hold_imu_norm04.reshape(4,len_normal).T
train_set_aluminum_hold_imu_norm05 = train_set_aluminum_hold_imu_norm05.reshape(4,len_normal).T
train_set_aluminum_hold_imu_norm06 = train_set_aluminum_hold_imu_norm06.reshape(4,len_normal).T
train_set_aluminum_hold_imu_norm07 = train_set_aluminum_hold_imu_norm07.reshape(4,len_normal).T
train_set_aluminum_hold_imu_norm08 = train_set_aluminum_hold_imu_norm08.reshape(4,len_normal).T
train_set_aluminum_hold_imu_norm09 = train_set_aluminum_hold_imu_norm09.reshape(4,len_normal).T
train_set_aluminum_hold_imu_norm10 = train_set_aluminum_hold_imu_norm10.reshape(4,len_normal).T
train_set_aluminum_hold_imu_norm11 = train_set_aluminum_hold_imu_norm11.reshape(4,len_normal).T
train_set_aluminum_hold_imu_norm12 = train_set_aluminum_hold_imu_norm12.reshape(4,len_normal).T
train_set_aluminum_hold_imu_norm13 = train_set_aluminum_hold_imu_norm13.reshape(4,len_normal).T
train_set_aluminum_hold_imu_norm14 = train_set_aluminum_hold_imu_norm14.reshape(4,len_normal).T
train_set_aluminum_hold_imu_norm15 = train_set_aluminum_hold_imu_norm15.reshape(4,len_normal).T
train_set_aluminum_hold_imu_norm16 = train_set_aluminum_hold_imu_norm16.reshape(4,len_normal).T
train_set_aluminum_hold_imu_norm17 = train_set_aluminum_hold_imu_norm17.reshape(4,len_normal).T
train_set_aluminum_hold_imu_norm18 = train_set_aluminum_hold_imu_norm18.reshape(4,len_normal).T
train_set_aluminum_hold_imu_norm19 = train_set_aluminum_hold_imu_norm19.reshape(4,len_normal).T
train_set_aluminum_hold_imu_norm20 = train_set_aluminum_hold_imu_norm20.reshape(4,len_normal).T
train_set_aluminum_hold_imu_norm21 = train_set_aluminum_hold_imu_norm21.reshape(4,len_normal).T
train_set_aluminum_hold_imu_norm22 = train_set_aluminum_hold_imu_norm22.reshape(4,len_normal).T
train_set_aluminum_hold_imu_norm23 = train_set_aluminum_hold_imu_norm23.reshape(4,len_normal).T
train_set_aluminum_hold_imu_norm24 = train_set_aluminum_hold_imu_norm24.reshape(4,len_normal).T
train_set_aluminum_hold_imu_norm25 = train_set_aluminum_hold_imu_norm25.reshape(4,len_normal).T
train_set_aluminum_hold_imu_norm26 = train_set_aluminum_hold_imu_norm26.reshape(4,len_normal).T
train_set_aluminum_hold_imu_norm27 = train_set_aluminum_hold_imu_norm27.reshape(4,len_normal).T
train_set_aluminum_hold_imu_norm28 = train_set_aluminum_hold_imu_norm28.reshape(4,len_normal).T
train_set_aluminum_hold_imu_norm29 = train_set_aluminum_hold_imu_norm29.reshape(4,len_normal).T
test_set_aluminum_hold_imu_norm = test_set_aluminum_hold_imu_norm.reshape(4,len_normal).T
train_set_aluminum_hold_imu_norm_full = np.array([train_set_aluminum_hold_imu_norm00,train_set_aluminum_hold_imu_norm01,train_set_aluminum_hold_imu_norm02,train_set_aluminum_hold_imu_norm03,train_set_aluminum_hold_imu_norm04,
                                    train_set_aluminum_hold_imu_norm05,train_set_aluminum_hold_imu_norm06,train_set_aluminum_hold_imu_norm07,train_set_aluminum_hold_imu_norm08,train_set_aluminum_hold_imu_norm09,
                                    train_set_aluminum_hold_imu_norm10,train_set_aluminum_hold_imu_norm11,train_set_aluminum_hold_imu_norm12,train_set_aluminum_hold_imu_norm13,train_set_aluminum_hold_imu_norm14,
                                    train_set_aluminum_hold_imu_norm15,train_set_aluminum_hold_imu_norm16,train_set_aluminum_hold_imu_norm17,train_set_aluminum_hold_imu_norm18,train_set_aluminum_hold_imu_norm19])
##########################################################################################
print('normalizing IMU data into same duration of spanner_handover')
# resampling imu signals of aluminum_hold
train_set_spanner_handover_imu_norm00=np.array([]);train_set_spanner_handover_imu_norm01=np.array([]);train_set_spanner_handover_imu_norm02=np.array([]);train_set_spanner_handover_imu_norm03=np.array([]);train_set_spanner_handover_imu_norm04=np.array([]);
train_set_spanner_handover_imu_norm05=np.array([]);train_set_spanner_handover_imu_norm06=np.array([]);train_set_spanner_handover_imu_norm07=np.array([]);train_set_spanner_handover_imu_norm08=np.array([]);train_set_spanner_handover_imu_norm09=np.array([]);
train_set_spanner_handover_imu_norm10=np.array([]);train_set_spanner_handover_imu_norm11=np.array([]);train_set_spanner_handover_imu_norm12=np.array([]);train_set_spanner_handover_imu_norm13=np.array([]);train_set_spanner_handover_imu_norm14=np.array([]);
train_set_spanner_handover_imu_norm15=np.array([]);train_set_spanner_handover_imu_norm16=np.array([]);train_set_spanner_handover_imu_norm17=np.array([]);train_set_spanner_handover_imu_norm18=np.array([]);train_set_spanner_handover_imu_norm19=np.array([]);
train_set_spanner_handover_imu_norm20=np.array([]);train_set_spanner_handover_imu_norm21=np.array([]);train_set_spanner_handover_imu_norm22=np.array([]);train_set_spanner_handover_imu_norm23=np.array([]);train_set_spanner_handover_imu_norm24=np.array([]);
train_set_spanner_handover_imu_norm25=np.array([]);train_set_spanner_handover_imu_norm26=np.array([]);train_set_spanner_handover_imu_norm27=np.array([]);train_set_spanner_handover_imu_norm28=np.array([]);train_set_spanner_handover_imu_norm29=np.array([]);
test_set_spanner_handover_imu_norm=np.array([]);
for ch_ex in range(4):
    train_set_spanner_handover_imu_norm00 = np.hstack(( train_set_spanner_handover_imu_norm00, np.interp(np.linspace(0, len(train_set_spanner_handover_imu_00)-1, len_normal), np.arange(0,len(train_set_spanner_handover_imu_00),1.), train_set_spanner_handover_imu_00[:,ch_ex]) ))
    train_set_spanner_handover_imu_norm01 = np.hstack(( train_set_spanner_handover_imu_norm01, np.interp(np.linspace(0, len(train_set_spanner_handover_imu_01)-1, len_normal), np.arange(0,len(train_set_spanner_handover_imu_01),1.), train_set_spanner_handover_imu_01[:,ch_ex]) ))
    train_set_spanner_handover_imu_norm02 = np.hstack(( train_set_spanner_handover_imu_norm02, np.interp(np.linspace(0, len(train_set_spanner_handover_imu_02)-1, len_normal), np.arange(0,len(train_set_spanner_handover_imu_02),1.), train_set_spanner_handover_imu_02[:,ch_ex]) ))
    train_set_spanner_handover_imu_norm03 = np.hstack(( train_set_spanner_handover_imu_norm03, np.interp(np.linspace(0, len(train_set_spanner_handover_imu_03)-1, len_normal), np.arange(0,len(train_set_spanner_handover_imu_03),1.), train_set_spanner_handover_imu_03[:,ch_ex]) ))
    train_set_spanner_handover_imu_norm04 = np.hstack(( train_set_spanner_handover_imu_norm04, np.interp(np.linspace(0, len(train_set_spanner_handover_imu_04)-1, len_normal), np.arange(0,len(train_set_spanner_handover_imu_04),1.), train_set_spanner_handover_imu_04[:,ch_ex]) ))
    train_set_spanner_handover_imu_norm05 = np.hstack(( train_set_spanner_handover_imu_norm05, np.interp(np.linspace(0, len(train_set_spanner_handover_imu_05)-1, len_normal), np.arange(0,len(train_set_spanner_handover_imu_05),1.), train_set_spanner_handover_imu_05[:,ch_ex]) ))
    train_set_spanner_handover_imu_norm06 = np.hstack(( train_set_spanner_handover_imu_norm06, np.interp(np.linspace(0, len(train_set_spanner_handover_imu_06)-1, len_normal), np.arange(0,len(train_set_spanner_handover_imu_06),1.), train_set_spanner_handover_imu_06[:,ch_ex]) ))
    train_set_spanner_handover_imu_norm07 = np.hstack(( train_set_spanner_handover_imu_norm07, np.interp(np.linspace(0, len(train_set_spanner_handover_imu_07)-1, len_normal), np.arange(0,len(train_set_spanner_handover_imu_07),1.), train_set_spanner_handover_imu_07[:,ch_ex]) ))
    train_set_spanner_handover_imu_norm08 = np.hstack(( train_set_spanner_handover_imu_norm08, np.interp(np.linspace(0, len(train_set_spanner_handover_imu_08)-1, len_normal), np.arange(0,len(train_set_spanner_handover_imu_08),1.), train_set_spanner_handover_imu_08[:,ch_ex]) ))
    train_set_spanner_handover_imu_norm09 = np.hstack(( train_set_spanner_handover_imu_norm09, np.interp(np.linspace(0, len(train_set_spanner_handover_imu_09)-1, len_normal), np.arange(0,len(train_set_spanner_handover_imu_09),1.), train_set_spanner_handover_imu_09[:,ch_ex]) ))
    train_set_spanner_handover_imu_norm10 = np.hstack(( train_set_spanner_handover_imu_norm10, np.interp(np.linspace(0, len(train_set_spanner_handover_imu_10)-1, len_normal), np.arange(0,len(train_set_spanner_handover_imu_10),1.), train_set_spanner_handover_imu_10[:,ch_ex]) ))
    train_set_spanner_handover_imu_norm11 = np.hstack(( train_set_spanner_handover_imu_norm11, np.interp(np.linspace(0, len(train_set_spanner_handover_imu_11)-1, len_normal), np.arange(0,len(train_set_spanner_handover_imu_11),1.), train_set_spanner_handover_imu_11[:,ch_ex]) ))
    train_set_spanner_handover_imu_norm12 = np.hstack(( train_set_spanner_handover_imu_norm12, np.interp(np.linspace(0, len(train_set_spanner_handover_imu_12)-1, len_normal), np.arange(0,len(train_set_spanner_handover_imu_12),1.), train_set_spanner_handover_imu_12[:,ch_ex]) ))
    train_set_spanner_handover_imu_norm13 = np.hstack(( train_set_spanner_handover_imu_norm13, np.interp(np.linspace(0, len(train_set_spanner_handover_imu_13)-1, len_normal), np.arange(0,len(train_set_spanner_handover_imu_13),1.), train_set_spanner_handover_imu_13[:,ch_ex]) ))
    train_set_spanner_handover_imu_norm14 = np.hstack(( train_set_spanner_handover_imu_norm14, np.interp(np.linspace(0, len(train_set_spanner_handover_imu_14)-1, len_normal), np.arange(0,len(train_set_spanner_handover_imu_14),1.), train_set_spanner_handover_imu_14[:,ch_ex]) ))
    train_set_spanner_handover_imu_norm15 = np.hstack(( train_set_spanner_handover_imu_norm15, np.interp(np.linspace(0, len(train_set_spanner_handover_imu_15)-1, len_normal), np.arange(0,len(train_set_spanner_handover_imu_15),1.), train_set_spanner_handover_imu_15[:,ch_ex]) ))
    train_set_spanner_handover_imu_norm16 = np.hstack(( train_set_spanner_handover_imu_norm16, np.interp(np.linspace(0, len(train_set_spanner_handover_imu_16)-1, len_normal), np.arange(0,len(train_set_spanner_handover_imu_16),1.), train_set_spanner_handover_imu_16[:,ch_ex]) ))
    train_set_spanner_handover_imu_norm17 = np.hstack(( train_set_spanner_handover_imu_norm17, np.interp(np.linspace(0, len(train_set_spanner_handover_imu_17)-1, len_normal), np.arange(0,len(train_set_spanner_handover_imu_17),1.), train_set_spanner_handover_imu_17[:,ch_ex]) ))
    train_set_spanner_handover_imu_norm18 = np.hstack(( train_set_spanner_handover_imu_norm18, np.interp(np.linspace(0, len(train_set_spanner_handover_imu_18)-1, len_normal), np.arange(0,len(train_set_spanner_handover_imu_18),1.), train_set_spanner_handover_imu_18[:,ch_ex]) ))
    train_set_spanner_handover_imu_norm19 = np.hstack(( train_set_spanner_handover_imu_norm19, np.interp(np.linspace(0, len(train_set_spanner_handover_imu_19)-1, len_normal), np.arange(0,len(train_set_spanner_handover_imu_19),1.), train_set_spanner_handover_imu_19[:,ch_ex]) ))
    train_set_spanner_handover_imu_norm20 = np.hstack((train_set_spanner_handover_imu_norm20, np.interp( np.linspace(0, len(train_set_spanner_handover_imu_20) - 1, len_normal), np.arange(0, len(train_set_spanner_handover_imu_20), 1.), train_set_spanner_handover_imu_20[:, ch_ex])))
    train_set_spanner_handover_imu_norm21 = np.hstack((train_set_spanner_handover_imu_norm21, np.interp( np.linspace(0, len(train_set_spanner_handover_imu_21) - 1, len_normal), np.arange(0, len(train_set_spanner_handover_imu_21), 1.), train_set_spanner_handover_imu_21[:, ch_ex])))
    train_set_spanner_handover_imu_norm22 = np.hstack((train_set_spanner_handover_imu_norm22, np.interp( np.linspace(0, len(train_set_spanner_handover_imu_22) - 1, len_normal), np.arange(0, len(train_set_spanner_handover_imu_22), 1.), train_set_spanner_handover_imu_22[:, ch_ex])))
    train_set_spanner_handover_imu_norm23 = np.hstack((train_set_spanner_handover_imu_norm23, np.interp( np.linspace(0, len(train_set_spanner_handover_imu_23) - 1, len_normal), np.arange(0, len(train_set_spanner_handover_imu_23), 1.), train_set_spanner_handover_imu_23[:, ch_ex])))
    train_set_spanner_handover_imu_norm24 = np.hstack((train_set_spanner_handover_imu_norm24, np.interp( np.linspace(0, len(train_set_spanner_handover_imu_24) - 1, len_normal), np.arange(0, len(train_set_spanner_handover_imu_24), 1.), train_set_spanner_handover_imu_24[:, ch_ex])))
    train_set_spanner_handover_imu_norm25 = np.hstack((train_set_spanner_handover_imu_norm25, np.interp( np.linspace(0, len(train_set_spanner_handover_imu_25) - 1, len_normal), np.arange(0, len(train_set_spanner_handover_imu_25), 1.), train_set_spanner_handover_imu_25[:, ch_ex])))
    train_set_spanner_handover_imu_norm26 = np.hstack((train_set_spanner_handover_imu_norm26, np.interp( np.linspace(0, len(train_set_spanner_handover_imu_26) - 1, len_normal), np.arange(0, len(train_set_spanner_handover_imu_26), 1.), train_set_spanner_handover_imu_26[:, ch_ex])))
    train_set_spanner_handover_imu_norm27 = np.hstack((train_set_spanner_handover_imu_norm27, np.interp( np.linspace(0, len(train_set_spanner_handover_imu_27) - 1, len_normal), np.arange(0, len(train_set_spanner_handover_imu_27), 1.), train_set_spanner_handover_imu_27[:, ch_ex])))
    train_set_spanner_handover_imu_norm28 = np.hstack((train_set_spanner_handover_imu_norm28, np.interp( np.linspace(0, len(train_set_spanner_handover_imu_28) - 1, len_normal), np.arange(0, len(train_set_spanner_handover_imu_28), 1.), train_set_spanner_handover_imu_28[:, ch_ex])))
    train_set_spanner_handover_imu_norm29 = np.hstack((train_set_spanner_handover_imu_norm29, np.interp( np.linspace(0, len(train_set_spanner_handover_imu_29) - 1, len_normal), np.arange(0, len(train_set_spanner_handover_imu_29), 1.), train_set_spanner_handover_imu_29[:, ch_ex])))
    test_set_spanner_handover_imu_norm = np.hstack(( test_set_spanner_handover_imu_norm, np.interp(np.linspace(0, len(test_set_spanner_handover_imu)-1, len_normal), np.arange(0,len(test_set_spanner_handover_imu),1.), test_set_spanner_handover_imu[:,ch_ex]) ))
train_set_spanner_handover_imu_norm00 = train_set_spanner_handover_imu_norm00.reshape(4,len_normal).T
train_set_spanner_handover_imu_norm01 = train_set_spanner_handover_imu_norm01.reshape(4,len_normal).T
train_set_spanner_handover_imu_norm02 = train_set_spanner_handover_imu_norm02.reshape(4,len_normal).T
train_set_spanner_handover_imu_norm03 = train_set_spanner_handover_imu_norm03.reshape(4,len_normal).T
train_set_spanner_handover_imu_norm04 = train_set_spanner_handover_imu_norm04.reshape(4,len_normal).T
train_set_spanner_handover_imu_norm05 = train_set_spanner_handover_imu_norm05.reshape(4,len_normal).T
train_set_spanner_handover_imu_norm06 = train_set_spanner_handover_imu_norm06.reshape(4,len_normal).T
train_set_spanner_handover_imu_norm07 = train_set_spanner_handover_imu_norm07.reshape(4,len_normal).T
train_set_spanner_handover_imu_norm08 = train_set_spanner_handover_imu_norm08.reshape(4,len_normal).T
train_set_spanner_handover_imu_norm09 = train_set_spanner_handover_imu_norm09.reshape(4,len_normal).T
train_set_spanner_handover_imu_norm10 = train_set_spanner_handover_imu_norm10.reshape(4,len_normal).T
train_set_spanner_handover_imu_norm11 = train_set_spanner_handover_imu_norm11.reshape(4,len_normal).T
train_set_spanner_handover_imu_norm12 = train_set_spanner_handover_imu_norm12.reshape(4,len_normal).T
train_set_spanner_handover_imu_norm13 = train_set_spanner_handover_imu_norm13.reshape(4,len_normal).T
train_set_spanner_handover_imu_norm14 = train_set_spanner_handover_imu_norm14.reshape(4,len_normal).T
train_set_spanner_handover_imu_norm15 = train_set_spanner_handover_imu_norm15.reshape(4,len_normal).T
train_set_spanner_handover_imu_norm16 = train_set_spanner_handover_imu_norm16.reshape(4,len_normal).T
train_set_spanner_handover_imu_norm17 = train_set_spanner_handover_imu_norm17.reshape(4,len_normal).T
train_set_spanner_handover_imu_norm18 = train_set_spanner_handover_imu_norm18.reshape(4,len_normal).T
train_set_spanner_handover_imu_norm19 = train_set_spanner_handover_imu_norm19.reshape(4,len_normal).T
train_set_spanner_handover_imu_norm20 = train_set_spanner_handover_imu_norm20.reshape(4,len_normal).T
train_set_spanner_handover_imu_norm21 = train_set_spanner_handover_imu_norm21.reshape(4,len_normal).T
train_set_spanner_handover_imu_norm22 = train_set_spanner_handover_imu_norm22.reshape(4,len_normal).T
train_set_spanner_handover_imu_norm23 = train_set_spanner_handover_imu_norm23.reshape(4,len_normal).T
train_set_spanner_handover_imu_norm24 = train_set_spanner_handover_imu_norm24.reshape(4,len_normal).T
train_set_spanner_handover_imu_norm25 = train_set_spanner_handover_imu_norm25.reshape(4,len_normal).T
train_set_spanner_handover_imu_norm26 = train_set_spanner_handover_imu_norm26.reshape(4,len_normal).T
train_set_spanner_handover_imu_norm27 = train_set_spanner_handover_imu_norm27.reshape(4,len_normal).T
train_set_spanner_handover_imu_norm28 = train_set_spanner_handover_imu_norm28.reshape(4,len_normal).T
train_set_spanner_handover_imu_norm29 = train_set_spanner_handover_imu_norm29.reshape(4,len_normal).T
test_set_spanner_handover_imu_norm = test_set_spanner_handover_imu_norm.reshape(4,len_normal).T
train_set_spanner_handover_imu_norm_full = np.array([train_set_spanner_handover_imu_norm00,train_set_spanner_handover_imu_norm01,train_set_spanner_handover_imu_norm02,train_set_spanner_handover_imu_norm03,train_set_spanner_handover_imu_norm04,
                                    train_set_spanner_handover_imu_norm05,train_set_spanner_handover_imu_norm06,train_set_spanner_handover_imu_norm07,train_set_spanner_handover_imu_norm08,train_set_spanner_handover_imu_norm09,
                                    train_set_spanner_handover_imu_norm10,train_set_spanner_handover_imu_norm11,train_set_spanner_handover_imu_norm12,train_set_spanner_handover_imu_norm13,train_set_spanner_handover_imu_norm14,
                                    train_set_spanner_handover_imu_norm15,train_set_spanner_handover_imu_norm16,train_set_spanner_handover_imu_norm17,train_set_spanner_handover_imu_norm18,train_set_spanner_handover_imu_norm19])
##########################################################################################
print('normalizing IMU data into same duration of tape_hold')
# resampling imu signals of aluminum_hold
train_set_tape_hold_imu_norm00=np.array([]);train_set_tape_hold_imu_norm01=np.array([]);train_set_tape_hold_imu_norm02=np.array([]);train_set_tape_hold_imu_norm03=np.array([]);train_set_tape_hold_imu_norm04=np.array([]);
train_set_tape_hold_imu_norm05=np.array([]);train_set_tape_hold_imu_norm06=np.array([]);train_set_tape_hold_imu_norm07=np.array([]);train_set_tape_hold_imu_norm08=np.array([]);train_set_tape_hold_imu_norm09=np.array([]);
train_set_tape_hold_imu_norm10=np.array([]);train_set_tape_hold_imu_norm11=np.array([]);train_set_tape_hold_imu_norm12=np.array([]);train_set_tape_hold_imu_norm13=np.array([]);train_set_tape_hold_imu_norm14=np.array([]);
train_set_tape_hold_imu_norm15=np.array([]);train_set_tape_hold_imu_norm16=np.array([]);train_set_tape_hold_imu_norm17=np.array([]);train_set_tape_hold_imu_norm18=np.array([]);train_set_tape_hold_imu_norm19=np.array([]);
train_set_tape_hold_imu_norm20=np.array([]);train_set_tape_hold_imu_norm21=np.array([]);train_set_tape_hold_imu_norm22=np.array([]);train_set_tape_hold_imu_norm23=np.array([]);train_set_tape_hold_imu_norm24=np.array([]);
train_set_tape_hold_imu_norm25=np.array([]);train_set_tape_hold_imu_norm26=np.array([]);train_set_tape_hold_imu_norm27=np.array([]);train_set_tape_hold_imu_norm28=np.array([]);train_set_tape_hold_imu_norm29=np.array([]);
test_set_tape_hold_imu_norm=np.array([]);
for ch_ex in range(4):
    train_set_tape_hold_imu_norm00 = np.hstack(( train_set_tape_hold_imu_norm00, np.interp(np.linspace(0, len(train_set_tape_hold_imu_00)-1, len_normal), np.arange(0,len(train_set_tape_hold_imu_00),1.), train_set_tape_hold_imu_00[:,ch_ex]) ))
    train_set_tape_hold_imu_norm01 = np.hstack(( train_set_tape_hold_imu_norm01, np.interp(np.linspace(0, len(train_set_tape_hold_imu_01)-1, len_normal), np.arange(0,len(train_set_tape_hold_imu_01),1.), train_set_tape_hold_imu_01[:,ch_ex]) ))
    train_set_tape_hold_imu_norm02 = np.hstack(( train_set_tape_hold_imu_norm02, np.interp(np.linspace(0, len(train_set_tape_hold_imu_02)-1, len_normal), np.arange(0,len(train_set_tape_hold_imu_02),1.), train_set_tape_hold_imu_02[:,ch_ex]) ))
    train_set_tape_hold_imu_norm03 = np.hstack(( train_set_tape_hold_imu_norm03, np.interp(np.linspace(0, len(train_set_tape_hold_imu_03)-1, len_normal), np.arange(0,len(train_set_tape_hold_imu_03),1.), train_set_tape_hold_imu_03[:,ch_ex]) ))
    train_set_tape_hold_imu_norm04 = np.hstack(( train_set_tape_hold_imu_norm04, np.interp(np.linspace(0, len(train_set_tape_hold_imu_04)-1, len_normal), np.arange(0,len(train_set_tape_hold_imu_04),1.), train_set_tape_hold_imu_04[:,ch_ex]) ))
    train_set_tape_hold_imu_norm05 = np.hstack(( train_set_tape_hold_imu_norm05, np.interp(np.linspace(0, len(train_set_tape_hold_imu_05)-1, len_normal), np.arange(0,len(train_set_tape_hold_imu_05),1.), train_set_tape_hold_imu_05[:,ch_ex]) ))
    train_set_tape_hold_imu_norm06 = np.hstack(( train_set_tape_hold_imu_norm06, np.interp(np.linspace(0, len(train_set_tape_hold_imu_06)-1, len_normal), np.arange(0,len(train_set_tape_hold_imu_06),1.), train_set_tape_hold_imu_06[:,ch_ex]) ))
    train_set_tape_hold_imu_norm07 = np.hstack(( train_set_tape_hold_imu_norm07, np.interp(np.linspace(0, len(train_set_tape_hold_imu_07)-1, len_normal), np.arange(0,len(train_set_tape_hold_imu_07),1.), train_set_tape_hold_imu_07[:,ch_ex]) ))
    train_set_tape_hold_imu_norm08 = np.hstack(( train_set_tape_hold_imu_norm08, np.interp(np.linspace(0, len(train_set_tape_hold_imu_08)-1, len_normal), np.arange(0,len(train_set_tape_hold_imu_08),1.), train_set_tape_hold_imu_08[:,ch_ex]) ))
    train_set_tape_hold_imu_norm09 = np.hstack(( train_set_tape_hold_imu_norm09, np.interp(np.linspace(0, len(train_set_tape_hold_imu_09)-1, len_normal), np.arange(0,len(train_set_tape_hold_imu_09),1.), train_set_tape_hold_imu_09[:,ch_ex]) ))
    train_set_tape_hold_imu_norm10 = np.hstack(( train_set_tape_hold_imu_norm10, np.interp(np.linspace(0, len(train_set_tape_hold_imu_10)-1, len_normal), np.arange(0,len(train_set_tape_hold_imu_10),1.), train_set_tape_hold_imu_10[:,ch_ex]) ))
    train_set_tape_hold_imu_norm11 = np.hstack(( train_set_tape_hold_imu_norm11, np.interp(np.linspace(0, len(train_set_tape_hold_imu_11)-1, len_normal), np.arange(0,len(train_set_tape_hold_imu_11),1.), train_set_tape_hold_imu_11[:,ch_ex]) ))
    train_set_tape_hold_imu_norm12 = np.hstack(( train_set_tape_hold_imu_norm12, np.interp(np.linspace(0, len(train_set_tape_hold_imu_12)-1, len_normal), np.arange(0,len(train_set_tape_hold_imu_12),1.), train_set_tape_hold_imu_12[:,ch_ex]) ))
    train_set_tape_hold_imu_norm13 = np.hstack(( train_set_tape_hold_imu_norm13, np.interp(np.linspace(0, len(train_set_tape_hold_imu_13)-1, len_normal), np.arange(0,len(train_set_tape_hold_imu_13),1.), train_set_tape_hold_imu_13[:,ch_ex]) ))
    train_set_tape_hold_imu_norm14 = np.hstack(( train_set_tape_hold_imu_norm14, np.interp(np.linspace(0, len(train_set_tape_hold_imu_14)-1, len_normal), np.arange(0,len(train_set_tape_hold_imu_14),1.), train_set_tape_hold_imu_14[:,ch_ex]) ))
    train_set_tape_hold_imu_norm15 = np.hstack(( train_set_tape_hold_imu_norm15, np.interp(np.linspace(0, len(train_set_tape_hold_imu_15)-1, len_normal), np.arange(0,len(train_set_tape_hold_imu_15),1.), train_set_tape_hold_imu_15[:,ch_ex]) ))
    train_set_tape_hold_imu_norm16 = np.hstack(( train_set_tape_hold_imu_norm16, np.interp(np.linspace(0, len(train_set_tape_hold_imu_16)-1, len_normal), np.arange(0,len(train_set_tape_hold_imu_16),1.), train_set_tape_hold_imu_16[:,ch_ex]) ))
    train_set_tape_hold_imu_norm17 = np.hstack(( train_set_tape_hold_imu_norm17, np.interp(np.linspace(0, len(train_set_tape_hold_imu_17)-1, len_normal), np.arange(0,len(train_set_tape_hold_imu_17),1.), train_set_tape_hold_imu_17[:,ch_ex]) ))
    train_set_tape_hold_imu_norm18 = np.hstack(( train_set_tape_hold_imu_norm18, np.interp(np.linspace(0, len(train_set_tape_hold_imu_18)-1, len_normal), np.arange(0,len(train_set_tape_hold_imu_18),1.), train_set_tape_hold_imu_18[:,ch_ex]) ))
    train_set_tape_hold_imu_norm19 = np.hstack(( train_set_tape_hold_imu_norm19, np.interp(np.linspace(0, len(train_set_tape_hold_imu_19)-1, len_normal), np.arange(0,len(train_set_tape_hold_imu_19),1.), train_set_tape_hold_imu_19[:,ch_ex]) ))
    train_set_tape_hold_imu_norm20 = np.hstack((train_set_tape_hold_imu_norm20, np.interp( np.linspace(0, len(train_set_tape_hold_imu_20) - 1, len_normal), np.arange(0, len(train_set_tape_hold_imu_20), 1.), train_set_tape_hold_imu_20[:, ch_ex])))
    train_set_tape_hold_imu_norm21 = np.hstack((train_set_tape_hold_imu_norm21, np.interp( np.linspace(0, len(train_set_tape_hold_imu_21) - 1, len_normal), np.arange(0, len(train_set_tape_hold_imu_21), 1.), train_set_tape_hold_imu_21[:, ch_ex])))
    train_set_tape_hold_imu_norm22 = np.hstack((train_set_tape_hold_imu_norm22, np.interp( np.linspace(0, len(train_set_tape_hold_imu_22) - 1, len_normal), np.arange(0, len(train_set_tape_hold_imu_22), 1.), train_set_tape_hold_imu_22[:, ch_ex])))
    train_set_tape_hold_imu_norm23 = np.hstack((train_set_tape_hold_imu_norm23, np.interp( np.linspace(0, len(train_set_tape_hold_imu_23) - 1, len_normal), np.arange(0, len(train_set_tape_hold_imu_23), 1.), train_set_tape_hold_imu_23[:, ch_ex])))
    train_set_tape_hold_imu_norm24 = np.hstack((train_set_tape_hold_imu_norm24, np.interp( np.linspace(0, len(train_set_tape_hold_imu_24) - 1, len_normal), np.arange(0, len(train_set_tape_hold_imu_24), 1.), train_set_tape_hold_imu_24[:, ch_ex])))
    train_set_tape_hold_imu_norm25 = np.hstack((train_set_tape_hold_imu_norm25, np.interp( np.linspace(0, len(train_set_tape_hold_imu_25) - 1, len_normal), np.arange(0, len(train_set_tape_hold_imu_25), 1.), train_set_tape_hold_imu_25[:, ch_ex])))
    train_set_tape_hold_imu_norm26 = np.hstack((train_set_tape_hold_imu_norm26, np.interp( np.linspace(0, len(train_set_tape_hold_imu_26) - 1, len_normal), np.arange(0, len(train_set_tape_hold_imu_26), 1.), train_set_tape_hold_imu_26[:, ch_ex])))
    train_set_tape_hold_imu_norm27 = np.hstack((train_set_tape_hold_imu_norm27, np.interp( np.linspace(0, len(train_set_tape_hold_imu_27) - 1, len_normal), np.arange(0, len(train_set_tape_hold_imu_27), 1.), train_set_tape_hold_imu_27[:, ch_ex])))
    train_set_tape_hold_imu_norm28 = np.hstack((train_set_tape_hold_imu_norm28, np.interp( np.linspace(0, len(train_set_tape_hold_imu_28) - 1, len_normal), np.arange(0, len(train_set_tape_hold_imu_28), 1.), train_set_tape_hold_imu_28[:, ch_ex])))
    train_set_tape_hold_imu_norm29 = np.hstack((train_set_tape_hold_imu_norm29, np.interp( np.linspace(0, len(train_set_tape_hold_imu_29) - 1, len_normal), np.arange(0, len(train_set_tape_hold_imu_29), 1.), train_set_tape_hold_imu_29[:, ch_ex])))
    test_set_tape_hold_imu_norm = np.hstack(( test_set_tape_hold_imu_norm, np.interp(np.linspace(0, len(test_set_tape_hold_imu)-1, len_normal), np.arange(0,len(test_set_tape_hold_imu),1.), test_set_tape_hold_imu[:,ch_ex]) ))
train_set_tape_hold_imu_norm00 = train_set_tape_hold_imu_norm00.reshape(4,len_normal).T
train_set_tape_hold_imu_norm01 = train_set_tape_hold_imu_norm01.reshape(4,len_normal).T
train_set_tape_hold_imu_norm02 = train_set_tape_hold_imu_norm02.reshape(4,len_normal).T
train_set_tape_hold_imu_norm03 = train_set_tape_hold_imu_norm03.reshape(4,len_normal).T
train_set_tape_hold_imu_norm04 = train_set_tape_hold_imu_norm04.reshape(4,len_normal).T
train_set_tape_hold_imu_norm05 = train_set_tape_hold_imu_norm05.reshape(4,len_normal).T
train_set_tape_hold_imu_norm06 = train_set_tape_hold_imu_norm06.reshape(4,len_normal).T
train_set_tape_hold_imu_norm07 = train_set_tape_hold_imu_norm07.reshape(4,len_normal).T
train_set_tape_hold_imu_norm08 = train_set_tape_hold_imu_norm08.reshape(4,len_normal).T
train_set_tape_hold_imu_norm09 = train_set_tape_hold_imu_norm09.reshape(4,len_normal).T
train_set_tape_hold_imu_norm10 = train_set_tape_hold_imu_norm10.reshape(4,len_normal).T
train_set_tape_hold_imu_norm11 = train_set_tape_hold_imu_norm11.reshape(4,len_normal).T
train_set_tape_hold_imu_norm12 = train_set_tape_hold_imu_norm12.reshape(4,len_normal).T
train_set_tape_hold_imu_norm13 = train_set_tape_hold_imu_norm13.reshape(4,len_normal).T
train_set_tape_hold_imu_norm14 = train_set_tape_hold_imu_norm14.reshape(4,len_normal).T
train_set_tape_hold_imu_norm15 = train_set_tape_hold_imu_norm15.reshape(4,len_normal).T
train_set_tape_hold_imu_norm16 = train_set_tape_hold_imu_norm16.reshape(4,len_normal).T
train_set_tape_hold_imu_norm17 = train_set_tape_hold_imu_norm17.reshape(4,len_normal).T
train_set_tape_hold_imu_norm18 = train_set_tape_hold_imu_norm18.reshape(4,len_normal).T
train_set_tape_hold_imu_norm19 = train_set_tape_hold_imu_norm19.reshape(4,len_normal).T
train_set_tape_hold_imu_norm20 = train_set_tape_hold_imu_norm20.reshape(4,len_normal).T
train_set_tape_hold_imu_norm21 = train_set_tape_hold_imu_norm21.reshape(4,len_normal).T
train_set_tape_hold_imu_norm22 = train_set_tape_hold_imu_norm22.reshape(4,len_normal).T
train_set_tape_hold_imu_norm23 = train_set_tape_hold_imu_norm23.reshape(4,len_normal).T
train_set_tape_hold_imu_norm24 = train_set_tape_hold_imu_norm24.reshape(4,len_normal).T
train_set_tape_hold_imu_norm25 = train_set_tape_hold_imu_norm25.reshape(4,len_normal).T
train_set_tape_hold_imu_norm26 = train_set_tape_hold_imu_norm26.reshape(4,len_normal).T
train_set_tape_hold_imu_norm27 = train_set_tape_hold_imu_norm27.reshape(4,len_normal).T
train_set_tape_hold_imu_norm28 = train_set_tape_hold_imu_norm28.reshape(4,len_normal).T
train_set_tape_hold_imu_norm29 = train_set_tape_hold_imu_norm29.reshape(4,len_normal).T
test_set_tape_hold_imu_norm = test_set_tape_hold_imu_norm.reshape(4,len_normal).T
train_set_tape_hold_imu_norm_full = np.array([train_set_tape_hold_imu_norm00,train_set_tape_hold_imu_norm01,train_set_tape_hold_imu_norm02,train_set_tape_hold_imu_norm03,train_set_tape_hold_imu_norm04,
                                    train_set_tape_hold_imu_norm05,train_set_tape_hold_imu_norm06,train_set_tape_hold_imu_norm07,train_set_tape_hold_imu_norm08,train_set_tape_hold_imu_norm09,
                                    train_set_tape_hold_imu_norm10,train_set_tape_hold_imu_norm11,train_set_tape_hold_imu_norm12,train_set_tape_hold_imu_norm13,train_set_tape_hold_imu_norm14,
                                    train_set_tape_hold_imu_norm15,train_set_tape_hold_imu_norm16,train_set_tape_hold_imu_norm17,train_set_tape_hold_imu_norm18,train_set_tape_hold_imu_norm19])

##################################################################################
# resampling the Pose data for experiencing the same duration
##################################################################################
# rospy.loginfo('normalizing data into same duration')
print('normalizing Pose data into same duration of aluminum_hold')
# resampling signals of aluminum_hold
train_set_aluminum_hold_pose_norm00=np.array([]);train_set_aluminum_hold_pose_norm01=np.array([]);train_set_aluminum_hold_pose_norm02=np.array([]);train_set_aluminum_hold_pose_norm03=np.array([]);train_set_aluminum_hold_pose_norm04=np.array([]);
train_set_aluminum_hold_pose_norm05=np.array([]);train_set_aluminum_hold_pose_norm06=np.array([]);train_set_aluminum_hold_pose_norm07=np.array([]);train_set_aluminum_hold_pose_norm08=np.array([]);train_set_aluminum_hold_pose_norm09=np.array([]);
train_set_aluminum_hold_pose_norm10=np.array([]);train_set_aluminum_hold_pose_norm11=np.array([]);train_set_aluminum_hold_pose_norm12=np.array([]);train_set_aluminum_hold_pose_norm13=np.array([]);train_set_aluminum_hold_pose_norm14=np.array([]);
train_set_aluminum_hold_pose_norm15=np.array([]);train_set_aluminum_hold_pose_norm16=np.array([]);train_set_aluminum_hold_pose_norm17=np.array([]);train_set_aluminum_hold_pose_norm18=np.array([]);train_set_aluminum_hold_pose_norm19=np.array([]);
train_set_aluminum_hold_pose_norm20=np.array([]);train_set_aluminum_hold_pose_norm21=np.array([]);train_set_aluminum_hold_pose_norm22=np.array([]);train_set_aluminum_hold_pose_norm23=np.array([]);train_set_aluminum_hold_pose_norm24=np.array([]);
train_set_aluminum_hold_pose_norm25=np.array([]);train_set_aluminum_hold_pose_norm26=np.array([]);train_set_aluminum_hold_pose_norm27=np.array([]);train_set_aluminum_hold_pose_norm28=np.array([]);train_set_aluminum_hold_pose_norm29=np.array([]);
test_set_aluminum_hold_pose_norm=np.array([]);
for ch_ex in range(7):
    train_set_aluminum_hold_pose_norm00 = np.hstack(( train_set_aluminum_hold_pose_norm00, np.interp(np.linspace(0, len(train_set_aluminum_hold_pose_00)-1, len_normal), np.arange(0,len(train_set_aluminum_hold_pose_00),1.), train_set_aluminum_hold_pose_00[:,ch_ex]) ))
    train_set_aluminum_hold_pose_norm01 = np.hstack(( train_set_aluminum_hold_pose_norm01, np.interp(np.linspace(0, len(train_set_aluminum_hold_pose_01)-1, len_normal), np.arange(0,len(train_set_aluminum_hold_pose_01),1.), train_set_aluminum_hold_pose_01[:,ch_ex]) ))
    train_set_aluminum_hold_pose_norm02 = np.hstack(( train_set_aluminum_hold_pose_norm02, np.interp(np.linspace(0, len(train_set_aluminum_hold_pose_02)-1, len_normal), np.arange(0,len(train_set_aluminum_hold_pose_02),1.), train_set_aluminum_hold_pose_02[:,ch_ex]) ))
    train_set_aluminum_hold_pose_norm03 = np.hstack(( train_set_aluminum_hold_pose_norm03, np.interp(np.linspace(0, len(train_set_aluminum_hold_pose_03)-1, len_normal), np.arange(0,len(train_set_aluminum_hold_pose_03),1.), train_set_aluminum_hold_pose_03[:,ch_ex]) ))
    train_set_aluminum_hold_pose_norm04 = np.hstack(( train_set_aluminum_hold_pose_norm04, np.interp(np.linspace(0, len(train_set_aluminum_hold_pose_04)-1, len_normal), np.arange(0,len(train_set_aluminum_hold_pose_04),1.), train_set_aluminum_hold_pose_04[:,ch_ex]) ))
    train_set_aluminum_hold_pose_norm05 = np.hstack(( train_set_aluminum_hold_pose_norm05, np.interp(np.linspace(0, len(train_set_aluminum_hold_pose_05)-1, len_normal), np.arange(0,len(train_set_aluminum_hold_pose_05),1.), train_set_aluminum_hold_pose_05[:,ch_ex]) ))
    train_set_aluminum_hold_pose_norm06 = np.hstack(( train_set_aluminum_hold_pose_norm06, np.interp(np.linspace(0, len(train_set_aluminum_hold_pose_06)-1, len_normal), np.arange(0,len(train_set_aluminum_hold_pose_06),1.), train_set_aluminum_hold_pose_06[:,ch_ex]) ))
    train_set_aluminum_hold_pose_norm07 = np.hstack(( train_set_aluminum_hold_pose_norm07, np.interp(np.linspace(0, len(train_set_aluminum_hold_pose_07)-1, len_normal), np.arange(0,len(train_set_aluminum_hold_pose_07),1.), train_set_aluminum_hold_pose_07[:,ch_ex]) ))
    train_set_aluminum_hold_pose_norm08 = np.hstack(( train_set_aluminum_hold_pose_norm08, np.interp(np.linspace(0, len(train_set_aluminum_hold_pose_08)-1, len_normal), np.arange(0,len(train_set_aluminum_hold_pose_08),1.), train_set_aluminum_hold_pose_08[:,ch_ex]) ))
    train_set_aluminum_hold_pose_norm09 = np.hstack(( train_set_aluminum_hold_pose_norm09, np.interp(np.linspace(0, len(train_set_aluminum_hold_pose_09)-1, len_normal), np.arange(0,len(train_set_aluminum_hold_pose_09),1.), train_set_aluminum_hold_pose_09[:,ch_ex]) ))
    train_set_aluminum_hold_pose_norm10 = np.hstack(( train_set_aluminum_hold_pose_norm10, np.interp(np.linspace(0, len(train_set_aluminum_hold_pose_10)-1, len_normal), np.arange(0,len(train_set_aluminum_hold_pose_10),1.), train_set_aluminum_hold_pose_10[:,ch_ex]) ))
    train_set_aluminum_hold_pose_norm11 = np.hstack(( train_set_aluminum_hold_pose_norm11, np.interp(np.linspace(0, len(train_set_aluminum_hold_pose_11)-1, len_normal), np.arange(0,len(train_set_aluminum_hold_pose_11),1.), train_set_aluminum_hold_pose_11[:,ch_ex]) ))
    train_set_aluminum_hold_pose_norm12 = np.hstack(( train_set_aluminum_hold_pose_norm12, np.interp(np.linspace(0, len(train_set_aluminum_hold_pose_12)-1, len_normal), np.arange(0,len(train_set_aluminum_hold_pose_12),1.), train_set_aluminum_hold_pose_12[:,ch_ex]) ))
    train_set_aluminum_hold_pose_norm13 = np.hstack(( train_set_aluminum_hold_pose_norm13, np.interp(np.linspace(0, len(train_set_aluminum_hold_pose_13)-1, len_normal), np.arange(0,len(train_set_aluminum_hold_pose_13),1.), train_set_aluminum_hold_pose_13[:,ch_ex]) ))
    train_set_aluminum_hold_pose_norm14 = np.hstack(( train_set_aluminum_hold_pose_norm14, np.interp(np.linspace(0, len(train_set_aluminum_hold_pose_14)-1, len_normal), np.arange(0,len(train_set_aluminum_hold_pose_14),1.), train_set_aluminum_hold_pose_14[:,ch_ex]) ))
    train_set_aluminum_hold_pose_norm15 = np.hstack(( train_set_aluminum_hold_pose_norm15, np.interp(np.linspace(0, len(train_set_aluminum_hold_pose_15)-1, len_normal), np.arange(0,len(train_set_aluminum_hold_pose_15),1.), train_set_aluminum_hold_pose_15[:,ch_ex]) ))
    train_set_aluminum_hold_pose_norm16 = np.hstack(( train_set_aluminum_hold_pose_norm16, np.interp(np.linspace(0, len(train_set_aluminum_hold_pose_16)-1, len_normal), np.arange(0,len(train_set_aluminum_hold_pose_16),1.), train_set_aluminum_hold_pose_16[:,ch_ex]) ))
    train_set_aluminum_hold_pose_norm17 = np.hstack(( train_set_aluminum_hold_pose_norm17, np.interp(np.linspace(0, len(train_set_aluminum_hold_pose_17)-1, len_normal), np.arange(0,len(train_set_aluminum_hold_pose_17),1.), train_set_aluminum_hold_pose_17[:,ch_ex]) ))
    train_set_aluminum_hold_pose_norm18 = np.hstack(( train_set_aluminum_hold_pose_norm18, np.interp(np.linspace(0, len(train_set_aluminum_hold_pose_18)-1, len_normal), np.arange(0,len(train_set_aluminum_hold_pose_18),1.), train_set_aluminum_hold_pose_18[:,ch_ex]) ))
    train_set_aluminum_hold_pose_norm19 = np.hstack(( train_set_aluminum_hold_pose_norm19, np.interp(np.linspace(0, len(train_set_aluminum_hold_pose_19)-1, len_normal), np.arange(0,len(train_set_aluminum_hold_pose_19),1.), train_set_aluminum_hold_pose_19[:,ch_ex]) ))
    train_set_aluminum_hold_pose_norm20 = np.hstack(( train_set_aluminum_hold_pose_norm20, np.interp(np.linspace(0, len(train_set_aluminum_hold_pose_20)-1, len_normal), np.arange(0,len(train_set_aluminum_hold_pose_20),1.), train_set_aluminum_hold_pose_20[:,ch_ex]) ))
    train_set_aluminum_hold_pose_norm21 = np.hstack(( train_set_aluminum_hold_pose_norm21, np.interp(np.linspace(0, len(train_set_aluminum_hold_pose_21)-1, len_normal), np.arange(0,len(train_set_aluminum_hold_pose_21),1.), train_set_aluminum_hold_pose_21[:,ch_ex]) ))
    train_set_aluminum_hold_pose_norm22 = np.hstack(( train_set_aluminum_hold_pose_norm22, np.interp(np.linspace(0, len(train_set_aluminum_hold_pose_22)-1, len_normal), np.arange(0,len(train_set_aluminum_hold_pose_22),1.), train_set_aluminum_hold_pose_22[:,ch_ex]) ))
    train_set_aluminum_hold_pose_norm23 = np.hstack(( train_set_aluminum_hold_pose_norm23, np.interp(np.linspace(0, len(train_set_aluminum_hold_pose_23)-1, len_normal), np.arange(0,len(train_set_aluminum_hold_pose_23),1.), train_set_aluminum_hold_pose_23[:,ch_ex]) ))
    train_set_aluminum_hold_pose_norm24 = np.hstack(( train_set_aluminum_hold_pose_norm24, np.interp(np.linspace(0, len(train_set_aluminum_hold_pose_24)-1, len_normal), np.arange(0,len(train_set_aluminum_hold_pose_24),1.), train_set_aluminum_hold_pose_24[:,ch_ex]) ))
    train_set_aluminum_hold_pose_norm25 = np.hstack(( train_set_aluminum_hold_pose_norm25, np.interp(np.linspace(0, len(train_set_aluminum_hold_pose_25)-1, len_normal), np.arange(0,len(train_set_aluminum_hold_pose_25),1.), train_set_aluminum_hold_pose_25[:,ch_ex]) ))
    train_set_aluminum_hold_pose_norm26 = np.hstack(( train_set_aluminum_hold_pose_norm26, np.interp(np.linspace(0, len(train_set_aluminum_hold_pose_26)-1, len_normal), np.arange(0,len(train_set_aluminum_hold_pose_26),1.), train_set_aluminum_hold_pose_26[:,ch_ex]) ))
    train_set_aluminum_hold_pose_norm27 = np.hstack(( train_set_aluminum_hold_pose_norm27, np.interp(np.linspace(0, len(train_set_aluminum_hold_pose_27)-1, len_normal), np.arange(0,len(train_set_aluminum_hold_pose_27),1.), train_set_aluminum_hold_pose_27[:,ch_ex]) ))
    train_set_aluminum_hold_pose_norm28 = np.hstack(( train_set_aluminum_hold_pose_norm28, np.interp(np.linspace(0, len(train_set_aluminum_hold_pose_28)-1, len_normal), np.arange(0,len(train_set_aluminum_hold_pose_28),1.), train_set_aluminum_hold_pose_28[:,ch_ex]) ))
    train_set_aluminum_hold_pose_norm29 = np.hstack(( train_set_aluminum_hold_pose_norm29, np.interp(np.linspace(0, len(train_set_aluminum_hold_pose_29)-1, len_normal), np.arange(0,len(train_set_aluminum_hold_pose_29),1.), train_set_aluminum_hold_pose_29[:,ch_ex]) ))
    test_set_aluminum_hold_pose_norm = np.hstack(( test_set_aluminum_hold_pose_norm, np.interp(np.linspace(0, len(test_set_aluminum_hold_pose)-1, len_normal), np.arange(0,len(test_set_aluminum_hold_pose),1.), test_set_aluminum_hold_pose[:,ch_ex]) ))
train_set_aluminum_hold_pose_norm00 = train_set_aluminum_hold_pose_norm00.reshape(7,len_normal).T
train_set_aluminum_hold_pose_norm01 = train_set_aluminum_hold_pose_norm01.reshape(7,len_normal).T
train_set_aluminum_hold_pose_norm02 = train_set_aluminum_hold_pose_norm02.reshape(7,len_normal).T
train_set_aluminum_hold_pose_norm03 = train_set_aluminum_hold_pose_norm03.reshape(7,len_normal).T
train_set_aluminum_hold_pose_norm04 = train_set_aluminum_hold_pose_norm04.reshape(7,len_normal).T
train_set_aluminum_hold_pose_norm05 = train_set_aluminum_hold_pose_norm05.reshape(7,len_normal).T
train_set_aluminum_hold_pose_norm06 = train_set_aluminum_hold_pose_norm06.reshape(7,len_normal).T
train_set_aluminum_hold_pose_norm07 = train_set_aluminum_hold_pose_norm07.reshape(7,len_normal).T
train_set_aluminum_hold_pose_norm08 = train_set_aluminum_hold_pose_norm08.reshape(7,len_normal).T
train_set_aluminum_hold_pose_norm09 = train_set_aluminum_hold_pose_norm09.reshape(7,len_normal).T
train_set_aluminum_hold_pose_norm10 = train_set_aluminum_hold_pose_norm10.reshape(7,len_normal).T
train_set_aluminum_hold_pose_norm11 = train_set_aluminum_hold_pose_norm11.reshape(7,len_normal).T
train_set_aluminum_hold_pose_norm12 = train_set_aluminum_hold_pose_norm12.reshape(7,len_normal).T
train_set_aluminum_hold_pose_norm13 = train_set_aluminum_hold_pose_norm13.reshape(7,len_normal).T
train_set_aluminum_hold_pose_norm14 = train_set_aluminum_hold_pose_norm14.reshape(7,len_normal).T
train_set_aluminum_hold_pose_norm15 = train_set_aluminum_hold_pose_norm15.reshape(7,len_normal).T
train_set_aluminum_hold_pose_norm16 = train_set_aluminum_hold_pose_norm16.reshape(7,len_normal).T
train_set_aluminum_hold_pose_norm17 = train_set_aluminum_hold_pose_norm17.reshape(7,len_normal).T
train_set_aluminum_hold_pose_norm18 = train_set_aluminum_hold_pose_norm18.reshape(7,len_normal).T
train_set_aluminum_hold_pose_norm19 = train_set_aluminum_hold_pose_norm19.reshape(7,len_normal).T
train_set_aluminum_hold_pose_norm20 = train_set_aluminum_hold_pose_norm20.reshape(7,len_normal).T
train_set_aluminum_hold_pose_norm21 = train_set_aluminum_hold_pose_norm21.reshape(7,len_normal).T
train_set_aluminum_hold_pose_norm22 = train_set_aluminum_hold_pose_norm22.reshape(7,len_normal).T
train_set_aluminum_hold_pose_norm23 = train_set_aluminum_hold_pose_norm23.reshape(7,len_normal).T
train_set_aluminum_hold_pose_norm24 = train_set_aluminum_hold_pose_norm24.reshape(7,len_normal).T
train_set_aluminum_hold_pose_norm25 = train_set_aluminum_hold_pose_norm25.reshape(7,len_normal).T
train_set_aluminum_hold_pose_norm26 = train_set_aluminum_hold_pose_norm26.reshape(7,len_normal).T
train_set_aluminum_hold_pose_norm27 = train_set_aluminum_hold_pose_norm27.reshape(7,len_normal).T
train_set_aluminum_hold_pose_norm28 = train_set_aluminum_hold_pose_norm28.reshape(7,len_normal).T
train_set_aluminum_hold_pose_norm29 = train_set_aluminum_hold_pose_norm29.reshape(7,len_normal).T
test_set_aluminum_hold_pose_norm = test_set_aluminum_hold_pose_norm.reshape(7,len_normal).T
train_set_aluminum_hold_pose_norm_full = np.array([train_set_aluminum_hold_pose_norm00,train_set_aluminum_hold_pose_norm01,train_set_aluminum_hold_pose_norm02,train_set_aluminum_hold_pose_norm03,train_set_aluminum_hold_pose_norm04,
                                    train_set_aluminum_hold_pose_norm05,train_set_aluminum_hold_pose_norm06,train_set_aluminum_hold_pose_norm07,train_set_aluminum_hold_pose_norm08,train_set_aluminum_hold_pose_norm09,
                                    train_set_aluminum_hold_pose_norm10,train_set_aluminum_hold_pose_norm11,train_set_aluminum_hold_pose_norm12,train_set_aluminum_hold_pose_norm13,train_set_aluminum_hold_pose_norm14,
                                    train_set_aluminum_hold_pose_norm15,train_set_aluminum_hold_pose_norm16,train_set_aluminum_hold_pose_norm17,train_set_aluminum_hold_pose_norm18,train_set_aluminum_hold_pose_norm19])
##########################################################################################
print('normalizing Pose data into same duration of spanner_handover')
# resampling pose signals of aluminum_hold
train_set_spanner_handover_pose_norm00=np.array([]);train_set_spanner_handover_pose_norm01=np.array([]);train_set_spanner_handover_pose_norm02=np.array([]);train_set_spanner_handover_pose_norm03=np.array([]);train_set_spanner_handover_pose_norm04=np.array([]);
train_set_spanner_handover_pose_norm05=np.array([]);train_set_spanner_handover_pose_norm06=np.array([]);train_set_spanner_handover_pose_norm07=np.array([]);train_set_spanner_handover_pose_norm08=np.array([]);train_set_spanner_handover_pose_norm09=np.array([]);
train_set_spanner_handover_pose_norm10=np.array([]);train_set_spanner_handover_pose_norm11=np.array([]);train_set_spanner_handover_pose_norm12=np.array([]);train_set_spanner_handover_pose_norm13=np.array([]);train_set_spanner_handover_pose_norm14=np.array([]);
train_set_spanner_handover_pose_norm15=np.array([]);train_set_spanner_handover_pose_norm16=np.array([]);train_set_spanner_handover_pose_norm17=np.array([]);train_set_spanner_handover_pose_norm18=np.array([]);train_set_spanner_handover_pose_norm19=np.array([]);
train_set_spanner_handover_pose_norm20=np.array([]);train_set_spanner_handover_pose_norm21=np.array([]);train_set_spanner_handover_pose_norm22=np.array([]);train_set_spanner_handover_pose_norm23=np.array([]);train_set_spanner_handover_pose_norm24=np.array([]);
train_set_spanner_handover_pose_norm25=np.array([]);train_set_spanner_handover_pose_norm26=np.array([]);train_set_spanner_handover_pose_norm27=np.array([]);train_set_spanner_handover_pose_norm28=np.array([]);train_set_spanner_handover_pose_norm29=np.array([]);
test_set_spanner_handover_pose_norm=np.array([]);
for ch_ex in range(7):
    train_set_spanner_handover_pose_norm00 = np.hstack(( train_set_spanner_handover_pose_norm00, np.interp(np.linspace(0, len(train_set_spanner_handover_pose_00)-1, len_normal), np.arange(0,len(train_set_spanner_handover_pose_00),1.), train_set_spanner_handover_pose_00[:,ch_ex]) ))
    train_set_spanner_handover_pose_norm01 = np.hstack(( train_set_spanner_handover_pose_norm01, np.interp(np.linspace(0, len(train_set_spanner_handover_pose_01)-1, len_normal), np.arange(0,len(train_set_spanner_handover_pose_01),1.), train_set_spanner_handover_pose_01[:,ch_ex]) ))
    train_set_spanner_handover_pose_norm02 = np.hstack(( train_set_spanner_handover_pose_norm02, np.interp(np.linspace(0, len(train_set_spanner_handover_pose_02)-1, len_normal), np.arange(0,len(train_set_spanner_handover_pose_02),1.), train_set_spanner_handover_pose_02[:,ch_ex]) ))
    train_set_spanner_handover_pose_norm03 = np.hstack(( train_set_spanner_handover_pose_norm03, np.interp(np.linspace(0, len(train_set_spanner_handover_pose_03)-1, len_normal), np.arange(0,len(train_set_spanner_handover_pose_03),1.), train_set_spanner_handover_pose_03[:,ch_ex]) ))
    train_set_spanner_handover_pose_norm04 = np.hstack(( train_set_spanner_handover_pose_norm04, np.interp(np.linspace(0, len(train_set_spanner_handover_pose_04)-1, len_normal), np.arange(0,len(train_set_spanner_handover_pose_04),1.), train_set_spanner_handover_pose_04[:,ch_ex]) ))
    train_set_spanner_handover_pose_norm05 = np.hstack(( train_set_spanner_handover_pose_norm05, np.interp(np.linspace(0, len(train_set_spanner_handover_pose_05)-1, len_normal), np.arange(0,len(train_set_spanner_handover_pose_05),1.), train_set_spanner_handover_pose_05[:,ch_ex]) ))
    train_set_spanner_handover_pose_norm06 = np.hstack(( train_set_spanner_handover_pose_norm06, np.interp(np.linspace(0, len(train_set_spanner_handover_pose_06)-1, len_normal), np.arange(0,len(train_set_spanner_handover_pose_06),1.), train_set_spanner_handover_pose_06[:,ch_ex]) ))
    train_set_spanner_handover_pose_norm07 = np.hstack(( train_set_spanner_handover_pose_norm07, np.interp(np.linspace(0, len(train_set_spanner_handover_pose_07)-1, len_normal), np.arange(0,len(train_set_spanner_handover_pose_07),1.), train_set_spanner_handover_pose_07[:,ch_ex]) ))
    train_set_spanner_handover_pose_norm08 = np.hstack(( train_set_spanner_handover_pose_norm08, np.interp(np.linspace(0, len(train_set_spanner_handover_pose_08)-1, len_normal), np.arange(0,len(train_set_spanner_handover_pose_08),1.), train_set_spanner_handover_pose_08[:,ch_ex]) ))
    train_set_spanner_handover_pose_norm09 = np.hstack(( train_set_spanner_handover_pose_norm09, np.interp(np.linspace(0, len(train_set_spanner_handover_pose_09)-1, len_normal), np.arange(0,len(train_set_spanner_handover_pose_09),1.), train_set_spanner_handover_pose_09[:,ch_ex]) ))
    train_set_spanner_handover_pose_norm10 = np.hstack(( train_set_spanner_handover_pose_norm10, np.interp(np.linspace(0, len(train_set_spanner_handover_pose_10)-1, len_normal), np.arange(0,len(train_set_spanner_handover_pose_10),1.), train_set_spanner_handover_pose_10[:,ch_ex]) ))
    train_set_spanner_handover_pose_norm11 = np.hstack(( train_set_spanner_handover_pose_norm11, np.interp(np.linspace(0, len(train_set_spanner_handover_pose_11)-1, len_normal), np.arange(0,len(train_set_spanner_handover_pose_11),1.), train_set_spanner_handover_pose_11[:,ch_ex]) ))
    train_set_spanner_handover_pose_norm12 = np.hstack(( train_set_spanner_handover_pose_norm12, np.interp(np.linspace(0, len(train_set_spanner_handover_pose_12)-1, len_normal), np.arange(0,len(train_set_spanner_handover_pose_12),1.), train_set_spanner_handover_pose_12[:,ch_ex]) ))
    train_set_spanner_handover_pose_norm13 = np.hstack(( train_set_spanner_handover_pose_norm13, np.interp(np.linspace(0, len(train_set_spanner_handover_pose_13)-1, len_normal), np.arange(0,len(train_set_spanner_handover_pose_13),1.), train_set_spanner_handover_pose_13[:,ch_ex]) ))
    train_set_spanner_handover_pose_norm14 = np.hstack(( train_set_spanner_handover_pose_norm14, np.interp(np.linspace(0, len(train_set_spanner_handover_pose_14)-1, len_normal), np.arange(0,len(train_set_spanner_handover_pose_14),1.), train_set_spanner_handover_pose_14[:,ch_ex]) ))
    train_set_spanner_handover_pose_norm15 = np.hstack(( train_set_spanner_handover_pose_norm15, np.interp(np.linspace(0, len(train_set_spanner_handover_pose_15)-1, len_normal), np.arange(0,len(train_set_spanner_handover_pose_15),1.), train_set_spanner_handover_pose_15[:,ch_ex]) ))
    train_set_spanner_handover_pose_norm16 = np.hstack(( train_set_spanner_handover_pose_norm16, np.interp(np.linspace(0, len(train_set_spanner_handover_pose_16)-1, len_normal), np.arange(0,len(train_set_spanner_handover_pose_16),1.), train_set_spanner_handover_pose_16[:,ch_ex]) ))
    train_set_spanner_handover_pose_norm17 = np.hstack(( train_set_spanner_handover_pose_norm17, np.interp(np.linspace(0, len(train_set_spanner_handover_pose_17)-1, len_normal), np.arange(0,len(train_set_spanner_handover_pose_17),1.), train_set_spanner_handover_pose_17[:,ch_ex]) ))
    train_set_spanner_handover_pose_norm18 = np.hstack(( train_set_spanner_handover_pose_norm18, np.interp(np.linspace(0, len(train_set_spanner_handover_pose_18)-1, len_normal), np.arange(0,len(train_set_spanner_handover_pose_18),1.), train_set_spanner_handover_pose_18[:,ch_ex]) ))
    train_set_spanner_handover_pose_norm19 = np.hstack(( train_set_spanner_handover_pose_norm19, np.interp(np.linspace(0, len(train_set_spanner_handover_pose_19)-1, len_normal), np.arange(0,len(train_set_spanner_handover_pose_19),1.), train_set_spanner_handover_pose_19[:,ch_ex]) ))
    train_set_spanner_handover_pose_norm20 = np.hstack((train_set_spanner_handover_pose_norm20, np.interp( np.linspace(0, len(train_set_spanner_handover_pose_20) - 1, len_normal), np.arange(0, len(train_set_spanner_handover_pose_20), 1.), train_set_spanner_handover_pose_20[:, ch_ex])))
    train_set_spanner_handover_pose_norm21 = np.hstack((train_set_spanner_handover_pose_norm21, np.interp( np.linspace(0, len(train_set_spanner_handover_pose_21) - 1, len_normal), np.arange(0, len(train_set_spanner_handover_pose_21), 1.), train_set_spanner_handover_pose_21[:, ch_ex])))
    train_set_spanner_handover_pose_norm22 = np.hstack((train_set_spanner_handover_pose_norm22, np.interp( np.linspace(0, len(train_set_spanner_handover_pose_22) - 1, len_normal), np.arange(0, len(train_set_spanner_handover_pose_22), 1.), train_set_spanner_handover_pose_22[:, ch_ex])))
    train_set_spanner_handover_pose_norm23 = np.hstack((train_set_spanner_handover_pose_norm23, np.interp( np.linspace(0, len(train_set_spanner_handover_pose_23) - 1, len_normal), np.arange(0, len(train_set_spanner_handover_pose_23), 1.), train_set_spanner_handover_pose_23[:, ch_ex])))
    train_set_spanner_handover_pose_norm24 = np.hstack((train_set_spanner_handover_pose_norm24, np.interp( np.linspace(0, len(train_set_spanner_handover_pose_24) - 1, len_normal), np.arange(0, len(train_set_spanner_handover_pose_24), 1.), train_set_spanner_handover_pose_24[:, ch_ex])))
    train_set_spanner_handover_pose_norm25 = np.hstack((train_set_spanner_handover_pose_norm25, np.interp( np.linspace(0, len(train_set_spanner_handover_pose_25) - 1, len_normal), np.arange(0, len(train_set_spanner_handover_pose_25), 1.), train_set_spanner_handover_pose_25[:, ch_ex])))
    train_set_spanner_handover_pose_norm26 = np.hstack((train_set_spanner_handover_pose_norm26, np.interp( np.linspace(0, len(train_set_spanner_handover_pose_26) - 1, len_normal), np.arange(0, len(train_set_spanner_handover_pose_26), 1.), train_set_spanner_handover_pose_26[:, ch_ex])))
    train_set_spanner_handover_pose_norm27 = np.hstack((train_set_spanner_handover_pose_norm27, np.interp( np.linspace(0, len(train_set_spanner_handover_pose_27) - 1, len_normal), np.arange(0, len(train_set_spanner_handover_pose_27), 1.), train_set_spanner_handover_pose_27[:, ch_ex])))
    train_set_spanner_handover_pose_norm28 = np.hstack((train_set_spanner_handover_pose_norm28, np.interp( np.linspace(0, len(train_set_spanner_handover_pose_28) - 1, len_normal), np.arange(0, len(train_set_spanner_handover_pose_28), 1.), train_set_spanner_handover_pose_28[:, ch_ex])))
    train_set_spanner_handover_pose_norm29 = np.hstack((train_set_spanner_handover_pose_norm29, np.interp( np.linspace(0, len(train_set_spanner_handover_pose_29) - 1, len_normal), np.arange(0, len(train_set_spanner_handover_pose_29), 1.), train_set_spanner_handover_pose_29[:, ch_ex])))
    test_set_spanner_handover_pose_norm = np.hstack(( test_set_spanner_handover_pose_norm, np.interp(np.linspace(0, len(test_set_spanner_handover_pose)-1, len_normal), np.arange(0,len(test_set_spanner_handover_pose),1.), test_set_spanner_handover_pose[:,ch_ex]) ))
train_set_spanner_handover_pose_norm00 = train_set_spanner_handover_pose_norm00.reshape(7,len_normal).T
train_set_spanner_handover_pose_norm01 = train_set_spanner_handover_pose_norm01.reshape(7,len_normal).T
train_set_spanner_handover_pose_norm02 = train_set_spanner_handover_pose_norm02.reshape(7,len_normal).T
train_set_spanner_handover_pose_norm03 = train_set_spanner_handover_pose_norm03.reshape(7,len_normal).T
train_set_spanner_handover_pose_norm04 = train_set_spanner_handover_pose_norm04.reshape(7,len_normal).T
train_set_spanner_handover_pose_norm05 = train_set_spanner_handover_pose_norm05.reshape(7,len_normal).T
train_set_spanner_handover_pose_norm06 = train_set_spanner_handover_pose_norm06.reshape(7,len_normal).T
train_set_spanner_handover_pose_norm07 = train_set_spanner_handover_pose_norm07.reshape(7,len_normal).T
train_set_spanner_handover_pose_norm08 = train_set_spanner_handover_pose_norm08.reshape(7,len_normal).T
train_set_spanner_handover_pose_norm09 = train_set_spanner_handover_pose_norm09.reshape(7,len_normal).T
train_set_spanner_handover_pose_norm10 = train_set_spanner_handover_pose_norm10.reshape(7,len_normal).T
train_set_spanner_handover_pose_norm11 = train_set_spanner_handover_pose_norm11.reshape(7,len_normal).T
train_set_spanner_handover_pose_norm12 = train_set_spanner_handover_pose_norm12.reshape(7,len_normal).T
train_set_spanner_handover_pose_norm13 = train_set_spanner_handover_pose_norm13.reshape(7,len_normal).T
train_set_spanner_handover_pose_norm14 = train_set_spanner_handover_pose_norm14.reshape(7,len_normal).T
train_set_spanner_handover_pose_norm15 = train_set_spanner_handover_pose_norm15.reshape(7,len_normal).T
train_set_spanner_handover_pose_norm16 = train_set_spanner_handover_pose_norm16.reshape(7,len_normal).T
train_set_spanner_handover_pose_norm17 = train_set_spanner_handover_pose_norm17.reshape(7,len_normal).T
train_set_spanner_handover_pose_norm18 = train_set_spanner_handover_pose_norm18.reshape(7,len_normal).T
train_set_spanner_handover_pose_norm19 = train_set_spanner_handover_pose_norm19.reshape(7,len_normal).T
train_set_spanner_handover_pose_norm20 = train_set_spanner_handover_pose_norm20.reshape(7,len_normal).T
train_set_spanner_handover_pose_norm21 = train_set_spanner_handover_pose_norm21.reshape(7,len_normal).T
train_set_spanner_handover_pose_norm22 = train_set_spanner_handover_pose_norm22.reshape(7,len_normal).T
train_set_spanner_handover_pose_norm23 = train_set_spanner_handover_pose_norm23.reshape(7,len_normal).T
train_set_spanner_handover_pose_norm24 = train_set_spanner_handover_pose_norm24.reshape(7,len_normal).T
train_set_spanner_handover_pose_norm25 = train_set_spanner_handover_pose_norm25.reshape(7,len_normal).T
train_set_spanner_handover_pose_norm26 = train_set_spanner_handover_pose_norm26.reshape(7,len_normal).T
train_set_spanner_handover_pose_norm27 = train_set_spanner_handover_pose_norm27.reshape(7,len_normal).T
train_set_spanner_handover_pose_norm28 = train_set_spanner_handover_pose_norm28.reshape(7,len_normal).T
train_set_spanner_handover_pose_norm29 = train_set_spanner_handover_pose_norm29.reshape(7,len_normal).T
test_set_spanner_handover_pose_norm = test_set_spanner_handover_pose_norm.reshape(7,len_normal).T
train_set_spanner_handover_pose_norm_full = np.array([train_set_spanner_handover_pose_norm00,train_set_spanner_handover_pose_norm01,train_set_spanner_handover_pose_norm02,train_set_spanner_handover_pose_norm03,train_set_spanner_handover_pose_norm04,
                                    train_set_spanner_handover_pose_norm05,train_set_spanner_handover_pose_norm06,train_set_spanner_handover_pose_norm07,train_set_spanner_handover_pose_norm08,train_set_spanner_handover_pose_norm09,
                                    train_set_spanner_handover_pose_norm10,train_set_spanner_handover_pose_norm11,train_set_spanner_handover_pose_norm12,train_set_spanner_handover_pose_norm13,train_set_spanner_handover_pose_norm14,
                                    train_set_spanner_handover_pose_norm15,train_set_spanner_handover_pose_norm16,train_set_spanner_handover_pose_norm17,train_set_spanner_handover_pose_norm18,train_set_spanner_handover_pose_norm19])
##########################################################################################
print('normalizing Pose data into same duration of tape_hold')
# resampling pose signals of aluminum_hold
train_set_tape_hold_pose_norm00=np.array([]);train_set_tape_hold_pose_norm01=np.array([]);train_set_tape_hold_pose_norm02=np.array([]);train_set_tape_hold_pose_norm03=np.array([]);train_set_tape_hold_pose_norm04=np.array([]);
train_set_tape_hold_pose_norm05=np.array([]);train_set_tape_hold_pose_norm06=np.array([]);train_set_tape_hold_pose_norm07=np.array([]);train_set_tape_hold_pose_norm08=np.array([]);train_set_tape_hold_pose_norm09=np.array([]);
train_set_tape_hold_pose_norm10=np.array([]);train_set_tape_hold_pose_norm11=np.array([]);train_set_tape_hold_pose_norm12=np.array([]);train_set_tape_hold_pose_norm13=np.array([]);train_set_tape_hold_pose_norm14=np.array([]);
train_set_tape_hold_pose_norm15=np.array([]);train_set_tape_hold_pose_norm16=np.array([]);train_set_tape_hold_pose_norm17=np.array([]);train_set_tape_hold_pose_norm18=np.array([]);train_set_tape_hold_pose_norm19=np.array([]);
train_set_tape_hold_pose_norm20=np.array([]);train_set_tape_hold_pose_norm21=np.array([]);train_set_tape_hold_pose_norm22=np.array([]);train_set_tape_hold_pose_norm23=np.array([]);train_set_tape_hold_pose_norm24=np.array([]);
train_set_tape_hold_pose_norm25=np.array([]);train_set_tape_hold_pose_norm26=np.array([]);train_set_tape_hold_pose_norm27=np.array([]);train_set_tape_hold_pose_norm28=np.array([]);train_set_tape_hold_pose_norm29=np.array([]);
test_set_tape_hold_pose_norm=np.array([]);
for ch_ex in range(7):
    train_set_tape_hold_pose_norm00 = np.hstack(( train_set_tape_hold_pose_norm00, np.interp(np.linspace(0, len(train_set_tape_hold_pose_00)-1, len_normal), np.arange(0,len(train_set_tape_hold_pose_00),1.), train_set_tape_hold_pose_00[:,ch_ex]) ))
    train_set_tape_hold_pose_norm01 = np.hstack(( train_set_tape_hold_pose_norm01, np.interp(np.linspace(0, len(train_set_tape_hold_pose_01)-1, len_normal), np.arange(0,len(train_set_tape_hold_pose_01),1.), train_set_tape_hold_pose_01[:,ch_ex]) ))
    train_set_tape_hold_pose_norm02 = np.hstack(( train_set_tape_hold_pose_norm02, np.interp(np.linspace(0, len(train_set_tape_hold_pose_02)-1, len_normal), np.arange(0,len(train_set_tape_hold_pose_02),1.), train_set_tape_hold_pose_02[:,ch_ex]) ))
    train_set_tape_hold_pose_norm03 = np.hstack(( train_set_tape_hold_pose_norm03, np.interp(np.linspace(0, len(train_set_tape_hold_pose_03)-1, len_normal), np.arange(0,len(train_set_tape_hold_pose_03),1.), train_set_tape_hold_pose_03[:,ch_ex]) ))
    train_set_tape_hold_pose_norm04 = np.hstack(( train_set_tape_hold_pose_norm04, np.interp(np.linspace(0, len(train_set_tape_hold_pose_04)-1, len_normal), np.arange(0,len(train_set_tape_hold_pose_04),1.), train_set_tape_hold_pose_04[:,ch_ex]) ))
    train_set_tape_hold_pose_norm05 = np.hstack(( train_set_tape_hold_pose_norm05, np.interp(np.linspace(0, len(train_set_tape_hold_pose_05)-1, len_normal), np.arange(0,len(train_set_tape_hold_pose_05),1.), train_set_tape_hold_pose_05[:,ch_ex]) ))
    train_set_tape_hold_pose_norm06 = np.hstack(( train_set_tape_hold_pose_norm06, np.interp(np.linspace(0, len(train_set_tape_hold_pose_06)-1, len_normal), np.arange(0,len(train_set_tape_hold_pose_06),1.), train_set_tape_hold_pose_06[:,ch_ex]) ))
    train_set_tape_hold_pose_norm07 = np.hstack(( train_set_tape_hold_pose_norm07, np.interp(np.linspace(0, len(train_set_tape_hold_pose_07)-1, len_normal), np.arange(0,len(train_set_tape_hold_pose_07),1.), train_set_tape_hold_pose_07[:,ch_ex]) ))
    train_set_tape_hold_pose_norm08 = np.hstack(( train_set_tape_hold_pose_norm08, np.interp(np.linspace(0, len(train_set_tape_hold_pose_08)-1, len_normal), np.arange(0,len(train_set_tape_hold_pose_08),1.), train_set_tape_hold_pose_08[:,ch_ex]) ))
    train_set_tape_hold_pose_norm09 = np.hstack(( train_set_tape_hold_pose_norm09, np.interp(np.linspace(0, len(train_set_tape_hold_pose_09)-1, len_normal), np.arange(0,len(train_set_tape_hold_pose_09),1.), train_set_tape_hold_pose_09[:,ch_ex]) ))
    train_set_tape_hold_pose_norm10 = np.hstack(( train_set_tape_hold_pose_norm10, np.interp(np.linspace(0, len(train_set_tape_hold_pose_10)-1, len_normal), np.arange(0,len(train_set_tape_hold_pose_10),1.), train_set_tape_hold_pose_10[:,ch_ex]) ))
    train_set_tape_hold_pose_norm11 = np.hstack(( train_set_tape_hold_pose_norm11, np.interp(np.linspace(0, len(train_set_tape_hold_pose_11)-1, len_normal), np.arange(0,len(train_set_tape_hold_pose_11),1.), train_set_tape_hold_pose_11[:,ch_ex]) ))
    train_set_tape_hold_pose_norm12 = np.hstack(( train_set_tape_hold_pose_norm12, np.interp(np.linspace(0, len(train_set_tape_hold_pose_12)-1, len_normal), np.arange(0,len(train_set_tape_hold_pose_12),1.), train_set_tape_hold_pose_12[:,ch_ex]) ))
    train_set_tape_hold_pose_norm13 = np.hstack(( train_set_tape_hold_pose_norm13, np.interp(np.linspace(0, len(train_set_tape_hold_pose_13)-1, len_normal), np.arange(0,len(train_set_tape_hold_pose_13),1.), train_set_tape_hold_pose_13[:,ch_ex]) ))
    train_set_tape_hold_pose_norm14 = np.hstack(( train_set_tape_hold_pose_norm14, np.interp(np.linspace(0, len(train_set_tape_hold_pose_14)-1, len_normal), np.arange(0,len(train_set_tape_hold_pose_14),1.), train_set_tape_hold_pose_14[:,ch_ex]) ))
    train_set_tape_hold_pose_norm15 = np.hstack(( train_set_tape_hold_pose_norm15, np.interp(np.linspace(0, len(train_set_tape_hold_pose_15)-1, len_normal), np.arange(0,len(train_set_tape_hold_pose_15),1.), train_set_tape_hold_pose_15[:,ch_ex]) ))
    train_set_tape_hold_pose_norm16 = np.hstack(( train_set_tape_hold_pose_norm16, np.interp(np.linspace(0, len(train_set_tape_hold_pose_16)-1, len_normal), np.arange(0,len(train_set_tape_hold_pose_16),1.), train_set_tape_hold_pose_16[:,ch_ex]) ))
    train_set_tape_hold_pose_norm17 = np.hstack(( train_set_tape_hold_pose_norm17, np.interp(np.linspace(0, len(train_set_tape_hold_pose_17)-1, len_normal), np.arange(0,len(train_set_tape_hold_pose_17),1.), train_set_tape_hold_pose_17[:,ch_ex]) ))
    train_set_tape_hold_pose_norm18 = np.hstack(( train_set_tape_hold_pose_norm18, np.interp(np.linspace(0, len(train_set_tape_hold_pose_18)-1, len_normal), np.arange(0,len(train_set_tape_hold_pose_18),1.), train_set_tape_hold_pose_18[:,ch_ex]) ))
    train_set_tape_hold_pose_norm19 = np.hstack(( train_set_tape_hold_pose_norm19, np.interp(np.linspace(0, len(train_set_tape_hold_pose_19)-1, len_normal), np.arange(0,len(train_set_tape_hold_pose_19),1.), train_set_tape_hold_pose_19[:,ch_ex]) ))
    train_set_tape_hold_pose_norm20 = np.hstack((train_set_tape_hold_pose_norm20, np.interp( np.linspace(0, len(train_set_tape_hold_pose_20) - 1, len_normal), np.arange(0, len(train_set_tape_hold_pose_20), 1.), train_set_tape_hold_pose_20[:, ch_ex])))
    train_set_tape_hold_pose_norm21 = np.hstack((train_set_tape_hold_pose_norm21, np.interp( np.linspace(0, len(train_set_tape_hold_pose_21) - 1, len_normal), np.arange(0, len(train_set_tape_hold_pose_21), 1.), train_set_tape_hold_pose_21[:, ch_ex])))
    train_set_tape_hold_pose_norm22 = np.hstack((train_set_tape_hold_pose_norm22, np.interp( np.linspace(0, len(train_set_tape_hold_pose_22) - 1, len_normal), np.arange(0, len(train_set_tape_hold_pose_22), 1.), train_set_tape_hold_pose_22[:, ch_ex])))
    train_set_tape_hold_pose_norm23 = np.hstack((train_set_tape_hold_pose_norm23, np.interp( np.linspace(0, len(train_set_tape_hold_pose_23) - 1, len_normal), np.arange(0, len(train_set_tape_hold_pose_23), 1.), train_set_tape_hold_pose_23[:, ch_ex])))
    train_set_tape_hold_pose_norm24 = np.hstack((train_set_tape_hold_pose_norm24, np.interp( np.linspace(0, len(train_set_tape_hold_pose_24) - 1, len_normal), np.arange(0, len(train_set_tape_hold_pose_24), 1.), train_set_tape_hold_pose_24[:, ch_ex])))
    train_set_tape_hold_pose_norm25 = np.hstack((train_set_tape_hold_pose_norm25, np.interp( np.linspace(0, len(train_set_tape_hold_pose_25) - 1, len_normal), np.arange(0, len(train_set_tape_hold_pose_25), 1.), train_set_tape_hold_pose_25[:, ch_ex])))
    train_set_tape_hold_pose_norm26 = np.hstack((train_set_tape_hold_pose_norm26, np.interp( np.linspace(0, len(train_set_tape_hold_pose_26) - 1, len_normal), np.arange(0, len(train_set_tape_hold_pose_26), 1.), train_set_tape_hold_pose_26[:, ch_ex])))
    train_set_tape_hold_pose_norm27 = np.hstack((train_set_tape_hold_pose_norm27, np.interp( np.linspace(0, len(train_set_tape_hold_pose_27) - 1, len_normal), np.arange(0, len(train_set_tape_hold_pose_27), 1.), train_set_tape_hold_pose_27[:, ch_ex])))
    train_set_tape_hold_pose_norm28 = np.hstack((train_set_tape_hold_pose_norm28, np.interp( np.linspace(0, len(train_set_tape_hold_pose_28) - 1, len_normal), np.arange(0, len(train_set_tape_hold_pose_28), 1.), train_set_tape_hold_pose_28[:, ch_ex])))
    train_set_tape_hold_pose_norm29 = np.hstack((train_set_tape_hold_pose_norm29, np.interp( np.linspace(0, len(train_set_tape_hold_pose_29) - 1, len_normal), np.arange(0, len(train_set_tape_hold_pose_29), 1.), train_set_tape_hold_pose_29[:, ch_ex])))
    test_set_tape_hold_pose_norm = np.hstack(( test_set_tape_hold_pose_norm, np.interp(np.linspace(0, len(test_set_tape_hold_pose)-1, len_normal), np.arange(0,len(test_set_tape_hold_pose),1.), test_set_tape_hold_pose[:,ch_ex]) ))
train_set_tape_hold_pose_norm00 = train_set_tape_hold_pose_norm00.reshape(7,len_normal).T
train_set_tape_hold_pose_norm01 = train_set_tape_hold_pose_norm01.reshape(7,len_normal).T
train_set_tape_hold_pose_norm02 = train_set_tape_hold_pose_norm02.reshape(7,len_normal).T
train_set_tape_hold_pose_norm03 = train_set_tape_hold_pose_norm03.reshape(7,len_normal).T
train_set_tape_hold_pose_norm04 = train_set_tape_hold_pose_norm04.reshape(7,len_normal).T
train_set_tape_hold_pose_norm05 = train_set_tape_hold_pose_norm05.reshape(7,len_normal).T
train_set_tape_hold_pose_norm06 = train_set_tape_hold_pose_norm06.reshape(7,len_normal).T
train_set_tape_hold_pose_norm07 = train_set_tape_hold_pose_norm07.reshape(7,len_normal).T
train_set_tape_hold_pose_norm08 = train_set_tape_hold_pose_norm08.reshape(7,len_normal).T
train_set_tape_hold_pose_norm09 = train_set_tape_hold_pose_norm09.reshape(7,len_normal).T
train_set_tape_hold_pose_norm10 = train_set_tape_hold_pose_norm10.reshape(7,len_normal).T
train_set_tape_hold_pose_norm11 = train_set_tape_hold_pose_norm11.reshape(7,len_normal).T
train_set_tape_hold_pose_norm12 = train_set_tape_hold_pose_norm12.reshape(7,len_normal).T
train_set_tape_hold_pose_norm13 = train_set_tape_hold_pose_norm13.reshape(7,len_normal).T
train_set_tape_hold_pose_norm14 = train_set_tape_hold_pose_norm14.reshape(7,len_normal).T
train_set_tape_hold_pose_norm15 = train_set_tape_hold_pose_norm15.reshape(7,len_normal).T
train_set_tape_hold_pose_norm16 = train_set_tape_hold_pose_norm16.reshape(7,len_normal).T
train_set_tape_hold_pose_norm17 = train_set_tape_hold_pose_norm17.reshape(7,len_normal).T
train_set_tape_hold_pose_norm18 = train_set_tape_hold_pose_norm18.reshape(7,len_normal).T
train_set_tape_hold_pose_norm19 = train_set_tape_hold_pose_norm19.reshape(7,len_normal).T
train_set_tape_hold_pose_norm20 = train_set_tape_hold_pose_norm20.reshape(7,len_normal).T
train_set_tape_hold_pose_norm21 = train_set_tape_hold_pose_norm21.reshape(7,len_normal).T
train_set_tape_hold_pose_norm22 = train_set_tape_hold_pose_norm22.reshape(7,len_normal).T
train_set_tape_hold_pose_norm23 = train_set_tape_hold_pose_norm23.reshape(7,len_normal).T
train_set_tape_hold_pose_norm24 = train_set_tape_hold_pose_norm24.reshape(7,len_normal).T
train_set_tape_hold_pose_norm25 = train_set_tape_hold_pose_norm25.reshape(7,len_normal).T
train_set_tape_hold_pose_norm26 = train_set_tape_hold_pose_norm26.reshape(7,len_normal).T
train_set_tape_hold_pose_norm27 = train_set_tape_hold_pose_norm27.reshape(7,len_normal).T
train_set_tape_hold_pose_norm28 = train_set_tape_hold_pose_norm28.reshape(7,len_normal).T
train_set_tape_hold_pose_norm29 = train_set_tape_hold_pose_norm29.reshape(7,len_normal).T
test_set_tape_hold_pose_norm = test_set_tape_hold_pose_norm.reshape(7,len_normal).T
train_set_tape_hold_pose_norm_full = np.array([train_set_tape_hold_pose_norm00,train_set_tape_hold_pose_norm01,train_set_tape_hold_pose_norm02,train_set_tape_hold_pose_norm03,train_set_tape_hold_pose_norm04,
                                    train_set_tape_hold_pose_norm05,train_set_tape_hold_pose_norm06,train_set_tape_hold_pose_norm07,train_set_tape_hold_pose_norm08,train_set_tape_hold_pose_norm09,
                                    train_set_tape_hold_pose_norm10,train_set_tape_hold_pose_norm11,train_set_tape_hold_pose_norm12,train_set_tape_hold_pose_norm13,train_set_tape_hold_pose_norm14,
                                    train_set_tape_hold_pose_norm15,train_set_tape_hold_pose_norm16,train_set_tape_hold_pose_norm17,train_set_tape_hold_pose_norm18,train_set_tape_hold_pose_norm19])



# create a 3 tasks iProMP
ipromp_aluminum_hold = iprompslib_imu_emg_joint.IProMP(num_joints=19, nrBasis=11, sigma=0.05, num_samples=101)
ipromp_spanner_handover = iprompslib_imu_emg_joint.IProMP(num_joints=19, nrBasis=11, sigma=0.05, num_samples=101)
ipromp_tape_hold = iprompslib_imu_emg_joint.IProMP(num_joints=19, nrBasis=11, sigma=0.05, num_samples=101)

# add demostration
for idx in range(0, nrDemo):
    # add demonstration of aluminum_hold
    demo_temp = np.hstack([train_set_aluminum_hold_imu_norm_full[idx], train_set_aluminum_hold_emg_norm_full[idx]])
    demo_temp = np.hstack([demo_temp, train_set_aluminum_hold_pose_norm_full[idx]])
    ipromp_aluminum_hold.add_demonstration(demo_temp)
    # add demonstration of spanner_handover
    demo_temp = np.hstack([train_set_spanner_handover_imu_norm_full[idx], train_set_spanner_handover_emg_norm_full[idx]])
    demo_temp = np.hstack([demo_temp, train_set_spanner_handover_pose_norm_full[idx]])
    ipromp_spanner_handover.add_demonstration(demo_temp)
    # add demonstration of tape_hold
    demo_temp = np.hstack([train_set_tape_hold_imu_norm_full[idx], train_set_tape_hold_emg_norm_full[idx]])
    demo_temp = np.hstack([demo_temp, train_set_tape_hold_pose_norm_full[idx]])
    ipromp_tape_hold.add_demonstration(demo_temp)


# aluminum hold
test_set_temp = np.hstack((train_set_aluminum_hold_imu_norm21, train_set_aluminum_hold_emg_norm21))
test_set = np.hstack((test_set_temp, np.zeros([len_normal, 7])))
##
test_set_temp = np.hstack((train_set_aluminum_hold_imu_norm22, train_set_aluminum_hold_emg_norm22))
test_set = np.hstack((test_set_temp, np.zeros([len_normal, 7])))
##
test_set_temp = np.hstack((train_set_aluminum_hold_imu_norm23, train_set_aluminum_hold_emg_norm23))
test_set = np.hstack((test_set_temp, np.zeros([len_normal, 7])))
##
test_set_temp = np.hstack((train_set_aluminum_hold_imu_norm24, train_set_aluminum_hold_emg_norm24))
test_set = np.hstack((test_set_temp, np.zeros([len_normal, 7])))
##
test_set_temp = np.hstack((train_set_aluminum_hold_imu_norm25, train_set_aluminum_hold_emg_norm25))
test_set = np.hstack((test_set_temp, np.zeros([len_normal, 7])))
##
test_set_temp = np.hstack((train_set_aluminum_hold_imu_norm26, train_set_aluminum_hold_emg_norm26))
test_set = np.hstack((test_set_temp, np.zeros([len_normal, 7])))
##
test_set_temp = np.hstack((train_set_aluminum_hold_imu_norm27, train_set_aluminum_hold_emg_norm27))
test_set = np.hstack((test_set_temp, np.zeros([len_normal, 7])))
##
test_set_temp = np.hstack((train_set_aluminum_hold_imu_norm28, train_set_aluminum_hold_emg_norm28))
test_set = np.hstack((test_set_temp, np.zeros([len_normal, 7])))
# ##
# test_set_temp = np.hstack((train_set_aluminum_hold_imu_norm29, train_set_aluminum_hold_emg_norm29))
# test_set = np.hstack((test_set_temp, np.zeros([len_normal, 7])))
##

# # spanner handover
# test_set_temp = np.hstack((train_set_spanner_handover_imu_norm21, train_set_spanner_handover_emg_norm21))
# test_set = np.hstack((test_set_temp, np.zeros([len_normal, 7])))
# ##
# test_set_temp = np.hstack((train_set_spanner_handover_imu_norm22, train_set_spanner_handover_emg_norm22))
# test_set = np.hstack((test_set_temp, np.zeros([len_normal, 7])))
# ##
# test_set_temp = np.hstack((train_set_spanner_handover_imu_norm23, train_set_spanner_handover_emg_norm23))
# test_set = np.hstack((test_set_temp, np.zeros([len_normal, 7])))
# ##
# test_set_temp = np.hstack((train_set_spanner_handover_imu_norm24, train_set_spanner_handover_emg_norm24))
# test_set = np.hstack((test_set_temp, np.zeros([len_normal, 7])))
# ##
# test_set_temp = np.hstack((train_set_spanner_handover_imu_norm25, train_set_spanner_handover_emg_norm25))
# test_set = np.hstack((test_set_temp, np.zeros([len_normal, 7])))
# ##
# test_set_temp = np.hstack((train_set_spanner_handover_imu_norm26, train_set_spanner_handover_emg_norm26))
# test_set = np.hstack((test_set_temp, np.zeros([len_normal, 7])))
# ##
# test_set_temp = np.hstack((train_set_spanner_handover_imu_norm27, train_set_spanner_handover_emg_norm27))
# test_set = np.hstack((test_set_temp, np.zeros([len_normal, 7])))
# ##
# test_set_temp = np.hstack((train_set_spanner_handover_imu_norm28, train_set_spanner_handover_emg_norm28))
# test_set = np.hstack((test_set_temp, np.zeros([len_normal, 7])))
# ##
# test_set_temp = np.hstack((train_set_spanner_handover_imu_norm29, train_set_spanner_handover_emg_norm29))
# test_set = np.hstack((test_set_temp, np.zeros([len_normal, 7])))
# ##
#
# # tape hold
# test_set_temp = np.hstack((train_set_tape_hold_imu_norm21, train_set_tape_hold_emg_norm21))
# test_set = np.hstack((test_set_temp, np.zeros([len_normal, 7])))
# ##
# test_set_temp = np.hstack((train_set_tape_hold_imu_norm22, train_set_tape_hold_emg_norm22))
# test_set = np.hstack((test_set_temp, np.zeros([len_normal, 7])))
# ##
# test_set_temp = np.hstack((train_set_tape_hold_imu_norm23, train_set_tape_hold_emg_norm23))
# test_set = np.hstack((test_set_temp, np.zeros([len_normal, 7])))
# ##
# test_set_temp = np.hstack((train_set_tape_hold_imu_norm24, train_set_tape_hold_emg_norm24))
# test_set = np.hstack((test_set_temp, np.zeros([len_normal, 7])))
# ##
# test_set_temp = np.hstack((train_set_tape_hold_imu_norm25, train_set_tape_hold_emg_norm25))
# test_set = np.hstack((test_set_temp, np.zeros([len_normal, 7])))
# ##
# test_set_temp = np.hstack((train_set_tape_hold_imu_norm26, train_set_tape_hold_emg_norm26))
# test_set = np.hstack((test_set_temp, np.zeros([len_normal, 7])))
# ##
# test_set_temp = np.hstack((train_set_tape_hold_imu_norm27, train_set_tape_hold_emg_norm27))
# test_set = np.hstack((test_set_temp, np.zeros([len_normal, 7])))
# ##
# test_set_temp = np.hstack((train_set_tape_hold_imu_norm28, train_set_tape_hold_emg_norm28))
# test_set = np.hstack((test_set_temp, np.zeros([len_normal, 7])))
# ##
# test_set_temp = np.hstack((train_set_tape_hold_imu_norm29, train_set_tape_hold_emg_norm29))
# test_set = np.hstack((test_set_temp, np.zeros([len_normal, 7])))
# # ##


# add via point as observation
imu_meansurement_noise_cov = np.eye((4))*1000
emg_meansurement_noise_cov = np.eye((8))*250
pose_meansurement_noise_cov = np.eye((7))*1.0
meansurement_noise_cov_full = scipy.linalg.block_diag(imu_meansurement_noise_cov, emg_meansurement_noise_cov, pose_meansurement_noise_cov)

#################################################################################################
# add via points to update the distribution
#################################################################################################
ipromp_aluminum_hold.add_viapoint(0.00, test_set[0,:], meansurement_noise_cov_full)
ipromp_aluminum_hold.add_viapoint(0.06, test_set[6,:], meansurement_noise_cov_full)
ipromp_aluminum_hold.add_viapoint(0.12, test_set[12,:], meansurement_noise_cov_full)
# ipromp_aluminum_hold.add_viapoint(0.18, test_set[18,:], meansurement_noise_cov_full)
# ipromp_aluminum_hold.add_viapoint(0.24, test_set[24,:], meansurement_noise_cov_full)
# ipromp_aluminum_hold.add_viapoint(0.30, test_set[30,:], meansurement_noise_cov_full)

ipromp_spanner_handover.add_viapoint(0.00, test_set[0,:], meansurement_noise_cov_full)
ipromp_spanner_handover.add_viapoint(0.06, test_set[6,:], meansurement_noise_cov_full)
ipromp_spanner_handover.add_viapoint(0.12, test_set[12,:], meansurement_noise_cov_full)
# ipromp_spanner_handover.add_viapoint(0.18, test_set[18,:], meansurement_noise_cov_full)
# ipromp_spanner_handover.add_viapoint(0.24, test_set[24,:], meansurement_noise_cov_full)
# ipromp_spanner_handover.add_viapoint(0.30, test_set[30,:], meansurement_noise_cov_full)

ipromp_tape_hold.add_viapoint(0.00, test_set[0,:], meansurement_noise_cov_full)
ipromp_tape_hold.add_viapoint(0.06, test_set[6,:], meansurement_noise_cov_full)
ipromp_tape_hold.add_viapoint(0.12, test_set[12,:], meansurement_noise_cov_full)
# ipromp_tape_hold.add_viapoint(0.18, test_set[18,:], meansurement_noise_cov_full)
# ipromp_tape_hold.add_viapoint(0.24, test_set[24,:], meansurement_noise_cov_full)
# ipromp_tape_hold.add_viapoint(0.30, test_set[30,:], meansurement_noise_cov_full)



#################################################################################################
## add observation
#################################################################################################
ipromp_aluminum_hold.add_obs(t=0.00, obsy=test_set[0,:], sigmay=meansurement_noise_cov_full)
ipromp_aluminum_hold.add_obs(t=0.06, obsy=test_set[6,:], sigmay=meansurement_noise_cov_full)
ipromp_aluminum_hold.add_obs(t=0.12, obsy=test_set[12,:], sigmay=meansurement_noise_cov_full)
# ipromp_aluminum_hold.add_obs(t=0.18, obsy=test_set[18,:], sigmay=meansurement_noise_cov_full)
# ipromp_aluminum_hold.add_obs(t=0.24, obsy=test_set[24,:], sigmay=meansurement_noise_cov_full)
# ipromp_aluminum_hold.add_obs(t=0.30, obsy=test_set[30,:], sigmay=meansurement_noise_cov_full)
prob_aluminum_hold = ipromp_aluminum_hold.prob_obs()
print('from obs, the log pro of aluminum_hold is', prob_aluminum_hold)
ipromp_spanner_handover.add_obs(t=0.00, obsy=test_set[0,:], sigmay=meansurement_noise_cov_full)
ipromp_spanner_handover.add_obs(t=0.06, obsy=test_set[6,:], sigmay=meansurement_noise_cov_full)
ipromp_spanner_handover.add_obs(t=0.12, obsy=test_set[12,:], sigmay=meansurement_noise_cov_full)
# ipromp_spanner_handover.add_obs(t=0.18, obsy=test_set[18,:], sigmay=meansurement_noise_cov_full)
# ipromp_spanner_handover.add_obs(t=0.24, obsy=test_set[24,:], sigmay=meansurement_noise_cov_full)
# ipromp_spanner_handover.add_obs(t=0.30, obsy=test_set[30,:], sigmay=meansurement_noise_cov_full)
prob_spanner_handover = ipromp_spanner_handover.prob_obs()
print('from obs, the log pro of spanner_handover is', prob_spanner_handover)
ipromp_tape_hold.add_obs(t=0.00, obsy=test_set[0,:], sigmay=meansurement_noise_cov_full)
ipromp_tape_hold.add_obs(t=0.06, obsy=test_set[6,:], sigmay=meansurement_noise_cov_full)
ipromp_tape_hold.add_obs(t=0.12, obsy=test_set[12,:], sigmay=meansurement_noise_cov_full)
# ipromp_tape_hold.add_obs(t=0.18, obsy=test_set[18,:], sigmay=meansurement_noise_cov_full)
# ipromp_tape_hold.add_obs(t=0.24, obsy=test_set[24,:], sigmay=meansurement_noise_cov_full)
# ipromp_tape_hold.add_obs(t=0.30, obsy=test_set[30,:], sigmay=meansurement_noise_cov_full)
prob_tape_hold = ipromp_tape_hold.prob_obs()
print('from obs, the log pro of tape_hold is', prob_tape_hold)

idx_max_pro = np.argmax([prob_aluminum_hold, prob_spanner_handover, prob_tape_hold])
if idx_max_pro == 0:
    print('the obs comes from aluminum_hold')
elif idx_max_pro == 1:
    print('the obs comes from spanner_handover')
elif idx_max_pro == 2:
    print('the obs comes from tape_hold')



########################################################
# plot everythings
########################################################
###################################################################################################


##################################################################################
##################################################################################
# the data have been resampled successfully as above
##################################################################################
##################################################################################

# #########################################
# # plot raw data
# #########################################
# plt.figure(0)
# for ch_ex in range(8):
#    plt.subplot(421+ch_ex)
#    plt.plot(range(len(train_set_aluminum_hold_emg_00)), train_set_aluminum_hold_emg_00[:,ch_ex])
#    plt.plot(range(len(train_set_aluminum_hold_emg_01)), train_set_aluminum_hold_emg_01[:,ch_ex])
#    plt.plot(range(len(train_set_aluminum_hold_emg_02)), train_set_aluminum_hold_emg_02[:,ch_ex])
#    plt.plot(range(len(train_set_aluminum_hold_emg_03)), train_set_aluminum_hold_emg_03[:,ch_ex])
#    plt.plot(range(len(train_set_aluminum_hold_emg_04)), train_set_aluminum_hold_emg_04[:,ch_ex])
#    plt.plot(range(len(train_set_aluminum_hold_emg_05)), train_set_aluminum_hold_emg_05[:,ch_ex])
#    plt.plot(range(len(train_set_aluminum_hold_emg_06)), train_set_aluminum_hold_emg_06[:,ch_ex])
#    plt.plot(range(len(train_set_aluminum_hold_emg_07)), train_set_aluminum_hold_emg_07[:,ch_ex])
#    plt.plot(range(len(train_set_aluminum_hold_emg_08)), train_set_aluminum_hold_emg_08[:,ch_ex])
#    plt.plot(range(len(train_set_aluminum_hold_emg_09)), train_set_aluminum_hold_emg_09[:,ch_ex])
#    plt.plot(range(len(train_set_aluminum_hold_emg_10)), train_set_aluminum_hold_emg_10[:,ch_ex])
#    plt.plot(range(len(train_set_aluminum_hold_emg_11)), train_set_aluminum_hold_emg_11[:,ch_ex])
#    plt.plot(range(len(train_set_aluminum_hold_emg_12)), train_set_aluminum_hold_emg_12[:,ch_ex])
#    plt.plot(range(len(train_set_aluminum_hold_emg_13)), train_set_aluminum_hold_emg_13[:,ch_ex])
#    plt.plot(range(len(train_set_aluminum_hold_emg_14)), train_set_aluminum_hold_emg_14[:,ch_ex])
#    plt.plot(range(len(train_set_aluminum_hold_emg_15)), train_set_aluminum_hold_emg_15[:,ch_ex])
#    plt.plot(range(len(train_set_aluminum_hold_emg_16)), train_set_aluminum_hold_emg_16[:,ch_ex])
#    plt.plot(range(len(train_set_aluminum_hold_emg_17)), train_set_aluminum_hold_emg_17[:,ch_ex])
#    plt.plot(range(len(train_set_aluminum_hold_emg_18)), train_set_aluminum_hold_emg_18[:,ch_ex])
#    plt.plot(range(len(train_set_aluminum_hold_emg_19)), train_set_aluminum_hold_emg_19[:,ch_ex])
# plt.figure(1)
# for ch_ex in range(8):
#    plt.subplot(421+ch_ex)
#    plt.plot(range(len(train_set_spanner_handover_emg_00)), train_set_spanner_handover_emg_00[:,ch_ex])
#    plt.plot(range(len(train_set_spanner_handover_emg_01)), train_set_spanner_handover_emg_01[:,ch_ex])
#    plt.plot(range(len(train_set_spanner_handover_emg_02)), train_set_spanner_handover_emg_02[:,ch_ex])
#    plt.plot(range(len(train_set_spanner_handover_emg_03)), train_set_spanner_handover_emg_03[:,ch_ex])
#    plt.plot(range(len(train_set_spanner_handover_emg_04)), train_set_spanner_handover_emg_04[:,ch_ex])
#    plt.plot(range(len(train_set_spanner_handover_emg_05)), train_set_spanner_handover_emg_05[:,ch_ex])
#    plt.plot(range(len(train_set_spanner_handover_emg_06)), train_set_spanner_handover_emg_06[:,ch_ex])
#    plt.plot(range(len(train_set_spanner_handover_emg_07)), train_set_spanner_handover_emg_07[:,ch_ex])
#    plt.plot(range(len(train_set_spanner_handover_emg_08)), train_set_spanner_handover_emg_08[:,ch_ex])
#    plt.plot(range(len(train_set_spanner_handover_emg_09)), train_set_spanner_handover_emg_09[:,ch_ex])
#    plt.plot(range(len(train_set_spanner_handover_emg_10)), train_set_spanner_handover_emg_10[:,ch_ex])
#    plt.plot(range(len(train_set_spanner_handover_emg_11)), train_set_spanner_handover_emg_11[:,ch_ex])
#    plt.plot(range(len(train_set_spanner_handover_emg_12)), train_set_spanner_handover_emg_12[:,ch_ex])
#    plt.plot(range(len(train_set_spanner_handover_emg_13)), train_set_spanner_handover_emg_13[:,ch_ex])
#    plt.plot(range(len(train_set_spanner_handover_emg_14)), train_set_spanner_handover_emg_14[:,ch_ex])
#    plt.plot(range(len(train_set_spanner_handover_emg_15)), train_set_spanner_handover_emg_15[:,ch_ex])
#    plt.plot(range(len(train_set_spanner_handover_emg_16)), train_set_spanner_handover_emg_16[:,ch_ex])
#    plt.plot(range(len(train_set_spanner_handover_emg_17)), train_set_spanner_handover_emg_17[:,ch_ex])
#    plt.plot(range(len(train_set_spanner_handover_emg_18)), train_set_spanner_handover_emg_18[:,ch_ex])
#    plt.plot(range(len(train_set_spanner_handover_emg_19)), train_set_spanner_handover_emg_19[:,ch_ex])
# plt.figure(2)
# for ch_ex in range(8):
#    plt.subplot(421+ch_ex)
#    plt.plot(range(len(train_set_tape_hold_emg_00)), train_set_tape_hold_emg_00[:,ch_ex])
#    plt.plot(range(len(train_set_tape_hold_emg_01)), train_set_tape_hold_emg_01[:,ch_ex])
#    plt.plot(range(len(train_set_tape_hold_emg_02)), train_set_tape_hold_emg_02[:,ch_ex])
#    plt.plot(range(len(train_set_tape_hold_emg_03)), train_set_tape_hold_emg_03[:,ch_ex])
#    plt.plot(range(len(train_set_tape_hold_emg_04)), train_set_tape_hold_emg_04[:,ch_ex])
#    plt.plot(range(len(train_set_tape_hold_emg_05)), train_set_tape_hold_emg_05[:,ch_ex])
#    plt.plot(range(len(train_set_tape_hold_emg_06)), train_set_tape_hold_emg_06[:,ch_ex])
#    plt.plot(range(len(train_set_tape_hold_emg_07)), train_set_tape_hold_emg_07[:,ch_ex])
#    plt.plot(range(len(train_set_tape_hold_emg_08)), train_set_tape_hold_emg_08[:,ch_ex])
#    plt.plot(range(len(train_set_tape_hold_emg_09)), train_set_tape_hold_emg_09[:,ch_ex])
#    plt.plot(range(len(train_set_tape_hold_emg_10)), train_set_tape_hold_emg_10[:,ch_ex])
#    plt.plot(range(len(train_set_tape_hold_emg_11)), train_set_tape_hold_emg_11[:,ch_ex])
#    plt.plot(range(len(train_set_tape_hold_emg_12)), train_set_tape_hold_emg_12[:,ch_ex])
#    plt.plot(range(len(train_set_tape_hold_emg_13)), train_set_tape_hold_emg_13[:,ch_ex])
#    plt.plot(range(len(train_set_tape_hold_emg_14)), train_set_tape_hold_emg_14[:,ch_ex])
#    plt.plot(range(len(train_set_tape_hold_emg_15)), train_set_tape_hold_emg_15[:,ch_ex])
#    plt.plot(range(len(train_set_tape_hold_emg_16)), train_set_tape_hold_emg_16[:,ch_ex])
#    plt.plot(range(len(train_set_tape_hold_emg_17)), train_set_tape_hold_emg_17[:,ch_ex])
#    plt.plot(range(len(train_set_tape_hold_emg_18)), train_set_tape_hold_emg_18[:,ch_ex])
#    plt.plot(range(len(train_set_tape_hold_emg_19)), train_set_tape_hold_emg_19[:,ch_ex])
# ###########################################################################################
# plt.figure(3)
# for ch_ex in range(4):
#    plt.subplot(411+ch_ex)
#    plt.plot(range(len(train_set_aluminum_hold_imu_00)), train_set_aluminum_hold_imu_00[:,ch_ex])
#    plt.plot(range(len(train_set_aluminum_hold_imu_01)), train_set_aluminum_hold_imu_01[:,ch_ex])
#    plt.plot(range(len(train_set_aluminum_hold_imu_02)), train_set_aluminum_hold_imu_02[:,ch_ex])
#    plt.plot(range(len(train_set_aluminum_hold_imu_03)), train_set_aluminum_hold_imu_03[:,ch_ex])
#    plt.plot(range(len(train_set_aluminum_hold_imu_04)), train_set_aluminum_hold_imu_04[:,ch_ex])
#    plt.plot(range(len(train_set_aluminum_hold_imu_05)), train_set_aluminum_hold_imu_05[:,ch_ex])
#    plt.plot(range(len(train_set_aluminum_hold_imu_06)), train_set_aluminum_hold_imu_06[:,ch_ex])
#    plt.plot(range(len(train_set_aluminum_hold_imu_07)), train_set_aluminum_hold_imu_07[:,ch_ex])
#    plt.plot(range(len(train_set_aluminum_hold_imu_08)), train_set_aluminum_hold_imu_08[:,ch_ex])
#    plt.plot(range(len(train_set_aluminum_hold_imu_09)), train_set_aluminum_hold_imu_09[:,ch_ex])
#    plt.plot(range(len(train_set_aluminum_hold_imu_10)), train_set_aluminum_hold_imu_10[:,ch_ex])
#    plt.plot(range(len(train_set_aluminum_hold_imu_11)), train_set_aluminum_hold_imu_11[:,ch_ex])
#    plt.plot(range(len(train_set_aluminum_hold_imu_12)), train_set_aluminum_hold_imu_12[:,ch_ex])
#    plt.plot(range(len(train_set_aluminum_hold_imu_13)), train_set_aluminum_hold_imu_13[:,ch_ex])
#    plt.plot(range(len(train_set_aluminum_hold_imu_14)), train_set_aluminum_hold_imu_14[:,ch_ex])
#    plt.plot(range(len(train_set_aluminum_hold_imu_15)), train_set_aluminum_hold_imu_15[:,ch_ex])
#    plt.plot(range(len(train_set_aluminum_hold_imu_16)), train_set_aluminum_hold_imu_16[:,ch_ex])
#    plt.plot(range(len(train_set_aluminum_hold_imu_17)), train_set_aluminum_hold_imu_17[:,ch_ex])
#    plt.plot(range(len(train_set_aluminum_hold_imu_18)), train_set_aluminum_hold_imu_18[:,ch_ex])
#    plt.plot(range(len(train_set_aluminum_hold_imu_19)), train_set_aluminum_hold_imu_19[:,ch_ex])
# plt.figure(4)
# for ch_ex in range(4):
#    plt.subplot(411+ch_ex)
#    plt.plot(range(len(train_set_spanner_handover_imu_00)), train_set_spanner_handover_imu_00[:,ch_ex])
#    plt.plot(range(len(train_set_spanner_handover_imu_01)), train_set_spanner_handover_imu_01[:,ch_ex])
#    plt.plot(range(len(train_set_spanner_handover_imu_02)), train_set_spanner_handover_imu_02[:,ch_ex])
#    plt.plot(range(len(train_set_spanner_handover_imu_03)), train_set_spanner_handover_imu_03[:,ch_ex])
#    plt.plot(range(len(train_set_spanner_handover_imu_04)), train_set_spanner_handover_imu_04[:,ch_ex])
#    plt.plot(range(len(train_set_spanner_handover_imu_05)), train_set_spanner_handover_imu_05[:,ch_ex])
#    plt.plot(range(len(train_set_spanner_handover_imu_06)), train_set_spanner_handover_imu_06[:,ch_ex])
#    plt.plot(range(len(train_set_spanner_handover_imu_07)), train_set_spanner_handover_imu_07[:,ch_ex])
#    plt.plot(range(len(train_set_spanner_handover_imu_08)), train_set_spanner_handover_imu_08[:,ch_ex])
#    plt.plot(range(len(train_set_spanner_handover_imu_09)), train_set_spanner_handover_imu_09[:,ch_ex])
#    plt.plot(range(len(train_set_spanner_handover_imu_10)), train_set_spanner_handover_imu_10[:,ch_ex])
#    plt.plot(range(len(train_set_spanner_handover_imu_11)), train_set_spanner_handover_imu_11[:,ch_ex])
#    plt.plot(range(len(train_set_spanner_handover_imu_12)), train_set_spanner_handover_imu_12[:,ch_ex])
#    plt.plot(range(len(train_set_spanner_handover_imu_13)), train_set_spanner_handover_imu_13[:,ch_ex])
#    plt.plot(range(len(train_set_spanner_handover_imu_14)), train_set_spanner_handover_imu_14[:,ch_ex])
#    plt.plot(range(len(train_set_spanner_handover_imu_15)), train_set_spanner_handover_imu_15[:,ch_ex])
#    plt.plot(range(len(train_set_spanner_handover_imu_16)), train_set_spanner_handover_imu_16[:,ch_ex])
#    plt.plot(range(len(train_set_spanner_handover_imu_17)), train_set_spanner_handover_imu_17[:,ch_ex])
#    plt.plot(range(len(train_set_spanner_handover_imu_18)), train_set_spanner_handover_imu_18[:,ch_ex])
#    plt.plot(range(len(train_set_spanner_handover_imu_19)), train_set_spanner_handover_imu_19[:,ch_ex])
# plt.figure(5)
# for ch_ex in range(4):
#    plt.subplot(411+ch_ex)
#    plt.plot(range(len(train_set_tape_hold_imu_00)), train_set_tape_hold_imu_00[:,ch_ex])
#    plt.plot(range(len(train_set_tape_hold_imu_01)), train_set_tape_hold_imu_01[:,ch_ex])
#    plt.plot(range(len(train_set_tape_hold_imu_02)), train_set_tape_hold_imu_02[:,ch_ex])
#    plt.plot(range(len(train_set_tape_hold_imu_03)), train_set_tape_hold_imu_03[:,ch_ex])
#    plt.plot(range(len(train_set_tape_hold_imu_04)), train_set_tape_hold_imu_04[:,ch_ex])
#    plt.plot(range(len(train_set_tape_hold_imu_05)), train_set_tape_hold_imu_05[:,ch_ex])
#    plt.plot(range(len(train_set_tape_hold_imu_06)), train_set_tape_hold_imu_06[:,ch_ex])
#    plt.plot(range(len(train_set_tape_hold_imu_07)), train_set_tape_hold_imu_07[:,ch_ex])
#    plt.plot(range(len(train_set_tape_hold_imu_08)), train_set_tape_hold_imu_08[:,ch_ex])
#    plt.plot(range(len(train_set_tape_hold_imu_09)), train_set_tape_hold_imu_09[:,ch_ex])
#    plt.plot(range(len(train_set_tape_hold_imu_10)), train_set_tape_hold_imu_10[:,ch_ex])
#    plt.plot(range(len(train_set_tape_hold_imu_11)), train_set_tape_hold_imu_11[:,ch_ex])
#    plt.plot(range(len(train_set_tape_hold_imu_12)), train_set_tape_hold_imu_12[:,ch_ex])
#    plt.plot(range(len(train_set_tape_hold_imu_13)), train_set_tape_hold_imu_13[:,ch_ex])
#    plt.plot(range(len(train_set_tape_hold_imu_14)), train_set_tape_hold_imu_14[:,ch_ex])
#    plt.plot(range(len(train_set_tape_hold_imu_15)), train_set_tape_hold_imu_15[:,ch_ex])
#    plt.plot(range(len(train_set_tape_hold_imu_16)), train_set_tape_hold_imu_16[:,ch_ex])
#    plt.plot(range(len(train_set_tape_hold_imu_17)), train_set_tape_hold_imu_17[:,ch_ex])
#    plt.plot(range(len(train_set_tape_hold_imu_18)), train_set_tape_hold_imu_18[:,ch_ex])
#    plt.plot(range(len(train_set_tape_hold_imu_19)), train_set_tape_hold_imu_19[:,ch_ex])
# ###########################################################################################
# plt.figure(6)
# for ch_ex in range(7):
#    plt.subplot(711+ch_ex)
#    plt.plot(range(len(train_set_aluminum_hold_pose_00)), train_set_aluminum_hold_pose_00[:,ch_ex])
#    plt.plot(range(len(train_set_aluminum_hold_pose_01)), train_set_aluminum_hold_pose_01[:,ch_ex])
#    plt.plot(range(len(train_set_aluminum_hold_pose_02)), train_set_aluminum_hold_pose_02[:,ch_ex])
#    plt.plot(range(len(train_set_aluminum_hold_pose_03)), train_set_aluminum_hold_pose_03[:,ch_ex])
#    plt.plot(range(len(train_set_aluminum_hold_pose_04)), train_set_aluminum_hold_pose_04[:,ch_ex])
#    plt.plot(range(len(train_set_aluminum_hold_pose_05)), train_set_aluminum_hold_pose_05[:,ch_ex])
#    plt.plot(range(len(train_set_aluminum_hold_pose_06)), train_set_aluminum_hold_pose_06[:,ch_ex])
#    plt.plot(range(len(train_set_aluminum_hold_pose_07)), train_set_aluminum_hold_pose_07[:,ch_ex])
#    plt.plot(range(len(train_set_aluminum_hold_pose_08)), train_set_aluminum_hold_pose_08[:,ch_ex])
#    plt.plot(range(len(train_set_aluminum_hold_pose_09)), train_set_aluminum_hold_pose_09[:,ch_ex])
#    plt.plot(range(len(train_set_aluminum_hold_pose_10)), train_set_aluminum_hold_pose_10[:,ch_ex])
#    plt.plot(range(len(train_set_aluminum_hold_pose_11)), train_set_aluminum_hold_pose_11[:,ch_ex])
#    plt.plot(range(len(train_set_aluminum_hold_pose_12)), train_set_aluminum_hold_pose_12[:,ch_ex])
#    plt.plot(range(len(train_set_aluminum_hold_pose_13)), train_set_aluminum_hold_pose_13[:,ch_ex])
#    plt.plot(range(len(train_set_aluminum_hold_pose_14)), train_set_aluminum_hold_pose_14[:,ch_ex])
#    plt.plot(range(len(train_set_aluminum_hold_pose_15)), train_set_aluminum_hold_pose_15[:,ch_ex])
#    plt.plot(range(len(train_set_aluminum_hold_pose_16)), train_set_aluminum_hold_pose_16[:,ch_ex])
#    plt.plot(range(len(train_set_aluminum_hold_pose_17)), train_set_aluminum_hold_pose_17[:,ch_ex])
#    plt.plot(range(len(train_set_aluminum_hold_pose_18)), train_set_aluminum_hold_pose_18[:,ch_ex])
#    plt.plot(range(len(train_set_aluminum_hold_pose_19)), train_set_aluminum_hold_pose_19[:,ch_ex])
# plt.figure(7)
# for ch_ex in range(7):
#    plt.subplot(711+ch_ex)
#    plt.plot(range(len(train_set_spanner_handover_pose_00)), train_set_spanner_handover_pose_00[:,ch_ex])
#    plt.plot(range(len(train_set_spanner_handover_pose_01)), train_set_spanner_handover_pose_01[:,ch_ex])
#    plt.plot(range(len(train_set_spanner_handover_pose_02)), train_set_spanner_handover_pose_02[:,ch_ex])
#    plt.plot(range(len(train_set_spanner_handover_pose_03)), train_set_spanner_handover_pose_03[:,ch_ex])
#    plt.plot(range(len(train_set_spanner_handover_pose_04)), train_set_spanner_handover_pose_04[:,ch_ex])
#    plt.plot(range(len(train_set_spanner_handover_pose_05)), train_set_spanner_handover_pose_05[:,ch_ex])
#    plt.plot(range(len(train_set_spanner_handover_pose_06)), train_set_spanner_handover_pose_06[:,ch_ex])
#    plt.plot(range(len(train_set_spanner_handover_pose_07)), train_set_spanner_handover_pose_07[:,ch_ex])
#    plt.plot(range(len(train_set_spanner_handover_pose_08)), train_set_spanner_handover_pose_08[:,ch_ex])
#    plt.plot(range(len(train_set_spanner_handover_pose_09)), train_set_spanner_handover_pose_09[:,ch_ex])
#    plt.plot(range(len(train_set_spanner_handover_pose_10)), train_set_spanner_handover_pose_10[:,ch_ex])
#    plt.plot(range(len(train_set_spanner_handover_pose_11)), train_set_spanner_handover_pose_11[:,ch_ex])
#    plt.plot(range(len(train_set_spanner_handover_pose_12)), train_set_spanner_handover_pose_12[:,ch_ex])
#    plt.plot(range(len(train_set_spanner_handover_pose_13)), train_set_spanner_handover_pose_13[:,ch_ex])
#    plt.plot(range(len(train_set_spanner_handover_pose_14)), train_set_spanner_handover_pose_14[:,ch_ex])
#    plt.plot(range(len(train_set_spanner_handover_pose_15)), train_set_spanner_handover_pose_15[:,ch_ex])
#    plt.plot(range(len(train_set_spanner_handover_pose_16)), train_set_spanner_handover_pose_16[:,ch_ex])
#    plt.plot(range(len(train_set_spanner_handover_pose_17)), train_set_spanner_handover_pose_17[:,ch_ex])
#    plt.plot(range(len(train_set_spanner_handover_pose_18)), train_set_spanner_handover_pose_18[:,ch_ex])
#    plt.plot(range(len(train_set_spanner_handover_pose_19)), train_set_spanner_handover_pose_19[:,ch_ex])
# plt.figure(8)
# for ch_ex in range(7):
#    plt.subplot(711+ch_ex)
#    plt.plot(range(len(train_set_tape_hold_pose_00)), train_set_tape_hold_pose_00[:,ch_ex])
#    plt.plot(range(len(train_set_tape_hold_pose_01)), train_set_tape_hold_pose_01[:,ch_ex])
#    plt.plot(range(len(train_set_tape_hold_pose_02)), train_set_tape_hold_pose_02[:,ch_ex])
#    plt.plot(range(len(train_set_tape_hold_pose_03)), train_set_tape_hold_pose_03[:,ch_ex])
#    plt.plot(range(len(train_set_tape_hold_pose_04)), train_set_tape_hold_pose_04[:,ch_ex])
#    plt.plot(range(len(train_set_tape_hold_pose_05)), train_set_tape_hold_pose_05[:,ch_ex])
#    plt.plot(range(len(train_set_tape_hold_pose_06)), train_set_tape_hold_pose_06[:,ch_ex])
#    plt.plot(range(len(train_set_tape_hold_pose_07)), train_set_tape_hold_pose_07[:,ch_ex])
#    plt.plot(range(len(train_set_tape_hold_pose_08)), train_set_tape_hold_pose_08[:,ch_ex])
#    plt.plot(range(len(train_set_tape_hold_pose_09)), train_set_tape_hold_pose_09[:,ch_ex])
#    plt.plot(range(len(train_set_tape_hold_pose_10)), train_set_tape_hold_pose_10[:,ch_ex])
#    plt.plot(range(len(train_set_tape_hold_pose_11)), train_set_tape_hold_pose_11[:,ch_ex])
#    plt.plot(range(len(train_set_tape_hold_pose_12)), train_set_tape_hold_pose_12[:,ch_ex])
#    plt.plot(range(len(train_set_tape_hold_pose_13)), train_set_tape_hold_pose_13[:,ch_ex])
#    plt.plot(range(len(train_set_tape_hold_pose_14)), train_set_tape_hold_pose_14[:,ch_ex])
#    plt.plot(range(len(train_set_tape_hold_pose_15)), train_set_tape_hold_pose_15[:,ch_ex])
#    plt.plot(range(len(train_set_tape_hold_pose_16)), train_set_tape_hold_pose_16[:,ch_ex])
#    plt.plot(range(len(train_set_tape_hold_pose_17)), train_set_tape_hold_pose_17[:,ch_ex])
#    plt.plot(range(len(train_set_tape_hold_pose_18)), train_set_tape_hold_pose_18[:,ch_ex])
#    plt.plot(range(len(train_set_tape_hold_pose_19)), train_set_tape_hold_pose_19[:,ch_ex])

# #########################################
# # plot EMG norm data
# #########################################
# # plot the norm emg data of aluminum_hold
# plt.figure(10)
# for ch_ex in range(8):
#    plt.subplot(421+ch_ex)
#    plt.plot(np.arange(0,1.01,0.01), train_set_aluminum_hold_emg_norm00[:,ch_ex])
#    plt.plot(np.arange(0,1.01,0.01), train_set_aluminum_hold_emg_norm01[:,ch_ex])
#    plt.plot(np.arange(0,1.01,0.01), train_set_aluminum_hold_emg_norm02[:,ch_ex])
#    plt.plot(np.arange(0,1.01,0.01), train_set_aluminum_hold_emg_norm03[:,ch_ex])
#    plt.plot(np.arange(0,1.01,0.01), train_set_aluminum_hold_emg_norm04[:,ch_ex])
#    plt.plot(np.arange(0,1.01,0.01), train_set_aluminum_hold_emg_norm05[:,ch_ex])
#    plt.plot(np.arange(0,1.01,0.01), train_set_aluminum_hold_emg_norm06[:,ch_ex])
#    plt.plot(np.arange(0,1.01,0.01), train_set_aluminum_hold_emg_norm07[:,ch_ex])
#    plt.plot(np.arange(0,1.01,0.01), train_set_aluminum_hold_emg_norm08[:,ch_ex])
#    plt.plot(np.arange(0,1.01,0.01), train_set_aluminum_hold_emg_norm09[:,ch_ex])
#    plt.plot(np.arange(0,1.01,0.01), train_set_aluminum_hold_emg_norm10[:,ch_ex])
#    plt.plot(np.arange(0,1.01,0.01), train_set_aluminum_hold_emg_norm11[:,ch_ex])
#    plt.plot(np.arange(0,1.01,0.01), train_set_aluminum_hold_emg_norm12[:,ch_ex])
#    plt.plot(np.arange(0,1.01,0.01), train_set_aluminum_hold_emg_norm13[:,ch_ex])
#    plt.plot(np.arange(0,1.01,0.01), train_set_aluminum_hold_emg_norm14[:,ch_ex])
#    plt.plot(np.arange(0,1.01,0.01), train_set_aluminum_hold_emg_norm15[:,ch_ex])
#    plt.plot(np.arange(0,1.01,0.01), train_set_aluminum_hold_emg_norm16[:,ch_ex])
#    plt.plot(np.arange(0,1.01,0.01), train_set_aluminum_hold_emg_norm17[:,ch_ex])
#    plt.plot(np.arange(0,1.01,0.01), train_set_aluminum_hold_emg_norm18[:,ch_ex])
#    plt.plot(np.arange(0,1.01,0.01), train_set_aluminum_hold_emg_norm19[:,ch_ex])
# # plot the norm emg data of spanner_handover
# plt.figure(11)
# for ch_ex in range(8):
#    plt.subplot(421+ch_ex)
#    plt.plot(np.arange(0,1.01,0.01), train_set_spanner_handover_emg_norm00[:,ch_ex])
#    plt.plot(np.arange(0,1.01,0.01), train_set_spanner_handover_emg_norm01[:,ch_ex])
#    plt.plot(np.arange(0,1.01,0.01), train_set_spanner_handover_emg_norm02[:,ch_ex])
#    plt.plot(np.arange(0,1.01,0.01), train_set_spanner_handover_emg_norm03[:,ch_ex])
#    plt.plot(np.arange(0,1.01,0.01), train_set_spanner_handover_emg_norm04[:,ch_ex])
#    plt.plot(np.arange(0,1.01,0.01), train_set_spanner_handover_emg_norm05[:,ch_ex])
#    plt.plot(np.arange(0,1.01,0.01), train_set_spanner_handover_emg_norm06[:,ch_ex])
#    plt.plot(np.arange(0,1.01,0.01), train_set_spanner_handover_emg_norm07[:,ch_ex])
#    plt.plot(np.arange(0,1.01,0.01), train_set_spanner_handover_emg_norm08[:,ch_ex])
#    plt.plot(np.arange(0,1.01,0.01), train_set_spanner_handover_emg_norm09[:,ch_ex])
#    plt.plot(np.arange(0,1.01,0.01), train_set_spanner_handover_emg_norm10[:,ch_ex])
#    plt.plot(np.arange(0,1.01,0.01), train_set_spanner_handover_emg_norm11[:,ch_ex])
#    plt.plot(np.arange(0,1.01,0.01), train_set_spanner_handover_emg_norm12[:,ch_ex])
#    plt.plot(np.arange(0,1.01,0.01), train_set_spanner_handover_emg_norm13[:,ch_ex])
#    plt.plot(np.arange(0,1.01,0.01), train_set_spanner_handover_emg_norm14[:,ch_ex])
#    plt.plot(np.arange(0,1.01,0.01), train_set_spanner_handover_emg_norm15[:,ch_ex])
#    plt.plot(np.arange(0,1.01,0.01), train_set_spanner_handover_emg_norm16[:,ch_ex])
#    plt.plot(np.arange(0,1.01,0.01), train_set_spanner_handover_emg_norm17[:,ch_ex])
#    plt.plot(np.arange(0,1.01,0.01), train_set_spanner_handover_emg_norm18[:,ch_ex])
#    plt.plot(np.arange(0,1.01,0.01), train_set_spanner_handover_emg_norm19[:,ch_ex])
# # plot the norm emg data of tape_hold
# plt.figure(12)
# for ch_ex in range(8):
#    plt.subplot(421+ch_ex)
#    plt.plot(np.arange(0,1.01,0.01), train_set_tape_hold_emg_norm00[:,ch_ex])
#    plt.plot(np.arange(0,1.01,0.01), train_set_tape_hold_emg_norm01[:,ch_ex])
#    plt.plot(np.arange(0,1.01,0.01), train_set_tape_hold_emg_norm02[:,ch_ex])
#    plt.plot(np.arange(0,1.01,0.01), train_set_tape_hold_emg_norm03[:,ch_ex])
#    plt.plot(np.arange(0,1.01,0.01), train_set_tape_hold_emg_norm04[:,ch_ex])
#    plt.plot(np.arange(0,1.01,0.01), train_set_tape_hold_emg_norm05[:,ch_ex])
#    plt.plot(np.arange(0,1.01,0.01), train_set_tape_hold_emg_norm06[:,ch_ex])
#    plt.plot(np.arange(0,1.01,0.01), train_set_tape_hold_emg_norm07[:,ch_ex])
#    plt.plot(np.arange(0,1.01,0.01), train_set_tape_hold_emg_norm08[:,ch_ex])
#    plt.plot(np.arange(0,1.01,0.01), train_set_tape_hold_emg_norm09[:,ch_ex])
#    plt.plot(np.arange(0,1.01,0.01), train_set_tape_hold_emg_norm10[:,ch_ex])
#    plt.plot(np.arange(0,1.01,0.01), train_set_tape_hold_emg_norm11[:,ch_ex])
#    plt.plot(np.arange(0,1.01,0.01), train_set_tape_hold_emg_norm12[:,ch_ex])
#    plt.plot(np.arange(0,1.01,0.01), train_set_tape_hold_emg_norm13[:,ch_ex])
#    plt.plot(np.arange(0,1.01,0.01), train_set_tape_hold_emg_norm14[:,ch_ex])
#    plt.plot(np.arange(0,1.01,0.01), train_set_tape_hold_emg_norm15[:,ch_ex])
#    plt.plot(np.arange(0,1.01,0.01), train_set_tape_hold_emg_norm16[:,ch_ex])
#    plt.plot(np.arange(0,1.01,0.01), train_set_tape_hold_emg_norm17[:,ch_ex])
#    plt.plot(np.arange(0,1.01,0.01), train_set_tape_hold_emg_norm18[:,ch_ex])
#    plt.plot(np.arange(0,1.01,0.01), train_set_tape_hold_emg_norm19[:,ch_ex])
#
# #########################################
# # plot IMU norm data
# #########################################
# # plot the norm imu data of aluminum_hold
# plt.figure(13)
# for ch_ex in range(4):
#    plt.subplot(411 + ch_ex)
#    plt.plot(np.arange(0, 1.01, 0.01), train_set_aluminum_hold_imu_norm00[:, ch_ex])
#    plt.plot(np.arange(0, 1.01, 0.01), train_set_aluminum_hold_imu_norm01[:, ch_ex])
#    plt.plot(np.arange(0, 1.01, 0.01), train_set_aluminum_hold_imu_norm02[:, ch_ex])
#    plt.plot(np.arange(0, 1.01, 0.01), train_set_aluminum_hold_imu_norm03[:, ch_ex])
#    plt.plot(np.arange(0, 1.01, 0.01), train_set_aluminum_hold_imu_norm04[:, ch_ex])
#    plt.plot(np.arange(0, 1.01, 0.01), train_set_aluminum_hold_imu_norm05[:, ch_ex])
#    plt.plot(np.arange(0, 1.01, 0.01), train_set_aluminum_hold_imu_norm06[:, ch_ex])
#    plt.plot(np.arange(0, 1.01, 0.01), train_set_aluminum_hold_imu_norm07[:, ch_ex])
#    plt.plot(np.arange(0, 1.01, 0.01), train_set_aluminum_hold_imu_norm08[:, ch_ex])
#    plt.plot(np.arange(0, 1.01, 0.01), train_set_aluminum_hold_imu_norm09[:, ch_ex])
#    plt.plot(np.arange(0, 1.01, 0.01), train_set_aluminum_hold_imu_norm10[:, ch_ex])
#    plt.plot(np.arange(0, 1.01, 0.01), train_set_aluminum_hold_imu_norm11[:, ch_ex])
#    plt.plot(np.arange(0, 1.01, 0.01), train_set_aluminum_hold_imu_norm12[:, ch_ex])
#    plt.plot(np.arange(0, 1.01, 0.01), train_set_aluminum_hold_imu_norm13[:, ch_ex])
#    plt.plot(np.arange(0, 1.01, 0.01), train_set_aluminum_hold_imu_norm14[:, ch_ex])
#    plt.plot(np.arange(0, 1.01, 0.01), train_set_aluminum_hold_imu_norm15[:, ch_ex])
#    plt.plot(np.arange(0, 1.01, 0.01), train_set_aluminum_hold_imu_norm16[:, ch_ex])
#    plt.plot(np.arange(0, 1.01, 0.01), train_set_aluminum_hold_imu_norm17[:, ch_ex])
#    plt.plot(np.arange(0, 1.01, 0.01), train_set_aluminum_hold_imu_norm18[:, ch_ex])
#    plt.plot(np.arange(0, 1.01, 0.01), train_set_aluminum_hold_imu_norm19[:, ch_ex])
# # plot the norm imu data of spanner_handover
# plt.figure(14)
# for ch_ex in range(4):
#    plt.subplot(411 + ch_ex)
#    plt.plot(np.arange(0, 1.01, 0.01), train_set_spanner_handover_imu_norm00[:, ch_ex])
#    plt.plot(np.arange(0, 1.01, 0.01), train_set_spanner_handover_imu_norm01[:, ch_ex])
#    plt.plot(np.arange(0, 1.01, 0.01), train_set_spanner_handover_imu_norm02[:, ch_ex])
#    plt.plot(np.arange(0, 1.01, 0.01), train_set_spanner_handover_imu_norm03[:, ch_ex])
#    plt.plot(np.arange(0, 1.01, 0.01), train_set_spanner_handover_imu_norm04[:, ch_ex])
#    plt.plot(np.arange(0, 1.01, 0.01), train_set_spanner_handover_imu_norm05[:, ch_ex])
#    plt.plot(np.arange(0, 1.01, 0.01), train_set_spanner_handover_imu_norm06[:, ch_ex])
#    plt.plot(np.arange(0, 1.01, 0.01), train_set_spanner_handover_imu_norm07[:, ch_ex])
#    plt.plot(np.arange(0, 1.01, 0.01), train_set_spanner_handover_imu_norm08[:, ch_ex])
#    plt.plot(np.arange(0, 1.01, 0.01), train_set_spanner_handover_imu_norm09[:, ch_ex])
#    plt.plot(np.arange(0, 1.01, 0.01), train_set_spanner_handover_imu_norm10[:, ch_ex])
#    plt.plot(np.arange(0, 1.01, 0.01), train_set_spanner_handover_imu_norm11[:, ch_ex])
#    plt.plot(np.arange(0, 1.01, 0.01), train_set_spanner_handover_imu_norm12[:, ch_ex])
#    plt.plot(np.arange(0, 1.01, 0.01), train_set_spanner_handover_imu_norm13[:, ch_ex])
#    plt.plot(np.arange(0, 1.01, 0.01), train_set_spanner_handover_imu_norm14[:, ch_ex])
#    plt.plot(np.arange(0, 1.01, 0.01), train_set_spanner_handover_imu_norm15[:, ch_ex])
#    plt.plot(np.arange(0, 1.01, 0.01), train_set_spanner_handover_imu_norm16[:, ch_ex])
#    plt.plot(np.arange(0, 1.01, 0.01), train_set_spanner_handover_imu_norm17[:, ch_ex])
#    plt.plot(np.arange(0, 1.01, 0.01), train_set_spanner_handover_imu_norm18[:, ch_ex])
#    plt.plot(np.arange(0, 1.01, 0.01), train_set_spanner_handover_imu_norm19[:, ch_ex])
# # plot the norm imu data of tape_hold
# plt.figure(15)
# for ch_ex in range(4):
#    plt.subplot(411 + ch_ex)
#    plt.plot(np.arange(0, 1.01, 0.01), train_set_tape_hold_imu_norm00[:, ch_ex])
#    plt.plot(np.arange(0, 1.01, 0.01), train_set_tape_hold_imu_norm01[:, ch_ex])
#    plt.plot(np.arange(0, 1.01, 0.01), train_set_tape_hold_imu_norm02[:, ch_ex])
#    plt.plot(np.arange(0, 1.01, 0.01), train_set_tape_hold_imu_norm03[:, ch_ex])
#    plt.plot(np.arange(0, 1.01, 0.01), train_set_tape_hold_imu_norm04[:, ch_ex])
#    plt.plot(np.arange(0, 1.01, 0.01), train_set_tape_hold_imu_norm05[:, ch_ex])
#    plt.plot(np.arange(0, 1.01, 0.01), train_set_tape_hold_imu_norm06[:, ch_ex])
#    plt.plot(np.arange(0, 1.01, 0.01), train_set_tape_hold_imu_norm07[:, ch_ex])
#    plt.plot(np.arange(0, 1.01, 0.01), train_set_tape_hold_imu_norm08[:, ch_ex])
#    plt.plot(np.arange(0, 1.01, 0.01), train_set_tape_hold_imu_norm09[:, ch_ex])
#    plt.plot(np.arange(0, 1.01, 0.01), train_set_tape_hold_imu_norm10[:, ch_ex])
#    plt.plot(np.arange(0, 1.01, 0.01), train_set_tape_hold_imu_norm11[:, ch_ex])
#    plt.plot(np.arange(0, 1.01, 0.01), train_set_tape_hold_imu_norm12[:, ch_ex])
#    plt.plot(np.arange(0, 1.01, 0.01), train_set_tape_hold_imu_norm13[:, ch_ex])
#    plt.plot(np.arange(0, 1.01, 0.01), train_set_tape_hold_imu_norm14[:, ch_ex])
#    plt.plot(np.arange(0, 1.01, 0.01), train_set_tape_hold_imu_norm15[:, ch_ex])
#    plt.plot(np.arange(0, 1.01, 0.01), train_set_tape_hold_imu_norm16[:, ch_ex])
#    plt.plot(np.arange(0, 1.01, 0.01), train_set_tape_hold_imu_norm17[:, ch_ex])
#    plt.plot(np.arange(0, 1.01, 0.01), train_set_tape_hold_imu_norm18[:, ch_ex])
#    plt.plot(np.arange(0, 1.01, 0.01), train_set_tape_hold_imu_norm19[:, ch_ex])
#
# #########################################
# # plot Pose norm data
# #########################################
# # plot the norm pose data of aluminum_hold
# plt.figure(16)
# for ch_ex in range(7):
#    plt.subplot(711 + ch_ex)
#    plt.plot(np.arange(0, 1.01, 0.01), train_set_aluminum_hold_pose_norm00[:, ch_ex])
#    plt.plot(np.arange(0, 1.01, 0.01), train_set_aluminum_hold_pose_norm01[:, ch_ex])
#    plt.plot(np.arange(0, 1.01, 0.01), train_set_aluminum_hold_pose_norm02[:, ch_ex])
#    plt.plot(np.arange(0, 1.01, 0.01), train_set_aluminum_hold_pose_norm03[:, ch_ex])
#    plt.plot(np.arange(0, 1.01, 0.01), train_set_aluminum_hold_pose_norm04[:, ch_ex])
#    plt.plot(np.arange(0, 1.01, 0.01), train_set_aluminum_hold_pose_norm05[:, ch_ex])
#    plt.plot(np.arange(0, 1.01, 0.01), train_set_aluminum_hold_pose_norm06[:, ch_ex])
#    plt.plot(np.arange(0, 1.01, 0.01), train_set_aluminum_hold_pose_norm07[:, ch_ex])
#    plt.plot(np.arange(0, 1.01, 0.01), train_set_aluminum_hold_pose_norm08[:, ch_ex])
#    plt.plot(np.arange(0, 1.01, 0.01), train_set_aluminum_hold_pose_norm09[:, ch_ex])
#    plt.plot(np.arange(0, 1.01, 0.01), train_set_aluminum_hold_pose_norm10[:, ch_ex])
#    plt.plot(np.arange(0, 1.01, 0.01), train_set_aluminum_hold_pose_norm11[:, ch_ex])
#    plt.plot(np.arange(0, 1.01, 0.01), train_set_aluminum_hold_pose_norm12[:, ch_ex])
#    plt.plot(np.arange(0, 1.01, 0.01), train_set_aluminum_hold_pose_norm13[:, ch_ex])
#    plt.plot(np.arange(0, 1.01, 0.01), train_set_aluminum_hold_pose_norm14[:, ch_ex])
#    plt.plot(np.arange(0, 1.01, 0.01), train_set_aluminum_hold_pose_norm15[:, ch_ex])
#    plt.plot(np.arange(0, 1.01, 0.01), train_set_aluminum_hold_pose_norm16[:, ch_ex])
#    plt.plot(np.arange(0, 1.01, 0.01), train_set_aluminum_hold_pose_norm17[:, ch_ex])
#    plt.plot(np.arange(0, 1.01, 0.01), train_set_aluminum_hold_pose_norm18[:, ch_ex])
#    plt.plot(np.arange(0, 1.01, 0.01), train_set_aluminum_hold_pose_norm19[:, ch_ex])
# # plot the norm pose data of spanner_handover
# plt.figure(17)
# for ch_ex in range(7):
#    plt.subplot(711 + ch_ex)
#    plt.plot(np.arange(0, 1.01, 0.01), train_set_spanner_handover_pose_norm00[:, ch_ex])
#    plt.plot(np.arange(0, 1.01, 0.01), train_set_spanner_handover_pose_norm01[:, ch_ex])
#    plt.plot(np.arange(0, 1.01, 0.01), train_set_spanner_handover_pose_norm02[:, ch_ex])
#    plt.plot(np.arange(0, 1.01, 0.01), train_set_spanner_handover_pose_norm03[:, ch_ex])
#    plt.plot(np.arange(0, 1.01, 0.01), train_set_spanner_handover_pose_norm04[:, ch_ex])
#    plt.plot(np.arange(0, 1.01, 0.01), train_set_spanner_handover_pose_norm05[:, ch_ex])
#    plt.plot(np.arange(0, 1.01, 0.01), train_set_spanner_handover_pose_norm06[:, ch_ex])
#    plt.plot(np.arange(0, 1.01, 0.01), train_set_spanner_handover_pose_norm07[:, ch_ex])
#    plt.plot(np.arange(0, 1.01, 0.01), train_set_spanner_handover_pose_norm08[:, ch_ex])
#    plt.plot(np.arange(0, 1.01, 0.01), train_set_spanner_handover_pose_norm09[:, ch_ex])
#    plt.plot(np.arange(0, 1.01, 0.01), train_set_spanner_handover_pose_norm10[:, ch_ex])
#    plt.plot(np.arange(0, 1.01, 0.01), train_set_spanner_handover_pose_norm11[:, ch_ex])
#    plt.plot(np.arange(0, 1.01, 0.01), train_set_spanner_handover_pose_norm12[:, ch_ex])
#    plt.plot(np.arange(0, 1.01, 0.01), train_set_spanner_handover_pose_norm13[:, ch_ex])
#    plt.plot(np.arange(0, 1.01, 0.01), train_set_spanner_handover_pose_norm14[:, ch_ex])
#    plt.plot(np.arange(0, 1.01, 0.01), train_set_spanner_handover_pose_norm15[:, ch_ex])
#    plt.plot(np.arange(0, 1.01, 0.01), train_set_spanner_handover_pose_norm16[:, ch_ex])
#    plt.plot(np.arange(0, 1.01, 0.01), train_set_spanner_handover_pose_norm17[:, ch_ex])
#    plt.plot(np.arange(0, 1.01, 0.01), train_set_spanner_handover_pose_norm18[:, ch_ex])
#    plt.plot(np.arange(0, 1.01, 0.01), train_set_spanner_handover_pose_norm19[:, ch_ex])
# # plot the norm pose data of tape_hold
# plt.figure(18)
# for ch_ex in range(7):
#    plt.subplot(711 + ch_ex)
#    plt.plot(np.arange(0, 1.01, 0.01), train_set_tape_hold_pose_norm00[:, ch_ex])
#    plt.plot(np.arange(0, 1.01, 0.01), train_set_tape_hold_pose_norm01[:, ch_ex])
#    plt.plot(np.arange(0, 1.01, 0.01), train_set_tape_hold_pose_norm02[:, ch_ex])
#    plt.plot(np.arange(0, 1.01, 0.01), train_set_tape_hold_pose_norm03[:, ch_ex])
#    plt.plot(np.arange(0, 1.01, 0.01), train_set_tape_hold_pose_norm04[:, ch_ex])
#    plt.plot(np.arange(0, 1.01, 0.01), train_set_tape_hold_pose_norm05[:, ch_ex])
#    plt.plot(np.arange(0, 1.01, 0.01), train_set_tape_hold_pose_norm06[:, ch_ex])
#    plt.plot(np.arange(0, 1.01, 0.01), train_set_tape_hold_pose_norm07[:, ch_ex])
#    plt.plot(np.arange(0, 1.01, 0.01), train_set_tape_hold_pose_norm08[:, ch_ex])
#    plt.plot(np.arange(0, 1.01, 0.01), train_set_tape_hold_pose_norm09[:, ch_ex])
#    plt.plot(np.arange(0, 1.01, 0.01), train_set_tape_hold_pose_norm10[:, ch_ex])
#    plt.plot(np.arange(0, 1.01, 0.01), train_set_tape_hold_pose_norm11[:, ch_ex])
#    plt.plot(np.arange(0, 1.01, 0.01), train_set_tape_hold_pose_norm12[:, ch_ex])
#    plt.plot(np.arange(0, 1.01, 0.01), train_set_tape_hold_pose_norm13[:, ch_ex])
#    plt.plot(np.arange(0, 1.01, 0.01), train_set_tape_hold_pose_norm14[:, ch_ex])
#    plt.plot(np.arange(0, 1.01, 0.01), train_set_tape_hold_pose_norm15[:, ch_ex])
#    plt.plot(np.arange(0, 1.01, 0.01), train_set_tape_hold_pose_norm16[:, ch_ex])
#    plt.plot(np.arange(0, 1.01, 0.01), train_set_tape_hold_pose_norm17[:, ch_ex])
#    plt.plot(np.arange(0, 1.01, 0.01), train_set_tape_hold_pose_norm18[:, ch_ex])
#    plt.plot(np.arange(0, 1.01, 0.01), train_set_tape_hold_pose_norm19[:, ch_ex])


# plot the prior distributioin
# plot ipromp_aluminum_hold
plt.figure(20)
for i in range(4):
    plt.subplot(411+i)
    ipromp_aluminum_hold.promps[i].plot(np.arange(0,1.01,0.01), color='g', legend='alumnium hold model, imu');plt.legend()
plt.figure(21)
for i in range(8):
    plt.subplot(421+i)
    ipromp_aluminum_hold.promps[4+i].plot(np.arange(0,1.01,0.01), color='g', legend='alumnium hold model, emg');plt.legend()
plt.figure(22)
for i in range(7):
    plt.subplot(711+i)
    ipromp_aluminum_hold.promps[4+8+i].plot(np.arange(0,1.01,0.01), color='g', legend='alumnium hold model, pose');plt.legend()
# plot ipromp_spanner_handover
plt.figure(23)
for i in range(4):
    plt.subplot(411+i)
    ipromp_spanner_handover.promps[i].plot(np.arange(0,1.01,0.01), color='g', legend='spanner handover model, imu');plt.legend()
plt.figure(24)
for i in range(8):
    plt.subplot(421+i)
    ipromp_spanner_handover.promps[4+i].plot(np.arange(0,1.01,0.01), color='g', legend='spanner handover model, emg');plt.legend()
plt.figure(25)
for i in range(7):
    plt.subplot(711+i)
    ipromp_spanner_handover.promps[4+8+i].plot(np.arange(0,1.01,0.01), color='g', legend='spanner handover model, pose');plt.legend()
# plot ipromp_tape_hold
plt.figure(26)
for i in range(4):
    plt.subplot(411+i)
    ipromp_tape_hold.promps[i].plot(np.arange(0,1.01,0.01), color='g', legend='tape hold model, imu');plt.legend()
plt.figure(27)
for i in range(8):
    plt.subplot(421+i)
    ipromp_tape_hold.promps[4+i].plot(np.arange(0,1.01,0.01), color='g', legend='tape hold model, emg');plt.legend()
plt.figure(28)
for i in range(7):
    plt.subplot(711+i)
    ipromp_tape_hold.promps[4+8+i].plot(np.arange(0,1.01,0.01), color='g', legend='tape hold model, pose');plt.legend()

# plot the updated distributioin
# plot ipromp_aluminum_hold
plt.figure(20)
for i in range(4):
    plt.subplot(411+i)
    plt.plot(ipromp_aluminum_hold.x, test_set[:, i], color='r', linewidth=3);
    ipromp_aluminum_hold.promps[i].plot_updated(np.arange(0,1.01,0.01), color='b', via_show=True);
plt.figure(21)
for i in range(8):
    plt.subplot(421+i)
    plt.plot(ipromp_aluminum_hold.x, test_set[:, 4+i], color='r', linewidth=3);
    ipromp_aluminum_hold.promps[4+i].plot_updated(np.arange(0,1.01,0.01), color='b', via_show=True);
plt.figure(22)
for i in range(7):
    plt.subplot(711+i)
    plt.plot(ipromp_aluminum_hold.x, test_set[:, 4+8+i], color='r', linewidth=3);
    ipromp_aluminum_hold.promps[4+8+i].plot_updated(np.arange(0,1.01,0.01), color='b', via_show=False);
# plot ipromp_spanner_handover
plt.figure(23)
for i in range(4):
    plt.subplot(411+i)
    plt.plot(ipromp_aluminum_hold.x, test_set[:, i], color='r', linewidth=3);
    ipromp_spanner_handover.promps[i].plot_updated(np.arange(0,1.01,0.01), color='b', via_show=True);
plt.figure(24)
for i in range(8):
    plt.subplot(421+i)
    plt.plot(ipromp_aluminum_hold.x, test_set[:, 4+i], color='r', linewidth=3);
    ipromp_spanner_handover.promps[4+i].plot_updated(np.arange(0,1.01,0.01), color='b', via_show=True);
plt.figure(25)
for i in range(7):
    plt.subplot(711+i)
    plt.plot(ipromp_aluminum_hold.x, test_set[:, 4+8+i], color='r', linewidth=3);
    ipromp_spanner_handover.promps[4+8+i].plot_updated(np.arange(0,1.01,0.01), color='b', via_show=False);
# plot ipromp_tape_hold
plt.figure(26)
for i in range(4):
    plt.subplot(411+i)
    plt.plot(ipromp_aluminum_hold.x, test_set[:, i], color='r', linewidth=3);
    ipromp_tape_hold.promps[i].plot_updated(np.arange(0,1.01,0.01), color='b', via_show=True);
plt.figure(27)
for i in range(8):
    plt.subplot(421+i)
    plt.plot(ipromp_aluminum_hold.x, test_set[:, 4+i], color='r', linewidth=3);
    ipromp_tape_hold.promps[4+i].plot_updated(np.arange(0,1.01,0.01), color='b', via_show=True);
plt.figure(28)
for i in range(7):
    plt.subplot(711+i)
    plt.plot(ipromp_aluminum_hold.x, test_set[:, 4+8+i], color='r', linewidth=3);
    ipromp_tape_hold.promps[4+8+i].plot_updated(np.arange(0,1.01,0.01), color='b', via_show=False);

# plt.show()