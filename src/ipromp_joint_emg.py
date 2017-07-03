#!/usr/bin/python
# Filename: ndpromp_joint_emg.py

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.signal as signal
import ipromps

plt.close('all')    # close all windows
len_normal = 101    # the len of normalized traj
nrDemo = 20         # number of trajectoreis for training


#########################################
# load date sets
#########################################
# read emg csv files
dir_prefix = '../../recorder/datasets/joint_emg/hold/csv/'
train_set_emg_00_pd = pd.read_csv(dir_prefix + '2017-05-27-11-44-31-myo_raw_pub.csv')
train_set_emg_01_pd = pd.read_csv(dir_prefix + '2017-05-27-11-44-56-myo_raw_pub.csv')
train_set_emg_02_pd = pd.read_csv(dir_prefix + '2017-05-27-11-45-20-myo_raw_pub.csv')
train_set_emg_03_pd = pd.read_csv(dir_prefix + '2017-05-27-11-46-31-myo_raw_pub.csv')
train_set_emg_04_pd = pd.read_csv(dir_prefix + '2017-05-27-11-46-55-myo_raw_pub.csv')
train_set_emg_05_pd = pd.read_csv(dir_prefix + '2017-05-27-11-47-26-myo_raw_pub.csv')
train_set_emg_06_pd = pd.read_csv(dir_prefix + '2017-05-27-11-47-53-myo_raw_pub.csv')
train_set_emg_07_pd = pd.read_csv(dir_prefix + '2017-05-27-11-48-15-myo_raw_pub.csv')
train_set_emg_08_pd = pd.read_csv(dir_prefix + '2017-05-27-11-48-36-myo_raw_pub.csv')
train_set_emg_09_pd = pd.read_csv(dir_prefix + '2017-05-27-11-48-59-myo_raw_pub.csv')
train_set_emg_10_pd = pd.read_csv(dir_prefix + '2017-05-27-11-49-27-myo_raw_pub.csv')
train_set_emg_11_pd = pd.read_csv(dir_prefix + '2017-05-27-11-49-55-myo_raw_pub.csv')
train_set_emg_12_pd = pd.read_csv(dir_prefix + '2017-05-27-11-50-19-myo_raw_pub.csv')
train_set_emg_13_pd = pd.read_csv(dir_prefix + '2017-05-27-11-50-42-myo_raw_pub.csv')
train_set_emg_14_pd = pd.read_csv(dir_prefix + '2017-05-27-11-51-10-myo_raw_pub.csv')
train_set_emg_15_pd = pd.read_csv(dir_prefix + '2017-05-27-11-51-32-myo_raw_pub.csv')
train_set_emg_16_pd = pd.read_csv(dir_prefix + '2017-05-27-11-51-59-myo_raw_pub.csv')
train_set_emg_17_pd = pd.read_csv(dir_prefix + '2017-05-27-11-52-23-myo_raw_pub.csv')
train_set_emg_18_pd = pd.read_csv(dir_prefix + '2017-05-27-11-52-48-myo_raw_pub.csv')
train_set_emg_19_pd = pd.read_csv(dir_prefix + '2017-05-27-11-53-09-myo_raw_pub.csv')
test_set_emg_pd = pd.read_csv(dir_prefix + '2017-05-27-11-53-38-myo_raw_pub.csv')
# invert the object to float32 for easy computing
train_set_emg_00 = np.float32(train_set_emg_00_pd.values[:,5:13])
train_set_emg_01 = np.float32(train_set_emg_01_pd.values[:,5:13])
train_set_emg_02 = np.float32(train_set_emg_02_pd.values[:,5:13])
train_set_emg_03 = np.float32(train_set_emg_03_pd.values[:,5:13])
train_set_emg_04 = np.float32(train_set_emg_04_pd.values[:,5:13])
train_set_emg_05 = np.float32(train_set_emg_05_pd.values[:,5:13])
train_set_emg_06 = np.float32(train_set_emg_06_pd.values[:,5:13])
train_set_emg_07 = np.float32(train_set_emg_07_pd.values[:,5:13])
train_set_emg_08 = np.float32(train_set_emg_08_pd.values[:,5:13])
train_set_emg_09 = np.float32(train_set_emg_09_pd.values[:,5:13])
train_set_emg_10 = np.float32(train_set_emg_10_pd.values[:,5:13])
train_set_emg_11 = np.float32(train_set_emg_11_pd.values[:,5:13])
train_set_emg_12 = np.float32(train_set_emg_12_pd.values[:,5:13])
train_set_emg_13 = np.float32(train_set_emg_13_pd.values[:,5:13])
train_set_emg_14 = np.float32(train_set_emg_14_pd.values[:,5:13])
train_set_emg_15 = np.float32(train_set_emg_15_pd.values[:,5:13])
train_set_emg_16 = np.float32(train_set_emg_16_pd.values[:,5:13])
train_set_emg_17 = np.float32(train_set_emg_17_pd.values[:,5:13])
train_set_emg_18 = np.float32(train_set_emg_18_pd.values[:,5:13])
train_set_emg_19 = np.float32(train_set_emg_19_pd.values[:,5:13])
test_set_emg = np.float32(test_set_emg_pd.values[:,5:13])

# read the joint txt file for fast reading
train_set_joint_00 = np.loadtxt(dir_prefix + 'train_set_joint_00.txt')
train_set_joint_01 = np.loadtxt(dir_prefix + 'train_set_joint_01.txt')
train_set_joint_02 = np.loadtxt(dir_prefix + 'train_set_joint_02.txt')
train_set_joint_03 = np.loadtxt(dir_prefix + 'train_set_joint_03.txt')
train_set_joint_04 = np.loadtxt(dir_prefix + 'train_set_joint_04.txt')
train_set_joint_05 = np.loadtxt(dir_prefix + 'train_set_joint_05.txt')
train_set_joint_06 = np.loadtxt(dir_prefix + 'train_set_joint_06.txt')
train_set_joint_07 = np.loadtxt(dir_prefix + 'train_set_joint_07.txt')
train_set_joint_08 = np.loadtxt(dir_prefix + 'train_set_joint_08.txt')
train_set_joint_09 = np.loadtxt(dir_prefix + 'train_set_joint_09.txt')
train_set_joint_10 = np.loadtxt(dir_prefix + 'train_set_joint_10.txt')
train_set_joint_11 = np.loadtxt(dir_prefix + 'train_set_joint_11.txt')
train_set_joint_12 = np.loadtxt(dir_prefix + 'train_set_joint_12.txt')
train_set_joint_13 = np.loadtxt(dir_prefix + 'train_set_joint_13.txt')
train_set_joint_14 = np.loadtxt(dir_prefix + 'train_set_joint_14.txt')
train_set_joint_15 = np.loadtxt(dir_prefix + 'train_set_joint_15.txt')
train_set_joint_16 = np.loadtxt(dir_prefix + 'train_set_joint_16.txt')
train_set_joint_17 = np.loadtxt(dir_prefix + 'train_set_joint_17.txt')
train_set_joint_18 = np.loadtxt(dir_prefix + 'train_set_joint_18.txt')
train_set_joint_19 = np.loadtxt(dir_prefix + 'train_set_joint_19.txt')
test_set_joint = np.loadtxt(dir_prefix + 'test_set_joint.txt')

# #########################################
# # plot raw data
# #########################################
# # plot the origin emg data
# plt.figure(0)
# for ch_ex in range(8):
#    plt.subplot(421+ch_ex)
#    plt.plot(range(len(train_set_emg_00)), train_set_emg_00[:,ch_ex])
#    plt.plot(range(len(train_set_emg_01)), train_set_emg_01[:,ch_ex])
#    plt.plot(range(len(train_set_emg_02)), train_set_emg_02[:,ch_ex])
#    plt.plot(range(len(train_set_emg_03)), train_set_emg_03[:,ch_ex])
#    plt.plot(range(len(train_set_emg_04)), train_set_emg_04[:,ch_ex])
#    plt.plot(range(len(train_set_emg_05)), train_set_emg_05[:,ch_ex])
#    plt.plot(range(len(train_set_emg_06)), train_set_emg_06[:,ch_ex])
#    plt.plot(range(len(train_set_emg_07)), train_set_emg_07[:,ch_ex])
#    plt.plot(range(len(train_set_emg_08)), train_set_emg_08[:,ch_ex])
#    plt.plot(range(len(train_set_emg_09)), train_set_emg_09[:,ch_ex])
#    plt.plot(range(len(train_set_emg_10)), train_set_emg_10[:,ch_ex])
#    plt.plot(range(len(train_set_emg_11)), train_set_emg_11[:,ch_ex])
#    plt.plot(range(len(train_set_emg_12)), train_set_emg_12[:,ch_ex])
#    plt.plot(range(len(train_set_emg_13)), train_set_emg_13[:,ch_ex])
#    plt.plot(range(len(train_set_emg_14)), train_set_emg_14[:,ch_ex])
#    plt.plot(range(len(train_set_emg_15)), train_set_emg_15[:,ch_ex])
#    plt.plot(range(len(train_set_emg_16)), train_set_emg_16[:,ch_ex])
#    plt.plot(range(len(train_set_emg_17)), train_set_emg_17[:,ch_ex])
#    plt.plot(range(len(train_set_emg_18)), train_set_emg_18[:,ch_ex])
#    plt.plot(range(len(train_set_emg_19)), train_set_emg_19[:,ch_ex])
# # plot the origin joint data
# plt.figure(1)
# for ch_ex in range(7):
#    plt.subplot(711+ch_ex)
#    plt.plot(range(len(train_set_joint_00)), train_set_joint_00[:,9+ch_ex])
#    plt.plot(range(len(train_set_joint_01)), train_set_joint_01[:,9+ch_ex])
#    plt.plot(range(len(train_set_joint_02)), train_set_joint_02[:,9+ch_ex])
#    plt.plot(range(len(train_set_joint_03)), train_set_joint_03[:,9+ch_ex])
#    plt.plot(range(len(train_set_joint_04)), train_set_joint_04[:,9+ch_ex])
#    plt.plot(range(len(train_set_joint_05)), train_set_joint_05[:,9+ch_ex])
#    plt.plot(range(len(train_set_joint_06)), train_set_joint_06[:,9+ch_ex])
#    plt.plot(range(len(train_set_joint_07)), train_set_joint_07[:,9+ch_ex])
#    plt.plot(range(len(train_set_joint_08)), train_set_joint_08[:,9+ch_ex])
#    plt.plot(range(len(train_set_joint_09)), train_set_joint_09[:,9+ch_ex])
#    plt.plot(range(len(train_set_joint_10)), train_set_joint_10[:,9+ch_ex])
#    plt.plot(range(len(train_set_joint_11)), train_set_joint_11[:,9+ch_ex])
#    plt.plot(range(len(train_set_joint_12)), train_set_joint_12[:,9+ch_ex])
#    plt.plot(range(len(train_set_joint_13)), train_set_joint_13[:,9+ch_ex])
#    plt.plot(range(len(train_set_joint_14)), train_set_joint_14[:,9+ch_ex])
#    plt.plot(range(len(train_set_joint_15)), train_set_joint_15[:,9+ch_ex])
#    plt.plot(range(len(train_set_joint_16)), train_set_joint_16[:,9+ch_ex])
#    plt.plot(range(len(train_set_joint_17)), train_set_joint_17[:,9+ch_ex])
#    plt.plot(range(len(train_set_joint_18)), train_set_joint_18[:,9+ch_ex])
#    plt.plot(range(len(train_set_joint_19)), train_set_joint_19[:,9+ch_ex])


#########################################
# resampling the signals for experiencing the same duration
#########################################
# resampling emg signals
train_set_emg_norm00=np.array([]);train_set_emg_norm01=np.array([]);train_set_emg_norm02=np.array([]);train_set_emg_norm03=np.array([]);train_set_emg_norm04=np.array([]);
train_set_emg_norm05=np.array([]);train_set_emg_norm06=np.array([]);train_set_emg_norm07=np.array([]);train_set_emg_norm08=np.array([]);train_set_emg_norm09=np.array([]);
train_set_emg_norm10=np.array([]);train_set_emg_norm11=np.array([]);train_set_emg_norm12=np.array([]);train_set_emg_norm13=np.array([]);train_set_emg_norm14=np.array([]);
train_set_emg_norm15=np.array([]);train_set_emg_norm16=np.array([]);train_set_emg_norm17=np.array([]);train_set_emg_norm18=np.array([]);train_set_emg_norm19=np.array([]);
test_set_emg_norm=np.array([]);
for ch_ex in range(8):
    train_set_emg_norm00 = np.hstack(( train_set_emg_norm00, np.interp(np.linspace(0, len(train_set_emg_00)-1, len_normal), np.arange(0,len(train_set_emg_00),1.), train_set_emg_00[:,ch_ex]) ))
    train_set_emg_norm01 = np.hstack(( train_set_emg_norm01, np.interp(np.linspace(0, len(train_set_emg_01)-1, len_normal), np.arange(0,len(train_set_emg_01),1.), train_set_emg_01[:,ch_ex]) ))
    train_set_emg_norm02 = np.hstack(( train_set_emg_norm02, np.interp(np.linspace(0, len(train_set_emg_02)-1, len_normal), np.arange(0,len(train_set_emg_02),1.), train_set_emg_02[:,ch_ex]) ))
    train_set_emg_norm03 = np.hstack(( train_set_emg_norm03, np.interp(np.linspace(0, len(train_set_emg_03)-1, len_normal), np.arange(0,len(train_set_emg_03),1.), train_set_emg_03[:,ch_ex]) ))
    train_set_emg_norm04 = np.hstack(( train_set_emg_norm04, np.interp(np.linspace(0, len(train_set_emg_04)-1, len_normal), np.arange(0,len(train_set_emg_04),1.), train_set_emg_04[:,ch_ex]) ))
    train_set_emg_norm05 = np.hstack(( train_set_emg_norm05, np.interp(np.linspace(0, len(train_set_emg_05)-1, len_normal), np.arange(0,len(train_set_emg_05),1.), train_set_emg_05[:,ch_ex]) ))
    train_set_emg_norm06 = np.hstack(( train_set_emg_norm06, np.interp(np.linspace(0, len(train_set_emg_06)-1, len_normal), np.arange(0,len(train_set_emg_06),1.), train_set_emg_06[:,ch_ex]) ))
    train_set_emg_norm07 = np.hstack(( train_set_emg_norm07, np.interp(np.linspace(0, len(train_set_emg_07)-1, len_normal), np.arange(0,len(train_set_emg_07),1.), train_set_emg_07[:,ch_ex]) ))
    train_set_emg_norm08 = np.hstack(( train_set_emg_norm08, np.interp(np.linspace(0, len(train_set_emg_08)-1, len_normal), np.arange(0,len(train_set_emg_08),1.), train_set_emg_08[:,ch_ex]) ))
    train_set_emg_norm09 = np.hstack(( train_set_emg_norm09, np.interp(np.linspace(0, len(train_set_emg_09)-1, len_normal), np.arange(0,len(train_set_emg_09),1.), train_set_emg_09[:,ch_ex]) ))
    train_set_emg_norm10 = np.hstack(( train_set_emg_norm10, np.interp(np.linspace(0, len(train_set_emg_10)-1, len_normal), np.arange(0,len(train_set_emg_10),1.), train_set_emg_10[:,ch_ex]) ))
    train_set_emg_norm11 = np.hstack(( train_set_emg_norm11, np.interp(np.linspace(0, len(train_set_emg_11)-1, len_normal), np.arange(0,len(train_set_emg_11),1.), train_set_emg_11[:,ch_ex]) ))
    train_set_emg_norm12 = np.hstack(( train_set_emg_norm12, np.interp(np.linspace(0, len(train_set_emg_12)-1, len_normal), np.arange(0,len(train_set_emg_12),1.), train_set_emg_12[:,ch_ex]) ))
    train_set_emg_norm13 = np.hstack(( train_set_emg_norm13, np.interp(np.linspace(0, len(train_set_emg_13)-1, len_normal), np.arange(0,len(train_set_emg_13),1.), train_set_emg_13[:,ch_ex]) ))
    train_set_emg_norm14 = np.hstack(( train_set_emg_norm14, np.interp(np.linspace(0, len(train_set_emg_14)-1, len_normal), np.arange(0,len(train_set_emg_14),1.), train_set_emg_14[:,ch_ex]) ))
    train_set_emg_norm15 = np.hstack(( train_set_emg_norm15, np.interp(np.linspace(0, len(train_set_emg_15)-1, len_normal), np.arange(0,len(train_set_emg_15),1.), train_set_emg_15[:,ch_ex]) ))
    train_set_emg_norm16 = np.hstack(( train_set_emg_norm16, np.interp(np.linspace(0, len(train_set_emg_16)-1, len_normal), np.arange(0,len(train_set_emg_16),1.), train_set_emg_16[:,ch_ex]) ))
    train_set_emg_norm17 = np.hstack(( train_set_emg_norm17, np.interp(np.linspace(0, len(train_set_emg_17)-1, len_normal), np.arange(0,len(train_set_emg_17),1.), train_set_emg_17[:,ch_ex]) ))
    train_set_emg_norm18 = np.hstack(( train_set_emg_norm18, np.interp(np.linspace(0, len(train_set_emg_18)-1, len_normal), np.arange(0,len(train_set_emg_18),1.), train_set_emg_18[:,ch_ex]) ))
    train_set_emg_norm19 = np.hstack(( train_set_emg_norm19, np.interp(np.linspace(0, len(train_set_emg_19)-1, len_normal), np.arange(0,len(train_set_emg_19),1.), train_set_emg_19[:,ch_ex]) ))
    test_set_emg_norm = np.hstack(( test_set_emg_norm, np.interp(np.linspace(0, len(test_set_emg)-1, len_normal), np.arange(0,len(test_set_emg),1.), test_set_emg[:,ch_ex]) ))
train_set_emg_norm00 = train_set_emg_norm00.reshape(8,len_normal).T
train_set_emg_norm01 = train_set_emg_norm01.reshape(8,len_normal).T
train_set_emg_norm02 = train_set_emg_norm02.reshape(8,len_normal).T
train_set_emg_norm03 = train_set_emg_norm03.reshape(8,len_normal).T
train_set_emg_norm04 = train_set_emg_norm04.reshape(8,len_normal).T
train_set_emg_norm05 = train_set_emg_norm05.reshape(8,len_normal).T
train_set_emg_norm06 = train_set_emg_norm06.reshape(8,len_normal).T
train_set_emg_norm07 = train_set_emg_norm07.reshape(8,len_normal).T
train_set_emg_norm08 = train_set_emg_norm08.reshape(8,len_normal).T
train_set_emg_norm09 = train_set_emg_norm09.reshape(8,len_normal).T
train_set_emg_norm10 = train_set_emg_norm10.reshape(8,len_normal).T
train_set_emg_norm11 = train_set_emg_norm11.reshape(8,len_normal).T
train_set_emg_norm12 = train_set_emg_norm12.reshape(8,len_normal).T
train_set_emg_norm13 = train_set_emg_norm13.reshape(8,len_normal).T
train_set_emg_norm14 = train_set_emg_norm14.reshape(8,len_normal).T
train_set_emg_norm15 = train_set_emg_norm15.reshape(8,len_normal).T
train_set_emg_norm16 = train_set_emg_norm16.reshape(8,len_normal).T
train_set_emg_norm17 = train_set_emg_norm17.reshape(8,len_normal).T
train_set_emg_norm18 = train_set_emg_norm18.reshape(8,len_normal).T
train_set_emg_norm19 = train_set_emg_norm19.reshape(8,len_normal).T
test_set_emg_norm = test_set_emg_norm.reshape(8,len_normal).T
train_set_emg_norm_full = np.array([train_set_emg_norm00,train_set_emg_norm01,train_set_emg_norm02,train_set_emg_norm03,train_set_emg_norm04,
                                    train_set_emg_norm05,train_set_emg_norm06,train_set_emg_norm07,train_set_emg_norm08,train_set_emg_norm09,
                                    train_set_emg_norm10,train_set_emg_norm11,train_set_emg_norm12,train_set_emg_norm13,train_set_emg_norm14,
                                    train_set_emg_norm15,train_set_emg_norm16,train_set_emg_norm17,train_set_emg_norm18,train_set_emg_norm19])
# resampling joint signals
train_set_joint_norm00=np.array([]);train_set_joint_norm01=np.array([]);train_set_joint_norm02=np.array([]);train_set_joint_norm03=np.array([]);train_set_joint_norm04=np.array([]);
train_set_joint_norm05=np.array([]);train_set_joint_norm06=np.array([]);train_set_joint_norm07=np.array([]);train_set_joint_norm08=np.array([]);train_set_joint_norm09=np.array([]);
train_set_joint_norm10=np.array([]);train_set_joint_norm11=np.array([]);train_set_joint_norm12=np.array([]);train_set_joint_norm13=np.array([]);train_set_joint_norm14=np.array([]);
train_set_joint_norm15=np.array([]);train_set_joint_norm16=np.array([]);train_set_joint_norm17=np.array([]);train_set_joint_norm18=np.array([]);train_set_joint_norm19=np.array([]);
test_set_joint_norm=np.array([]);
for ch_ex in range(7):
    train_set_joint_norm00 = np.hstack(( train_set_joint_norm00, np.interp(np.linspace(0, len(train_set_joint_00)-1, len_normal), np.arange(0,len(train_set_joint_00),1.), train_set_joint_00[:,ch_ex+9]) ))
    train_set_joint_norm01 = np.hstack(( train_set_joint_norm01, np.interp(np.linspace(0, len(train_set_joint_01)-1, len_normal), np.arange(0,len(train_set_joint_01),1.), train_set_joint_01[:,ch_ex+9]) ))
    train_set_joint_norm02 = np.hstack(( train_set_joint_norm02, np.interp(np.linspace(0, len(train_set_joint_02)-1, len_normal), np.arange(0,len(train_set_joint_02),1.), train_set_joint_02[:,ch_ex+9]) ))
    train_set_joint_norm03 = np.hstack(( train_set_joint_norm03, np.interp(np.linspace(0, len(train_set_joint_03)-1, len_normal), np.arange(0,len(train_set_joint_03),1.), train_set_joint_03[:,ch_ex+9]) ))
    train_set_joint_norm04 = np.hstack(( train_set_joint_norm04, np.interp(np.linspace(0, len(train_set_joint_04)-1, len_normal), np.arange(0,len(train_set_joint_04),1.), train_set_joint_04[:,ch_ex+9]) ))
    train_set_joint_norm05 = np.hstack(( train_set_joint_norm05, np.interp(np.linspace(0, len(train_set_joint_05)-1, len_normal), np.arange(0,len(train_set_joint_05),1.), train_set_joint_05[:,ch_ex+9]) ))
    train_set_joint_norm06 = np.hstack(( train_set_joint_norm06, np.interp(np.linspace(0, len(train_set_joint_06)-1, len_normal), np.arange(0,len(train_set_joint_06),1.), train_set_joint_06[:,ch_ex+9]) ))
    train_set_joint_norm07 = np.hstack(( train_set_joint_norm07, np.interp(np.linspace(0, len(train_set_joint_07)-1, len_normal), np.arange(0,len(train_set_joint_07),1.), train_set_joint_07[:,ch_ex+9]) ))
    train_set_joint_norm08 = np.hstack(( train_set_joint_norm08, np.interp(np.linspace(0, len(train_set_joint_08)-1, len_normal), np.arange(0,len(train_set_joint_08),1.), train_set_joint_08[:,ch_ex+9]) ))
    train_set_joint_norm09 = np.hstack(( train_set_joint_norm09, np.interp(np.linspace(0, len(train_set_joint_09)-1, len_normal), np.arange(0,len(train_set_joint_09),1.), train_set_joint_09[:,ch_ex+9]) ))
    train_set_joint_norm10 = np.hstack(( train_set_joint_norm10, np.interp(np.linspace(0, len(train_set_joint_10)-1, len_normal), np.arange(0,len(train_set_joint_10),1.), train_set_joint_10[:,ch_ex+9]) ))
    train_set_joint_norm11 = np.hstack(( train_set_joint_norm11, np.interp(np.linspace(0, len(train_set_joint_11)-1, len_normal), np.arange(0,len(train_set_joint_11),1.), train_set_joint_11[:,ch_ex+9]) ))
    train_set_joint_norm12 = np.hstack(( train_set_joint_norm12, np.interp(np.linspace(0, len(train_set_joint_12)-1, len_normal), np.arange(0,len(train_set_joint_12),1.), train_set_joint_12[:,ch_ex+9]) ))
    train_set_joint_norm13 = np.hstack(( train_set_joint_norm13, np.interp(np.linspace(0, len(train_set_joint_13)-1, len_normal), np.arange(0,len(train_set_joint_13),1.), train_set_joint_13[:,ch_ex+9]) ))
    train_set_joint_norm14 = np.hstack(( train_set_joint_norm14, np.interp(np.linspace(0, len(train_set_joint_14)-1, len_normal), np.arange(0,len(train_set_joint_14),1.), train_set_joint_14[:,ch_ex+9]) ))
    train_set_joint_norm15 = np.hstack(( train_set_joint_norm15, np.interp(np.linspace(0, len(train_set_joint_15)-1, len_normal), np.arange(0,len(train_set_joint_15),1.), train_set_joint_15[:,ch_ex+9]) ))
    train_set_joint_norm16 = np.hstack(( train_set_joint_norm16, np.interp(np.linspace(0, len(train_set_joint_16)-1, len_normal), np.arange(0,len(train_set_joint_16),1.), train_set_joint_16[:,ch_ex+9]) ))
    train_set_joint_norm17 = np.hstack(( train_set_joint_norm17, np.interp(np.linspace(0, len(train_set_joint_17)-1, len_normal), np.arange(0,len(train_set_joint_17),1.), train_set_joint_17[:,ch_ex+9]) ))
    train_set_joint_norm18 = np.hstack(( train_set_joint_norm18, np.interp(np.linspace(0, len(train_set_joint_18)-1, len_normal), np.arange(0,len(train_set_joint_18),1.), train_set_joint_18[:,ch_ex+9]) ))
    train_set_joint_norm19 = np.hstack(( train_set_joint_norm19, np.interp(np.linspace(0, len(train_set_joint_19)-1, len_normal), np.arange(0,len(train_set_joint_19),1.), train_set_joint_19[:,ch_ex+9]) ))
    test_set_joint_norm = np.hstack(( test_set_joint_norm, np.interp(np.linspace(0, len(test_set_joint)-1, len_normal), np.arange(0,len(test_set_joint),1.), test_set_joint[:,ch_ex]) ))
train_set_joint_norm00 = train_set_joint_norm00.reshape(7,len_normal).T
train_set_joint_norm01 = train_set_joint_norm01.reshape(7,len_normal).T
train_set_joint_norm02 = train_set_joint_norm02.reshape(7,len_normal).T
train_set_joint_norm03 = train_set_joint_norm03.reshape(7,len_normal).T
train_set_joint_norm04 = train_set_joint_norm04.reshape(7,len_normal).T
train_set_joint_norm05 = train_set_joint_norm05.reshape(7,len_normal).T
train_set_joint_norm06 = train_set_joint_norm06.reshape(7,len_normal).T
train_set_joint_norm07 = train_set_joint_norm07.reshape(7,len_normal).T
train_set_joint_norm08 = train_set_joint_norm08.reshape(7,len_normal).T
train_set_joint_norm09 = train_set_joint_norm09.reshape(7,len_normal).T
train_set_joint_norm10 = train_set_joint_norm10.reshape(7,len_normal).T
train_set_joint_norm11 = train_set_joint_norm11.reshape(7,len_normal).T
train_set_joint_norm12 = train_set_joint_norm12.reshape(7,len_normal).T
train_set_joint_norm13 = train_set_joint_norm13.reshape(7,len_normal).T
train_set_joint_norm14 = train_set_joint_norm14.reshape(7,len_normal).T
train_set_joint_norm15 = train_set_joint_norm15.reshape(7,len_normal).T
train_set_joint_norm16 = train_set_joint_norm16.reshape(7,len_normal).T
train_set_joint_norm17 = train_set_joint_norm17.reshape(7,len_normal).T
train_set_joint_norm18 = train_set_joint_norm18.reshape(7,len_normal).T
train_set_joint_norm19 = train_set_joint_norm19.reshape(7,len_normal).T
test_set_joint_norm = test_set_joint_norm.reshape(7,len_normal).T
train_set_joint_norm_full = np.array([train_set_joint_norm00, train_set_joint_norm01, train_set_joint_norm02, train_set_joint_norm03, train_set_joint_norm04, 
                                      train_set_joint_norm05, train_set_joint_norm06, train_set_joint_norm07, train_set_joint_norm08, train_set_joint_norm09, 
                                      train_set_joint_norm10, train_set_joint_norm11, train_set_joint_norm12, train_set_joint_norm13, train_set_joint_norm14, 
                                      train_set_joint_norm15, train_set_joint_norm16, train_set_joint_norm17, train_set_joint_norm18, train_set_joint_norm19])


#########################################
# plot norm data
#########################################
# plot the norm emg data
plt.figure(2)
for ch_ex in range(8):
   plt.subplot(421+ch_ex)
   plt.plot(np.arange(0,1.01,0.01), train_set_emg_norm00[:,ch_ex])
   plt.plot(np.arange(0,1.01,0.01), train_set_emg_norm01[:,ch_ex])
   plt.plot(np.arange(0,1.01,0.01), train_set_emg_norm02[:,ch_ex])
   plt.plot(np.arange(0,1.01,0.01), train_set_emg_norm03[:,ch_ex])
   plt.plot(np.arange(0,1.01,0.01), train_set_emg_norm04[:,ch_ex])
   plt.plot(np.arange(0,1.01,0.01), train_set_emg_norm05[:,ch_ex])
   plt.plot(np.arange(0,1.01,0.01), train_set_emg_norm06[:,ch_ex])
   plt.plot(np.arange(0,1.01,0.01), train_set_emg_norm07[:,ch_ex])
   plt.plot(np.arange(0,1.01,0.01), train_set_emg_norm08[:,ch_ex])
   plt.plot(np.arange(0,1.01,0.01), train_set_emg_norm09[:,ch_ex])
   plt.plot(np.arange(0,1.01,0.01), train_set_emg_norm10[:,ch_ex])
   plt.plot(np.arange(0,1.01,0.01), train_set_emg_norm11[:,ch_ex])
   plt.plot(np.arange(0,1.01,0.01), train_set_emg_norm12[:,ch_ex])
   plt.plot(np.arange(0,1.01,0.01), train_set_emg_norm13[:,ch_ex])
   plt.plot(np.arange(0,1.01,0.01), train_set_emg_norm14[:,ch_ex])
   plt.plot(np.arange(0,1.01,0.01), train_set_emg_norm15[:,ch_ex])
   plt.plot(np.arange(0,1.01,0.01), train_set_emg_norm16[:,ch_ex])
   plt.plot(np.arange(0,1.01,0.01), train_set_emg_norm17[:,ch_ex])
   plt.plot(np.arange(0,1.01,0.01), train_set_emg_norm18[:,ch_ex])
   plt.plot(np.arange(0,1.01,0.01), train_set_emg_norm19[:,ch_ex])
# plot the norm joint data
plt.figure(3)
for ch_ex in range(7):
   plt.subplot(711+ch_ex)
   plt.plot(np.arange(0,1.01,0.01), train_set_joint_norm00[:,ch_ex])
   plt.plot(np.arange(0,1.01,0.01), train_set_joint_norm01[:,ch_ex])
   plt.plot(np.arange(0,1.01,0.01), train_set_joint_norm02[:,ch_ex])
   plt.plot(np.arange(0,1.01,0.01), train_set_joint_norm03[:,ch_ex])
   plt.plot(np.arange(0,1.01,0.01), train_set_joint_norm04[:,ch_ex])
   plt.plot(np.arange(0,1.01,0.01), train_set_joint_norm05[:,ch_ex])
   plt.plot(np.arange(0,1.01,0.01), train_set_joint_norm06[:,ch_ex])
   plt.plot(np.arange(0,1.01,0.01), train_set_joint_norm07[:,ch_ex])
   plt.plot(np.arange(0,1.01,0.01), train_set_joint_norm08[:,ch_ex])
   plt.plot(np.arange(0,1.01,0.01), train_set_joint_norm09[:,ch_ex])
   plt.plot(np.arange(0,1.01,0.01), train_set_joint_norm10[:,ch_ex])
   plt.plot(np.arange(0,1.01,0.01), train_set_joint_norm11[:,ch_ex])
   plt.plot(np.arange(0,1.01,0.01), train_set_joint_norm12[:,ch_ex])
   plt.plot(np.arange(0,1.01,0.01), train_set_joint_norm13[:,ch_ex])
   plt.plot(np.arange(0,1.01,0.01), train_set_joint_norm14[:,ch_ex])
   plt.plot(np.arange(0,1.01,0.01), train_set_joint_norm15[:,ch_ex])
   plt.plot(np.arange(0,1.01,0.01), train_set_joint_norm16[:,ch_ex])
   plt.plot(np.arange(0,1.01,0.01), train_set_joint_norm17[:,ch_ex])
   plt.plot(np.arange(0,1.01,0.01), train_set_joint_norm18[:,ch_ex])
   plt.plot(np.arange(0,1.01,0.01), train_set_joint_norm19[:,ch_ex])


# create a n-dimensional iProMP
ipromp = ipromps.IProMP(num_joints=15, nrBasis=11, sigma=0.05, num_samples=101)

# add demostration
for idx in range(0, nrDemo):
    demo_temp = np.hstack([train_set_emg_norm_full[idx], train_set_joint_norm_full[idx]])
    ipromp.add_demonstration(demo_temp)

# plot the prior distributioin
plt.figure(4)
for i in range(8):
    plt.subplot(421+i)
    ipromp.promps[i].plot(np.arange(0,1.01,0.01), color='g')
plt.figure(5)
for i in range(7):
    plt.subplot(711+i)
    ipromp.promps[8+i].plot(np.arange(0,1.01,0.01), color='g')

# filter emg and joint data
# test_set_emg_norm_filtered = signal.medfilt(test_set_emg_norm, [11,1])
# test_set_joint_norm_filtered = signal.medfilt(test_set_joint_norm, [5,1])
test_set_emg_norm_filtered = signal.medfilt(train_set_emg_norm19, [11,1])
test_set_joint_norm_filtered = signal.medfilt(train_set_joint_norm19, [5,1])
test_set_joint_norm_filtered_zero = np.zeros([101,7])

# stack emg and joint data as a new matrix
test_set_norm_filtered = np.hstack((test_set_emg_norm_filtered, test_set_joint_norm_filtered_zero))

# add via point as observation
ipromp.add_viapoint(0.00, test_set_norm_filtered[0,:], 0.1)
ipromp.add_viapoint(0.08, test_set_norm_filtered[8,:], 0.1)
# ipromp.add_viapoint(0.18, test_set_norm_filtered[18,:], 0.1)
# ipromp.add_viapoint(0.28, test_set_norm_filtered[28,:], 0.1)
# ipromp.add_viapoint(0.40, test_set_norm_filtered[40,:], 0.1)
# ipromp.add_viapoint(0.50, test_set_norm_filtered[50,:], 0.1)

# plot the updated distribution
plt.figure(4)
for i in range(8):
    plt.subplot(421+i)
    plt.plot(ipromp.x, test_set_emg_norm_filtered[:,i], color='b', linewidth=3)
    ipromp.promps[i].plot_updated(ipromp.x, color='r')
plt.figure(5)
for i in range(7):
    plt.subplot(711+i)
    plt.plot(ipromp.x, test_set_joint_norm_filtered[:,i], color='b', linewidth=3)
    ipromp.promps[8+i].plot_updated(ipromp.x, color='r', via_show=False)

ipromp.add_obsy(t=0.00, obsy=test_set_norm_filtered[0,:], sigmay=0.1)
ipromp.add_obsy(t=0.08, obsy=test_set_norm_filtered[8,:], sigmay=0.1)
prob = ipromp.prob_obs()
print prob

plt.show()