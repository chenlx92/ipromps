#!/usr/bin/python
# Filename: ndpromp_emg.py

import numpy as np
import pandas as pds
import matplotlib.pyplot as plt
import scipy.signal as signal
import ipromps

# close the all windows
plt.close('all')

# read txt
dir_prefix = '../../recorder/datasets/joint_emg/hold/csv/'
train_set0_pd = pds.read_csv(dir_prefix + '2017-05-27-11-44-31-myo_raw_pub.csv')
train_set1_pd = pds.read_csv(dir_prefix + '2017-05-27-11-44-56-myo_raw_pub.csv')
train_set2_pd = pds.read_csv(dir_prefix + '2017-05-27-11-45-20-myo_raw_pub.csv')
train_set3_pd = pds.read_csv(dir_prefix + '2017-05-27-11-46-31-myo_raw_pub.csv')
train_set4_pd = pds.read_csv(dir_prefix + '2017-05-27-11-46-55-myo_raw_pub.csv')
train_set5_pd = pds.read_csv(dir_prefix + '2017-05-27-11-47-26-myo_raw_pub.csv')
train_set6_pd = pds.read_csv(dir_prefix + '2017-05-27-11-47-53-myo_raw_pub.csv')
train_set7_pd = pds.read_csv(dir_prefix + '2017-05-27-11-48-15-myo_raw_pub.csv')
train_set8_pd = pds.read_csv(dir_prefix + '2017-05-27-11-48-36-myo_raw_pub.csv')
train_set9_pd = pds.read_csv(dir_prefix + '2017-05-27-11-48-59-myo_raw_pub.csv')
train_set10_pd = pds.read_csv(dir_prefix + '2017-05-27-11-49-27-myo_raw_pub.csv')
train_set11_pd = pds.read_csv(dir_prefix + '2017-05-27-11-49-55-myo_raw_pub.csv')
train_set12_pd = pds.read_csv(dir_prefix + '2017-05-27-11-50-19-myo_raw_pub.csv')
train_set13_pd = pds.read_csv(dir_prefix + '2017-05-27-11-50-42-myo_raw_pub.csv')
train_set14_pd = pds.read_csv(dir_prefix + '2017-05-27-11-51-10-myo_raw_pub.csv')
train_set15_pd = pds.read_csv(dir_prefix + '2017-05-27-11-51-32-myo_raw_pub.csv')
train_set16_pd = pds.read_csv(dir_prefix + '2017-05-27-11-51-59-myo_raw_pub.csv')
train_set17_pd = pds.read_csv(dir_prefix + '2017-05-27-11-52-23-myo_raw_pub.csv')
train_set18_pd = pds.read_csv(dir_prefix + '2017-05-27-11-52-48-myo_raw_pub.csv')
train_set19_pd = pds.read_csv(dir_prefix + '2017-05-27-11-53-09-myo_raw_pub.csv')
test_set_pd = pds.read_csv(dir_prefix + '2017-05-27-11-53-38-myo_raw_pub.csv')

train_set0 = float32(train_set0_pd.values[:,5:12])
train_set1 = float32(train_set1_pd.values[:,5:12])
train_set2 = float32(train_set2_pd.values[:,5:12])
train_set3 = float32(train_set3_pd.values[:,5:12])
train_set4 = float32(train_set4_pd.values[:,5:12])
train_set5 = float32(train_set5_pd.values[:,5:12])
train_set6 = float32(train_set6_pd.values[:,5:12])
train_set7 = float32(train_set7_pd.values[:,5:12])
train_set8 = float32(train_set8_pd.values[:,5:12])
train_set9 = float32(train_set9_pd.values[:,5:12])
train_set10 = float32(train_set10_pd.values[:,5:12])
train_set11 = float32(train_set11_pd.values[:,5:12])
train_set12 = float32(train_set12_pd.values[:,5:12])
train_set13 = float32(train_set13_pd.values[:,5:12])
train_set14 = float32(train_set14_pd.values[:,5:12])
train_set15 = float32(train_set15_pd.values[:,5:12])
train_set16 = float32(train_set16_pd.values[:,5:12])
train_set17 = float32(train_set17_pd.values[:,5:12])
train_set18 = float32(train_set18_pd.values[:,5:12])
train_set19 = float32(train_set19_pd.values[:,5:12])
test_set = float32(test_set_pd.values[:,5:12])

# plot the origin data,ch0
plt.figure(0)
ch_ex = 2
plt.plot(range(len(train_set0)), train_set0[:,ch_ex])
plt.plot(range(len(train_set1)), train_set1[:,ch_ex])
plt.plot(range(len(train_set2)), train_set2[:,ch_ex])
plt.plot(range(len(train_set3)), train_set3[:,ch_ex])
plt.plot(range(len(train_set4)), train_set4[:,ch_ex])
plt.plot(range(len(train_set5)), train_set5[:,ch_ex])
plt.plot(range(len(train_set6)), train_set6[:,ch_ex])
plt.plot(range(len(train_set7)), train_set7[:,ch_ex])
plt.plot(range(len(train_set8)), train_set8[:,ch_ex])
plt.plot(range(len(train_set9)), train_set9[:,ch_ex])
plt.plot(range(len(train_set10)), train_set10[:,ch_ex])
plt.plot(range(len(train_set11)), train_set11[:,ch_ex])
plt.plot(range(len(train_set12)), train_set12[:,ch_ex])
plt.plot(range(len(train_set13)), train_set13[:,ch_ex])
plt.plot(range(len(train_set14)), train_set14[:,ch_ex])
plt.plot(range(len(train_set15)), train_set15[:,ch_ex])
plt.plot(range(len(train_set16)), train_set16[:,ch_ex])
plt.plot(range(len(train_set17)), train_set17[:,ch_ex])
plt.plot(range(len(train_set18)), train_set18[:,ch_ex])
plt.plot(range(len(train_set19)), train_set19[:,ch_ex])
plt.plot(range(len(test_set)), test_set[:,ch_ex])

len_normal = 101.0
resampling_x = np.linspace(0, len(train_set0)-1, len_normal)

# norm the traj
train_set_norm0 = np.interp(resampling_x, np.arange(0,len(train_set0),1.), train_set0[:,ch_ex])
train_set_norm1 = np.interp(resampling_x, np.arange(0,len(train_set1),1.), train_set1[:,ch_ex])
train_set_norm2 = np.interp(resampling_x, np.arange(0,len(train_set2),1.), train_set2[:,ch_ex])
train_set_norm3 = np.interp(resampling_x, np.arange(0,len(train_set3),1.), train_set3[:,ch_ex])
train_set_norm4 = np.interp(resampling_x, np.arange(0,len(train_set4),1.), train_set4[:,ch_ex])
train_set_norm5 = np.interp(resampling_x, np.arange(0,len(train_set5),1.), train_set5[:,ch_ex])
train_set_norm6 = np.interp(resampling_x, np.arange(0,len(train_set6),1.), train_set6[:,ch_ex])
train_set_norm7 = np.interp(resampling_x, np.arange(0,len(train_set7),1.), train_set7[:,ch_ex])
train_set_norm8 = np.interp(resampling_x, np.arange(0,len(train_set8),1.), train_set8[:,ch_ex])
train_set_norm9 = np.interp(resampling_x, np.arange(0,len(train_set9),1.), train_set9[:,ch_ex])
train_set_norm10 = np.interp(resampling_x, np.arange(0,len(train_set10),1.), train_set10[:,ch_ex])
train_set_norm11 = np.interp(resampling_x, np.arange(0,len(train_set11),1.), train_set11[:,ch_ex])
train_set_norm12 = np.interp(resampling_x, np.arange(0,len(train_set12),1.), train_set12[:,ch_ex])
train_set_norm13 = np.interp(resampling_x, np.arange(0,len(train_set13),1.), train_set13[:,ch_ex])
train_set_norm14 = np.interp(resampling_x, np.arange(0,len(train_set14),1.), train_set14[:,ch_ex])
train_set_norm15 = np.interp(resampling_x, np.arange(0,len(train_set15),1.), train_set15[:,ch_ex])
train_set_norm16 = np.interp(resampling_x, np.arange(0,len(train_set16),1.), train_set16[:,ch_ex])
train_set_norm17 = np.interp(resampling_x, np.arange(0,len(train_set17),1.), train_set17[:,ch_ex])
train_set_norm18 = np.interp(resampling_x, np.arange(0,len(train_set18),1.), train_set18[:,ch_ex])
train_set_norm19 = np.interp(resampling_x, np.arange(0,len(train_set19),1.), train_set19[:,ch_ex])
test_set_norm = np.interp(resampling_x, np.arange(0,len(test_set),1.), test_set[:,ch_ex])

train_set_norm_full = np.array([train_set_norm0,train_set_norm1,train_set_norm2,train_set_norm3,
                                train_set_norm4,train_set_norm5,train_set_norm6,train_set_norm7,
                                train_set_norm8,train_set_norm9,train_set_norm10,train_set_norm11,
                                train_set_norm12,train_set_norm13,train_set_norm14,train_set_norm15,
                                train_set_norm16,train_set_norm17,train_set_norm18,train_set_norm19]).T

# plot the norm traj           
plt.figure(1)
plt.plot(resampling_x, train_set_norm_full)

# create a ProMP object
p = ipromps.ProMP(nrBasis=11, sigma=0.05, num_samples=len_normal)

# number of trajectoreis for training
nrTraj = len(train_set_norm_full.T) 

# add demonstration
for traj in range(0, nrTraj):
    p.add_demonstration(train_set_norm_full[:,traj])

# gaussian filtered data
train_set_norm11_filtered = signal.medfilt(test_set_norm,11)

# plot the trained model and generated traj
plt.figure(2)
p.plot(x=p.x, color='r')
plt.plot(p.x, p.generate_trajectory(), 'g',linewidth=3)

# add via point as observation
#p.add_viapoint(0.00, train_set_norm11_filtered[0.00*100], 10.0)
#p.add_viapoint(0.05, train_set_norm11_filtered[0.05*100], 10.0)
#p.add_viapoint(0.10, train_set_norm11_filtered[0.10*100], 10.0)
#p.add_viapoint(0.15, train_set_norm11_filtered[0.15*100], 10.0)
#p.add_viapoint(0.25, train_set_norm11_filtered[0.25*100], 10.0)
#p.add_viapoint(0.30, train_set_norm11_filtered[0.30*100], 10.0)
#p.add_viapoint(0.35, train_set_norm11_filtered[0.35*100], 10.0)
#p.add_viapoint(0.40, train_set_norm11_filtered[0.40*100], 10.0)
#p.add_viapoint(0.45, train_set_norm11_filtered[0.45*100], 10.0)
#p.add_viapoint(0.50, train_set_norm11_filtered[0.50*100], 10.0)
#p.add_viapoint(0.55, train_set_norm11_filtered[0.55*100], 10.0)
#p.add_viapoint(0.60, train_set_norm11_filtered[0.60*100], 10.0)
#p.add_viapoint(0.65, train_set_norm11_filtered[0.65*100], 10.0)
p.add_viapoint(0.00, train_set_norm11_filtered[0.00*100], 10.0)
p.add_viapoint(0.02, train_set_norm11_filtered[0.02*100], 10.0)
p.add_viapoint(0.04, train_set_norm11_filtered[0.04*100], 10.0)
p.add_viapoint(0.06, train_set_norm11_filtered[0.06*100], 10.0)
p.add_viapoint(0.08, train_set_norm11_filtered[0.08*100], 10.0)
p.add_viapoint(0.10, train_set_norm11_filtered[0.10*100], 10.0)
p.add_viapoint(0.12, train_set_norm11_filtered[0.12*100], 10.0)
p.add_viapoint(0.14, train_set_norm11_filtered[0.14*100], 10.0)
p.add_viapoint(0.16, train_set_norm11_filtered[0.16*100], 10.0)



# plot the trained model and generated traj
plt.figure(3)
p.plot(x=p.x, color='r')
#p.plot_unit(x=p.x, color='r')
#p.plot_updated(x=p.x)
plt.plot(p.x, train_set_norm11)
plt.plot(p.x, p.generate_trajectory(), 'g',linewidth=3)


#plt.figure(4)
plt.plot(p.x, train_set_norm11_filtered,'blue',linewidth=4)

# show the plot
#plt.legend()
plt.show()
