#!/usr/bin/python
# Filename: ndpromp_emg.py

import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as signal
import ipromps

# close the all windows
plt.close('all')

# read data sets from files
train_set00 = np.loadtxt('../../recorder/datasets/emg_data_no_timestamp/emg_data0.txt');
train_set01 = np.loadtxt('../../recorder/datasets/emg_data_no_timestamp/emg_data1.txt');
train_set02 = np.loadtxt('../../recorder/datasets/emg_data_no_timestamp/emg_data2.txt');
train_set03 = np.loadtxt('../../recorder/datasets/emg_data_no_timestamp/emg_data3.txt');
train_set04 = np.loadtxt('../../recorder/datasets/emg_data_no_timestamp/emg_data4.txt');
train_set05 = np.loadtxt('../../recorder/datasets/emg_data_no_timestamp/emg_data5.txt');
train_set06 = np.loadtxt('../../recorder/datasets/emg_data_no_timestamp/emg_data6.txt');
train_set07 = np.loadtxt('../../recorder/datasets/emg_data_no_timestamp/emg_data7.txt');
train_set08 = np.loadtxt('../../recorder/datasets/emg_data_no_timestamp/emg_data8.txt');
train_set09 = np.loadtxt('../../recorder/datasets/emg_data_no_timestamp/emg_data9.txt');
train_set10 = np.loadtxt('../../recorder/datasets/emg_data_no_timestamp/emg_data10.txt');
train_set11 = np.loadtxt('../../recorder/datasets/emg_data_no_timestamp/emg_data11.txt');
train_set12 = np.loadtxt('../../recorder/datasets/emg_data_no_timestamp/emg_data12.txt');
train_set13 = np.loadtxt('../../recorder/datasets/emg_data_no_timestamp/emg_data13.txt');
train_set14 = np.loadtxt('../../recorder/datasets/emg_data_no_timestamp/emg_data14.txt');
train_set15 = np.loadtxt('../../recorder/datasets/emg_data_no_timestamp/emg_data15.txt');
train_set16 = np.loadtxt('../../recorder/datasets/emg_data_no_timestamp/emg_data16.txt');
train_set17 = np.loadtxt('../../recorder/datasets/emg_data_no_timestamp/emg_data17.txt');
train_set18 = np.loadtxt('../../recorder/datasets/emg_data_no_timestamp/emg_data18.txt');
train_set19 = np.loadtxt('../../recorder/datasets/emg_data_no_timestamp/emg_data19.txt');
train_set20 = np.loadtxt('../../recorder/datasets/emg_data_no_timestamp/emg_data20.txt');
train_set21 = np.loadtxt('../../recorder/datasets/emg_data_no_timestamp/emg_data21.txt');
train_set22 = np.loadtxt('../../recorder/datasets/emg_data_no_timestamp/emg_data22.txt');
train_set23 = np.loadtxt('../../recorder/datasets/emg_data_no_timestamp/emg_data23.txt');
train_set24 = np.loadtxt('../../recorder/datasets/emg_data_no_timestamp/emg_data24.txt');
test_set = np.loadtxt('../../recorder/datasets/emg_data_no_timestamp/emg_data31.txt');

# plot the origin data,ch0 is example here
plt.figure(0)
plt.title('the raw training sets',fontsize=23)
plt.xlabel('time',fontsize=20);plt.ylabel('emg signal',fontsize=20)
plt.plot(range(len(train_set00)), train_set00[:,0])
plt.plot(range(len(train_set01)), train_set01[:,0])
plt.plot(range(len(train_set02)), train_set02[:,0])
plt.plot(range(len(train_set03)), train_set03[:,0])
plt.plot(range(len(train_set04)), train_set04[:,0])
plt.plot(range(len(train_set05)), train_set05[:,0])
plt.plot(range(len(train_set06)), train_set06[:,0])
plt.plot(range(len(train_set07)), train_set07[:,0])
plt.plot(range(len(train_set08)), train_set08[:,0])
plt.plot(range(len(train_set09)), train_set09[:,0])
plt.plot(range(len(train_set10)), train_set10[:,0])
plt.plot(range(len(train_set11)), train_set11[:,0])
plt.plot(range(len(train_set12)), train_set12[:,0])
plt.plot(range(len(train_set13)), train_set13[:,0])
plt.plot(range(len(train_set14)), train_set14[:,0])
plt.plot(range(len(train_set15)), train_set15[:,0])
plt.plot(range(len(train_set16)), train_set16[:,0])
plt.plot(range(len(train_set17)), train_set17[:,0])
plt.plot(range(len(train_set18)), train_set18[:,0])
plt.plot(range(len(train_set19)), train_set19[:,0])
plt.plot(range(len(train_set20)), train_set20[:,0])
plt.plot(range(len(train_set21)), train_set21[:,0])
plt.plot(range(len(train_set22)), train_set22[:,0])
plt.plot(range(len(train_set23)), train_set23[:,0])
plt.plot(range(len(train_set24)), train_set24[:,0])

# define the norm traj len
len_normal = 101.0

# norm the traj, by resampling it
train_set_norm00 = np.interp(np.linspace(0, len(train_set00)-1, len_normal), np.arange(0,len(train_set00),1.), train_set00[:,0])
train_set_norm01 = np.interp(np.linspace(0, len(train_set01)-1, len_normal), np.arange(0,len(train_set01),1.), train_set01[:,0])
train_set_norm02 = np.interp(np.linspace(0, len(train_set02)-1, len_normal), np.arange(0,len(train_set02),1.), train_set02[:,0])
train_set_norm03 = np.interp(np.linspace(0, len(train_set03)-1, len_normal), np.arange(0,len(train_set03),1.), train_set03[:,0])
train_set_norm04 = np.interp(np.linspace(0, len(train_set04)-1, len_normal), np.arange(0,len(train_set04),1.), train_set04[:,0])
train_set_norm05 = np.interp(np.linspace(0, len(train_set05)-1, len_normal), np.arange(0,len(train_set05),1.), train_set05[:,0])
train_set_norm06 = np.interp(np.linspace(0, len(train_set06)-1, len_normal), np.arange(0,len(train_set06),1.), train_set06[:,0])
train_set_norm07 = np.interp(np.linspace(0, len(train_set07)-1, len_normal), np.arange(0,len(train_set07),1.), train_set07[:,0])
train_set_norm08 = np.interp(np.linspace(0, len(train_set08)-1, len_normal), np.arange(0,len(train_set08),1.), train_set08[:,0])
train_set_norm09 = np.interp(np.linspace(0, len(train_set09)-1, len_normal), np.arange(0,len(train_set09),1.), train_set09[:,0])
train_set_norm10 = np.interp(np.linspace(0, len(train_set10)-1, len_normal), np.arange(0,len(train_set10),1.), train_set10[:,0])
train_set_norm11 = np.interp(np.linspace(0, len(train_set11)-1, len_normal), np.arange(0,len(train_set11),1.), train_set11[:,0])
train_set_norm12 = np.interp(np.linspace(0, len(train_set12)-1, len_normal), np.arange(0,len(train_set12),1.), train_set12[:,0])
train_set_norm13 = np.interp(np.linspace(0, len(train_set13)-1, len_normal), np.arange(0,len(train_set13),1.), train_set13[:,0])
train_set_norm14 = np.interp(np.linspace(0, len(train_set14)-1, len_normal), np.arange(0,len(train_set14),1.), train_set14[:,0])
train_set_norm15 = np.interp(np.linspace(0, len(train_set15)-1, len_normal), np.arange(0,len(train_set15),1.), train_set15[:,0])
train_set_norm16 = np.interp(np.linspace(0, len(train_set16)-1, len_normal), np.arange(0,len(train_set16),1.), train_set16[:,0])
train_set_norm17 = np.interp(np.linspace(0, len(train_set17)-1, len_normal), np.arange(0,len(train_set17),1.), train_set17[:,0])
train_set_norm18 = np.interp(np.linspace(0, len(train_set18)-1, len_normal), np.arange(0,len(train_set18),1.), train_set18[:,0])
train_set_norm19 = np.interp(np.linspace(0, len(train_set19)-1, len_normal), np.arange(0,len(train_set19),1.), train_set19[:,0])
train_set_norm20 = np.interp(np.linspace(0, len(train_set20)-1, len_normal), np.arange(0,len(train_set20),1.), train_set20[:,0])
train_set_norm21 = np.interp(np.linspace(0, len(train_set21)-1, len_normal), np.arange(0,len(train_set21),1.), train_set21[:,0])
train_set_norm22 = np.interp(np.linspace(0, len(train_set22)-1, len_normal), np.arange(0,len(train_set22),1.), train_set22[:,0])
train_set_norm23 = np.interp(np.linspace(0, len(train_set23)-1, len_normal), np.arange(0,len(train_set23),1.), train_set23[:,0])
train_set_norm24 = np.interp(np.linspace(0, len(train_set24)-1, len_normal), np.arange(0,len(train_set24),1.), train_set24[:,0])
test_set_norm = np.interp(np.linspace(0, len(test_set)-1, len_normal), np.arange(0,len(test_set),1.), test_set[:,0])

# the full training sets
train_set_norm_full = np.array([train_set_norm00,train_set_norm01,train_set_norm02,train_set_norm03,
                                train_set_norm04,train_set_norm05,train_set_norm06,train_set_norm07,
                                train_set_norm08,train_set_norm09,train_set_norm10,train_set_norm11,
                                train_set_norm12,train_set_norm13,train_set_norm14,train_set_norm15,
                                train_set_norm16,train_set_norm17,train_set_norm18,train_set_norm19,
                                train_set_norm20,train_set_norm21,train_set_norm22,train_set_norm23,
                                train_set_norm24]).T

# plot the norm traj           
plt.figure(1)
plt.title('the normalized training sets',fontsize=23)
plt.xlabel('time',fontsize=20);plt.ylabel('emg signal',fontsize=20)
plt.plot(np.arange(0.0, 1.01, 0.01), train_set_norm_full)
plt.show()

# create a ProMP object
p = ipromps.ProMP(nrBasis=11, sigma=0.05, num_samples=len_normal)

# number of trajectoreis for training
nrTraj = len(train_set_norm_full.T) 

# add demonstration
for traj in range(0, nrTraj):
    p.add_demonstration(train_set_norm_full[:,traj])

# median filtered data
test_set_norm_filtered = signal.medfilt(test_set_norm,21)

# plot the trained model and generated traj
plt.figure(2)
plt.title('the generated model from training sets',fontsize=23)
plt.xlabel('time',fontsize=20);plt.ylabel('emg signal',fontsize=20)
p.plot(x=p.x, color='r')
plt.plot(p.x, p.generate_trajectory(), 'r',linewidth=3)

# add via point as observation
p.add_viapoint(0.00, test_set_norm_filtered[0.00*100], 10.0)
p.add_viapoint(0.02, test_set_norm_filtered[0.02*100], 10.0)
p.add_viapoint(0.04, test_set_norm_filtered[0.04*100], 10.0)
p.add_viapoint(0.06, test_set_norm_filtered[0.06*100], 10.0)
p.add_viapoint(0.08, test_set_norm_filtered[0.08*100], 10.0)
p.add_viapoint(0.10, test_set_norm_filtered[0.10*100], 10.0)
p.add_viapoint(0.12, test_set_norm_filtered[0.12*100], 10.0)
p.add_viapoint(0.14, test_set_norm_filtered[0.14*100], 10.0)
p.add_viapoint(0.16, test_set_norm_filtered[0.16*100], 10.0)

# plot the trained model and generated traj
plt.figure(3)
plt.title('the prediction from new observation',fontsize=23)
plt.xlabel('time',fontsize=20);plt.ylabel('emg signal',fontsize=20)
p.plot(x=p.x, color='r', legend='prior distribution')
plt.plot(p.x, test_set_norm, alpha=0.5, label='raw emg data with norm')
plt.plot(p.x, test_set_norm_filtered, 'b', linewidth=3, alpha=0.5, label='emg data with median filter')
plt.plot(p.x, p.generate_trajectory(), 'g',linewidth=6, label='prediction trajectory')
p.plot_unit(x=p.x, color='g')
plt.legend()
plt.show()
