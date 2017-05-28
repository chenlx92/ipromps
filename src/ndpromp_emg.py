#!/usr/bin/python
# Filename: ndpromp_emg.py

import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as signal
import ipromps

# close the all windows
plt.close('all')

# read txt
train_set0 = np.loadtxt('../../recorder/datasets/emg_data_no_timestamp/emg_data0.txt');
train_set1 = np.loadtxt('../../recorder/datasets/emg_data_no_timestamp/emg_data1.txt');
train_set2 = np.loadtxt('../../recorder/datasets/emg_data_no_timestamp/emg_data2.txt');
train_set3 = np.loadtxt('../../recorder/datasets/emg_data_no_timestamp/emg_data3.txt');
train_set4 = np.loadtxt('../../recorder/datasets/emg_data_no_timestamp/emg_data4.txt');
train_set5 = np.loadtxt('../../recorder/datasets/emg_data_no_timestamp/emg_data5.txt');
train_set6 = np.loadtxt('../../recorder/datasets/emg_data_no_timestamp/emg_data6.txt');
train_set7 = np.loadtxt('../../recorder/datasets/emg_data_no_timestamp/emg_data7.txt');
train_set8 = np.loadtxt('../../recorder/datasets/emg_data_no_timestamp/emg_data8.txt');
train_set9 = np.loadtxt('../../recorder/datasets/emg_data_no_timestamp/emg_data9.txt');
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


test_set = np.loadtxt('../../recorder/datasets/emg_data_no_timestamp/emg_data31.txt');

# plot the origin data,ch0
plt.figure(0)
plt.plot(range(len(train_set0)), train_set0[:,0])
plt.plot(range(len(train_set1)), train_set1[:,0])
plt.plot(range(len(train_set2)), train_set2[:,0])
plt.plot(range(len(train_set3)), train_set3[:,0])
plt.plot(range(len(train_set4)), train_set4[:,0])
plt.plot(range(len(train_set5)), train_set5[:,0])
plt.plot(range(len(train_set6)), train_set6[:,0])
plt.plot(range(len(train_set7)), train_set7[:,0])
plt.plot(range(len(train_set8)), train_set8[:,0])
plt.plot(range(len(train_set9)), train_set9[:,0])
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

# define the norm traj len
len_normal = 101.0
#resampling_x = np.linspace(0, len(train_set0)-1, len_normal)

# norm the traj
train_set_norm0 = np.interp(np.linspace(0, len(train_set0)-1, len_normal), np.arange(0,len(train_set0),1.), train_set0[:,0])
train_set_norm1 = np.interp(np.linspace(0, len(train_set1)-1, len_normal), np.arange(0,len(train_set1),1.), train_set1[:,0])
train_set_norm2 = np.interp(np.linspace(0, len(train_set2)-1, len_normal), np.arange(0,len(train_set2),1.), train_set2[:,0])
train_set_norm3 = np.interp(np.linspace(0, len(train_set3)-1, len_normal), np.arange(0,len(train_set3),1.), train_set3[:,0])
train_set_norm4 = np.interp(np.linspace(0, len(train_set4)-1, len_normal), np.arange(0,len(train_set4),1.), train_set4[:,0])
train_set_norm5 = np.interp(np.linspace(0, len(train_set5)-1, len_normal), np.arange(0,len(train_set5),1.), train_set5[:,0])
train_set_norm6 = np.interp(np.linspace(0, len(train_set6)-1, len_normal), np.arange(0,len(train_set6),1.), train_set6[:,0])
train_set_norm7 = np.interp(np.linspace(0, len(train_set7)-1, len_normal), np.arange(0,len(train_set7),1.), train_set7[:,0])
train_set_norm8 = np.interp(np.linspace(0, len(train_set8)-1, len_normal), np.arange(0,len(train_set8),1.), train_set8[:,0])
train_set_norm9 = np.interp(np.linspace(0, len(train_set9)-1, len_normal), np.arange(0,len(train_set9),1.), train_set9[:,0])
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
test_set_norm = np.interp(np.linspace(0, len(test_set)-1, len_normal), np.arange(0,len(test_set),1.), test_set[:,0])

train_set_norm_full = np.array([train_set_norm0,train_set_norm1,train_set_norm2,train_set_norm3,
                                train_set_norm4,train_set_norm5,train_set_norm6,train_set_norm7,
                                train_set_norm8,train_set_norm9,train_set_norm10,train_set_norm11,
                                train_set_norm12,train_set_norm13,train_set_norm14,train_set_norm15,
                                train_set_norm16,train_set_norm17,train_set_norm18,train_set_norm19]).T

# plot the norm traj           
plt.figure(1)
plt.plot(np.arange(0.0, 1.01, 0.01), train_set_norm_full)

# create a ProMP object
p = ipromps.ProMP(nrBasis=11, sigma=0.05, num_samples=len_normal)

# number of trajectoreis for training
nrTraj = len(train_set_norm_full.T) 

# add demonstration
for traj in range(0, nrTraj):
    p.add_demonstration(train_set_norm_full[:,traj])

# gaussian filtered data
test_set_norm_filtered = signal.medfilt(test_set_norm,21)

# plot the trained model and generated traj
plt.figure(2)
p.plot(x=p.x, color='r')
plt.plot(p.x, p.generate_trajectory(), 'g',linewidth=3)

# add via point as observation
#p.add_viapoint(0.00, test_set_norm_filtered[0.00*100], 10.0)
#p.add_viapoint(0.05, test_set_norm_filtered[0.05*100], 10.0)
#p.add_viapoint(0.10, test_set_norm_filtered[0.10*100], 10.0)
#p.add_viapoint(0.15, test_set_norm_filtered[0.15*100], 10.0)
#p.add_viapoint(0.25, test_set_norm_filtered[0.25*100], 10.0)
#p.add_viapoint(0.30, test_set_norm_filtered[0.30*100], 10.0)
#p.add_viapoint(0.35, test_set_norm_filtered[0.35*100], 10.0)
#p.add_viapoint(0.40, test_set_norm_filtered[0.40*100], 10.0)
#p.add_viapoint(0.45, test_set_norm_filtered[0.45*100], 10.0)
#p.add_viapoint(0.50, test_set_norm_filtered[0.50*100], 10.0)
#p.add_viapoint(0.55, test_set_norm_filtered[0.55*100], 10.0)
#p.add_viapoint(0.60, test_set_norm_filtered[0.60*100], 10.0)
#p.add_viapoint(0.65, test_set_norm_filtered[0.65*100], 10.0)
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
p.plot(x=p.x, color='r')
#p.plot_unit(x=p.x, color='r')
#p.plot_updated(x=p.x)
plt.plot(p.x, train_set_norm11)
plt.plot(p.x, p.generate_trajectory(), 'g',linewidth=3)


#plt.figure(4)
plt.plot(p.x, test_set_norm_filtered,'blue',linewidth=4)

# show the plot
#plt.legend()
plt.show()
