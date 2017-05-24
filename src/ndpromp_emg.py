#!/usr/bin/python
# Filename: ndpromp_emg.py

import numpy as np
import matplotlib.pyplot as plt
import ipromps

# close the all windows
plt.close('all')

# read txt
train_set0 = np.loadtxt('../../recorder/datasets/emg_data0.txt');
train_set1 = np.loadtxt('../../recorder/datasets/emg_data1.txt');
train_set2 = np.loadtxt('../../recorder/datasets/emg_data2.txt');
train_set3 = np.loadtxt('../../recorder/datasets/emg_data3.txt');
train_set4 = np.loadtxt('../../recorder/datasets/emg_data4.txt');
train_set5 = np.loadtxt('../../recorder/datasets/emg_data5.txt');
train_set6 = np.loadtxt('../../recorder/datasets/emg_data6.txt');
train_set7 = np.loadtxt('../../recorder/datasets/emg_data7.txt');
train_set8 = np.loadtxt('../../recorder/datasets/emg_data8.txt');
train_set9 = np.loadtxt('../../recorder/datasets/emg_data9.txt');
train_set11 = np.loadtxt('../../recorder/datasets/emg_data11.txt');

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

len_normal = 101.0
resampling_x = np.linspace(0, len(train_set0)-1, len_normal)

# norm the traj
train_set_norm0 = np.interp(resampling_x, np.arange(0,len(train_set0),1.), train_set0[:,0])
train_set_norm1 = np.interp(resampling_x, np.arange(0,len(train_set1),1.), train_set1[:,0])
train_set_norm2 = np.interp(resampling_x, np.arange(0,len(train_set2),1.), train_set2[:,0])
train_set_norm3 = np.interp(resampling_x, np.arange(0,len(train_set3),1.), train_set3[:,0])
train_set_norm4 = np.interp(resampling_x, np.arange(0,len(train_set4),1.), train_set4[:,0])
train_set_norm5 = np.interp(resampling_x, np.arange(0,len(train_set5),1.), train_set5[:,0])
train_set_norm6 = np.interp(resampling_x, np.arange(0,len(train_set6),1.), train_set6[:,0])
train_set_norm7 = np.interp(resampling_x, np.arange(0,len(train_set7),1.), train_set7[:,0])
train_set_norm8 = np.interp(resampling_x, np.arange(0,len(train_set8),1.), train_set8[:,0])
train_set_norm9 = np.interp(resampling_x, np.arange(0,len(train_set9),1.), train_set9[:,0])
train_set_norm11 = np.interp(resampling_x, np.arange(0,len(train_set11),1.), train_set11[:,0])

train_set_norm_full = np.array([train_set_norm0,train_set_norm1,train_set_norm2,train_set_norm3,
                                train_set_norm4,train_set_norm5,train_set_norm6,train_set_norm7,
                                train_set_norm8,train_set_norm9]).T

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

# plot the trained model and generated traj
plt.figure(2)
p.plot(x=p.x, color='r')
plt.plot(p.x, p.generate_trajectory(), 'g',linewidth=3)

# add via point as observation
#p.add_viapoint(0.05, 40, 1.0)
p.add_viapoint(0.00, train_set_norm11[0.00*100], 1000.0)
p.add_viapoint(0.05, train_set_norm11[0.05*100], 1000.0)
p.add_viapoint(0.10, train_set_norm11[0.10*100], 1000.0)
p.add_viapoint(0.15, train_set_norm11[0.15*100], 1000.0)
p.add_viapoint(0.25, train_set_norm11[0.25*100], 1000.0)
p.add_viapoint(0.35, train_set_norm11[0.35*100], 1000.0)
#p.add_viapoint(0.40, train_set_norm11[0.40*100],50.0)
#p.add_viapoint(0.45, train_set_norm11[0.45*100],50.0)
#p.add_viapoint(0.50, train_set_norm11[0.50*100],50.0)

# plot the trained model and generated traj
plt.figure(3)
p.plot(x=p.x, color='r')
#p.plot_unit(x=p.x, color='r')
#p.plot_updated(x=p.x)
plt.plot(p.x, train_set_norm11)
#plt.plot(p.x, p.generate_trajectory(), 'g',linewidth=3)

# show the plot
#plt.legend()
plt.show()
