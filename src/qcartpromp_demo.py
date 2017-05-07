#!/usr/bin/python
# Filename: ndpromp_demo.py

import qcartpromp
import numpy as np
import matplotlib.pyplot as plt

# close the all windows
plt.close('all')

# create a ProMP object
p = qcartpromp.QCartProMP()

# Generate and plot trajectory Data
x = np.arange(0,1.01,0.01)           # time points for trajectories
nrTraj=50                            # number of trajectoreis for training
sigmaNoise=0.02                      # noise on training trajectories
len_traj = 125.0

train_set = np.loadtxt('../datasets/TrainingSet.txt');
test_set = np.loadtxt('../datasets/TestingSet.txt');

# add demonstration
for traj in range(0, nrTraj):    
    resampling_x = np.arange(0, len_traj+len_traj/100, len_traj/100)
    train_set_normal_1st = np.interp(resampling_x, np.arange(0,125.,1.), train_set[:,2*traj]);
    train_set_normal_2rd = np.interp(resampling_x, np.arange(0,125.,1.), train_set[:,2*traj+1]);
    
    samples = np.array([ train_set_normal_1st, train_set_normal_2rd ]).T
    p.add_demonstration(samples)
    label = 'training set' if traj==0 else ''
    plt.figure(1)
    plt.plot(x, samples[:,0], 'grey', label=label)
    plt.figure(2)
    plt.plot(x, samples[:,1], 'grey', label=label)
    plt.figure(3)
    plt.plot(samples[:,0], samples[:,1], 'grey', label=label)

# add via point as observation
p.add_viapoint(0.2, np.array([0.4, 0.6]))

# plot the trained model and generated traj
#p.plot(x=p.x, output_randomess=0.0)

# show the plot
#plt.legend()
#plt.show()