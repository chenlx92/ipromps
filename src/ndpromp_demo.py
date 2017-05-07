#!/usr/bin/python
# Filename: ndpromp_demo.py

import numpy as np
import matplotlib.pyplot as plt
import ipromps

# close the all windows
plt.close('all')

train_set = np.loadtxt('../datasets/TrainingSet.txt');
test_set = np.loadtxt('../datasets/TestingSet.txt');

# create a ProMP object
ndpromp = ipromps.NDProMP(num_joints=2, nrBasis=30)

# Generate and plot trajectory Data
x = np.arange(0,1.01,0.01)           # time points for trajectories
nrDemo=50                            # number of demo
sigmaNoise=0.02                      # noise on training trajectories
len_traj = 125.0
len_normal = 101.0

# add demonstration
for traj in range(0, nrDemo):
    # resampling
    resampling_x = np.linspace(0, len_traj-1, len_normal)
    train_sets_normal_1st = np.interp(resampling_x, np.arange(0,len_traj,1.), train_set[:,2*traj]);
    train_sets_normal_2rd = np.interp(resampling_x, np.arange(0,len_traj,1.), train_set[:,2*traj+1]);
    samples = np.array([ train_sets_normal_1st, train_sets_normal_2rd ]).T
    
    ndpromp.add_demonstration(samples)
    
    # plot the training set
    plt.figure(0)
    label = 'training sets 1st dim' if traj==0 else ''
    plt.plot(x, samples[:,0], 'grey', label=label, alpha=0.3)
    plt.legend()
    
    plt.figure(1)
    label = 'training sets 2nd dim' if traj==0 else ''
    plt.plot(x, samples[:,1], 'grey', label=label, alpha=0.3)
    plt.legend()
    
    plt.figure(2)
    label = 'training sets whole traj' if traj==0 else ''
    plt.plot(samples[:,0], samples[:,1], 'grey', label=label, alpha=0.3)
    plt.legend()

plt.figure(0)
ndpromp.promps[0].plot(x)
plt.figure(1)
ndpromp.promps[1].plot(x)

# testing set
#for traj in range(0, nrDemo):
#    # resampling
#    resampling_x = np.linspace(0, len_traj-1, len_normal)
#    test_sets_normal_1st = np.interp(resampling_x, np.arange(0,len_traj,1.), test_set[:,2*traj]);
#    test_sets_normal_2rd = np.interp(resampling_x, np.arange(0,len_traj,1.), test_set[:,2*traj+1]);
#    samples_test = np.array([ test_sets_normal_1st, test_sets_normal_2rd ]).T
#    
#    plt.figure(5)
#    label = 'testing sets 1st dim' if traj==0 else ''
#    plt.plot(x, samples_test[:,0], 'grey', label=label, alpha=0.3)
#    plt.legend()


# add via point as observation
ndpromp.add_viapoint(0.1, np.array([0.12, 0.30]))
ndpromp.add_viapoint(0.2, np.array([0.37, 0.65]))
ndpromp.add_viapoint(0.4, np.array([0.48, 0.65]))

# plot the updated distributions
plt.figure(0)
ndpromp.promps[0].plot_updated(ndpromp.x, color='r')
plt.figure(1)
ndpromp.promps[1].plot_updated(ndpromp.x, color='r')

# plot
plt.legend()
plt.show()

