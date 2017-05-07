#!/usr/bin/python
# Filename: ipromp_demo.py
import ipromp
import numpy as np
import matplotlib.pyplot as plt

# close the current windows
plt.close()
plt.close()
plt.close()
plt.figure(1)

# create a ProMP object
ipromp_demo = ipromp.IProMP(num_joints=3)

# Generate and plot trajectory Data
x = np.arange(0,1.01,0.01)           # time points for trajectories
nrTraj=30                            # number of trajectoreis for training
sigmaNoise=0.02                      # noise on training trajectories
A1 = np.array([.5, .0, .0])    # the weight of different func 
A2 = np.array([.0, .5, .0])    # the weight of different func 
A3 = np.array([.0, .0, .5])    # the weight of different func 
X = np.vstack( (np.sin(5*x), x**3, x ))    # the basis func


# add demonstration
for traj in range(0, nrTraj):
    sample1 = np.dot(A1 + sigmaNoise * np.random.randn(1,3), X)[0]
    sample2 = np.dot(A2 + sigmaNoise * np.random.randn(1,3), X)[0]
    sample3 = np.dot(A3 + sigmaNoise * np.random.randn(1,3), X)[0]
    samples = np.array([sample1, sample2, sample3]).T
    ipromp_demo.add_demonstration(samples)
    label = 'training set' if traj==0 else ''
    plt.plot(x, samples, 'grey', label=label, alpha=0.2)

# add via point as observation
ipromp_demo.add_viapoint(0.33, np.array([0.5+0.02, 0.02+0.02, 0]))
# plot the trained model and generated traj
ipromp_demo.plot(x=ipromp_demo.x)


plt.figure(2)
ipromp_demo.add_viapoint(0.50, np.array([0.32+0.02, 0.07+0.02, 0]))
ipromp_demo.plot(x=ipromp_demo.x)

plt.figure(3)
ipromp_demo.add_viapoint(0.60, np.array([0.07+0.02, 0.11+0.02, 0]))
ipromp_demo.plot(x=ipromp_demo.x)

# show the plot
#plt.legend()
plt.show()
