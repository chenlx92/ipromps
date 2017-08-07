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
import sys
from optparse import OptionParser

dataset_aluminum_hold = joblib.load('./pkl/dataset_aluminum_hold.pkl')
# dataset_spanner_handover = joblib.load('./pkl/dataset_spanner_handover.pkl')
# dataset_tape_hold = joblib.load('./pkl/dataset_tape_hold.pkl')

plt.figure(0)
for i in range(20):
    x = np.array([0, 1, 2, 3])
    data = dataset_aluminum_hold[i]["imu"][0,:]
    my_xticks = ['quat[0]', 'quat[1]', 'quat[2]', 'quat[2]']
    plt.xticks(x, my_xticks)
    plt.plot(x, data, 'o', markersize=15)
plt.show()