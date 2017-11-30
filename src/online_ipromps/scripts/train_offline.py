#!/usr/bin/python

import numpy as np
import matplotlib.pyplot as plt
import ipromps_lib
import scipy.linalg
from sklearn.externals import joblib
from sklearn import preprocessing
import scipy.stats as stats
import pylab as pl

# ipromps model
num_joints = 28
num_demos = 20
num_basis = 31
sigma_basis = 0.05
num_samples = 101
num_obs_joints = 12

# measurement noise
imu_noise = 1.0
emg_noise = 2.0
pose_noise = 1.0

# phase estimation para
num_alpha_candidate = 10
nominal_duration = 1.0
nominal_interval = nominal_duration / (num_samples-1)
states_rate = 50.0

# the measurement noise cov matrix
meansurement_noise_cov_full = scipy.linalg.block_diag(np.eye(8) * imu_noise,
                                                      np.eye(3) * emg_noise,
                                                      np.eye(17) * pose_noise)


# load norm date sets
datasets_norm = joblib.load('../datasets/pkl/handover_20171128/datasets_norm.pkl')
datasets4train = [x[0:num_demos] for x in datasets_norm]

# create a 3 tasks iProMPs
ipromps_set = [ipromps_lib.IProMP(num_joints=num_joints, num_basis=num_basis, sigma_basis=sigma_basis,
                                  num_samples=num_samples, num_obs_joints=num_obs_joints,
                                  sigmay=meansurement_noise_cov_full)]*len(datasets4train)

# add demostration and alpha var for each IProMPs
for idx, task_idx in enumerate(ipromps_set):
    print('training the task %d IProMP'%(idx))
    # for demo_idx in range(num_demos):
    for demo_idx in datasets4train[idx]:
        demo_temp = np.hstack([demo_idx['emg'], demo_idx['left_hand'], demo_idx['robot_joints']]),
        task_idx.add_demonstration(demo_temp[0])    # spital variance demo
        task_idx.add_alpha(demo_idx['alpha'])       # temporal variance demo

# plt.figure(0)
# h = ipromps_set[0].alpha
# h.sort()
# hmean = np.mean(h)
# hstd = np.std(h)
# pdf = stats.norm.pdf(h, hmean, hstd)
# pl.hist(h,normed=True,color='b')
# plt.plot(h, pdf, linewidth=5, color='r', marker='o',markersize=10) # including h here is crucial
# candidate = ipromps_set[0].alpha_candidate(num_alpha_candidate)
# candidate_x = [x['candidate'] for x in candidate]
# prob = [x['prob'] for x in candidate]
# plt.plot(candidate_x, prob, linewidth=0, color='g', marker='o', markersize=14);
# print("the aluminum_hold alpha mean is ", hmean)
# print("the aluminum_hold alpha std is hmean", hstd)
#
# plt.show()```


# save the trained models as pkl
print('saving the trained models')
joblib.dump(ipromps_set, "../trained_models/ipromps_set.pkl")

print('Everyone is happy!!! You trained the IProMPs successfully!')
