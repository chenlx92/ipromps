#!/usr/bin/python
# Filename: load_data.py

from __future__ import print_function

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.interpolate import griddata
from sklearn.externals import joblib

len_normal = 101    # the len of normalized traj
dir_init = '../../../recorder/datasets/imu_emg_joint_pose_3_task/'

#################################################
# load raw datasets
#################################################
# load aluminum hold dateset
print('loading aluminum hold dataset')
dir_aluminum_hold_prefix = dir_init + 'aluminum_hold/csv/'
file_prefix_aluminum_hold = [line.strip() for line in open(dir_aluminum_hold_prefix+'filename_aluminum_hold.cfg', 'r')]
dataset_aluminum_hold = []
for i in range(len(file_prefix_aluminum_hold)):
    emg = pd.read_csv(dir_aluminum_hold_prefix + file_prefix_aluminum_hold[i] + 'myo_raw_emg_pub.csv')
    imu = pd.read_csv(dir_aluminum_hold_prefix + file_prefix_aluminum_hold[i] + 'myo_raw_imu_pub.csv')
    pose = pd.read_csv(dir_aluminum_hold_prefix + file_prefix_aluminum_hold[i] + 'robot-limb-left-endpoint_state.csv')
    # saved as list -->dict -->ndarray
    dataset_aluminum_hold.append({"emg":emg.values[:, 5:13], "imu":imu.values[:, 5:9], "pose":pose.values[:, 5:12]})

# load spanner handover dateset
print('loading spanner handover dataset')
# read emg csv files of aluminum_hold
dir_spanner_handover_prefix = dir_init + 'spanner_handover/csv/'
file_prefix_spanner_handover = [line.strip() for line in open(dir_spanner_handover_prefix+'filename_spanner_handover.cfg', 'r')]
dataset_spanner_handover = []
for i in range(len(file_prefix_spanner_handover)):
    emg = pd.read_csv(dir_spanner_handover_prefix + file_prefix_spanner_handover[i] + 'myo_raw_emg_pub.csv')
    imu = pd.read_csv(dir_spanner_handover_prefix + file_prefix_spanner_handover[i] + 'myo_raw_imu_pub.csv')
    pose = pd.read_csv(dir_spanner_handover_prefix + file_prefix_spanner_handover[i] + 'robot-limb-left-endpoint_state.csv')
    # saved as list -->dict -->ndarray
    dataset_spanner_handover.append({"emg":emg.values[:, 5:13], "imu":imu.values[:, 5:9], "pose":pose.values[:, 5:12]})

# load tape hold date set
print('loading tape hold dataset')
dir_tape_hold_prefix = dir_init + 'tape_hold/csv/'
file_prefix_tape_hold = [line.strip() for line in open(dir_tape_hold_prefix+'filename_tape_hold.cfg', 'r')]
dataset_tape_hold = []
for i in range(len(file_prefix_tape_hold)):
    emg = pd.read_csv(dir_tape_hold_prefix + file_prefix_tape_hold[i] + 'myo_raw_emg_pub.csv')
    imu = pd.read_csv(dir_tape_hold_prefix + file_prefix_tape_hold[i] + 'myo_raw_imu_pub.csv')
    pose = pd.read_csv(dir_tape_hold_prefix + file_prefix_tape_hold[i] + 'robot-limb-left-endpoint_state.csv')
    # saved as list -->dict -->ndarray
    dataset_tape_hold.append({"emg":emg.values[:, 5:13], "imu":imu.values[:, 5:9], "pose":pose.values[:, 5:12]})

# joblib.dump(dataset_aluminum_hold, "dataset_aluminum_hold.pkl")


#################################################
# resampling the dataset for the same duration
#################################################
print('resampling the aluminum hold dataset for the same duration')
dataset_aluminum_hold_norm = []
for i in range(len(file_prefix_aluminum_hold)):
    ## resampling for emg
    emg_points = np.arange(len(dataset_aluminum_hold[i]["emg"]))
    emg_grid = np.linspace(0, len(dataset_aluminum_hold[i]["emg"])-1, len_normal)
    emg_norm = griddata(emg_points, dataset_aluminum_hold[i]["emg"], emg_grid, method='linear')
    ## resampling for imu
    imu_points = np.arange(len(dataset_aluminum_hold[i]["imu"]))
    imu_grid = np.linspace(0, len(dataset_aluminum_hold[i]["imu"]) - 1, len_normal)
    imu_norm = griddata(imu_points, dataset_aluminum_hold[i]["imu"], imu_grid, method='linear')
    ## resampling for emg
    pose_points = np.arange(len(dataset_aluminum_hold[i]["pose"]))
    pose_grid = np.linspace(0, len(dataset_aluminum_hold[i]["pose"]) - 1, len_normal)
    pose_norm = griddata(pose_points, dataset_aluminum_hold[i]["pose"], pose_grid, method='linear')
    # saved as the list-->dict-->ndarray
    dataset_aluminum_hold_norm.append({"emg":emg_norm, "imu":imu_norm, "pose":pose_norm})

print('resampling the spanner handover dataset for the same duration')
dataset_spanner_handover_norm = []
for i in range(len(file_prefix_spanner_handover)):
    ## resampling for emg
    emg_points = np.arange(len(dataset_spanner_handover[i]["emg"]))
    emg_grid = np.linspace(0, len(dataset_spanner_handover[i]["emg"])-1, len_normal)
    emg_norm = griddata(emg_points, dataset_spanner_handover[i]["emg"], emg_grid, method='linear')
    ## resampling for imu
    imu_points = np.arange(len(dataset_spanner_handover[i]["imu"]))
    imu_grid = np.linspace(0, len(dataset_spanner_handover[i]["imu"]) - 1, len_normal)
    imu_norm = griddata(imu_points, dataset_spanner_handover[i]["imu"], imu_grid, method='linear')
    ## resampling for emg
    pose_points = np.arange(len(dataset_spanner_handover[i]["pose"]))
    pose_grid = np.linspace(0, len(dataset_spanner_handover[i]["pose"]) - 1, len_normal)
    pose_norm = griddata(pose_points, dataset_spanner_handover[i]["pose"], pose_grid, method='linear')
    # saved as the list-->dict-->ndarray
    dataset_spanner_handover_norm.append({"emg":emg_norm, "imu":imu_norm, "pose":pose_norm})

print('resampling the tape hold dataset for the same duration')
dataset_tape_hold_norm = []
for i in range(len(file_prefix_tape_hold)):
    ## resampling for emg
    emg_points = np.arange(len(dataset_tape_hold[i]["emg"]))
    emg_grid = np.linspace(0, len(dataset_tape_hold[i]["emg"])-1, len_normal)
    emg_norm = griddata(emg_points, dataset_tape_hold[i]["emg"], emg_grid, method='linear')
    ## resampling for imu
    imu_points = np.arange(len(dataset_tape_hold[i]["imu"]))
    imu_grid = np.linspace(0, len(dataset_tape_hold[i]["imu"]) - 1, len_normal)
    imu_norm = griddata(imu_points, dataset_tape_hold[i]["imu"], imu_grid, method='linear')
    ## resampling for emg
    pose_points = np.arange(len(dataset_tape_hold[i]["pose"]))
    pose_grid = np.linspace(0, len(dataset_tape_hold[i]["pose"]) - 1, len_normal)
    pose_norm = griddata(pose_points, dataset_tape_hold[i]["pose"], pose_grid, method='linear')
    # saved as the list-->dict-->ndarray
    dataset_tape_hold_norm.append({"emg":emg_norm, "imu":imu_norm, "pose":pose_norm})


#################################################
# save the dataset as pkl
#################################################
## the raw datasets
joblib.dump(dataset_aluminum_hold, "./pkl/dataset_aluminum_hold.pkl")
joblib.dump(dataset_spanner_handover, "./pkl/dataset_spanner_handover.pkl")
joblib.dump(dataset_tape_hold, "./pkl/dataset_tape_hold.pkl")
# the normalized dataset
joblib.dump(dataset_aluminum_hold_norm, "./pkl/dataset_aluminum_hold_norm.pkl")
joblib.dump(dataset_spanner_handover_norm, "./pkl/dataset_spanner_handover_norm.pkl")
joblib.dump(dataset_tape_hold_norm, "./pkl/dataset_tape_hold_norm.pkl")


# plt.figure(0)
# plt.plot(range(len(dataset_aluminum_hold_norm[0]["emg"])), dataset_aluminum_hold_norm[0]["emg"][:, 1])
# plt.show()

