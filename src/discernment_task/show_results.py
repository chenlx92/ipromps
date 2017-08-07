import glob
import os
import sys
import subprocess

path = "./"
dataset_num = 30
for i in range(dataset_num):
    subprocess.Popen(['python imu_emg_pose_test_compact.py -t %s' %str(i)], shell = True)