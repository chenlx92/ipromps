#!/usr/bin/env python
# run it in terminal: rosrun
# the dir tree is : /dataset/task n/ .rosbag
import glob
import os
import subprocess

path = "../datasets/raw/handover_dataset_20171128/wrench"
all_files = glob.glob(os.path.join(path, "*.bag")) #make list of paths
print(all_files)
for file_idx in all_files:
    subprocess.Popen(['./bag_to_csv.sh ' + file_idx + ' ' +path+'/csv'], shell=True)

print('everyone is happy!!!')