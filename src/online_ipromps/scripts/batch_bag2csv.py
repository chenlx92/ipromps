#!/usr/bin/env python
# run it in terminal:
# i.e. python batch_bag2csv.py -d ../datasets/handover_20171128/raw/
#   or rosrun ipromps batch_bag2csv.py -d ../datasets/handover_20171128/raw/

import glob
import os
import subprocess
from optparse import OptionParser


parser = OptionParser()
parser.add_option("-d", "--ddbk", action="store",
                  dest="dataset_dir",
                  default=None,
                  help="the directory of dataset")
(options, args) = parser.parse_args()
path = options.dataset_dir


if path is None:
    print('please input the datasets directory.')

else:
    task_list = glob.glob(os.path.join(path, "*"))
    for task_list_idx in task_list:
        file_list = glob.glob(os.path.join(task_list_idx, "*.bag"))
        if file_list == []:
            print('No rosbag found. ')
            continue
        for file_idx in file_list:
            subprocess.Popen(['./bag_to_csv.sh ' + file_idx + ' ' + task_list_idx + '/csv'], shell=True)