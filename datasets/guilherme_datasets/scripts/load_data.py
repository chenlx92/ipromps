#!/usr/bin/python
import numpy as np
import pandas as pd
import glob
import os
from sklearn.externals import joblib
import scipy.signal as signal
from scipy.ndimage.filters import gaussian_filter1d
import scipy.io as sio


data_mat = sio.loadmat('./../taskProMP.mat')
data = data_mat['taskProMP'][0]

datasets_raw = []
for task_idx in range(len(data)):
    demo_temp = []
    for demo_idx in range(len(data[task_idx])):
        # the info of interest: convert the object to int / float
        demo_temp.append({
            'stamp': demo_idx,
            'left_hand': data[task_idx][demo_idx][0][:, 0:3].astype(float),
            'left_joints': data[task_idx][demo_idx][0][:, 10:13].astype(float)
        })
    datasets_raw.append(demo_temp)





print('Saving the datasets as pkl ...')
joblib.dump(datasets_raw, '../datasets_raw.pkl')
