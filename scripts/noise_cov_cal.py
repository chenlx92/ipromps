#!/usr/bin/python
import numpy as np
import pandas as pd
import os
from sklearn.externals import joblib
import ConfigParser

# read conf file
file_path = os.path.dirname(__file__)
cp = ConfigParser.SafeConfigParser()
cp.read(os.path.join(file_path, '../cfg/params.cfg'))
# the datasets path
datasets_path = os.path.join(file_path, cp.get('datasets', 'path'))


def main():
    # read csv file
    csv_path = os.path.join(datasets_path, 'info/noise/multiModal_states.csv')
    data = pd.read_csv(csv_path)

    # extract the all signals data
    # emg = data.values[:, 7:15].astype(float)
    left_hand = data.values[:, 207:210].astype(float)
    left_joints = data.values[:, 317:320].astype(float)     # robot ee actually
    # left_joints = data.values[:, 99:106].astype(float)

    # stack them as a big matrix
    # full_data = np.hstack([emg, left_hand, left_joints])
    full_data = np.hstack([left_hand, left_joints])

    # compute the noise observation covariance matrix
    noise_cov = np.cov(full_data.T)

    # save it in pkl
    joblib.dump(noise_cov, os.path.join(datasets_path, 'pkl/noise_cov.pkl'))

    print('Saved the noise covariance matrix successfully!')


if __name__ == '__main__':
    main()
