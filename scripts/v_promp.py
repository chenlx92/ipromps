#!/usr/bin/python
import numpy as np
import ipromps_lib
from sklearn.externals import joblib
import os
import ConfigParser
import matplotlib.pyplot as plt

# the current file path
file_path = os.path.dirname(__file__)

# read models cfg file
cp_models = ConfigParser.SafeConfigParser()
cp_models.read(os.path.join(file_path, '../cfg/models.cfg'))
# read models params
datasets_path = os.path.join(file_path, cp_models.get('datasets', 'path'))
len_norm = cp_models.getint('datasets', 'len_norm')
num_basis = cp_models.getint('basisFunc', 'num_basisFunc')
sigma_basis = cp_models.getfloat('basisFunc', 'sigma_basisFunc')
datasets_norm_preproc = joblib.load(os.path.join(datasets_path, 'pkl/datasets_norm_preproc.pkl'))

num_demo = 20

promp = ipromps_lib.ProMP()
for idx_demo in range(num_demo):
    promp.add_demonstration(datasets_norm_preproc[0][idx_demo]['left_joints'][:,0])
promp.plot_prior(b_regression=False, linewidth_mean=5)

promp.add_viapoint(0.1, datasets_norm_preproc[0][2]['left_joints'][10,0])
promp.add_viapoint(0.2, datasets_norm_preproc[0][2]['left_joints'][20,0])
promp.add_viapoint(0.3, datasets_norm_preproc[0][2]['left_joints'][30,0])
promp.add_viapoint(0.4, datasets_norm_preproc[0][2]['left_joints'][40,0])
promp.add_viapoint(0.5, datasets_norm_preproc[0][2]['left_joints'][50,0])
promp.gen_uTrajectory()
promp.plot_uUpdated()

plt.plot(promp.x, datasets_norm_preproc[0][2]['left_joints'][:,0], color='g', linewidth=5)

plt.show()

