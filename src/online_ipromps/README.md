# Interaction ProMP

This package is the implement of Interaction ProMP described [here](http://www.ausy.tu-darmstadt.de/uploads/Team/PubGJMaeda/phase_estim_IJRR.pdf).

# Requirements

- python >=2.6
- numpy
- sklearn
- scipy >= 0.19.1

## how to upgrade scipy 
1. Install the gfortran. If neccessory, you should install gfortran-5 for dependency.
2. Upgrade the scipy with "sudo easy_install --upgrade scipy"
3. If fail, maybe you should upgrade the numpy first with "sudo easy_install --upgrade numpy"
ref: https://askubuntu.com/questions/682825/how-to-update-to-the-latest-numpy-and-scipy-on-ubuntu-14-04lts

# examples
`bag_to_csv.sh`: a script to convert single rosbag to csv
`batch_bag2csv.py`: a batch python script to convert rosbag to csv, run it in terminal for some ros shell script
`load_data.py` : load the data from csv file and resample the time sequence as same duration
`train_offline`: train the Interaction ProMPs from the demonstrations
`test_online`: test the trained models 
