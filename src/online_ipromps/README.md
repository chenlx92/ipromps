# Interaction ProMP

This package is the implement of Interaction ProMP described [here](http://www.ausy.tu-darmstadt.de/uploads/Team/PubGJMaeda/phase_estim_IJRR.pdf).

# Requirements

- python >=2.6
- numpy
- sklearn
- scipy >= 0.19.1

## upgrade scipy 
1. Install the gfortran. If neccessory, you should install gfortran-5 for dependency.
2. Upgrade the scipy with "sudo easy_install --upgrade scipy"
3. If fail, maybe you should upgrade the numpy first with "sudo easy_install --upgrade numpy"
ref: https://askubuntu.com/questions/682825/how-to-update-to-the-latest-numpy-and-scipy-on-ubuntu-14-04lts

# The script descriptioins  
`bag_to_csv.sh`: a script to convert single rosbag to csv  
`batch_bag2csv.py`: a batch python script to convert rosbag to csv, run it in terminal for some ros shell script  
`load_data.py` : load the data from csv file and resample the time sequence as same duration  
`train_offline`: train the Interaction ProMPs from the demonstrations 
`test_online`: test the trained models  

# The preparation command to run IProMPs
All command run in **baxter.sh** space.  
1. `roslaunch openni_launch openni.launch`: open the xtion  
2. `roslaunch aruco_hand_eye baxter_xtion_cal_pub.launch`: load the baxter-xtion calibration result  
3. `rosrun openni_tracker openni_tracker`: open the human skeleton tracking node  
4. `roslaunch myo_driver myo_raw_pub.launch`: start the Myo armband node  
5. `rosrun states_manager states_pub.py`: open the state manager node  

# The training set 
note of collection: 
1. check the csv file derived from rosbag 
2. start to record all interest data and demonstrate the collaborative task motion (human and robot) in the same time 
3. demonstrate with spatio-temporal variance 
4. no occlusion in demonstration,  check the demo human tracking data 
5. no robot orientation change in simple test
6. take photos of armband wear position, object handle position and record training process video

option of improvement:
1. use robot joint/cartesian space/twist
2. consider the EMG signals' effect (correlate it with robot motion?)
3. design the task with different motion of robot arm, human hand, elbow, shoulder...
4. use double robot arm

