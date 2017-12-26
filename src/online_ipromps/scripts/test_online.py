#!/usr/bin/python

import rospy
from states_manager.msg import multiModal
import numpy as np
import threading
import scipy.signal as signal
import baxter_interface
from baxter_interface import CHECK_VERSION
import time
import sys
import os
from sklearn.externals import joblib
import matplotlib.pyplot as plt

import subprocess
subprocess.call(['speech-dispatcher'])        #start speech dispatcher
subprocess.call(['spd-say', '"your process has finished"'])

path = '/../datasets/handover_20171128/pkl'
num_alpha_candidate = 10
timer_interval = 1
ready_time = 5

###########################


def make_command(line, t):
    """
    cleans a single line of recorded joint positions
    :param line: the line described in a list to process
    :param t: the row index of the array
    :return: the list cmd
    """
    joint_cmd_names = ['left_s0', 'left_s1', 'left_e0', 'left_e1', 'left_w0', 'left_w1', 'left_w2']
    data_line = [line[t][2], line[t][3], line[t][0], line[t][1], line[t][4], line[t][5], line[t][6]]
    command = dict(zip(joint_cmd_names, data_line))
    return command


def fun_timer():
    """
    the timer callback func
    :return:
    """
    rospy.loginfo('Time out!!!')
    global flag_record
    flag_record = False  # stop record the msg
    global ipromps_set, min_max_scaler, obs_data_list, filt_kernel
    rospy.loginfo('The len of observed data is %d', len(obs_data_list))
    obs_data = np.array([]).reshape([0, 18])
    timestamp = []
    for obs_data_list_idx in obs_data_list:
        emg = obs_data_list_idx['emg']
        left_hand = obs_data_list_idx['left_hand']
        left_joints = obs_data_list_idx['left_joints']
        full_data = np.hstack([emg, left_hand, left_joints])
        obs_data = np.vstack([obs_data, full_data])
        timestamp.append(obs_data_list_idx['stamp'])

    # filter the data
    obs_data = signal.medfilt(obs_data, filt_kernel)

    # preprocessing for the data
    obs_data_post_arr = min_max_scaler.transform(obs_data)
    # consider the unobserved info
    obs_data_post_arr[:, 11:19] = 0.0

    # phase estimation
    rospy.loginfo('Phase estimating...')
    alpha_max_list = []
    for ipromp in ipromps_set:
        alpha_temp = ipromp.alpha_candidate(num_alpha_candidate)
        idx_max = ipromp.estimate_alpha(alpha_temp, obs_data_post_arr, timestamp)
        alpha_max_list.append(alpha_temp[idx_max]['candidate'])
        ipromp.set_alpha(alpha_temp[idx_max]['candidate'])

    # task recognition
    rospy.loginfo('Adding via points in each trained model...')
    for task_idx, ipromp in enumerate(ipromps_set):
        for idx in range(len(obs_data_list)):
            ipromp.add_viapoint(obs_data_list[idx]['stamp']/alpha_max_list[task_idx], obs_data_post_arr[idx, :])
        ipromp.param_update(unit_update=True)
    rospy.loginfo('Computing the likelihood for each model under observations...')

    prob_task = []
    for ipromp in ipromps_set:
        prob_task_temp = ipromp.prob_obs()
        prob_task.append(prob_task_temp)
    idx_max_pro = np.argmax(prob_task)
    # idx_max_pro = 2
    rospy.loginfo('The max fit model index is task %d', idx_max_pro)

    # robot motion generation
    [traj_time, traj] = ipromps_set[idx_max_pro].gen_real_traj(alpha_max_list[idx_max_pro])
    traj = min_max_scaler.inverse_transform(traj)
    robot_traj = traj[:, 11:18]

    # save the robot traj
    rospy.loginfo('Saving the robot traj...')
    joblib.dump(robot_traj, current_path+path+'/robot_traj.pkl')

    # robot start point
    global left
    rospy.loginfo('Moving to start position...')
    left_start = make_command(robot_traj, 0)
    print(left_start)
    left.move_to_joint_positions(left_start)

    # move the robot along the trajectory
    rospy.loginfo('Moving along the trajectory...')
    start_time = rospy.get_time()
    for t in range(len(traj_time)):
        l_cmd = make_command(robot_traj, t)
        while (rospy.get_time()-start_time) < traj_time[t]:
            left.set_joint_positions(l_cmd)
    rospy.loginfo('The whole trajectory has been run!')

    # save the conditional result
    rospy.loginfo('Saving the post IProMPs...')
    joblib.dump(ipromps_set, current_path+path+'/ipromps_set_post.pkl')

    rospy.loginfo('All finished!!!')


def callback(data):
    global flag_record
    if not flag_record:
        return

    global init_time
    if init_time is None:
        init_time = data.header.stamp

    # emg
    emg_data = np.array([data.emgStates.ch0, data.emgStates.ch1, data.emgStates.ch2,
                         data.emgStates.ch3, data.emgStates.ch4, data.emgStates.ch5,
                         data.emgStates.ch6, data.emgStates.ch7]).reshape([1, 8])
    # left_hand
    left_hand = np.array([data.tf_of_interest.transforms[5].transform.translation.x,
                          data.tf_of_interest.transforms[5].transform.translation.y,
                          data.tf_of_interest.transforms[5].transform.translation.z]).reshape([1, 3])
    # left_joints
    left_joints = np.array(data.jointStates.position[2:9]).reshape([1, 7])
    left_gripper = np.zeros_like(left_joints)

    global obs_data_list
    time_stamp = (data.header.stamp - init_time).secs + (data.header.stamp - init_time).nsecs*1e-9
    obs_data_list.append({'emg': emg_data,
                          'left_hand': left_hand,
                          'left_joints': left_gripper,
                          'stamp': time_stamp})
    rospy.loginfo(obs_data_list[-1])


def ready_go(count):
    """
    the func for press START key and countdown
    :param count: the time for countdown
    :return:
    """
    global flag_record
    while not flag_record:
        input_str = raw_input('Press ENTER to start: ')
        if '' == input_str:
            flag_record = True
    for idx in range(count):
        time.sleep(1.0)
        rospy.loginfo('The remaining time: %ds', count-idx-1)


if __name__ == '__main__':

    # init node
    rospy.init_node('online_ipromps_node', anonymous=True)
    rospy.loginfo('Created the ROS node!')

    # load datasets
    rospy.loginfo('Loading the datasets...')
    current_path = os.path.split(os.path.abspath(sys.argv[0]))[0]   # the directory of this script
    [ipromps_set, datasets4train_post, min_max_scaler, filt_kernel] = joblib.load(current_path+path+'/ipromps_set.pkl')

    # the flag var of starting info record
    flag_record = False
    # to save the online data
    obs_data_list = []
    # create a timer
    timer = threading.Timer(timer_interval, fun_timer)

    # baxter init
    rospy.loginfo("Getting robot state... ")
    rs = baxter_interface.RobotEnable(CHECK_VERSION)
    init_state = rs.state().enabled
    rospy.loginfo("Enabling robot... ")
    rs.enable()
    left = baxter_interface.Limb('left')

    # the start cmd
    ready_go(ready_time)

    # the init time
    init_time = None
    # subscribe the /multiModal_states topic
    rospy.Subscriber("/multiModal_states", multiModal, callback)

    # start the timer
    timer.start()

    # spin() simply keeps python from exiting until this node is stopped
    rospy.spin()
