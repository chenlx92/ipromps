#!/usr/bin/python
import rospy
from states_manager.msg import multiModal
import numpy as np
import threading
import scipy.signal as signal
import baxter_interface
from baxter_interface import CHECK_VERSION
import time
import os
import ConfigParser
from sklearn.externals import joblib

# read conf file
file_path = os.path.dirname(__file__)
cp = ConfigParser.SafeConfigParser()
cp.read(os.path.join(file_path, '../config/model.conf'))

# load param
datasets_path = os.path.join(file_path, cp.get('datasets', 'path'))
num_alpha_candidate = cp.getint('phase', 'num_phaseCandidate')
timer_interval = cp.getfloat('online', 'timer_interval')
ready_time = cp.getint('online', 'ready_time')
task_name_path = os.path.join(datasets_path, 'pkl/task_name_list.pkl')
task_name_list = joblib.load(task_name_path)
filter_kernel = np.fromstring(cp.get('filter', 'filter_kernel'), dtype=int, sep=',')


def make_command(arr, t):
    """
    cleans a single line of recorded joint positions
    :param arr: the line described in a list to process
    :param t: the row index of the array
    :return: the cmd list
    """
    joint_cmd_names = ['left_s0', 'left_s1', 'left_e0', 'left_e1', 'left_w0', 'left_w1', 'left_w2']
    data_line = [arr[t][2], arr[t][3], arr[t][0], arr[t][1], arr[t][4], arr[t][5], arr[t][6]]
    command = dict(zip(joint_cmd_names, data_line))
    return command


def fun_timer():
    """
    the timer callback function
    :return:
    """
    rospy.loginfo('Time out!!!')
    global flag_record
    flag_record = False  # stop record the msg
    global ipromps_set, obs_data_list
    rospy.loginfo('The len of observed data is %d', len(obs_data_list))
    obs_data = np.array([]).reshape([0, ipromps_set[0].num_joints])
    timestamp = []
    for obs_data_list_idx in obs_data_list:
        emg = obs_data_list_idx['emg']
        left_hand = obs_data_list_idx['left_hand']
        left_joints = obs_data_list_idx['left_joints']
        full_data = np.hstack([emg, left_hand, left_joints])
        obs_data = np.vstack([obs_data, full_data])
        timestamp.append(obs_data_list_idx['stamp'])

    # filter the data
    obs_data = signal.medfilt(obs_data, filter_kernel)

    # preprocessing for the data
    obs_data_post_arr = ipromps_set[0].min_max_scaler.transform(obs_data)
    # consider the unobserved info
    obs_data_post_arr[:, 11:19] = 0.0

    # phase estimation
    rospy.loginfo('Phase estimating...')
    alpha_max_list = []
    for ipromp in ipromps_set:
        alpha_temp = ipromp.alpha_candidate()
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
    idx_max_prob = np.argmax(prob_task)
    rospy.loginfo('The max fit model index is task %s', task_name_list[idx_max_prob])

    # robot motion generation
    [traj_time, traj] = ipromps_set[idx_max_prob].gen_real_traj(alpha_max_list[idx_max_prob])
    traj = ipromps_set[idx_max_prob].min_max_scaler.inverse_transform(traj)
    robot_traj = traj[:, 11:18]

    # robot start point
    global left
    rospy.loginfo('Moving to start position...')
    left_start = make_command(robot_traj, 0)
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
    joblib.dump(ipromps_set, os.path.join(datasets_path, 'pkl/ipromps_set_post.pkl'))
    # save the robot traj
    rospy.loginfo('Saving the robot traj...')
    joblib.dump(robot_traj, os.path.join(datasets_path, 'pkl/robot_traj.pkl'))

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
    :param count: the time duration for countdown
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


def main():
    # init node
    rospy.init_node('online_ipromps_node', anonymous=True)
    rospy.loginfo('Created the ROS node!')

    # load datasets
    rospy.loginfo('Loading the datasets...')
    global ipromps_set
    ipromps_set = joblib.load(os.path.join(datasets_path, 'pkl/ipromps_set.pkl'))

    # the flag var of starting info record
    global flag_record, obs_data_list
    flag_record = False
    # to save the online data
    obs_data_list = []
    # create a timer
    timer = threading.Timer(timer_interval, fun_timer)

    # baxter init
    global left
    rospy.loginfo("Getting robot state... ")
    rs = baxter_interface.RobotEnable(CHECK_VERSION)
    init_state = rs.state().enabled

    def clean_shutdown():
        print("\nExiting example...")
        if not init_state:
            print("Disabling robot...")
            rs.disable()
    rospy.on_shutdown(clean_shutdown)

    rospy.loginfo("Enabling robot... ")
    rs.enable()
    left = baxter_interface.Limb('left')

    # the start cmd
    ready_go(ready_time)

    # the init time
    global init_time
    init_time = None
    # subscribe the /multiModal_states topic
    rospy.Subscriber("/multiModal_states", multiModal, callback)

    # start the timer
    timer.start()

    # spin() simply keeps python from exiting until this node is stopped
    rospy.spin()


if __name__ == '__main__':
    main()
