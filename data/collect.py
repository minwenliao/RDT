# -- coding: UTF-8
import os
import time
import numpy as np
import h5py
import argparse
import dm_env
import collections
from collections import deque
import rospy
from sensor_msgs.msg import JointState, Image
from nav_msgs.msg import Odometry
from cv_bridge import CvBridge
import sys
import cv2

# 保存数据函数
def save_data(args, timesteps, actions, dataset_path):
    # 数据字典
    data_size = len(actions)
    data_dict = {
        '/observations/qpos': [],
        '/observations/qvel': [],
        '/observations/effort': [],
        '/action': [],
        '/base_action': [],
    }

    # 相机字典
    for cam_name in args.camera_names:
        data_dict[f'/observations/images/{cam_name}'] = []
        if args.use_depth_image:
            data_dict[f'/observations/images_depth/{cam_name}'] = []

    # 遍历动作数据
    while actions:
        action = actions.pop(0)   # 动作  当前动作
        ts = timesteps.pop(0)     # 当前时刻的奖励

        # 往字典里面添加数据
        data_dict['/observations/qpos'].append(ts.observation['qpos'])
        data_dict['/observations/qvel'].append(ts.observation['qvel'])
        data_dict['/observations/effort'].append(ts.observation['effort'])
        data_dict['/action'].append(action)
        data_dict['/base_action'].append(ts.observation['base_vel'])

        # 相机数据
        for cam_name in args.camera_names:
            data_dict[f'/observations/images/{cam_name}'].append(ts.observation['images'][cam_name])
            if args.use_depth_image:
                data_dict[f'/observations/images_depth/{cam_name}'].append(ts.observation['images_depth'][cam_name])

    t0 = time.time()
    with h5py.File(dataset_path + '.hdf5', 'w', rdcc_nbytes=1024**2*2) as root:
        # 添加属性
        root.attrs['sim'] = False
        root.attrs['compress'] = False

        # 创建数据组
        obs = root.create_group('observations')
        image = obs.create_group('images')
        for cam_name in args.camera_names:
            _ = image.create_dataset(cam_name, (data_size, 480, 640, 3), dtype='uint8',
                                         chunks=(1, 480, 640, 3), )
        if args.use_depth_image:
            image_depth = obs.create_group('images_depth')
            for cam_name in args.camera_names:
                _ = image_depth.create_dataset(cam_name, (data_size, 480, 640), dtype='uint16',
                                             chunks=(1, 480, 640), )

        _ = obs.create_dataset('qpos', (data_size, 14))
        _ = obs.create_dataset('qvel', (data_size, 14))
        _ = obs.create_dataset('effort', (data_size, 14))
        _ = root.create_dataset('action', (data_size, 14))
        _ = root.create_dataset('base_action', (data_size, 2))

        # 将数据写入HDF5文件
        for name, array in data_dict.items():  
            root[name][...] = array
    print(f'\033[32m\nSaving: {time.time() - t0:.1f} secs. %s \033[0m\n'%dataset_path)

class RosOperator:
    def __init__(self, args):
        # 初始化传感器和关节数据队列
        self.img_left_deque = deque()
        self.img_right_deque = deque()
        self.img_front_deque = deque()
        self.img_left_depth_deque = deque()
        self.img_right_depth_deque = deque()
        self.img_front_depth_deque = deque()
        self.master_arm_left_deque = deque()
        self.master_arm_right_deque = deque()
        self.puppet_arm_left_deque = deque()
        self.puppet_arm_right_deque = deque()
        self.robot_base_deque = deque()
        self.bridge = CvBridge()
        self.args = args
        self.init_ros()

    def init_ros(self):
        rospy.init_node('data_collector', anonymous=True)
        rospy.Subscriber(self.args.img_front_topic, Image, self.img_front_callback)
        rospy.Subscriber(self.args.img_left_topic, Image, self.img_left_callback)
        rospy.Subscriber(self.args.img_right_topic, Image, self.img_right_callback)
        rospy.Subscriber(self.args.master_arm_left_topic, JointState, self.master_arm_left_callback)
        rospy.Subscriber(self.args.master_arm_right_topic, JointState, self.master_arm_right_callback)
        rospy.Subscriber(self.args.robot_base_topic, Odometry, self.robot_base_callback)

    def img_front_callback(self, msg):
        self.img_front_deque.append(msg)

    def img_left_callback(self, msg):
        self.img_left_deque.append(msg)

    def img_right_callback(self, msg):
        self.img_right_deque.append(msg)

    def master_arm_left_callback(self, msg):
        self.master_arm_left_deque.append(msg)

    def master_arm_right_callback(self, msg):
        self.master_arm_right_deque.append(msg)

    def robot_base_callback(self, msg):
        self.robot_base_deque.append(msg)

    def get_frame(self):
        # Check that all necessary data is available
        if len(self.img_left_deque) == 0 or len(self.img_right_deque) == 0 or len(self.img_front_deque) == 0:
            return False
        
        # Retrieve the most recent data for each sensor
        img_left = self.bridge.imgmsg_to_cv2(self.img_left_deque.pop(), 'passthrough')
        img_right = self.bridge.imgmsg_to_cv2(self.img_right_deque.pop(), 'passthrough')
        img_front = self.bridge.imgmsg_to_cv2(self.img_front_deque.pop(), 'passthrough')
        master_arm_left = self.master_arm_left_deque.pop()
        master_arm_right = self.master_arm_right_deque.pop()
        robot_base = self.robot_base_deque.pop()

        # Return the collected data
        return img_front, img_left, img_right, master_arm_left, master_arm_right, robot_base

    def process(self):
        timesteps = []
        actions = []
        count = 0

        rate = rospy.Rate(self.args.frame_rate)
        while count < self.args.max_timesteps and not rospy.is_shutdown():
            result = self.get_frame()
            if not result:
                rate.sleep()
                continue
            count += 1

            img_front, img_left, img_right, master_arm_left, master_arm_right, robot_base = result

            # Create observation data
            obs = {
                'images': {
                    'front': img_front,
                    'left': img_left,
                    'right': img_right
                },
                'qpos': np.concatenate([master_arm_left.position, master_arm_right.position]),
                'qvel': np.concatenate([master_arm_left.velocity, master_arm_right.velocity]),
                'base_vel': [robot_base.twist.twist.linear.x, robot_base.twist.twist.angular.z],
            }

            # Create action data
            action = np.concatenate([master_arm_left.position, master_arm_right.position])

            # Append to lists
            timesteps.append(dm_env.TimeStep(
                step_type=dm_env.StepType.MID, reward=None, discount=None, observation=obs))
            actions.append(action)

            rate.sleep()

        return timesteps, actions

def get_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_dir', type=str, default='./data', help='Dataset directory')
    parser.add_argument('--episode_idx', type=int, default=0, help='Episode index')
    parser.add_argument('--max_timesteps', type=int, default=500, help='Maximum number of timesteps')
    parser.add_argument('--camera_names', type=str, nargs='+', default=['cam_high', 'cam_left_wrist', 'cam_right_wrist'])
    parser.add_argument('--img_front_topic', type=str, default='/camera_f/color/image_raw')
    parser.add_argument('--img_left_topic', type=str, default='/camera_l/color/image_raw')
    parser.add_argument('--img_right_topic', type=str, default='/camera_r/color/image_raw')
    parser.add_argument('--master_arm_left_topic', type=str, default='/master/joint_left')
    parser.add_argument('--master_arm_right_topic', type=str, default='/master/joint_right')
    parser.add_argument('--robot_base_topic', type=str, default='/odom')
    args = parser.parse_args()
    return args

def main():
    args = get_arguments()
    ros_operator = RosOperator(args)
    timesteps, actions = ros_operator.process()
    dataset_dir = os.path.join(args.dataset_dir, f"episode_{args.episode_idx}")
    save_data(args, timesteps, actions, dataset_dir)

if __name__ == '__main__':
    main()
