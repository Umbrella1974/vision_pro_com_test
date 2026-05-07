""" 
test avp_client in mujoco simulator env 
Its communication base on the ROS1

"""

import time
from multiprocessing.managers import SharedMemoryManager
# import cv2
import pathlib
import numpy as np
import hydra
import scipy.spatial.transform as st
import rospy

# from pynput import keyboard
from functools import cache
import traceback

from termcolor import cprint
from std_msgs.msg import Float64MultiArray
from avp_client import AVPTeleopClient

# from diffusion_policy.real_world.keystroke_counter import KeystrokeCounter, KeyCode
def main():
    rospy.init_node("avp_client_node")
    desired_qpos_hand_pub = rospy.Publisher('desired_qpos_hand', Float64MultiArray, queue_size=1)
    desired_wrist_pose_pub = rospy.Publisher('desired_wrist_pose', Float64MultiArray, queue_size=1)
    with SharedMemoryManager() as shm_manager:
        with AVPTeleopClient(
            shm_manager=shm_manager,
            AVP_IP= "192.168.1.29",
            # arm_tcp_vive_euler=[-90, 0, 0],
            arm_tcp_vive_euler=[0, -90, 90],
            # arm_tcp_vive_euler=[0,0,0],
            arm_tcp_leap_euler=[180, 0, 90],
            launch_timeout=3,
            get_max_k=30,
            frequency=30,
            verbose=True,
            left_hand=False,
            right_hand=True
        ) as teleop_client:
            teleop_client.reset()

            while not rospy.is_shutdown():
            # while True:
                motion_state = teleop_client.get_motion_state()
                movement = motion_state["motion_event"]
                print(movement)
                qpos_hand = movement[:16]
                wrist_pose = movement[16:]
                desired_qpos_hand_pub.publish(Float64MultiArray(data=qpos_hand))
                desired_wrist_pose_pub.publish(Float64MultiArray(data=wrist_pose))
                # rospy.sleep(1/10)
                time.sleep(1/10)

if __name__ == "__main__":
    main()