#!/usr/bin/env python

import rospy
import numpy as np
from std_msgs.msg import Float64MultiArray

def send_control_commands():
    rospy.init_node('control_command_sender', anonymous=True)

    # 创建发布者
    desired_qpos_hand_pub = rospy.Publisher('desired_qpos_hand', Float64MultiArray, queue_size=1)
    desired_wrist_pose_pub = rospy.Publisher('desired_wrist_pose', Float64MultiArray, queue_size=1)

    rate = rospy.Rate(30)  # 30 Hz
    desired_angles = np.array([0.0, 0.0, 0.0, 0.0, 
                            0.0, 0.0, 0.0, 0.0, 
                            0.0, 0.0, 0.0, 0.0, 
                            0.0, 0.0, 0.0, 0.0])
    desired_wrist_pose = np.array([-0.365, 0.208, 0.397, 3.14, 0.0, 0.0])

    while not rospy.is_shutdown():
        # 假设的控制指令
        # desired_angles = np.random.rand(16)  # 随机生成16个关节角度
        # desired_wrist_pose = np.random.rand(6)  # 随机生成6个腕部姿态参数

        # desired_wrist_pose[0] += 0.01  # 逐渐增加第一个参数


        # 创建消息
        qpos_msg = Float64MultiArray(data=desired_angles.tolist())
        wrist_msg = Float64MultiArray(data=desired_wrist_pose.tolist())

        # 发布消息
        desired_qpos_hand_pub.publish(qpos_msg)
        desired_wrist_pose_pub.publish(wrist_msg)

        rospy.loginfo(f"Sent desired angles: {desired_angles}")
        rospy.loginfo(f"Sent desired wrist pose: {desired_wrist_pose}")

        rate.sleep()

if __name__ == '__main__':
    try:
        send_control_commands()
    except rospy.ROSInterruptException:
        pass