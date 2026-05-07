import pybullet as p
import numpy as np
import os
import time
from avp_stream import VisionProStreamer
from leap_hand_utils.dynamixel_client import *
import leap_hand_utils.leap_hand_utils as lhu
from termcolor import colored, cprint


'''
This is based off of https://github.com/Improbable-AI/VisionProTeleop by Younghyo Park et. al. which streams the AVP data.
We then runs inverse kinematics and output LEAP Hand joint angles.
Feed these joint angles into your preferred LEAP Hand API, Python, ROS1/ROS2 etc.

See https://github.com/leap-hand/Bidex_VisionPro_Teleop for further details.

Note how the fingertip positions are matching, but the joint angles between the two hands are not due to the IK solution.  :) 
This is not as accurate as Manus gloves but easier to use.
Inspired by Dexcap https://dex-cap.github.io/ by Wang et. al. and Robotic Telekinesis by Shaw et. al.
'''

#Set this to your IP address of you AVP
AVP_IP = "192.168.1.29"

class Leapv1PybulletIKPython():
    def __init__(self, is_left = False, viz = False):
        #start AVP         
        # self.vps = VisionProStreamer(ip = AVP_IP, record = True)
        # start pybullet
        #clid = p.connect(p.SHARED_MEMORY)
        #clid = p.connect(p.DIRECT)
        if viz:
            p.connect(p.GUI)
        else:
            p.connect(p.DIRECT)
        # load right leap hand           
        path_src = os.path.abspath(__file__)
        path_src = os.path.dirname(path_src)
        self.is_left = is_left
        self.glove_to_leap_mapping_scale = 1.6
        self.leapEndEffectorIndex = [3, 4, 8, 9, 13, 14, 18, 19]
        if self.is_left:
            path_src = os.path.join(path_src, "leap_hand_mesh_left/robot_pybullet.urdf")
            ##You may have to set this path for your setup on ROS2
            self.LeapId = p.loadURDF(
                path_src,
                [0.31, 0.01, 0.06],
                p.getQuaternionFromEuler([1.57, 0, 0]),   ##this is the correct frame and rotation for left hand.....use left hand data instead
                useFixedBase = True
            )
        else:
            path_src = os.path.join(path_src, "leap_hand_mesh_right/robot_pybullet.urdf")
            ##You may have to set this path for your setup on ROS2
            self.LeapId = p.loadURDF(
                path_src,
                [-0.22, 0.01, 0.03],
                p.getQuaternionFromEuler([1.57, 0, 3.14]),
                useFixedBase = True
            )

        self.numJoints = p.getNumJoints(self.LeapId)
        p.setGravity(0, 0, 0)
        useRealTimeSimulation = 0
        p.setRealTimeSimulation(useRealTimeSimulation)
        self.create_target_vis()
            
    def create_target_vis(self):
        # load balls
        small_ball_radius = 0.01
        small_ball_shape = p.createCollisionShape(p.GEOM_SPHERE, radius=small_ball_radius)
        ball_radius = 0.01
        ball_shape = p.createCollisionShape(p.GEOM_SPHERE, radius=ball_radius)
        baseMass = 0.001
        basePosition = [0.25, 0.25, 0]
        
        self.ballMbt = []
        for i in range(0,4):
            self.ballMbt.append(p.createMultiBody(baseMass=baseMass, baseCollisionShapeIndex=ball_shape, basePosition=basePosition)) # for base and finger tip joints    
            no_collision_group = 0
            no_collision_mask = 0
            p.setCollisionFilterGroupMask(self.ballMbt[i], -1, no_collision_group, no_collision_mask)
        p.changeVisualShape(self.ballMbt[0], -1, rgbaColor=[1, 0, 0, 1]) 
        p.changeVisualShape(self.ballMbt[1], -1, rgbaColor=[0, 1, 0, 1]) 
        p.changeVisualShape(self.ballMbt[2], -1, rgbaColor=[0, 0, 1, 1])  
        p.changeVisualShape(self.ballMbt[3], -1, rgbaColor=[1, 1, 1, 1])
        
    def update_target_vis(self, hand_pos):
        _, current_orientation = p.getBasePositionAndOrientation( self.ballMbt[0])
        p.resetBasePositionAndOrientation(self.ballMbt[0], hand_pos[3], current_orientation)
        _, current_orientation = p.getBasePositionAndOrientation(self.ballMbt[1])
        p.resetBasePositionAndOrientation(self.ballMbt[1], hand_pos[2], current_orientation)
        _, current_orientation = p.getBasePositionAndOrientation(self.ballMbt[2])
        p.resetBasePositionAndOrientation(self.ballMbt[2], hand_pos[7], current_orientation)
        _, current_orientation = p.getBasePositionAndOrientation(self.ballMbt[3])
        p.resetBasePositionAndOrientation(self.ballMbt[3], hand_pos[1], current_orientation)
        
    # def get_avp_data(self):
    #     #gets the data converts it and then computes IK and visualizes
    #     r = self.vps.latest              
    #     if self.is_left:
    #         hand_pose = np.asarray(r['left_fingers']).astype(float)  
    #     else:
    #         hand_pose = np.asarray(r['right_fingers']).astype(float) 
    #     indices = [3,4,8,9,13,14,18,19,23,24]
    #     hand_pos = hand_pose[indices, :3, 3]
    #     for i in range(0,10):
    #         hand_pos[i][0] = hand_pos[i][0] * 1.35 * 1.5
    #         hand_pos[i][1] = hand_pos[i][1] * 1.5
    #         hand_pos[i][2] = hand_pos[i][2] * 1.5
    #     output = self.compute_IK(hand_pos)
    #     self.update_target_vis(hand_pos)
    #     return output
        
    def compute_IK(self, hand_pos):
        p.stepSimulation()     

        rightHandIndex_middle_pos = hand_pos[2]
        rightHandIndex_pos = hand_pos[3]
        
        rightHandMiddle_middle_pos = hand_pos[4]
        rightHandMiddle_pos = hand_pos[5]
        
        rightHandRing_middle_pos = hand_pos[6]
        rightHandRing_pos = hand_pos[7]
        
        rightHandThumb_middle_pos = hand_pos[0]
        rightHandThumb_pos = hand_pos[1]
        
        leapEndEffectorPos = [
            rightHandIndex_middle_pos,
            rightHandIndex_pos,
            rightHandMiddle_middle_pos,
            rightHandMiddle_pos,
            rightHandRing_middle_pos,
            rightHandRing_pos,
            rightHandThumb_middle_pos,
            rightHandThumb_pos
        ]

        jointPoses = p.calculateInverseKinematics2(
            self.LeapId,
            self.leapEndEffectorIndex,
            leapEndEffectorPos,
            solver=p.IK_DLS,
            maxNumIterations=50,
            residualThreshold=0.0001,
        )
        
        combined_jointPoses = (jointPoses[0:4] + (0.0,) + jointPoses[4:8] + (0.0,) + jointPoses[8:12] + (0.0,) + jointPoses[12:16] + (0.0,))
        combined_jointPoses = list(combined_jointPoses)

        # update the hand joints
        for i in range(20):
            p.setJointMotorControl2(
                bodyIndex=self.LeapId,
                jointIndex=i,
                controlMode=p.POSITION_CONTROL,
                targetPosition=combined_jointPoses[i],
                targetVelocity=0,
                force=500,
                positionGain=0.3,
                velocityGain=1,
            )

        # map results to real robot
        real_robot_hand_q = np.array([float(0.0) for _ in range(16)])
        #real_left_robot_hand_q = np.array([0.0 for _ in range(16)])

        real_robot_hand_q[0:4] = jointPoses[0:4]
        real_robot_hand_q[4:8] = jointPoses[4:8]
        real_robot_hand_q[8:12] = jointPoses[8:12]
        real_robot_hand_q[12:16] = jointPoses[12:16]
        real_robot_hand_q[0:2] = real_robot_hand_q[0:2][::-1]
        real_robot_hand_q[4:6] = real_robot_hand_q[4:6][::-1]
        real_robot_hand_q[8:10] = real_robot_hand_q[8:10][::-1]
        return [float(i) for i in real_robot_hand_q]