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
    def __init__(self, is_left = False):
        #start AVP         
        self.vps = VisionProStreamer(ip = AVP_IP, record = True)
        # start pybullet
        #clid = p.connect(p.SHARED_MEMORY)
        #clid = p.connect(p.DIRECT)
        p.connect(p.GUI)
        # load right leap hand           
        path_src = os.path.abspath(__file__)
        path_src = os.path.dirname(path_src)
        self.is_left = is_left
        self.glove_to_leap_mapping_scale = 1.6
        self.leapEndEffectorIndex = [3, 4, 8, 9, 13, 14, 18, 19]
        if self.is_left:
            path_src = os.path.join(path_src, "leap_hand_utils/leap_hand_mesh_left/robot_pybullet.urdf")
            ##You may have to set this path for your setup on ROS2
            self.LeapId = p.loadURDF(
                path_src,
                [0.31, 0.01, 0.06],
                p.getQuaternionFromEuler([1.57, 0, 0]),   ##this is the correct frame and rotation for left hand.....use left hand data instead
                useFixedBase = True
            )
        else:
            path_src = os.path.join(path_src, "leap_hand_utils/leap_hand_mesh_right/robot_pybullet.urdf")
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
        
    def get_glove_data(self, hand_pos):
        #gets the data converts it and then computes IK and visualizes
        # hand_pos[2][0] = hand_pos[2][0] - 0.02  this isn't great because they won't oppose properly
        # hand_pos[3][0] = hand_pos[3][0] - 0.02    
        # hand_pos[6][0] = hand_pos[6][0] + 0.02
        # hand_pos[7][0] = hand_pos[7][0] + 0.02
        #hand_pos[2][1] = hand_pos[2][1] + 0.002
        hand_pos[4][1] = hand_pos[4][1] + 0.002
        hand_pos[6][1] = hand_pos[6][1] + 0.002
        output = self.compute_IK(hand_pos)
        self.update_target_vis(hand_pos)
        return output
    

    def get_avp_data(self):
        #gets the data converts it and then computes IK and visualizes
        r = self.vps.latest 
        if self.is_left:
            hand_pose = np.asarray(r['left_fingers']).astype(float)  
        else:
            hand_pose = np.asarray(r['right_fingers']).astype(float) 
        indices = [3,4,8,9,13,14,18,19,23,24]
        hand_pos = hand_pose[indices, :3, 3]
        for i in range(0,10):
            hand_pos[i][0] = hand_pos[i][0] * 1.35 * 1.5
            hand_pos[i][1] = hand_pos[i][1] * 1.5
            hand_pos[i][2] = hand_pos[i][2] * 1.5
        output = self.compute_IK(hand_pos)
        self.update_target_vis(hand_pos)
        return output
        
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


class LeapNode:
    def __init__(self):
        ####Some parameters
        # I recommend you keep the current limit from 350 for the lite, and 550 for the full hand
        # Increase KP if the hand is too weak, decrease if it's jittery.
        self.kP = 900
        self.kI = 0
        self.kD = 200
        self.curr_lim = 350
        self.prev_pos = self.pos = self.curr_pos = lhu.allegro_to_LEAPhand(np.zeros(16))
        #You can put the correct port here or have the node auto-search for a hand at the first 3 ports.
        # For example ls /dev/serial/by-id/* to find your LEAP Hand. Then use the result.  
        # For example: /dev/serial/by-id/usb-FTDI_USB__-__Serial_Converter_FT7W91VW-if00-port0
        self.motors = motors = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15]
        try:
            self.dxl_client = DynamixelClient(motors, '/dev/ttyUSB0', 4000000)
            self.dxl_client.connect()
        except Exception:
            try:
                self.dxl_client = DynamixelClient(motors, '/dev/ttyUSB1', 4000000)
                self.dxl_client.connect()
            except Exception:
                self.dxl_client = DynamixelClient(motors, 'COM13', 4000000)
                self.dxl_client.connect()
        #Enables position-current control mode and the default parameters, it commands a position and then caps the current so the motors don't overload
        self.dxl_client.sync_write(motors, np.ones(len(motors))*5, 11, 1)
        self.dxl_client.set_torque_enabled(motors, True)
        self.dxl_client.sync_write(motors, np.ones(len(motors)) * self.kP, 84, 2) # Pgain stiffness     
        self.dxl_client.sync_write([0,4,8], np.ones(3) * (self.kP * 0.75), 84, 2) # Pgain stiffness for side to side should be a bit less
        self.dxl_client.sync_write(motors, np.ones(len(motors)) * self.kI, 82, 2) # Igain
        self.dxl_client.sync_write(motors, np.ones(len(motors)) * self.kD, 80, 2) # Dgain damping
        self.dxl_client.sync_write([0,4,8], np.ones(3) * (self.kD * 0.75), 80, 2) # Dgain damping for side to side should be a bit less
        #Max at current (in unit 1ma) so don't overheat and grip too hard #500 normal or #350 for lite
        self.dxl_client.sync_write(motors, np.ones(len(motors)) * self.curr_lim, 102, 2)
        self.dxl_client.write_desired_pos(self.motors, self.curr_pos)

    #allegro compatibility joint angles.  It adds 180 to make the fully open position at 0 instead of 180
    def set_allegro(self, pose):
        pose = lhu.allegro_to_LEAPhand(pose, zeros=False)
        self.prev_pos = self.curr_pos
        self.curr_pos = np.array(pose)
        self.dxl_client.write_desired_pos(self.motors, self.curr_pos)

    #Receive LEAP pose and directly control the robot
    def set_leap(self, pose):
        self.prev_pos = self.curr_pos
        self.curr_pos = np.array(pose)
        self.dxl_client.write_desired_pos(self.motors, self.curr_pos)

        #Sim compatibility for policies, it assumes the ranges are [-1,1] and then convert to leap hand ranges.
    def set_ones(self, pose):
        pose = lhu.sim_ones_to_LEAPhand(np.array(pose))
        self.prev_pos = self.curr_pos
        self.curr_pos = np.array(pose)
        self.dxl_client.write_desired_pos(self.motors, self.curr_pos)
    #read position of the robot
    def read_pos(self):
        return self.dxl_client.read_pos()
    #read velocity
    def read_vel(self):
        return self.dxl_client.read_vel()
    #read current
    def read_cur(self):
        return self.dxl_client.read_cur()
    #These combined commands are faster FYI and return a list of data
    def pos_vel(self):
        return self.dxl_client.read_pos_vel()
    #These combined commands are faster FYI and return a list of data
    def pos_vel_eff_srv(self):
        return self.dxl_client.read_pos_vel_cur()
    



if __name__ == "__main__":
    pbik = Leapv1PybulletIKPython()
    # leap_hand = LeapNode()
    while True:
        output_joints = pbik.get_avp_data()
        # output_joints = np.zeros(16)
        # output_joints[0] = 0.1
        # print(output_joints)  ##This is the 16 dimensional LEAP Hand vector  Feed this into your python or ROS1/ROS2 code.  Ordering is the same!
        # leap_hand.set_leap(output_joints)
        # leap_hand.set_allegro(output_joints)
        # cprint("Actual Position: ",leap_hand.read_pos(), "red")
        time.sleep(0.033)