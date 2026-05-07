import time
import pybullet as p
import numpy as np
from pathlib import Path


class LeapRetargeting:
    def __init__(self, urdf_path):
        p.connect(p.DIRECT)
        # p.connect(p.GUI)

        urdf_path = Path(__file__).parent / urdf_path

        self.glove_to_leap_mapping_scale = 1.6
        self.leapEndEffectorIndex = [7, 6, 14, 13, 21, 20, 26, 28]
        self.short_idx = [3, 4, 8, 9, 13, 14, 18, 19, 23, 24]

        self.LeapId = p.loadURDF(
            str(urdf_path),
            [0, 0, 0],
            p.getQuaternionFromEuler([0, 0, 0]),
            useFixedBase=True,
        )

        self.numJoints = p.getNumJoints(self.LeapId)
        p.setGravity(0, 0, 0)
        useRealTimeSimulation = 0
        p.setRealTimeSimulation(useRealTimeSimulation)
        self.create_target_vis()
        # # print all the joint info
        # for i in range(self.numJoints):
        #     print(p.getJointInfo(self.LeapId, i))
        # print("Loaded Leap Hand")

        # find all active joints and corresponding joint indices
        self.activeJoints = []
        self.activeJointIndices = []
        for i in range(self.numJoints):
            jointInfo = p.getJointInfo(self.LeapId, i)
            qIndex = jointInfo[3]
            if qIndex > -1:
                self.activeJoints.append(jointInfo[1])
                self.activeJointIndices.append(i)

        time.sleep(1)
        # print("Active Joints: ", self.activeJoints)

    def create_target_vis(self):
        # load balls
        small_ball_radius = 0.01
        small_ball_shape = p.createCollisionShape(p.GEOM_SPHERE, radius=small_ball_radius)
        ball_radius = 0.01
        ball_shape = p.createCollisionShape(p.GEOM_SPHERE, radius=ball_radius)
        baseMass = 0.001
        basePosition = [0.25, 0.25, 0]

        self.ballMbt = []
        for i in range(0, 4):
            self.ballMbt.append(
                p.createMultiBody(
                    baseMass=baseMass, baseCollisionShapeIndex=ball_shape, basePosition=basePosition
                )
            )  # for base and finger tip joints
            no_collision_group = 0
            no_collision_mask = 0
            p.setCollisionFilterGroupMask(
                self.ballMbt[i], -1, no_collision_group, no_collision_mask
            )
        p.changeVisualShape(self.ballMbt[0], -1, rgbaColor=[1, 0, 0, 1])
        p.changeVisualShape(self.ballMbt[1], -1, rgbaColor=[0, 1, 0, 1])
        p.changeVisualShape(self.ballMbt[2], -1, rgbaColor=[0, 0, 1, 1])
        p.changeVisualShape(self.ballMbt[3], -1, rgbaColor=[1, 1, 1, 1])

    def update_target_vis(self, hand_pos):
        # p.resetBasePositionAndOrientation(self.LeapId, [0, 0, 0], hand_quat[0])

        _, current_orientation = p.getBasePositionAndOrientation(self.ballMbt[0])
        p.resetBasePositionAndOrientation(self.ballMbt[0], hand_pos[3], current_orientation)
        _, current_orientation = p.getBasePositionAndOrientation(self.ballMbt[1])
        p.resetBasePositionAndOrientation(self.ballMbt[1], hand_pos[5], current_orientation)
        _, current_orientation = p.getBasePositionAndOrientation(self.ballMbt[2])
        p.resetBasePositionAndOrientation(self.ballMbt[2], hand_pos[7], current_orientation)
        _, current_orientation = p.getBasePositionAndOrientation(self.ballMbt[3])
        p.resetBasePositionAndOrientation(self.ballMbt[3], hand_pos[1], current_orientation)

    def retarget(self, hand_pos):
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
            rightHandThumb_pos,
        ]

        jointPoses = p.calculateInverseKinematics2(
            self.LeapId,
            self.leapEndEffectorIndex,
            leapEndEffectorPos,
            solver=p.IK_DLS,
            maxNumIterations=500,
            residualThreshold=0.0001,
        )

        # calculate real qpos
        # swap 0,1 ; 4,5 ; 8,9
        # real_robot_hand_q = np.array(jointPoses[:16], dtype=np.float64)
        # real_robot_hand_q[[0, 1]] = real_robot_hand_q[[1, 0]]
        # real_robot_hand_q[[4, 5]] = real_robot_hand_q[[5, 4]]
        # real_robot_hand_q[[8, 9]] = real_robot_hand_q[[9, 8]]
                # map results to real robot
        real_robot_hand_q = np.array([float(0.0) for _ in range(16)])
        # real_left_robot_hand_q = np.array([0.0 for _ in range(16)])

        real_robot_hand_q[0:4] = jointPoses[0:4]
        real_robot_hand_q[4:8] = jointPoses[4:8]
        real_robot_hand_q[8:12] = jointPoses[8:12]
        real_robot_hand_q[12:16] = jointPoses[12:16]
        real_robot_hand_q[0:2] = real_robot_hand_q[0:2][::-1]
        real_robot_hand_q[4:6] = real_robot_hand_q[4:6][::-1]
        real_robot_hand_q[8:10] = real_robot_hand_q[8:10][::-1]

        # self.update_target_vis(leapEndEffectorPos)

        return real_robot_hand_q
