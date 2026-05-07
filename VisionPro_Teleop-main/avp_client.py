"""
A client to get the latest AVP data and convert its frame.
We try to realize the bimanual teleoperation task.
However, it only support right hand only.
"""
import multiprocessing as mp
import contextlib
from typing import List
import pathlib

# import enum
import numpy as np
from scipy.spatial.transform import Rotation as R
import time

from termcolor import cprint
import rospy
from geometry_msgs.msg import PoseWithCovarianceStamped, PoseArray
from avp_stream import VisionProStreamer

# from diffusion_policy.shared_memory.shared_memory_queue import (
#     SharedMemoryQueue, Empty)
from diffusion_policy.shared_memory.shared_memory_ring_buffer import SharedMemoryRingBuffer
from diffusion_policy.common.precise_sleep import precise_wait
from leap_hand_utils.leap_pybullet_IK import Leapv1PybulletIKPython

# from .manus_vive_utils.network import PoseStampedSubscriber, ManusPoseArraySubscriber
# from dex_retargeting.retargeting_config import RetargetingConfig


class AVPTeleopClient(mp.Process):
    def __init__(
        self,
        shm_manager,
        AVP_IP: str,
        arm_tcp_vive_euler: List[float],
        arm_tcp_leap_euler: List[float],
        launch_timeout=2,
        get_max_k=30,
        frequency=30,
        verbose=False,
        left_hand = False,
        right_hand = True,
    ):
        """
        Continuously listen to Manus & Vive events
        and update the latest state.
        """
        super().__init__(name="AVPTeleopClient")

        

        # Create retargeting solver
        # urdf_dir = pathlib.Path(__file__).parent / "leap_hand_utils" / retargeting_config.urdf_dir

        if left_hand:
            self.leap_IK_left = Leapv1PybulletIKPython(is_left = True)
        if right_hand:
            self.leap_IK_right = Leapv1PybulletIKPython(is_left = False)

        # Fixed frame rotations for wrist: UR5 End Effector Frame -> Vive Tracker Frame
        self.r_tcp_to_tracker = R.from_euler("XYZ", arm_tcp_vive_euler, degrees=True)  # type: ignore
        self.r_E_L = R.from_euler("XYZ", arm_tcp_leap_euler, degrees=True)  # type: ignore

        # self.rt = RotationTransformer(from_rep="axis_angle", to_rep="rotation_6d")

        # real_hand_joint_names = robot_hand_joint_names
        # retargeting_joint_names = self.retargeting.joint_names
        # self.retargeting_to_real = np.array(
        #     [retargeting_joint_names.index(name) for name in real_hand_joint_names]
        # ).astype(int)

        # build ring buffer
        self.wrist_pose_0 = {
            "left": None,
            "right": None 
        }
        example = {}
        example["motion_event"] = np.zeros((22,))  # 3 translation, 3 rotation, 1 period
        example["receive_timestamp"] = time.time()

        ring_buffer = SharedMemoryRingBuffer.create_from_examples(
            shm_manager=shm_manager,
            examples=example,
            get_max_k=get_max_k,
            get_time_budget=0.2,
            put_desired_frequency=frequency,
        )
        self.verbose = verbose
        self.AVP_IP = AVP_IP
        # self.retargeting_config = retargeting_config
        self.frequency = frequency
        # self.tcp_origin_pose = tcp_origin_pose
        self.launch_timeout = launch_timeout
        self.left_hand = left_hand
        self.right_hand = right_hand
        # shared variables
        self.ready_event = mp.Event()
        self.stop_event = mp.Event()
        self.reset_event = mp.Event()
        self.ring_buffer = ring_buffer

    # ========= launch method ===========
    def start(self, wait=True):
        super().start()
        if wait:
            self.start_wait()
        if self.verbose:
            print(f"[AVPTeleopClient] Client process spawned at {self.pid}")

    def stop(self, wait=True):
        self.stop_event.set()
        if wait:
            self.stop_wait()

    def start_wait(self):
        self.ready_event.wait(self.launch_timeout)
        assert self.is_alive()

    def stop_wait(self):
        self.join()

    @property
    def is_ready(self):
        return self.ready_event.is_set()

    # ========= context manager ===========
    def __enter__(self):
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()

    # ========= command methods ============
    def reset(self):
        self.reset_event.set()
        # wait for reset to finish
        while self.reset_event.is_set():
            time.sleep(0.010)

    # ========= receive APIs =============
    def get_motion_state(self, k=None, out=None):
        if k is None:
            return self.ring_buffer.get(out=out)
        else:
            return self.ring_buffer.get_last_k(k=k, out=out)

    # helper functions
    def reset_initial_wrist_pose(self, wait=True):
        r = self.vps.latest 
        # r = np.load("data.npy", allow_pickle=True).item()
        if self.left_hand:
            self.wrist_pose_0["left"] = np.asarray(r['left_wrist']).astype(float)
        if self.right_hand:
            self.wrist_pose_0["right"] = np.asarray(r['right_wrist']).astype(float)
        if wait:
            print("Recording initial human wrist pose in 3 seconds.")
            print("Please keep your palm downward.")
            for i in range(3):
                print(f"{3-i}...")
                time.sleep(1)
        print("Human Agent Reset Done")

    def calculate_desired_hand_joint_angles(self, hand_pose: np.ndarray, hand: str):
        assert hand_pose is not None, "AVP data not received yet."

        indices = [3,4,8,9,13,14,18,19,23,24]
        hand_pos = hand_pose[indices, :3, 3]
        for i in range(0,10):
            hand_pos[i][0] = hand_pos[i][0] * 1.35 * 1.5
            hand_pos[i][1] = hand_pos[i][1] * 1.5
            hand_pos[i][2] = hand_pos[i][2] * 1.5
        if hand == "right":
            output = self.leap_IK_right.compute_IK(hand_pos)
        if hand == "left":
            output = self.leap_IK_left.compute_IK(hand_pos)
        return output

    def calculate_desired_wrist_pose(
        self, tracker_pose: np.ndarray, hand: str, residual_pose = np.zeros(6)
    ) -> np.ndarray:
        """
        Calculate the desired wrist pose relative to the initial TCP frame.
        Args:
        - tracker_pose: rotation matrix representing the current wrist pose in the tracker frame.
        - residual_pose: 6D vector representing the residual pose from retargeting dummy joints.
        Returns:
        - desired_wrist_pose: 6D vector (XYZ + rotvec) representing the desired wrist pose relative to the initial TCP frame
        - if you want to use the pose, please convert it to absolute pose.
        """
        translation_t_base = tracker_pose[0][:3, 3]
        rotation_matrix_t_base = tracker_pose[0][:3, :3]

        if hand == "left":
            translation_0_base = self.wrist_pose_0["left"][0][:3, 3]
            rotation_matrix_0_base = self.wrist_pose_0["left"][0][:3, :3]
        elif hand == "right":
            translation_0_base = self.wrist_pose_0["right"][0][:3, 3]
            rotation_matrix_0_base = self.wrist_pose_0["right"][0][:3, :3]
        # wrist rotation
        r_H0_Ht = R.from_matrix(rotation_matrix_0_base).inv() * R.from_matrix(rotation_matrix_t_base)
        r_E0_Et = self.r_tcp_to_tracker * r_H0_Ht * self.r_tcp_to_tracker.inv()
        # wrist pos
        pos_tracker_t_0 = translation_t_base - translation_0_base
        s_E0 = (self.r_tcp_to_tracker * (R.from_matrix(rotation_matrix_0_base).inv())).apply(pos_tracker_t_0)
        # s_E0 = pos_tracker_t_0
        # residual pose from retargeting dummy joints
        r_Et_Etr = self.r_E_L * R.from_euler("XYZ", residual_pose[3:], degrees=False) * self.r_E_L.inv()  # type: ignore
        sr_Et = self.r_E_L.apply(residual_pose[:3])

        desired_wrist_pos = s_E0 + r_E0_Et.apply(sr_Et)
        desired_wrist_rotvec = (r_E0_Et * r_Et_Etr).as_rotvec()
        return np.hstack([desired_wrist_pos, desired_wrist_rotvec])
   
    def get_avp_data(self):
        #gets the data converts it and then computes IK and visualizes

        # fingers data
        # r = np.load("data.npy", allow_pickle=True).item()
        r = self.vps.latest 
        # r = np.load("data.npy", allow_pickle=True).item()
        hand_pose = {
            "left": None,
            "right": None
        }             
        if self.left_hand:
            left_hand_qpos_des = self.calculate_desired_hand_joint_angles(
                np.asarray(r['left_fingers']).astype(float), "left")
            left_wrist_pose = self.calculate_desired_wrist_pose(
                np.asarray(r['left_wrist']).astype(float), "left")
            hand_pose["left"] = np.concatenate([left_wrist_pose, left_hand_qpos_des])
        if self.right_hand:
            right_hand_qpos_des = self.calculate_desired_hand_joint_angles(
                np.asarray(r['right_fingers']).astype(float), "right")
            right_wrist_pose = self.calculate_desired_wrist_pose(
                np.asarray(r['right_wrist']).astype(float), "right")
            hand_pose["right"] = np.concatenate([right_wrist_pose, right_hand_qpos_des])

        # pinch distance
        # TODO
        return hand_pose

    # ========= main loop ==========
    def run(self):
        print("Glove & Tracker Subscribers started.")
        time.sleep(2)
        dt = 1.0 / self.frequency
        max_wait_s = 5
        target_pose = {
            "left": None,
            "right": None
        }
        self.vps = VisionProStreamer(ip = self.AVP_IP, record = False)
        try:
            motion_event = np.zeros((22,), dtype=np.int64)
            # send one message immediately so client can start reading
            self.ring_buffer.put({"motion_event": motion_event, "receive_timestamp": time.time()})

            # t_wait_start = time.time()

            # while True:
            #     if self.stop_event.is_set():
            #         break
            #     # glove_data = self.glove_sub.get_poses()
            #     glove_data = self.glove_poses
            #     # tracker_data = self.vive_tracker_sub.get_pose()
            #     tracker_data = self.tracker_pose
            #     if glove_data is not None and tracker_data is not None:
            #         print("Initial glove and tracker data received.")
            #         break
            #     if time.time() - t_wait_start > max_wait_s:
            #         print("Waiting for the initial glove and tracker data.")
            #         break
            # assert glove_data is not None, "Glove data not received yet."
            # assert tracker_data is not None, "Tracker data not received yet."
            
            self.reset_initial_wrist_pose(wait=False)
            self.reset_event.clear()
            self.ready_event.set()
            
            while not self.stop_event.is_set():
                t_start = time.monotonic()
                if self.reset_event.is_set():
                    print("Resetting the human agent.")
                    self.reset_initial_wrist_pose(wait=False)
                    self.reset_event.clear()
                else:
                    target_pose = self.get_avp_data()
                    receive_timestamp = time.time()
                    motion_event = target_pose["right"]
                    # finish integrating this round of events before sending over
                    self.ring_buffer.put(
                        {"motion_event": motion_event, "receive_timestamp": receive_timestamp}
                    )
                    precise_wait(t_start + dt)
                    # time.sleep(dt)

        except Exception as e:
            cprint(f"Error in AVPViveTeleopClient: {e}", "red")
            self.ready_event.clear()
            self.stop_event.set()

        finally:
            # self.glove_sub.stop()
            # self.glove_sub.unregister()
            # # self.vive_tracker_sub.stop()
            # self.vive_tracker_sub.unregister()
            self.stop_event.set()
            self.ready_event.clear()

            print("[AVPTeleopClient] Exiting worker process.")
