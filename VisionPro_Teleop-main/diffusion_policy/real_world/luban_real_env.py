from typing import Optional
import pathlib
import numpy as np
import time
import shutil
import math
from multiprocessing.managers import SharedMemoryManager
import scipy.spatial.transform as st

from termcolor import cprint
from diffusion_policy.real_world.rtde_interpolation_controller import RTDEInterpolationController
from diffusion_policy.real_world.leap_hand_interpolation_controller import (
    LeapHandInterpolationController,
)
from diffusion_policy.real_world.multi_realsense2 import MultiRealsense, SingleRealsense

# from diffusion_policy.real_world.uvc_camera import UvcCamera
from diffusion_policy.real_world.multi_dottip import MultiDotTip
from diffusion_policy.real_world.video_recorder_uvc import VideoRecorderUVC as VideoRecorder
from diffusion_policy.common.timestamp_accumulator import (
    TimestampObsAccumulator,
    TimestampActionAccumulator,
    # align_timestamps,
)
from diffusion_policy.real_world.multi_camera_visualizer import MultiCameraVisualizer
from diffusion_policy.common.replay_buffer import ReplayBuffer
from diffusion_policy.common.cv2_util import get_image_transform, optimal_row_cols
from diffusion_policy.model.common.rotation_transformer import RotationTransformer

DEFAULT_OBS_KEY_MAP = {
    # robot
    "ActualTCPPose": "robot_eef_pose",
    "ActualTCPSpeed": "robot_eef_pose_vel",
    "ActualQ": "robot_joint",
    "ActualQd": "robot_joint_vel",
    # timestamps
    "step_idx": "step_idx",
    "timestamp": "timestamp",
}


class LubanRealEnv:
    def __init__(
        self,
        # required params
        output_dir,
        robot_ip,
        leap_hand_port,
        realsense_serial_numbers,
        uvc_dev_path,
        tactile_sensor_names,
        # env params
        frequency=10,
        n_obs_steps=2,
        # obs
        obs_image_resolution=(640, 480),
        rotation_rep="rotation_6d",
        max_obs_buffer_size=30,
        obs_key_map=DEFAULT_OBS_KEY_MAP,
        # obs_float32=False,
        # action
        max_pos_speed=3.0,
        max_rot_speed=2.5,
        max_joint_speed=2.0,
        # robot
        tcp_offset=0.0,
        tcp_init_pose=[-0.475, 0.210, 0.400, 3.14159265, 0.0, 0.0],
        workspace_min_xyz_W=[-0.750, 0.020, 0.100],
        workspace_max_xyz_W=[-0.200, 0.400, 0.500],
        # video capture params
        video_capture_fps=30,
        video_capture_resolution=(640, 480),
        # saving params
        # record_raw_video=True,
        thread_per_video=2,
        video_crf=21,
        # vis params
        enable_multi_cam_vis=True,
        multi_cam_vis_resolution=(640, 480),
        # shared memory
        shm_manager=None,
        verbose=False,
    ):
        assert frequency <= video_capture_fps
        output_dir = pathlib.Path(output_dir)
        assert output_dir.parent.is_dir()
        video_dir = output_dir.joinpath("videos")
        video_dir.mkdir(parents=True, exist_ok=True)
        zarr_path = str(output_dir.joinpath("replay_buffer.zarr").absolute())
        replay_buffer = ReplayBuffer.create_from_path(zarr_path=zarr_path, mode="a")

        # =========== rotation transformer ===========
        rotation_transformer = RotationTransformer(from_rep="axis_angle", to_rep=rotation_rep)

        if shm_manager is None:
            shm_manager = SharedMemoryManager()
            shm_manager.start()


        # =========== ur arm ===========

        cube_diag = np.linalg.norm([1, 1, 1])
        # j_init = np.array([0, -90, -90, -90, 90, 0]) / 180 * np.pi
        # if not init_joints:
        #     j_init = None

        ur_arm = RTDEInterpolationController(
            shm_manager=shm_manager,
            robot_ip=robot_ip,
            frequency=125,  # UR5 CB3 RTDE
            lookahead_time=0.1,
            gain=300,
            max_pos_speed=max_pos_speed * cube_diag,
            max_rot_speed=max_rot_speed * cube_diag,
            workspace_min_xyz_W=workspace_min_xyz_W,
            workspace_max_xyz_W=workspace_max_xyz_W,
            launch_timeout=5,
            tcp_offset_pose=[0, 0, tcp_offset, 0, 0, 0],
            payload_mass=None,
            payload_cog=None,
            tcp_init_pose=tcp_init_pose,
            tcp_init_speed=0.05,
            soft_real_time=False,
            receive_keys=None,
            get_max_k=max_obs_buffer_size,
            verbose=verbose,
        )

        # =========== leap hand ===========
        init_qpos = np.zeros(16)
        leap_hand = LeapHandInterpolationController(
            shm_manager=shm_manager,
            robot_port=leap_hand_port,
            frequency=90,
            max_joint_speed=max_joint_speed,  # 5% of max speed
            launch_timeout=5,
            p_gain=800.0,
            d_gain=200.0,
            i_gain=0.0,
            curr_limit=550.0,
            joints_init=init_qpos,
            # joints_init_speed=1.05,
            joint_init_wait_s=1.0,
            soft_real_time=False,
            receive_keys=None,
            get_max_k=max_obs_buffer_size,
            verbose=verbose,
        )

        self.ur_arm = ur_arm
        self.leap_hand = leap_hand
        self.tcp_init_pose = tcp_init_pose
        self.hand_init_qpos = init_qpos


        self.frequency = frequency

        self.max_pos_speed = max_pos_speed
        self.max_rot_speed = max_rot_speed
        self.obs_key_map = obs_key_map
        # recording
        self.output_dir = output_dir
        self.replay_buffer = replay_buffer

        self.rotation_transformer = rotation_transformer

        self.start_time = None

    # ======== start-stop API =============
    @property
    def is_ready(self):
        return (
            self.ur_arm.is_ready
            and self.leap_hand.is_ready
        )

    @property
    def num_episodes(self):
        return self.replay_buffer.n_episodes

    def start(self, wait=True):
        self.ur_arm.start(wait=False)
        self.leap_hand.start(wait=False)
        if wait:
            self.start_wait()

    def stop(self, wait=True):
        self.ur_arm.stop(wait=False)
        self.leap_hand.stop(wait=False)
        self.end_episode()
        if wait:
            self.stop_wait()

    def start_wait(self):
        self.ur_arm.start_wait()
        self.leap_hand.start_wait()


    def stop_wait(self):
        self.ur_arm.stop_wait()
        self.leap_hand.stop_wait()


    def reset(self):
        self.ur_arm.servoL(self.tcp_init_pose, duration=3.0)
        self.leap_hand.servoJ(self.hand_init_qpos, duration=3.0)
        time.sleep(4)

    # ========= context manager ===========
    def __enter__(self):
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()

    # ========= async env API ===========
    def exec_actions(
        self, actions: np.ndarray, timestamps: np.ndarray, stages: Optional[np.ndarray] = None
    ):
        assert self.is_ready
        if not isinstance(actions, np.ndarray):
            actions = np.array(actions)
        if not isinstance(timestamps, np.ndarray):
            timestamps = np.array(timestamps)
        if stages is None:
            stages = np.zeros_like(timestamps, dtype=np.int64)
        elif not isinstance(stages, np.ndarray):
            stages = np.array(stages, dtype=np.int64)

        # convert action to pose
        receive_time = time.time()
        is_new = timestamps > receive_time
        if not np.any(is_new):
            cprint("No new actions to execute!", "yellow")
        new_actions = actions[is_new]
        new_timestamps = timestamps[is_new]
        new_stages = stages[is_new]

        # undo convert actions
        new_exec_actions = _undo_convert_actions(new_actions.copy(), self.rotation_transformer)
        for i in range(len(new_exec_actions)):
            self.ur_arm.schedule_waypoint(
                pose=new_exec_actions[i][:6], target_time=new_timestamps[i]
            )
            self.leap_hand.schedule_waypoint(
                qpos=new_exec_actions[i][6:], target_time=new_timestamps[i]
            )

        # record actions
        if self.action_accumulator is not None:
            self.action_accumulator.put(new_actions, new_timestamps)
        if self.stage_accumulator is not None:
            self.stage_accumulator.put(new_stages, new_timestamps)

    def get_robot_state(self):
        return self.ur_arm.get_state()  # TODO: add leap hand state?

    # recording API
    def start_episode(self, start_time=None):
        "Start recording and return first obs"
        if start_time is None:
            start_time = time.time()
        self.start_time = start_time

        assert self.is_ready

        # prepare recording stuff
        episode_id = self.replay_buffer.n_episodes

        # create accumulators
        self.obs_arm_accumulator = TimestampObsAccumulator(
            start_time=start_time, dt=1 / self.frequency
        )
        self.obs_hand_accumulator = TimestampObsAccumulator(
            start_time=start_time, dt=1 / self.frequency
        )
        self.action_accumulator = TimestampActionAccumulator(
            start_time=start_time, dt=1 / self.frequency
        )
        self.stage_accumulator = TimestampActionAccumulator(
            start_time=start_time, dt=1 / self.frequency
        )
        print(f"Episode {episode_id} started!")

    def end_episode(self):
        "Stop recording"
        assert self.is_ready


        if self.obs_arm_accumulator is not None and self.obs_hand_accumulator is not None:
            # saving low-dim obs and actions to disk
            cprint("Saving low-dim obs and actions to disk...", "yellow")
            # recording
            assert self.action_accumulator is not None
            assert self.stage_accumulator is not None

            # Since the only way to accumulate obs and action is by calling
            # get_obs and exec_actions, which will be in the same thread.
            # We don't need to worry new data come in here.
            obs_arm_data = self.obs_arm_accumulator.data
            obs_hand_data = self.obs_hand_accumulator.data
            obs_arm_timestamps = self.obs_arm_accumulator.timestamps
            obs_hand_timestamps = self.obs_hand_accumulator.timestamps

            actions = self.action_accumulator.actions
            action_timestamps = self.action_accumulator.timestamps
            stages = self.stage_accumulator.actions
            n_steps = min(len(obs_arm_timestamps), len(obs_hand_timestamps), len(action_timestamps))
            if n_steps > 0:
                episode = dict()
                episode["timestamp"] = obs_arm_timestamps[:n_steps]
                episode["action"] = actions[:n_steps]
                episode["stage"] = stages[:n_steps]
                for key, value in obs_arm_data.items():
                    episode[key] = value[:n_steps]
                for key, value in obs_hand_data.items():
                    episode[key] = value[:n_steps]
                self.replay_buffer.add_episode(episode, compressors="disk")
                episode_id = self.replay_buffer.n_episodes - 1
                print(f"Episode {episode_id} saved!")

            self.obs_arm_accumulator = None
            self.obs_hand_accumulator = None
            self.action_accumulator = None
            self.stage_accumulator = None

    def drop_episode(self):
        # self.end_episode()
        self.replay_buffer.drop_episode()
        episode_id = self.replay_buffer.n_episodes
        print(f"Episode {episode_id} dropped!")


def _convert_arm_eef_obs(raw_obs, rotation_transformer):
    # obs = raw_obs

    pos = raw_obs[..., :3]
    rot = raw_obs[..., 3:6]
    rot = rotation_transformer.forward(rot)
    raw_obs = np.concatenate([pos, rot], axis=-1).astype(np.float32)

    obs = raw_obs
    return obs


def _undo_convert_actions(raw_actions, rotation_transformer):
    # actions = raw_actions
    pos = raw_actions[..., :3]
    rot = raw_actions[..., 3:9]
    hand = raw_actions[..., 9:]
    rot = rotation_transformer.inverse(rot)
    raw_actions = np.concatenate([pos, rot, hand], axis=-1).astype(np.float32)

    actions = raw_actions
    return actions
