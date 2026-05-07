""" 
仅仅用于测试，存在线程退出异常
"""

import time
from multiprocessing.managers import SharedMemoryManager
# import cv2
import pathlib
import numpy as np
# import hydra
import scipy.spatial.transform as st
import rospy

# from pynput import keyboard
from functools import cache
import traceback

from termcolor import cprint
from std_msgs.msg import Float64MultiArray
from avp_client import AVPTeleopClient
from diffusion_policy.real_world.keystroke_counter import KeystrokeCounter, KeyCode
from diffusion_policy.real_world.luban_real_env import LubanRealEnv
from diffusion_policy.model.common.rotation_transformer import RotationTransformer
from diffusion_policy.common.precise_sleep import precise_wait

def convert_wrist_rel_to_abs(
    wrist_pose_rel: np.ndarray, tcp_origin_real: np.ndarray, rotation_transformer
):
    wrist_pos_rel, wrist_rotvec_rel = wrist_pose_rel[:3], wrist_pose_rel[3:]

    r_W_E_0 = st.Rotation.from_rotvec(tcp_origin_real[3:])
    wrist_pos_abs = tcp_origin_real[:3] + r_W_E_0.apply(wrist_pos_rel)
    wrist_rotvec_abs = (r_W_E_0 * st.Rotation.from_rotvec(wrist_rotvec_rel)).as_rotvec()
    # return np.hstack([wrist_pos_abs, wrist_rotvec_abs])
    wrist_rot6d_abs = rotation_transformer.forward(wrist_rotvec_abs)
    return np.hstack([wrist_pos_abs, wrist_rot6d_abs])

# from diffusion_policy.real_world.keystroke_counter import KeystrokeCounter, KeyCode
def main():

    
    rotation_transformer = RotationTransformer(from_rep="axis_angle", to_rep="rotation_6d")

    # Flags for foot pedal to control the recording flow
    stop_recording = False
    pause = True
    reset = True
    exit_curr_episode = False
    drop_curr_episode = False

    tcp_init_pose=[-0.475, 0.210, 0.400, 3.14159265, 0.0, 0.0]

    dt = 1.0 / 10
    with SharedMemoryManager() as shm_manager:
        with KeystrokeCounter() as key_counter,  AVPTeleopClient(
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
        ) as teleop_client, LubanRealEnv(
            frequency=10,
            output_dir="data",
            robot_ip='10.42.0.123',
            max_pos_speed=2.0,
            max_rot_speed=0.5,
            tcp_init_pose=tcp_init_pose,
            workspace_min_xyz_W=[-0.750, 0.020, 0.100],
            workspace_max_xyz_W=[-0.200, 0.400, 0.500],
              # sensors
            realsense_serial_numbers=[],
            uvc_dev_path=None,
            # uvc_dev_path: null
            tactile_sensor_names=[],
            # leap hand
            leap_hand_port='/dev/ttyDXL',
            max_joint_speed=3.0,
            shm_manager=shm_manager,
            verbose=False,
        ) as env:
            # teleop_client.reset()
            time.sleep(2)
            print("Ready!")
            while not stop_recording:
                cprint(f"Collecting Episode {env.num_episodes}", "green")
                t_start = time.monotonic()
                iter_idx = 0
                while True:
                   # calculate timing
                    t_loop_start = time.monotonic()
                    # t_cycle_end = t_start + (iter_idx + 1) * dt
                    t_cycle_end = t_loop_start + dt
                    t_command_target = t_cycle_end + dt

                    # handle key presses
                    press_events = key_counter.get_press_events()
                    release_events = key_counter.get_release_events()
                    for key_stroke in press_events:
                        if key_stroke == KeyCode(char="q"):
                            cprint("System is QUITTING", "yellow")
                            # Exit program
                            pause = True
                            stop_recording = True
                            exit_curr_episode = True
                        elif key_stroke == KeyCode(char="s"):
                            # Start recording
                            if reset:
                                # reset the client
                                teleop_client.reset()
                                reset = False
                            # env.start_episode(t_cycle_end + dt)
                            # env.start_episode(
                            #     t_start + (iter_idx + 2) * dt - time.monotonic() + time.time()
                            # )
                            env.start_episode(t_command_target + 0.5*dt - time.monotonic() + time.time())  # TODO: check this
                            # env.start_episode(t_loop_start)
                            key_counter.clear()
                            pause = False
                            cprint("Recording!", "green")
                        elif key_stroke == KeyCode(char="1"):
                            # Save recording (auto save)
                            drop_curr_episode = False
                            # env.reset()
                            key_counter.clear()
                            pause = True
                            reset = True
                            exit_curr_episode = True
                            cprint("Saved episode! Next episode will start recording.", "blue")
                        elif key_stroke == KeyCode(char="r"):
                            # drop recording
                            # env.drop_episode()
                            drop_curr_episode = True
                            # env.reset()
                            key_counter.clear()
                            pause = True
                            reset = True
                            exit_curr_episode = True
                            cprint("Drop episode! This episode will start recording again.", "red")

                    for key_stroke in release_events:
                        if key_stroke == KeyCode(char="p"):
                            # Stop recording
                            env.end_episode()
                            key_counter.clear()
                            pause = True
                            reset = True
                            cprint(
                                "Recording paused! Please choose to DROP (left pedal) or SAVE (middle pedal) this episode",
                                "dark_grey",
                            )

                    # pump obs
                    episode_id = env.replay_buffer.n_episodes
                    # text = f"Episode: {episode_id}, Stage: {stage}"
                    text = f"Episode: {episode_id}"
                    if not pause:
                        text += ", Recording!"
                    else:
                        text += ", Paused!"


                    # precise_wait(t_sample)  # TODO: check this

                    if pause:
                        # print("Paused...")
                        precise_wait(t_cycle_end)
                    else:
                        # Get teleop command
                        teleop_motion = teleop_client.get_motion_state()
                        # teleop_motion = {}
                        # teleop_motion["motion_event"] = np.zeros(22)
                        arm_tcp_motion_rel = teleop_motion["motion_event"][:6]
                        hand_qpos_motion_abs = teleop_motion["motion_event"][6:]
                        arm_tcp_motion_abs = convert_wrist_rel_to_abs(
                            wrist_pose_rel=arm_tcp_motion_rel,
                            tcp_origin_real=np.array(tcp_init_pose),
                            rotation_transformer=rotation_transformer,
                        )
                        assert arm_tcp_motion_abs.shape == (9,)
                        target_action = np.hstack([arm_tcp_motion_abs, hand_qpos_motion_abs])

                        # Execute teleop command
                        env.exec_actions(
                            actions=[target_action],
                            timestamps=[t_command_target - time.monotonic() + time.time()],
                            # stages=[stage],
                        )
                        print(target_action)

                        precise_wait(t_cycle_end)
                        actual_dt = time.monotonic() - t_loop_start
                        if actual_dt > dt + 0.001:
                            cprint(f"Warning: loop took {actual_dt:.3f}s", "red")
                        iter_idx += 1
                        # print(f"iter_idx: {iter_idx}")

                    if exit_curr_episode:
                        exit_curr_episode = False
                        break

                # >>>>> end of current episode <<<<<
                if not stop_recording:
                    if drop_curr_episode:
                        env.drop_episode()
                        drop_curr_episode = False
                    env.reset()



if __name__ == "__main__":
    main()