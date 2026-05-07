import os
import time
import enum
import multiprocessing as mp
from multiprocessing.managers import SharedMemoryManager
import scipy.interpolate as si
import scipy.spatial.transform as st
import numpy as np

from .leap_hand_utils.dynamixel_client import DynamixelClient
from .leap_hand_utils.pose_trajectory_interpolator import PoseTrajectoryInterpolator
from .leap_hand_utils import leap_hand_utils as lhu

from diffusion_policy.shared_memory.shared_memory_queue import SharedMemoryQueue, Empty
from diffusion_policy.shared_memory.shared_memory_ring_buffer import SharedMemoryRingBuffer
from diffusion_policy.common.precise_sleep import precise_wait


class Command(enum.Enum):
    STOP = 0
    SERVOJ = 1
    SCHEDULE_WAYPOINT = 2


class LeapHandInterpolationController(mp.Process):
    """
    To ensure sending command to the robot with predictable latency
    this controller need its separate process (due to python GIL)
    """

    def __init__(
        self,
        shm_manager: SharedMemoryManager,
        robot_port: str = "/dev/ttyDXL",
        frequency=90,
        max_joint_speed=1.0,  # 5% of max speed
        launch_timeout=3,
        p_gain: float = 800.0,
        d_gain: float = 200.0,
        i_gain: float = 0.0,
        curr_limit: float = 550.0,
        joints_init=None,
        # joints_init_speed=1.05,
        joint_init_wait_s=1.0,
        soft_real_time=False,
        verbose=False,
        receive_keys=None,
        get_max_k=128,
    ):
        # verify
        assert 0 < frequency <= 90
        assert 0 < max_joint_speed < 3.14  # TODO: XC330 motor speed limit
        assert curr_limit < 600

        if joints_init is not None:
            joints_init = np.array(joints_init)
            assert joints_init.shape == (16,)
        else:
            joints_init = np.zeros((16,), dtype=np.float64)

        super().__init__(name="LeapHandInterpolationController")
        self.robot_port = robot_port
        self.frequency = frequency
        self.max_joint_speed = max_joint_speed
        self.launch_timeout = launch_timeout
        self.joints_init = joints_init
        self.joint_init_wait_s = joint_init_wait_s
        # self.joints_init_speed = joints_init_speed
        self.soft_real_time = soft_real_time
        self.verbose = verbose

        # build input queue
        example = {
            "cmd": Command.SCHEDULE_WAYPOINT.value,
            "target_pose": np.zeros((16,), dtype=np.float64),
            "duration": 0.0,
            "target_time": 0.0,
        }
        input_queue = SharedMemoryQueue.create_from_examples(
            shm_manager=shm_manager, examples=example, buffer_size=256
        )

        # build ring buffer
        if receive_keys is None:
            receive_keys = [
                "actual_qpos",
                # 'ActualQVel',
                # 'targetqpos',
            ]

        motors = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]
        try:
            self.dxl_client = DynamixelClient(motors, robot_port, 4000000)
            self.dxl_client.connect()
        except Exception as e:
            raise f"{e}: Cannot connect to Dynamixel Client, please check the connection"

        self.dxl_client.sync_write(motors, np.ones(len(motors)) * 5, 11, 1)

        self.dxl_client.sync_write(motors, np.ones(len(motors)) * p_gain, 84, 2)  # Pgain stiffness
        self.dxl_client.sync_write(
            [0, 4, 8], np.ones(3) * (p_gain * 0.75), 84, 2
        )  # Pgain stiffness for side to side should be a bit less
        self.dxl_client.sync_write(motors, np.ones(len(motors)) * i_gain, 82, 2)  # Igain
        self.dxl_client.sync_write(motors, np.ones(len(motors)) * d_gain, 80, 2)  # Dgain damping
        self.dxl_client.sync_write(
            [0, 4, 8], np.ones(3) * (d_gain * 0.75), 80, 2
        )  # Dgain damping for side to side should be a bit less
        # Max at current (in unit 1ma) so don't overheat and grip too hard #500 normal or #350 for lite
        self.dxl_client.sync_write(motors, np.ones(len(motors)) * curr_limit, 102, 2)

        self.relax_qpos = self.get_actual_qpos()

        self.dxl_client.set_torque_enabled(motors, True)

        for key in receive_keys:
            example[key] = np.array(getattr(self, "get_" + key)())
        example["robot_receive_timestamp"] = time.time()
        ring_buffer = SharedMemoryRingBuffer.create_from_examples(
            shm_manager=shm_manager,
            examples=example,
            get_max_k=get_max_k,
            get_time_budget=0.2,
            put_desired_frequency=frequency,
        )

        self.motors = motors
        self.ready_event = mp.Event()
        self.input_queue = input_queue
        self.ring_buffer = ring_buffer
        self.receive_keys = receive_keys

    def get_actual_qpos(self):
        pos_raw = self.dxl_client.read_pos()
        curr_qpos = lhu.LEAPhand_to_allegro(pos_raw, zeros=False)
        assert curr_qpos.shape == (16,)
        return curr_qpos

    # ========= launch method ===========
    def start(self, wait=True):
        super().start()
        if wait:
            self.start_wait()
        if self.verbose:
            print(f"[LeapHandController] Controller process spawned at {self.pid}")

    def stop(self, wait=True):
        self.servoJ(self.relax_qpos, duration=1)
        time.sleep(1.5)
        message = {"cmd": Command.STOP.value}
        self.input_queue.put(message)
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
    def servoJ(self, qpos, duration=0.1):
        """
        duration: desired time to reach pose
        """
        assert self.is_alive()
        assert duration >= (1 / self.frequency)
        qpos = np.array(qpos)
        assert qpos.shape == (16,)

        message = {"cmd": Command.SERVOJ.value, "target_pose": qpos, "duration": duration}
        self.input_queue.put(message)

    def schedule_waypoint(self, qpos, target_time):
        assert target_time > time.time()
        qpos = np.array(qpos)
        assert qpos.shape == (16,)

        message = {
            "cmd": Command.SCHEDULE_WAYPOINT.value,
            "target_pose": qpos,
            "target_time": target_time,
        }
        self.input_queue.put(message)

    # ========= receive APIs =============
    def get_state(self, k=None, out=None):
        if k is None:
            return self.ring_buffer.get(out=out)
        else:
            return self.ring_buffer.get_last_k(k=k, out=out)

    def get_all_state(self):
        return self.ring_buffer.get_all()

    # ========= main loop in process ============
    def run(self):
        # enable soft real-time
        if self.soft_real_time:
            os.sched_setscheduler(0, os.SCHED_RR, os.sched_param(20))

        try:
            if self.verbose:
                print(f"[LeapHandPositionalController] Connect to robot: {self.robot_port}")

            # init qpos
            dt = 1.0 / self.frequency
            curr_qpos = self.get_actual_qpos()
            # use monotonic time to make sure the control loop never go backward
            start_loop_t = time.monotonic()
            last_waypoint_time = start_loop_t
            pose_interp = PoseTrajectoryInterpolator(times=[start_loop_t], poses=[curr_qpos])
            # pose_interp = pose_interp.schedule_waypoint(
            #     pose=self.joints_init,
            #     time=start_loop_t + self.joint_init_wait_s,
            #     curr_time=start_loop_t,
            #     last_waypoint_time=last_waypoint_time,
            # )
            pose_interp = pose_interp.drive_to_waypoint(
                pose=self.joints_init,
                time=start_loop_t + self.joint_init_wait_s,
                curr_time=start_loop_t,
                max_pos_speed=self.max_joint_speed,
            )

            # main loop
            iter_idx = 0
            keep_running = True
            while keep_running:
                # start control iteration
                # send command to robot
                t_start = t_now = time.monotonic()
                # diff = t_now - pose_interp.times[-1]
                # if diff > 0:
                #     print('extrapolate', diff)
                qpos_command = pose_interp(t_now)
                self.dxl_client.write_desired_pos(
                    self.motors, lhu.allegro_to_LEAPhand(qpos_command, zeros=False)
                )

                # update robot state
                state = dict()
                for key in self.receive_keys:
                    state[key] = np.array(getattr(self, "get_" + key)())
                state["robot_receive_timestamp"] = time.time()
                self.ring_buffer.put(state)

                # fetch command from queue
                try:
                    commands = self.input_queue.get_all()
                    n_cmd = len(commands["cmd"])
                except Empty:
                    n_cmd = 0

                # execute commands
                for i in range(n_cmd):
                    command = dict()
                    for key, value in commands.items():
                        command[key] = value[i]
                    cmd = command["cmd"]

                    if cmd == Command.STOP.value:
                        keep_running = False
                        # stop immediately, ignore later commands
                        break
                    elif cmd == Command.SERVOJ.value:
                        # since curr_pose always lag behind curr_target_pose
                        # if we start the next interpolation with curr_pose
                        # the command robot receive will have discontinouity
                        # and cause jittery robot behavior.
                        target_pose = command["target_pose"]
                        duration = float(command["duration"])
                        curr_time = t_now + dt
                        t_insert = curr_time + duration
                        pose_interp = pose_interp.drive_to_waypoint(
                            pose=target_pose,
                            time=t_insert,
                            curr_time=curr_time,
                            max_pos_speed=self.max_joint_speed,
                            # max_rot_speed=self.max_rot_speed,
                        )
                        last_waypoint_time = t_insert
                        if self.verbose:
                            print(
                                "[LeapHandPositionalController] New pose target:{} duration:{}s".format(
                                    target_pose, duration
                                )
                            )
                    elif cmd == Command.SCHEDULE_WAYPOINT.value:
                        target_pose = command["target_pose"]
                        target_time = float(command["target_time"])
                        # translate global time to monotonic time
                        target_time = time.monotonic() - time.time() + target_time
                        curr_time = t_now + dt
                        pose_interp = pose_interp.schedule_waypoint(
                            pose=target_pose,
                            time=target_time,
                            max_pos_speed=self.max_joint_speed,
                            curr_time=curr_time,
                            last_waypoint_time=last_waypoint_time,
                        )
                        last_waypoint_time = target_time
                    else:
                        keep_running = False
                        break

                # regulate frequency
                # rtde_c.waitPeriod(t_start)
                precise_wait(t_start + dt, slack_time=0.001)

                # first loop successful, ready to receive command
                # if iter_idx == 0:
                #     self.ready_event.set()
                if self.ready_event.is_set() is False and (
                    t_start + dt - start_loop_t > self.joint_init_wait_s * 1.2
                ):
                    self.ready_event.set()

                iter_idx += 1

                if self.verbose:
                    print(
                        f"[LeapHandPositionalController] Actual frequency {1/(time.time() - t_start)}"
                    )
        except Exception as e:
            print(f"[LeapHandPositionalController] Error: {e}")
        finally:
            # manditory cleanup
            # decelerate
            self.dxl_client.set_torque_enabled(self.motors, False)

            # terminate
            self.dxl_client.disconnect()
            self.ready_event.set()

            # if self.verbose:
            print(f"[LeapHandPositionalController] Disconnected from robot: {self.robot_port}")
