import time
from typing import Optional, Callable, Dict
import multiprocessing as mp
import contextlib
from typing import List
import enum
import pathlib
import numpy as np
from termcolor import cprint
import rospy
from cv_bridge import CvBridge
from sensor_msgs.msg import Image
from std_msgs.msg import Float32MultiArray
import cv2
from threadpoolctl import threadpool_limits
from multiprocessing.managers import SharedMemoryManager
from diffusion_policy.common.timestamp_accumulator import get_accumulate_timestamp_idxs
from diffusion_policy.shared_memory.shared_memory_ring_buffer import SharedMemoryRingBuffer
from diffusion_policy.shared_memory.shared_memory_queue import SharedMemoryQueue, Full, Empty
from diffusion_policy.real_world.video_recorder_uvc import VideoRecorderUVC as VideoRecorder
from diffusion_policy.common.precise_sleep import precise_wait


class Command(enum.Enum):
    # SET_COLOR_OPTION = 0
    # SET_DEPTH_OPTION = 1
    START_RECORDING = 2
    STOP_RECORDING = 3
    RESTART_PUT = 4


class SingleDotTip(mp.Process):
    MAX_PATH_LENGTH = 4096  # linux path has a limit of 4096 bytes

    def __init__(
        self,
        shm_manager: SharedMemoryManager,
        name: str,
        image_topic: str,
        contact_topic: str = "",
        use_contact_inference: bool = False,
        rotation: int = None,
        flip: str = None,
        resolution=(192, 192),
        capture_fps=30,
        put_fps=None,
        put_downsample=True,
        record_fps=None,
        get_max_k=30,
        transform: Optional[Callable[[Dict], Dict]] = None,
        video_recorder: Optional[VideoRecorder] = None,
        verbose=False,
    ):
        super().__init__()
        if put_fps is None:
            put_fps = capture_fps
        if record_fps is None:
            record_fps = capture_fps

        # create ring buffer
        resolution = tuple(resolution)
        shape = resolution[::-1]
        examples = dict()
        # examples["gray"] = np.empty(shape=shape + (3,), dtype=np.uint8)
        examples["gray"] = np.empty(shape=shape, dtype=np.uint8)

        examples["camera_capture_timestamp"] = 0.0
        examples["camera_receive_timestamp"] = 0.0
        examples["timestamp"] = 0.0
        examples["step_idx"] = 0

        vis_ring_buffer = SharedMemoryRingBuffer.create_from_examples(
            shm_manager=shm_manager,
            examples=examples if transform is None else transform(dict(examples)),
            get_max_k=1,
            get_time_budget=0.2,
            put_desired_frequency=capture_fps,
        )

        ring_buffer = SharedMemoryRingBuffer.create_from_examples(
            shm_manager=shm_manager,
            examples=examples if transform is None else transform(dict(examples)),
            get_max_k=get_max_k,
            get_time_budget=0.2,
            put_desired_frequency=put_fps,
        )

        self.video_recorder_example = np.empty(shape=shape, dtype=np.uint8)

        # create command queue
        examples = {
            "cmd": Command.STOP_RECORDING.value,
            "video_path": np.array("a" * self.MAX_PATH_LENGTH),
            "recording_start_time": 0.0,
            "put_start_time": 0.0,
        }

        command_queue = SharedMemoryQueue.create_from_examples(
            shm_manager=shm_manager, examples=examples, buffer_size=128
        )

        # create video recorder
        if video_recorder is None:
            # realsense uses bgr24 pixel format
            # default thread_type to FRAEM
            # i.e. each frame uses one core
            # instead of all cores working on all frames.
            # this prevents CPU over-subpscription and
            # improves performance significantly
            video_recorder = VideoRecorder.create_h264(
                fps=record_fps,
                codec="h264",
                input_pix_fmt="gray",
                crf=18,
                thread_type="FRAME",
                thread_count=4,
            )

        self.name = name
        self.image_topic = image_topic
        self.contact_topic = contact_topic
        self.use_contact_inference = use_contact_inference
        self.rotation = rotation
        self.flip = flip
        self.resolution = resolution
        self.capture_fps = capture_fps
        self.put_fps = put_fps
        self.put_downsample = put_downsample
        self.record_fps = record_fps
        self.transform = transform
        self.video_recorder = video_recorder
        self.verbose = verbose
        self.put_start_time = None

        # shared variables
        self.shm_manager = shm_manager
        self.stop_event = mp.Event()
        self.ready_event = mp.Event()
        self.ring_buffer = ring_buffer
        self.vis_ring_buffer = vis_ring_buffer
        self.command_queue = command_queue

    # ========= context manager ===========
    def __enter__(self):
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()

        # ========= user API ===========

    def start(self, wait=True, put_start_time=None):
        self.put_start_time = put_start_time
        # shape = self.resolution[::-1]
        # data_example = np.empty(shape=shape + (3,), dtype=np.uint8)
        self.video_recorder.start(
            shm_manager=self.shm_manager, data_example=self.video_recorder_example
        )
        super().start()
        if wait:
            self.start_wait()

    def stop(self, wait=True):
        self.video_recorder.stop()
        self.stop_event.set()
        if wait:
            self.end_wait()

    def start_wait(self):
        self.ready_event.wait()
        self.video_recorder.start_wait()

    def end_wait(self):
        self.join()
        self.video_recorder.end_wait()

    @property
    def is_ready(self):
        return self.ready_event.is_set()

    def get(self, k=None, out=None):
        if k is None:
            return self.ring_buffer.get(out=out)
        else:
            return self.ring_buffer.get_last_k(k, out=out)

    def get_vis(self, out=None):
        return self.vis_ring_buffer.get(out=out)

    def start_recording(self, video_path: str, start_time: float = -1):
        # assert self.enable_color

        path_len = len(video_path.encode("utf-8"))
        if path_len > self.MAX_PATH_LENGTH:
            raise RuntimeError("video_path too long.")
        self.command_queue.put(
            {
                "cmd": Command.START_RECORDING.value,
                "video_path": video_path,
                "recording_start_time": start_time,
            }
        )

    def stop_recording(self):
        self.command_queue.put({"cmd": Command.STOP_RECORDING.value})

    def restart_put(self, start_time):
        self.command_queue.put({"cmd": Command.RESTART_PUT.value, "put_start_time": start_time})

    def run(self):
        with contextlib.suppress(rospy.exceptions.ROSException):
            rospy.init_node(f"dotview_{self.name}_node", anonymous=True)

        sensor = ROSDotTip(
            name=self.name,
            image_topic=self.image_topic,
            contact_topic=self.contact_topic,
            use_contact_inference=self.use_contact_inference,
            rotation=self.rotation,
            flip=self.flip,
        )
        ts = time.time()
        while not sensor.is_connected:
            sensor.connect()
            if time.time() - ts > 5:
                raise f"Failed to connect to {self.name} DotTip sensor."
        print(f"Connected to {self.name} DotTip sensor.")
        time.sleep(1)

        # w, h = self.resolution
        fps = self.capture_fps
        dt = 1.0 / fps

        try:
            # put frequency regulation
            put_idx = None
            put_start_time = self.put_start_time
            if put_start_time is None:
                put_start_time = time.time()

            iter_idx = 0
            t_start = time.time()
            while not self.stop_event.is_set():
                t_loop_start = time.monotonic()
                raw_img = sensor.read_image()
                if self.use_contact_inference:
                    angle, force = sensor.read_contact_info()
                receive_time = time.time()

                # grab data
                data = dict()
                data["camera_receive_timestamp"] = receive_time
                # realsense report in ms
                data["camera_capture_timestamp"] = receive_time  # TODO: get the real timestamp
                data["gray"] = raw_img

                # apply transform
                put_data = data
                if self.transform is not None:
                    put_data = self.transform(dict(data))
                if self.put_downsample:
                    # put frequency regulation
                    local_idxs, global_idxs, put_idx = get_accumulate_timestamp_idxs(
                        timestamps=[receive_time],
                        start_time=put_start_time,
                        dt=1 / self.put_fps,
                        # this is non in first iteration
                        # and then replaced with a concrete number
                        next_global_idx=put_idx,
                        # continue to pump frames even if not started.
                        # start_time is simply used to align timestamps.
                        allow_negative=True,
                    )

                    for step_idx in global_idxs:
                        put_data["step_idx"] = step_idx
                        # put_data['timestamp'] = put_start_time + step_idx / self.put_fps
                        put_data["timestamp"] = receive_time
                        # print(step_idx, data['timestamp'])
                        self.ring_buffer.put(put_data, wait=False)
                else:
                    step_idx = int((receive_time - put_start_time) * self.put_fps)
                    put_data["step_idx"] = step_idx
                    put_data["timestamp"] = receive_time
                    self.ring_buffer.put(put_data, wait=False)

                # signal ready
                if iter_idx == 0:
                    self.ready_event.set()

                # put to vis
                vis_data = put_data
                self.vis_ring_buffer.put(vis_data, wait=False)

                # record frame
                rec_data = put_data
                if self.video_recorder.is_ready():
                    self.video_recorder.write_frame(rec_data["gray"], frame_time=receive_time)

                # perf
                t_end = time.time()
                duration = t_end - t_start
                frequency = np.round(1 / duration, 1)
                t_start = t_end
                if self.verbose:
                    print(f"[DotTip {self.name}] FPS {frequency}")

                # fetch command from queue
                try:
                    commands = self.command_queue.get_all()
                    n_cmd = len(commands["cmd"])
                except Empty:
                    n_cmd = 0

                # execute commands
                for i in range(n_cmd):
                    command = dict()
                    for key, value in commands.items():
                        command[key] = value[i]
                    cmd = command["cmd"]
                    if cmd == Command.START_RECORDING.value:
                        video_path = str(command["video_path"])
                        start_time = command["recording_start_time"]
                        if start_time < 0:
                            start_time = None
                        self.video_recorder.start_recording(video_path, start_time=start_time)
                    elif cmd == Command.STOP_RECORDING.value:
                        # self.video_recorder.stop()
                        self.video_recorder.stop_recording()
                        # stop need to flush all in-flight frames to disk, which might take longer than dt.
                        # soft-reset put to drop frames to prevent ring buffer overflow.
                        put_idx = None
                    elif cmd == Command.RESTART_PUT.value:
                        put_idx = None
                        put_start_time = command["put_start_time"]
                        # self.ring_buffer.clear()
                iter_idx += 1
                precise_wait(t_loop_start + dt)

        except Exception as e:
            cprint(f"DotTip {self.name} Error: {e}", "red")

        finally:
            self.video_recorder.stop()
            sensor.disconnect()
            self.ready_event.set()

        if self.verbose:
            print(f"DotTip {self.name} Exiting worker process.")


class ROSDotTip:
    def __init__(self, name, image_topic, contact_topic, use_contact_inference, rotation, flip):
        self.name = name
        self.camera_topic = image_topic
        self.contact_topic = contact_topic
        self.use_contact_inference = use_contact_inference

        self.rotation = None
        if rotation == -90:
            self.rotation = cv2.ROTATE_90_COUNTERCLOCKWISE
        elif rotation == 90:
            self.rotation = cv2.ROTATE_90_CLOCKWISE
        elif rotation == 180:
            self.rotation = cv2.ROTATE_180

        self.flip = None
        if flip == "horizontal":
            self.flip = np.fliplr
        elif flip == "vertical":
            self.flip = np.flipud

        # try:
        #     rospy.init_node("{}".format(f"dotview_{self.name}_node"), disable_signals=True)
        # except rospy.ROSException:
        #     pass

        self.bridge = CvBridge()
        self.image_sub = None
        self.contact_sub = None
        self.is_connected = False
        self.tactile_image = None
        self.contact_data = None
        self.contact_angle, self.contact_force = None, None
        self.logs = {}

    def connect(self):
        self.image_sub = rospy.Subscriber(
            self.camera_topic, Image, self._callback_image, queue_size=1, buff_size=2**24
        )
        rospy.sleep(0.5)
        if self.image_sub.get_num_connections() == 0:
            raise ValueError(
                f"No connection to the dotview image topic: {self.camera_topic}, please check the topic."
            )

        if self.use_contact_inference:
            self.contact_sub = rospy.Subscriber(
                self.contact_topic, Float32MultiArray, self._callback_contact, queue_size=1
            )
            rospy.sleep(0.5)
            if self.contact_sub.get_num_connections() == 0:
                raise ValueError(
                    f"No connection to the dotview contact info topic: {self.contact_topic}, please check the contact topic."
                )
        self.is_connected = True

    def _callback_image(self, msg):
        start_time = rospy.get_time()
        cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding="passthrough")
        if self.rotation is not None:
            cv_image = cv2.rotate(cv_image, self.rotation)
        if self.flip is not None:
            cv_image = self.flip(cv_image)

        # log the number of seconds it took to read the image
        self.logs["delta_timestamp_s"] = rospy.get_time() - start_time

        # log the utc time at which the image was received
        # self.logs["timestamp_utc"] = capture_timestamp_utc()
        self.tactile_image = cv_image

    def _callback_contact(self, msg):
        self.contact_data = np.array(list(msg.data))

    def read_image(self):
        if not self.is_connected:
            raise f"ROS DotTip {self.name} is not connected. Try running `camera.connect()` first."
        assert (
            self.tactile_image is not None
        ), "Tactile image is not available. Please check if the image topic is connected."
        return self.tactile_image

    # def read_all(self) -> np.ndarray | tuple[np.ndarray, np.ndarray, np.ndarray]:
    def read_all(self):
        if not self.is_connected:
            raise f"ROS DotTip {self.name} is not connected. Try running `camera.connect()` first."
        assert (
            self.tactile_image is not None
        ), "Tactile image is not available. Please check if the image topic is connected."
        if not self.use_contact_inference:  # Only read the image
            return self.tactile_image
        angle, force = self.read_contact_info()
        return self.tactile_image, angle, force

    def read_contact_info(self, smooth_alpha: float = 1.0) -> tuple[np.ndarray, np.ndarray]:
        if self.contact_data is None:
            raise ValueError(
                "Contact data is not available. Please check if the contact topic is connected."
            )
        # delete normal force that is -
        if self.contact_data[-1] > 0:
            self.contact_data[-1] = 0

        if self.contact_angle is None or self.contact_force is None:
            self.contact_angle = self.contact_data[:2]
            self.contact_force = self.contact_data[2:]
        else:
            self.contact_angle = self.contact_angle + smooth_alpha * (
                self.contact_data[:2] - self.contact_angle
            )
            self.contact_force = self.contact_force + smooth_alpha * (
                self.contact_data[2:] - self.contact_force
            )

        return self.contact_angle, self.contact_force

    def disconnect(self):
        if not self.is_connected:
            raise f"ROS Web Camera {self.name} is not connected. Try running `camera.connect()` first."

        if self.image_sub is not None:
            self.image_sub.unregister()
        if self.contact_sub is not None:
            self.contact_sub.unregister()
        self.image_sub = None
        self.contact_sub = None
        self.is_connected = False

        print(f"Disconnected from {self.name} DotTip sensor.")
