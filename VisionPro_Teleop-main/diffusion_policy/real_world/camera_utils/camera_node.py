import pickle
import threading
import time
from typing import Optional, Tuple
from multiprocessing.managers import SharedMemoryManager
from diffusion_policy.shared_memory.shared_memory_ring_buffer import SharedMemoryRingBuffer

import numpy as np
import zmq

from diffusion_policy.real_world.camera_utils.cameras.camera import CameraDriver

DEFAULT_CAMERA_PORT = 5000


class ZMQClientCamera(CameraDriver):
    """A class representing a ZMQ client for a leader robot."""

    def __init__(self, port: int = DEFAULT_CAMERA_PORT, host: str = "127.0.0.1"):
        self._context = zmq.Context()
        self._socket = self._context.socket(zmq.REQ)
        self._socket.connect(f"tcp://{host}:{port}")

    def read(
        self,
        img_size: Optional[Tuple[int, int]] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Get the current state of the leader robot.

        Returns:
            T: The current state of the leader robot.
        """
        # pack the image_size and send it to the server
        send_message = pickle.dumps(img_size)
        self._socket.send(send_message)
        state_dict = pickle.loads(self._socket.recv())
        return state_dict


class ZMQServerCamera:
    def __init__(
        self,
        camera: CameraDriver,
        port: int = DEFAULT_CAMERA_PORT,
        host: str = "127.0.0.1",
    ):
        self._camera = camera
        self._context = zmq.Context()
        self._socket = self._context.socket(zmq.REP)
        addr = f"tcp://{host}:{port}"
        debug_message = f"Camera Sever Binding to {addr}, Camera: {camera}"
        print(debug_message)
        self._timout_message = f"Timeout in Camera Server, Camera: {camera}"
        self._socket.bind(addr)
        self._stop_event = threading.Event()
    

    def serve(self) -> None:
        """Serve the leader robot state over ZMQ."""
        self._socket.setsockopt(zmq.RCVTIMEO, 1000)  # Set timeout to 1000 ms
        while not self._stop_event.is_set():
            try:
                message = self._socket.recv()
                img_size = pickle.loads(message)
                camera_read = self._camera.read(img_size)
                self._socket.send(pickle.dumps(camera_read))
            except zmq.Again:
                print(self._timout_message)
                # Timeout occurred, check if the stop event is set

    def stop(self) -> None:
        """Signal the server to stop serving."""
        self._stop_event.set()


class ZMQServerCameraFaster:
    def __init__(
        self,
        camera: CameraDriver,
        port: int = DEFAULT_CAMERA_PORT,
        host: str = "127.0.0.1",
    ):
        self._camera = camera
        self._context = zmq.Context()
        self._socket = self._context.socket(zmq.REP)
        addr = f"tcp://{host}:{port}"
        debug_message = f"Camera Sever Binding to {addr}, Camera: {camera}"
        print(debug_message)
        self._timout_message = f"Timeout in Camera Server, Camera: {camera}"
        self._socket.bind(addr)
        self._stop_event = threading.Event()

        self.cam_buffer = None
        self.refresh_interval = 1 / 30  # Refresh every 1/30 second
        self.refresh_thread = threading.Thread(target=self._refresh_buffer)
        self.refresh_thread.daemon = True
        self.refresh_thread.start()

        shm_manager = SharedMemoryManager()
        shm_manager.start()
        examples = dict()
        examples["color"] = np.zeros((480, 640, 3), dtype=np.uint8)
        examples["depth"] = np.zeros((480, 640), dtype=np.uint16)
        examples["camera_capture_timestamp"] = 0.0

        ring_buffer = SharedMemoryRingBuffer.create_from_examples(
            shm_manager=shm_manager,
            examples=examples,
            get_max_k=2,
            get_time_budget=0.2,
            put_desired_frequency=30,
        )
        self.ring_buffer = ring_buffer

    def serve(self) -> None:
        """Serve the leader robot state over ZMQ."""
        self._socket.setsockopt(zmq.RCVTIMEO, 1000)  # Set timeout to 1000 ms
        data = None
        while not self._stop_event.is_set():
            try:
                _ = self._socket.recv()
                # data = self.ring_buffer.get()
                data = self.ring_buffer.get_last_k(k=2, out=data)
                # print(data)
                if data is not None:
                    self._socket.send(pickle.dumps(data))
                else:
                    self._socket.send(b"Buffer is empty.")         
                       
                # if self.cam_buffer is not None:
                #     self._socket.send(pickle.dumps(self.cam_buffer))
                # else:
                #     self._socket.send(b"Buffer is empty.")
            except zmq.Again:
                print(self._timout_message)

    def _refresh_buffer(self):
        """Periodically refresh the buffer."""
        data = dict()
        t_last = 0.0
        count = 1
        t_0 = time.time()
        t_temp = time.time()
        while not self._stop_event.is_set():
            t_1 = time.time()
            self.cam_buffer = self._camera.read()
            t_2 = time.time()
            data["color"], data["depth"], data["camera_capture_timestamp"] = self.cam_buffer
            _, _, t = self.cam_buffer
            if t_last == 0.0:
                t_last = t
            if t - t_last < 0.03:
                print("Too fast")
            elif t - t_last > 0.05:
                print("Too slow")
            t_last = t

            # print("fps:", 1 / (t_2 - t_temp))
            t_temp = t_2
            self.ring_buffer.put(data)
            # print("read time:", t_2 - t_1)
            # _, _, _time = self.cam_buffer
            # print(_time)
            # print("receive time:", time.time())
            # if t_2 - t_0 < self.refresh_interval * count:
            #     time.sleep(self.refresh_interval * count - (t_2 - t_0))
            # count += 1

    def stop(self) -> None:
        """Signal the server to stop serving."""
        self._stop_event.set()