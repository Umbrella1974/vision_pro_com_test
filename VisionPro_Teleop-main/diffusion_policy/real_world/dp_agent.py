import multiprocessing as mp
import threading

# from scipy.spatial.transform import Rotation as R
import time
from termcolor import cprint
import zmq
import pickle
import torch

from diffusion_policy.common.pytorch_util import dict_apply
from diffusion_policy.policy.base_image_policy import BaseImagePolicy
from diffusion_policy.common.temporal_ensemble import EnsembleBuffer


DEFAULT_INFERENCE_PORT = 5555


class DPInferenceZMQServer(mp.Process):
    # class DPInferenceZMQServer(mp.Process):
    def __init__(
        self,
        # shm_manager,
        shape_meta,
        policy: BaseImagePolicy,
        port: int = DEFAULT_INFERENCE_PORT,
        device: str = "cpu",
        verbose=False,
    ):
        super().__init__(name="DPAgent")

        # # build zmq socket
        # self.context = zmq.Context()
        # self.socket = self.context.socket(zmq.PAIR)
        # self.socket.bind(f"tcp://*:{port}")
        self.stop_event = mp.Event()
        self.ready_event = mp.Event()
        self.port = port
        self.policy = policy
        self.device = device
        self.shape_meta = shape_meta
        self.verbose = verbose
        self.context = None  # Initialize context and socket as None
        self.socket = None

    def get_obs(self):
        """
        Get observation from socket.
        """
        obs_dict = None
        while True:
            try:
                # check for a message, this will not block
                message = self.socket.recv(flags=zmq.NOBLOCK)
            except zmq.Again:
                break
            else:
                # deserialize message
                obs_dict = pickle.loads(message)

        if obs_dict is None:
            while True:
                message = self.socket.recv()
                obs_dict = pickle.loads(message)
                if "obs" not in obs_dict and "t" not in obs_dict:  # TODO:
                    if "num_diffusion_iters" in obs_dict:  # ignore and send success
                        self.socket.send_string("success")
                    continue
                break
        return obs_dict["obs"], obs_dict["t"]

    # def stop(self):
    #     self.stop_event.set()

    def act(self, obs):
        """
        Perform action on observation.
        """
        print("act")
        with torch.no_grad():
            # obs_dict_np = get_real_obs_dict(env_obs=obs, shape_meta=self.shape_meta)
            # obs_dict = dict_apply(
            #     obs_dict_np, lambda x: torch.from_numpy(x).unsqueeze(0).to(self.device)
            # )
            obs_dict = dict_apply(obs, lambda x: x.to(self.device))
            result = self.policy.predict_action(obs_dict)
            action = result["action"][0].detach().to("cpu").numpy()
        return action

    # ========= launch method ===========
    def start(self):
        super().start()
        # if wait:
        #     self.start_wait()
        if self.verbose:
            print(f"[DP Inference ZMQ Server] process spawned at {self.pid}")

    def stop(self):
        self.stop_event.set()

    # def start_wait(self):
    #     self.ready_event.wait(self.launch_timeout)
    #     assert self.is_alive()

    # def stop_wait(self):
    #     self.join()

    @property
    def is_ready(self):
        return self.ready_event.is_set()

    # ========= context manager ===========
    def __enter__(self):
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()

    def test_inference(self):
        cprint("Compiling inference function", "green")
        # message = self.socket.recv_string()
        # print(f"server received message: {message}")

        message = self.socket.recv()
        # # start_time = time.time()
        state_dict = pickle.loads(message)
        # self.num_diffusion_iters = state_dict["num_diffusion_iters"]
        example_obs = state_dict["example_obs"]
        print(f"received compilation request: # diff iters = {state_dict['num_diffusion_iters']}")
        print(f"example_obs: {example_obs}")
        self.socket.send_string("success")
        self.ready_event.set()

    def compile_inference(self, precision="high"):
        cprint("Compiling inference function", "green")
        message = self.socket.recv()
        start_time = time.time()
        state_dict = pickle.loads(message)
        # self.num_diffusion_iters = state_dict["num_diffusion_iters"]
        example_obs = state_dict["example_obs"]
        print(f"received compilation request: # diff iters = {state_dict['num_diffusion_iters']}")

        torch.set_float32_matmul_precision(precision)
        self.policy.forward = torch.compile(torch.no_grad(self.policy.forward))

        for i in range(2):  # burn in
            self.act(example_obs)
        print("success, compile time: " + str(time.time() - start_time))
        self.socket.send_string("success")
        self.ready_event.set()

    def run(self):
        # build zmq socket
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.PAIR)
        self.socket.bind(f"tcp://*:{self.port}")
        print("[DPAgent] Inference server Started!")
        self.compile_inference()
        # self.test_inference()
        print("[DPAgent] Inference server is ready!")
        try:
            while not self.stop_event.is_set():
                obs, t = self.get_obs()
                if self.verbose:
                    ts = time.time()
                    cprint(f"Received obs at time={t}. Inference start! ", "green")

                pred = self.act(obs)

                if self.verbose:
                    cprint(
                        f"Inference cost {time.time()-ts}! Sending back to real world! ", "green"
                    )

                message = pickle.dumps({"acts": pred, "t": t})
                self.socket.send(message)
        except Exception as e:
            print(f"[DPAgent] Error: {e}")
        finally:
            self.socket.close()
            self.context.term()


class DPInferenceZMQClient:
    def __init__(
        self,
        port: int = DEFAULT_INFERENCE_PORT,
        host="localhost",
        temporal_ensemble_mode="new",
        # default_action=None,
        verbose=False,
    ):
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.PAIR)
        self.socket.connect(f"tcp://{host}:{port}")
        cprint(f"Connected to {host}:{port}", "green")

        # self.t = 0
        self.ensemble_buffer = EnsembleBuffer(mode=temporal_ensemble_mode)
        # self.last_act = default_action
        self.verbose = verbose

    def test_inference(self, example_obs, num_diffusion_iters):
        message = pickle.dumps(
            {"example_obs": example_obs, "num_diffusion_iters": num_diffusion_iters}
        )
        # self.socket.send_string("msg sent by client")
        self.socket.send(message)
        print("sent compile request")

        message = self.socket.recv()
        print(f"received message: {message}")
        assert message == b"success"

    def compile_inference(self, example_obs, num_diffusion_iters):
        message = pickle.dumps(
            {"example_obs": example_obs, "num_diffusion_iters": num_diffusion_iters}
        )
        self.socket.send(message)
        print("sent compile request")

        message = self.socket.recv()
        assert message == b"success"

    def act(self, obs, t):
        # self.t += 1
        # final_act = self.last_act

        # send obs to server
        # message = pickle.dumps({"obs": obs, "t": self.t})
        message = pickle.dumps({"obs": obs, "t": t})
        self.socket.send(message)

        # receive action from server
        while True:
            try:
                message = self.socket.recv(flags=zmq.NOBLOCK)
            except zmq.Again:
                break
            else:
                act_dict = pickle.loads(message)
                acts, pt = act_dict["acts"], act_dict["t"]
                if self.verbose:
                    cprint(f"Received action at time={pt}.", "green")
                self.ensemble_buffer.add_action(acts, t)

        # get action from ensemble buffer
        final_action = self.ensemble_buffer.get_action()
        # self.t += 1

        return final_action

    def reset(self):
        self.ensemble_buffer.reset_buffer()
        # clear message buffer
        while True:
            try:
                message = self.socket.recv(flags=zmq.NOBLOCK)
            except zmq.Again:
                break

    def __del__(self):
        self.socket.close()
        self.context.term()

    def close(self):
        self.socket.close()
        self.context.term()


class DPInferenceZMQ:
    def __init__(
        self,
        # shm_manager,
        shape_meta,
        policy: BaseImagePolicy,
        port: int = DEFAULT_INFERENCE_PORT,
        device: str = "cpu",
        verbose=False,
    ):

        # build zmq socket
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.PAIR)
        self.socket.bind(f"tcp://*:{port}")
        self.policy = policy
        self.device = device
        self.shape_meta = shape_meta
        self.verbose = verbose
        self.stop_event = threading.Event()

    def stop(self):
        self.stop_event.set()

    def get_obs(self):
        """
        Get observation from socket.
        """
        obs_dict = None
        while True:
            try:
                # check for a message, this will not block
                message = self.socket.recv(flags=zmq.NOBLOCK)
            except zmq.Again:
                break
            else:
                # deserialize message
                obs_dict = pickle.loads(message)

        if obs_dict is None:
            while True:
                message = self.socket.recv()
                obs_dict = pickle.loads(message)
                if "obs" not in obs_dict and "t" not in obs_dict:
                    if "num_diffusion_iters" in obs_dict:  # ignore and send success
                        self.socket.send_string("success")
                    continue
                break
        return obs_dict["obs"], obs_dict["t"]

    def act(self, obs):
        """
        Perform action on observation.
        """
        with torch.no_grad():
            # obs_dict_np = get_real_obs_dict(env_obs=obs, shape_meta=self.shape_meta)
            # obs_dict = dict_apply(
            #     obs_dict_np, lambda x: torch.from_numpy(x).unsqueeze(0).to(self.device)
            # )
            obs_dict = dict_apply(obs, lambda x: x.to(self.device))
            result = self.policy.predict_action(obs_dict)
            action = result["action"][0].detach().to("cpu").numpy()
        return action

    def test_inference(self):
        cprint("Compiling inference function", "green")
        # message = self.socket.recv_string()
        # print(f"server received message: {message}")

        message = self.socket.recv()
        # # start_time = time.time()
        state_dict = pickle.loads(message)
        # self.num_diffusion_iters = state_dict["num_diffusion_iters"]
        example_obs = state_dict["example_obs"]
        print(f"received compilation request: # diff iters = {state_dict['num_diffusion_iters']}")
        print(f"example_obs: {example_obs}")
        self.socket.send_string("success")

    def compile_inference(self, precision="high"):
        cprint("Compiling inference function", "green")
        message = self.socket.recv()
        start_time = time.time()
        state_dict = pickle.loads(message)
        # self.num_diffusion_iters = state_dict["num_diffusion_iters"]
        example_obs = state_dict["example_obs"]
        print(f"received compilation request: # diff iters = {state_dict['num_diffusion_iters']}")

        torch.set_float32_matmul_precision(precision)
        self.policy.forward = torch.compile(torch.no_grad(self.policy.forward))

        print("start burn in")

        for i in range(10):  # burn in
            self.act(example_obs)
        print("success, compile time: " + str(time.time() - start_time))
        self.socket.send_string("success")

    def run(self):
        # print("[DPAgent] Inference server Started!")
        # # self.compile_inference()
        # self.test_inference()
        print("[DPAgent] Inference server is started!")
        try:
            while not self.stop_event.is_set():
                obs, t = self.get_obs()
                if self.verbose:
                    ts = time.time()
                    cprint(f"Received obs at time={t}. Inference start! ", "green")

                pred = self.act(obs)

                if self.verbose:
                    cprint(
                        f"Inference cost {time.time()-ts}! Sending back to real world! ", "green"
                    )

                message = pickle.dumps({"acts": pred, "t": t})
                self.socket.send(message)
        except Exception as e:
            print(f"[DPAgent] Error: {e}")
        finally:
            self.socket.close()
            self.context.term()
