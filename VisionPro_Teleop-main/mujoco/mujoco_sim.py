import mujoco
import mujoco.viewer
import numpy as np
import time
import rospy
import hydra
# from hydra import OmegaConf 
from std_msgs.msg import Float64MultiArray
from sensor_msgs.msg import JointState
# from avatar_dex.constants import *
# from avatar_dex.utils.files import get_yaml_data, get_path_in_package
from scipy.spatial.transform import Rotation as R

# a class for allegro hand simulation
class AvatarMujocoSim():
    def __init__(self):
        rospy.init_node("mujoco_sim_teleop", anonymous=True)

        self.desired_angles = None
        self.desired_wrist_pose = None
        self.target_vis_vectors = None

        self.fps = 30

        self.num_steps = 0
        self.wrist_pos_0, self.mocap_pos_0 = None, None

        self._create_subscriber()
        self._create_publisher()
        self._create_mujoco_model()

    def _create_subscriber(self):
        rospy.Subscriber("desired_qpos_hand", Float64MultiArray, self._callback_teleop_hand, queue_size=1)
        rospy.Subscriber("desired_wrist_pose", Float64MultiArray, self._callback_wrist_pose, queue_size=1)
        # rospy.Subscriber(ALLEGRO_TARGET_VIS_TOPIC, Float64MultiArray, self._callback_target_vis, queue_size=1)


    def _create_publisher(self):
        pass
        # self.joint_state_pub = rospy.Publisher(ALLEGRO_JOINT_STATE_TOPIC, JointState, queue_size=1)

    def _create_mujoco_model(self): 
        # self.model = mujoco.MjModel.from_xml_path(get_path_in_package(f"components/simulation/xml/{self.sim_config['scene_name']}.xml"))
        # self.model = mujoco.MjModel.from_xml_path(self.config.xml_path)
        self.model = mujoco.MjModel.from_xml_path("mujoco/assets/scene/scene_leap_right.xml")
        self.data = mujoco.MjData(self.model)
        self.timestep = 1/self.fps
    
    def _callback_teleop_hand(self, teleop_angles):
        self.desired_angles = np.array(list(teleop_angles.data)).reshape(16, )
    
    def _callback_wrist_pose(self, wrist_pose):
        self.desired_wrist_pose = np.array(list(wrist_pose.data)).reshape(6, )

    def _callback_target_vis(self, target_vis):
        self.target_vis_vectors = np.array(list(target_vis.data)).reshape(-1, 3)


    def _get_joint_state(self):
        joint_state = JointState()
        joint_state.header.stamp = rospy.Time.now()
        # joint_state.name = self.model.joint_names
        joint_state.position = self.data.qpos
        joint_state.velocity = self.data.qvel
        joint_state.effort = self.data.qfrc_inverse
        return joint_state
    
    def _publish_joint_state(self):
        pass
        # joint_state = self._get_joint_state()
        # self.joint_state_pub.publish(joint_state)

    def _set_desired_angles(self):
        if self.desired_angles is not None:
            self.data.ctrl[:] = self.desired_angles
        else:
            self.data.ctrl[:] = 0.0

    def _set_init_wrist_pose(self):
        self.init_pos = [-0.375, 0.208, 0.397]
        self.init_rotvec = [3.141, 0.016, 0.008]

        rot = R.from_rotvec(self.init_rotvec).as_quat()

        # convert x,y,z,w to w,x,y,z
        rot = np.array([rot[3], rot[0], rot[1], rot[2]])
        # rospy.loginfo(f"number of steps: {self.num_steps}")

        pos_0 = np.array([0.0, 0.0, 0.0])
        self.data.mocap_pos[0] = pos_0 + self.init_pos  # TODO: 
        self.data.mocap_quat[0] = rot

        # Visulaize target tip_pos
        if self.target_vis_vectors is not None :
            for i in range(4):
                r_W_tcp = R.from_rotvec(self.init_rotvec)
                r_tcp_hand = R.from_euler('XYZ', [np.pi, 0, np.pi/2])
                vis_pos_W = (r_W_tcp * r_tcp_hand).apply(self.target_vis_vectors[i]) + pos_0 + self.init_pos
                self.data.mocap_pos[i+1] = vis_pos_W
    
        self.num_steps += 1        

    def _set_desired_wrist_pose(self):
        if self.desired_wrist_pose is not None:
            pos = self.desired_wrist_pose[:3].copy()
            rotvec = self.desired_wrist_pose[3:].copy()

            # rot = R.from_rotvec(rotvec).as_quat()

            # convert x,y,z,w to w,x,y,z
            
            # rospy.loginfo(f"number of steps: {self.num_steps}")

            # pos_0 = np.array([0.0, 0.0, 0.0])
            pos_0 = np.array([-0.375, 0.208, 0.397])
            rot_0 = R.from_rotvec([3.141, 0.016, 0.008])
            rot = (rot_0 * R.from_rotvec(rotvec)).as_quat()
            rot = np.array([rot[3], rot[0], rot[1], rot[2]])
            self.data.mocap_pos[0] = pos_0 + rot_0.apply(pos)  # TODO: 
            self.data.mocap_quat[0] = rot

            # Visulaize target tip_pos
            if self.target_vis_vectors is not None :
                for i in range(4):
                    r_W_tcp = R.from_rotvec(rotvec)
                    r_tcp_hand = R.from_euler('XYZ', [np.pi, 0, np.pi/2])
                    vis_pos_W = (r_W_tcp * r_tcp_hand).apply(self.target_vis_vectors[i]) + pos_0 + pos
                    self.data.mocap_pos[i+1] = vis_pos_W
     
            self.num_steps += 1
        

    def _step(self): 
        self._publish_joint_state()
        self._set_desired_angles()
        self._set_desired_wrist_pose()
        # self.model.step(self.data)
        mujoco.mj_step(self.model, self.data)
        

    def simulate(self):
        print("Starting Mujoco simulation renderer")
        with mujoco.viewer.launch_passive(self.model, self.data) as viewer:
            start = time.time()
            self._set_init_wrist_pose()
            while viewer.is_running() and not rospy.is_shutdown():
                step_start = time.time()

                self._step()

                # Example modification of a viewer option: toggle contact points every two seconds.
                with viewer.lock():
                    viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTPOINT] = int(self.data.time % 2)

                # Pick up changes to the physics state, apply perturbations, update options from GUI.
                viewer.sync()

                # Rudimentary time keeping, will drift relative to wall clock.
                # time_until_next_step = model.opt.timestep - (time.time() - step_start)
                time_until_next_step = self.timestep - (time.time() - step_start)
                if time_until_next_step > 0:
                    time.sleep(time_until_next_step)

        # close mujoco
        viewer.close()
        print("\nSimulation terminated.")

# @hydra.main(config_path="", config_name="sim_config")
def main():
    # xml_path = "assets/scene/scene_leap_right.xml"
    # model = mujoco.MjModel.from_xml_path(xml_path)
    avatar_sim = AvatarMujocoSim()
    avatar_sim.simulate()

if __name__ == "__main__":
    main()