from aic_control_interfaces.srv import ChangeTargetMode
import json
import time
from pathlib import Path
from scipy.spatial.transform import Rotation as Rot

import cv2
import numpy as np
import rclpy
import tf2_ros
from cv_bridge import CvBridge
from message_filters import Subscriber, ApproximateTimeSynchronizer
from rclpy.node import Node
from sensor_msgs.msg import CameraInfo, Image
from aic_teleoperation.mag_test import Stereo_mag_square
from geometry_msgs.msg import Twist, Wrench, Vector3
from aic_control_interfaces.msg import MotionUpdate, TrajectoryGenerationMode, TargetMode

from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
import math
from aic_teleoperation.reference_pose_initializer import ReferencePoseInitializer,search_find_Node
from aic_teleoperation.insert import insert_task
from aic_teleoperation.port_finder import Stereo_nic_port
from aic_teleoperation.sc_port_finder import Stereo_sc_port
FAST_LINEAR_VEL = 0.1
FAST_ANGULAR_VEL = 0.35

NIC_TIP_QUAT_XYZW = np.array([0.180, 0.006, -0.027, 0.983], dtype=float)
SC_TIP_QUAT_XYZW = np.array([-0.161, 0.167, -0.694, -0.681], dtype=float)
NIC_GRIPPER_OFFSET = np.array([-0.000, -0.018, 0.048], dtype=float)
SC_GRIPPER_OFFSET = np.array([-0.001, -0.010, 0.008], dtype=float)


def _is_nic_module(module_name: str) -> bool:
    return module_name.startswith("nic")


def get_tip_pose_config(module_name: str) -> tuple[np.ndarray, np.ndarray]:
    if _is_nic_module(module_name):
        return NIC_TIP_QUAT_XYZW.copy(), NIC_GRIPPER_OFFSET.copy()
    return SC_TIP_QUAT_XYZW.copy(), SC_GRIPPER_OFFSET.copy()


class nic_task(Node):
    def __init__(self,goal_position, goal_quaternion,module_name="nic_card_0"):
        super().__init__("nic_task")

        self.controller_namespace = self.declare_parameter(
            "controller_namespace", "aic_controller"
        ).value

        self.base_frame = "base_link"
        self.ee_frame = "gripper/tcp"

        self.goal_position = np.array(
            goal_position,
            dtype=float,
        )
        self.goal_quaternion = np.array(
            goal_quaternion,
            dtype=float,
        )
        self.module_name = module_name
        self.q_tip_g, self.gripper_offset = get_tip_pose_config(module_name)

        self.pos_tolerance = float(self.declare_parameter("pos_tolerance", 0.005).value)
        self.rot_tolerance = float(self.declare_parameter("rot_tolerance", 0.002).value)

        self.kp_linear = 2.0
        self.kp_angular = 2.0
        self.max_linear_vel = FAST_LINEAR_VEL
        self.max_angular_vel = FAST_ANGULAR_VEL

        self.motion_update_publisher = self.create_publisher(
            MotionUpdate, f"/{self.controller_namespace}/pose_commands", 10
        )

        while self.motion_update_publisher.get_subscription_count() == 0:
            self.get_logger().info(
                f"Waiting for subscriber to '{self.controller_namespace}/pose_commands'..."
            )
            time.sleep(1.0)

        self.client = self.create_client(
            ChangeTargetMode, f"/{self.controller_namespace}/change_target_mode"
        )

        while not self.client.wait_for_service(timeout_sec=1.0):
            self.get_logger().info(
                f"Waiting for service '{self.controller_namespace}/change_target_mode'..."
            )

        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        self.goal_reached = False
        self._goal_logged = False

        ######need to remove later just for logging the mag results
#########################################################################################
        self.bridge = CvBridge()
        self.output_dir = Path.home() / "nic_final_images"
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.left_image = None
        self.right_image = None
        self.images_saved = False

        self.left_sub = Subscriber(self, Image, "/left_camera/image")
        self.right_sub = Subscriber(self, Image, "/right_camera/image")
        self.sync = ApproximateTimeSynchronizer(
            [self.left_sub, self.right_sub],
            queue_size=10,
            slop=0.05,
        )
        self.sync.registerCallback(self.image_callback)
##############################################################################################
        self.timer = self.create_timer(0.04, self.send_references)

    def quat_conj(self, q):
        return np.array([-q[0], -q[1], -q[2], q[3]], dtype=float)

    def quat_mul(self, q1, q2):
        x1, y1, z1, w1 = q1
        x2, y2, z2, w2 = q2
        return np.array([
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
        ], dtype=float)

    def quat_to_rotvec(self, q):
        if q[3] < 0.0:
            q = -q
        xyz_norm = np.linalg.norm(q[:3])
        if xyz_norm < 1e-9:
            return np.zeros(3)
        angle = 2.0 * math.atan2(xyz_norm, q[3])
        axis = q[:3] / xyz_norm
        return axis * angle

    def get_current_tcp_pose(self):
        try:
            tf = self.tf_buffer.lookup_transform(
                self.base_frame, self.ee_frame, rclpy.time.Time()
            )
            pos = np.array([
                tf.transform.translation.x,
                tf.transform.translation.y,
                tf.transform.translation.z,
            ], dtype=float)
            quat = np.array([
                tf.transform.rotation.x,
                tf.transform.rotation.y,
                tf.transform.rotation.z,
                tf.transform.rotation.w,
            ], dtype=float)
            return pos, quat
        except Exception as e:
            self.get_logger().warn(f"Could not get TCP transform: {e}")
            return None, None

    def generate_velocity_motion_update(self, twist, frame_id):
        msg = MotionUpdate()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = frame_id
        msg.velocity = twist
        msg.target_stiffness = np.diag([85.0] * 6).flatten()
        msg.target_damping = np.diag([75.0] * 6).flatten()
        msg.feedforward_wrench_at_tip = Wrench(
            force=Vector3(x=0.0, y=0.0, z=0.0),
            torque=Vector3(x=0.0, y=0.0, z=0.0),
        )
        msg.wrench_feedback_gains_at_tip = [0.0] * 6
        msg.trajectory_generation_mode.mode = TrajectoryGenerationMode.MODE_VELOCITY
        return msg

    def send_change_control_mode_req(self, mode):
        req = ChangeTargetMode.Request()
        req.target_mode.mode = mode
        future = self.client.call_async(req)
        rclpy.spin_until_future_complete(self, future)
        response = future.result()
        if response and response.success:
            self.get_logger().info(f"Changed control mode to {mode}")
        else:
            self.get_logger().info(f"Failed to change control mode to {mode}")

    def send_references(self):
        gripper_pos, gripper_quat = self.get_current_tcp_pose()
        if gripper_pos is None:
            return
        # pos_err = self.goal_position - gripper_pos

        q_tip_b = self.quat_mul(gripper_quat, self.q_tip_g)

        q_err = self.quat_mul(self.goal_quaternion, self.quat_conj(q_tip_b))


        rot_err = self.quat_to_rotvec(q_err)



        rot = Rot.from_quat(gripper_quat)
        plug_pos = gripper_pos + rot.apply(self.gripper_offset)


        pos_err = self.goal_position - plug_pos

        pos_norm = np.linalg.norm(pos_err)
        rot_norm = np.linalg.norm(rot_err)

        twist = Twist()

        if pos_norm < self.pos_tolerance and rot_norm < self.rot_tolerance:
            self.goal_reached = True
            self.motion_update_publisher.publish(
                self.generate_velocity_motion_update(twist, self.base_frame)
            )
            if not self._goal_logged:
                self.get_logger().info(f"final quaternion (tip in base frame) = {q_tip_b}")
                self.get_logger().info(f"Goal quaternion: {self.goal_quaternion}")
                self.get_logger().info(f"tip quaternion error: {q_err}")
                self.get_logger().info("Reference pose reached.")
                self._goal_logged = True
            if not self.images_saved:
                self.save_final_images()
            return
        self.goal_reached = False
        self._goal_logged = False

        linear_cmd = np.clip(
            self.kp_linear * pos_err,
            -self.max_linear_vel,
            self.max_linear_vel,
        )
        angular_cmd = np.clip(
            self.kp_angular * rot_err,
            -self.max_angular_vel,
            self.max_angular_vel,
        )

        twist.linear.x = float(linear_cmd[0])
        twist.linear.y = float(linear_cmd[1])
        twist.linear.z = float(linear_cmd[2])
        twist.angular.x = float(angular_cmd[0])
        twist.angular.y = float(angular_cmd[1])
        twist.angular.z = float(angular_cmd[2])

        self.motion_update_publisher.publish(
            self.generate_velocity_motion_update(twist, self.base_frame)
        )

        self.get_logger().info(
            f"Moving to reference | pos_err={pos_norm:.4f} m, rot_err={rot_norm:.4f} rad"
        )
    def image_callback(self, left_msg, right_msg):
        try:
            self.left_image = self.bridge.imgmsg_to_cv2(left_msg, desired_encoding="bgr8")
            self.right_image = self.bridge.imgmsg_to_cv2(right_msg, desired_encoding="bgr8")
        except Exception as e:
            self.get_logger().warn(f"Failed to convert camera images: {e}")

    def save_final_images(self):
        if self.left_image is None or self.right_image is None:
            self.get_logger().warn("No synchronized left/right images available to save.")
            return

        # create dataset folder
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # find next index
        existing_left = sorted(self.output_dir.glob("left_*.png"))

        if len(existing_left) == 0:
            idx = 1
        else:
            last_idx = max(
                int(p.stem.split("_")[-1]) for p in existing_left
            )
            idx = last_idx + 1

        left_path = self.output_dir / f"left_{idx:05d}.png"
        right_path = self.output_dir / f"right_{idx:05d}.png"

        cv2.imwrite(str(left_path), self.left_image)
        cv2.imwrite(str(right_path), self.right_image)

        self.get_logger().info(
            f"Saved dataset pair #{idx}:\n"
            f"  Left : {left_path}\n"
            f"  Right: {right_path}"
        )

        self.images_saved = True


import sys
import re
import numpy as np


def get_goal_position_tb(module_name: str):

    match = re.match(r"nic_card_(\d+)", module_name)

    if match is None:
        raise ValueError(
            f"Invalid module name: {module_name}"
        )

    card_idx = int(match.group(1))

    if card_idx < 0 or card_idx > 4:
        raise ValueError(
            "Card index must be between 0 and 4"
        )

    base_y = 0.182
    spacing = 0.039

    # nic_card_4 -> offset 0
    offset = (4 - card_idx) * spacing


    R_mb_t = np.diag([-1.0, 1.0, -1.0])
    return np.array([0.0, base_y + offset, 0.2]), R_mb_t

def _goal_position_tb(module_name: str) -> np.ndarray:
    sc_match = re.match(r"sc_port_?(\d+)$", module_name)
    if sc_match is not None:
        port_idx = int(sc_match.group(1))
        if port_idx == 0:
            return np.array([0.0, 0.120, 0.2])
        if port_idx == 1:
            return np.array([0.0, 0.0797, 0.2])
        raise ValueError("SC port index must be 0 or 1.")

    try:
        position, _ = get_goal_position_tb(module_name)
        return position
    except ValueError:
        match = re.search(r"(\d+)$", module_name)
        if match is None:
            raise
        position, _ = get_goal_position_tb(f"nic_card_{match.group(1)}")
        return position

def _board_to_module_rotation(module_name: str) -> np.ndarray:
    if re.match(r"sc_port_?(\d+)$", module_name):
        return np.array(
            [
                [0.0, 1.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.0, 0.0, -1.0],
            ]
        )
    return np.diag([-1.0, 1.0, -1.0])



def main():
    if len(sys.argv) < 2:
        print("Usage:")
        print(
            "pixi run ros2 run aic_teleoperation "
            "move2task nic_card_0"
        )
        return

    module_name = sys.argv[1]

    goal_position_tb = _goal_position_tb(module_name)
    R_mb_t = _board_to_module_rotation(module_name)

    print(f"Selected module : {module_name}")
    print(f"Goal position TB: {goal_position_tb}")
    total_start = time.perf_counter()
    rclpy.init()

    try:
        init_ref_node = ReferencePoseInitializer()
        init_ref_node.send_change_control_mode_req(TargetMode.MODE_CARTESIAN)
        while rclpy.ok() and not init_ref_node.goal_reached:
            rclpy.spin_once(init_ref_node, timeout_sec=0.1)

        search_node = search_find_Node()
        search_node.send_change_control_mode_req(TargetMode.MODE_CARTESIAN)
        while rclpy.ok() and not search_node.task_done:
            rclpy.spin_once(search_node, timeout_sec=0.1)

        mag_ref_node = Stereo_mag_square()
        while rclpy.ok() and mag_ref_node.latest_result is None:
            rclpy.spin_once(mag_ref_node, timeout_sec=0.1)

        result = mag_ref_node.latest_result
        if result is None:
            print("Failed to get mag result. Exiting.")
            return

        center = np.asarray(result["center"])
        Rz = np.asarray(result["Rz"])

        # goal_position_tb = np.array([0.0, 0.182, 0.25])
        goal_position = Rz @ goal_position_tb + center
        # print(f"Calculated goal position in base frame: {goal_position}")
        #add board config file later for refactoring
        #R_mb_t

        R_mb = Rz @ R_mb_t
        quat = Rot.from_matrix(R_mb).as_quat()
        goal_quaternion_tip = [quat[0], quat[1], quat[2], quat[3]]

        mag_ref_node.destroy_node()

        node = nic_task(goal_position, goal_quaternion_tip, module_name)
        while rclpy.ok() and not node.goal_reached:
            rclpy.spin_once(node, timeout_sec=0.1)

        stereo_port_node = Stereo_nic_port() if _is_nic_module(module_name) else Stereo_sc_port()

        if _is_nic_module(module_name):
            while rclpy.ok() and stereo_port_node.card_orientation is None:
                rclpy.spin_once(stereo_port_node, timeout_sec=0.1)

            if stereo_port_node.port1 is None or stereo_port_node.card_orientation is None:
                print("Failed to get NIC port pose. Exiting.")
                stereo_port_node.destroy_node()
                return

            print(goal_quaternion_tip, stereo_port_node.card_orientation)
            q_xyzw = stereo_port_node.card_orientation
            port_pos = stereo_port_node.port0
        else:
            while rclpy.ok() and stereo_port_node.port is None:
                rclpy.spin_once(stereo_port_node, timeout_sec=0.1)

            if stereo_port_node.port is None:
                print("Failed to get SC port pose. Exiting.")
                stereo_port_node.destroy_node()
                return

            q_xyzw = goal_quaternion_tip
            port_pos = stereo_port_node.port

        q_wxyz = (q_xyzw[3], q_xyzw[0], q_xyzw[1], q_xyzw[2])
        insert_node = insert_task(port_pos, q_wxyz, module_name)
        try:
            while rclpy.ok() and insert_node._phase != "done":
                rclpy.spin_once(insert_node, timeout_sec=0.1)
        finally:
            insert_node.destroy_node()
            stereo_port_node.destroy_node()

    except KeyboardInterrupt:
        print("Interrupted by user.")

    finally:
        total_time = time.perf_counter() - total_start
        print(f"[TIME] TOTAL EXECUTION TIME: {total_time:.3f} sec")
        rclpy.shutdown()
