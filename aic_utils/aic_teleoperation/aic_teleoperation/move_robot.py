#!/usr/bin/env python3

import sys
import time
import numpy as np
import rclpy
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
import tf2_ros
import math

from geometry_msgs.msg import Twist, Wrench, Vector3
from aic_control_interfaces.msg import MotionUpdate, TrajectoryGenerationMode, TargetMode
from aic_control_interfaces.srv import ChangeTargetMode

import json
import time
from pathlib import Path

import cv2
from cv_bridge import CvBridge
from message_filters import Subscriber, ApproximateTimeSynchronizer
from rclpy.node import Node
from sensor_msgs.msg import CameraInfo, Image

FAST_LINEAR_VEL = 0.1
FAST_ANGULAR_VEL = 0.5
GOAL_POS = np.array([-0.34564215,  0.20225491,  0.2102515], dtype=float)
GOAL_QUAT = np.array([-0.47557305,  0.87967623,  0.0  , 0.0], dtype=float)

class ReferencePoseInitializer(Node):
    def __init__(self):
        super().__init__("reference_pose_initializer")

        self.controller_namespace = self.declare_parameter(
            "controller_namespace", "aic_controller"
        ).value

        self.base_frame = "base_link"
        self.ee_frame = "gripper/tcp"

        self.goal_position = np.array(
            self.declare_parameter("goal_position", GOAL_POS.tolist()).value,
            dtype=float,
        )
        self.goal_quaternion = np.array(
            self.declare_parameter("goal_quaternion", GOAL_QUAT.tolist()).value,
            dtype=float,
        )

        self.pos_tolerance = float(self.declare_parameter("pos_tolerance", 0.005).value)
        self.rot_tolerance = float(self.declare_parameter("rot_tolerance", 0.002).value)
        self.motion_duration = float(
            self.declare_parameter("motion_duration", 3.0).value
        )
        if self.motion_duration <= 0.0:
            raise ValueError("motion_duration must be greater than zero")

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
        self.start_position = None
        self.start_quaternion = None
        self.motion_start_time = None

        self.bridge = CvBridge()
        self.output_dir = Path.home() / "sft_mount_ima"
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

    def quat_slerp(self, q0, q1, fraction):
        q0 = np.asarray(q0, dtype=float)
        q1 = np.asarray(q1, dtype=float)
        q0_norm = np.linalg.norm(q0)
        q1_norm = np.linalg.norm(q1)
        if q0_norm < 1e-12 or q1_norm < 1e-12:
            raise ValueError("Cannot interpolate a zero-length quaternion")

        q0 = q0 / q0_norm
        q1 = q1 / q1_norm
        dot = float(np.dot(q0, q1))

        # q and -q represent the same orientation. Flip the goal quaternion
        # when necessary so SLERP follows the shorter rotation.
        if dot < 0.0:
            q1 = -q1
            dot = -dot

        dot = float(np.clip(dot, -1.0, 1.0))
        if dot > 0.9995:
            result = q0 + fraction * (q1 - q0)
            return result / np.linalg.norm(result)

        angle = math.acos(dot)
        sin_angle = math.sin(angle)
        q0_weight = math.sin((1.0 - fraction) * angle) / sin_angle
        q1_weight = math.sin(fraction * angle) / sin_angle
        return q0_weight * q0 + q1_weight * q1

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
        current_pos, current_quat = self.get_current_tcp_pose()
        if current_pos is None:
            return

        if self.motion_start_time is None:
            self.start_position = current_pos.copy()
            self.start_quaternion = current_quat.copy()
            self.motion_start_time = time.monotonic()
            self.get_logger().info(
                f"Starting pose interpolation over {self.motion_duration:.2f} seconds."
            )

        elapsed = time.monotonic() - self.motion_start_time
        fraction = float(np.clip(elapsed / self.motion_duration, 0.0, 1.0))

        reference_pos = (
            (1.0 - fraction) * self.start_position
            + fraction * self.goal_position
        )
        reference_quat = self.quat_slerp(
            self.start_quaternion, self.goal_quaternion, fraction
        )

        pos_err = reference_pos - current_pos
        q_err = self.quat_mul(reference_quat, self.quat_conj(current_quat))
        rot_err = self.quat_to_rotvec(q_err)

        pos_norm = np.linalg.norm(pos_err)
        rot_norm = np.linalg.norm(rot_err)

        twist = Twist()

        if (
            fraction >= 1.0
            and pos_norm < self.pos_tolerance
            and rot_norm < self.rot_tolerance
        ):
            self.goal_reached = True
            self.motion_update_publisher.publish(
                self.generate_velocity_motion_update(twist, self.base_frame)
            )
            if not self._goal_logged:
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
            f"Moving to reference | progress={fraction:.1%}, "
            f"pos_err={pos_norm:.4f} m, rot_err={rot_norm:.4f} rad"
        )

    def image_callback(self, left_msg, right_msg):
        try:
            self.left_image = self.bridge.imgmsg_to_cv2(
                left_msg, desired_encoding="bgr8"
            )
            self.right_image = self.bridge.imgmsg_to_cv2(
                right_msg, desired_encoding="bgr8"
            )
        except Exception as e:
            self.get_logger().warn(f"Failed to convert camera images: {e}")

    def save_final_images(self):
        if self.left_image is None or self.right_image is None:
            self.get_logger().warn(
                "No synchronized left/right images available to save."
            )
            return

        self.output_dir.mkdir(parents=True, exist_ok=True)
        existing_left = sorted(self.output_dir.glob("left_*.png"))

        if len(existing_left) == 0:
            idx = 1
        else:
            last_idx = max(int(p.stem.split("_")[-1]) for p in existing_left)
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


def main(args=None):
    ref_node = None
    try:
        rclpy.init(args=args)

        ref_node = ReferencePoseInitializer()
        ref_node.send_change_control_mode_req(TargetMode.MODE_CARTESIAN)

        while rclpy.ok() and not ref_node.goal_reached:
            rclpy.spin_once(ref_node, timeout_sec=0.1)

        ref_node.get_logger().info("Starting second code now.")
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        if ref_node is not None:
            ref_node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
