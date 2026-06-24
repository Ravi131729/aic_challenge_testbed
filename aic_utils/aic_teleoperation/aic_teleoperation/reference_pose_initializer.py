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


class ReferencePoseInitializer(Node):
    def __init__(self):
        super().__init__("reference_pose_initializer")

        self.controller_namespace = self.declare_parameter(
            "controller_namespace", "aic_controller"
        ).value

        self.base_frame = "base_link"
        self.ee_frame = "gripper/tcp"

        self.goal_position = np.array(
            self.declare_parameter("goal_position", [-0.372, 0.193, 0.304]).value,
            dtype=float,
        )
        self.goal_quaternion = np.array(
            self.declare_parameter("goal_quaternion", [0.992, 0.0, 0.0, -0.127]).value,
            dtype=float,
        )

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
        current_pos, current_quat = self.get_current_tcp_pose()
        if current_pos is None:
            return

        pos_err = self.goal_position - current_pos
        q_err = self.quat_mul(self.goal_quaternion, self.quat_conj(current_quat))
        rot_err = self.quat_to_rotvec(q_err)

        pos_norm = np.linalg.norm(pos_err)
        rot_norm = np.linalg.norm(rot_err)

        twist = Twist()

        if pos_norm < self.pos_tolerance and rot_norm < self.rot_tolerance:
            self.goal_reached = True
            self.motion_update_publisher.publish(
                self.generate_velocity_motion_update(twist, self.base_frame)
            )
            if not self._goal_logged:
                self.get_logger().info("Reference pose reached.")
                self._goal_logged = True
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


class SecondTaskNode(Node):
    def __init__(self):
        super().__init__("second_task_node")

        self.bridge = CvBridge()

        self.base_frame = self.declare_parameter("base_frame", "base_link").value
        self.optical_frame = self.declare_parameter(
            "optical_frame", "center_camera/optical"
        ).value

        self.image_topic = self.declare_parameter(
            "image_topic", "/center_camera/image"
        ).value
        self.camera_info_topic = self.declare_parameter(
            "camera_info_topic", "/center_camera/camera_info"
        ).value

        self.output_dir = Path(
            self.declare_parameter("output_dir", "/tmp/center_camera_capture").value
        )
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        self.saved = False

        self.image_sub = Subscriber(self, Image, self.image_topic)
        self.info_sub = Subscriber(self, CameraInfo, self.camera_info_topic)

        self.sync = ApproximateTimeSynchronizer(
            [self.image_sub, self.info_sub],
            queue_size=10,
            slop=0.1,
        )
        self.sync.registerCallback(self.synced_capture_callback)

        self.get_logger().info("Second task node started.")
        self.get_logger().info("Waiting for center camera image + camera info...")

    def synced_capture_callback(self, image_msg: Image, info_msg: CameraInfo):
        if self.saved:
            return

        try:
            tf_msg = self.tf_buffer.lookup_transform(
                self.base_frame,
                self.optical_frame,
                rclpy.time.Time(),
            )
        except Exception as e:
            self.get_logger().warn(f"TF not available yet for {self.optical_frame}: {e}")
            return

        stamp_sec = image_msg.header.stamp.sec
        stamp_nsec = image_msg.header.stamp.nanosec
        stamp_name = f"{stamp_sec}_{stamp_nsec}"

        image_path = self.output_dir / f"center_camera_image_{stamp_name}.png"
        info_path = self.output_dir / f"center_camera_info_{stamp_name}.json"
        pose_path = self.output_dir / f"center_camera_optical_pose_{stamp_name}.npz"

        try:
            # Save image
            cv_image = self.bridge.imgmsg_to_cv2(image_msg, desired_encoding="bgr8")
            cv2.imwrite(str(image_path), cv_image)

            # Save camera intrinsics / info
            camera_info_data = {
                "header_frame_id": info_msg.header.frame_id,
                "width": int(info_msg.width),
                "height": int(info_msg.height),
                "distortion_model": info_msg.distortion_model,
                "d": list(info_msg.d),
                "k": list(info_msg.k),
                "r": list(info_msg.r),
                "p": list(info_msg.p),
            }
            with open(info_path, "w") as f:
                json.dump(camera_info_data, f, indent=2)

            # Save optical pose as transform from base_frame -> optical_frame
            pose_data = {
                "translation": np.array(
                    [
                        tf_msg.transform.translation.x,
                        tf_msg.transform.translation.y,
                        tf_msg.transform.translation.z,
                    ],
                    dtype=float,
                ),
                "rotation": np.array(
                    [
                        tf_msg.transform.rotation.x,
                        tf_msg.transform.rotation.y,
                        tf_msg.transform.rotation.z,
                        tf_msg.transform.rotation.w,
                    ],
                    dtype=float,
                ),
                "parent_frame": self.base_frame,
                "child_frame": self.optical_frame,
            }
            np.savez(pose_path, **pose_data)

            self.saved = True
            self.get_logger().info(f"Saved image to: {image_path}")
            self.get_logger().info(f"Saved camera info to: {info_path}")
            self.get_logger().info(f"Saved optical pose to: {pose_path}")
            self.get_logger().info("Second task data capture complete.")

            # Put your real second task here after saving
            self.run_task_once()

        except Exception as e:
            self.get_logger().error(f"Failed to save camera data: {e}")

    def run_task_once(self):
        self.get_logger().info("Running second code here.")
        # Put your actual task code here

class search_find_Node(Node):
    def __init__(self):
        super().__init__("search_find_node")

        self.bridge = CvBridge()

        self.controller_namespace = self.declare_parameter(
            "controller_namespace", "aic_controller"
        ).value

        self.base_frame = self.declare_parameter("base_frame", "base_link").value
        self.image_topic = self.declare_parameter(
            "image_topic", "/center_camera/image"
        ).value

        self.magenta_h_low = int(self.declare_parameter("magenta_h_low", 135).value)
        self.magenta_h_high = int(self.declare_parameter("magenta_h_high", 175).value)
        self.magenta_s_low = int(self.declare_parameter("magenta_s_low", 70).value)
        self.magenta_v_low = int(self.declare_parameter("magenta_v_low", 40).value)

        self.min_blob_area = float(self.declare_parameter("min_blob_area", 250.0).value)
        self.center_tolerance_px = float(
            self.declare_parameter("center_tolerance_px", 25.0).value
        )

        self.max_xy_vel = float(self.declare_parameter("max_xy_vel", 0.04).value)
        self.kp_xy = float(self.declare_parameter("kp_xy", 0.0005).value)

        # in __init__
        self.search_state = 0
        self.search_state_start = self.get_clock().now()

        self.search_move_time = float(self.declare_parameter("search_move_time", 0.6).value)
        self.search_step_vel = float(self.declare_parameter("search_step_vel", 0.1).value)
        self.search_step_x = float(self.declare_parameter("search_step_x", 0.004).value)

        # Flip these if your robot moves opposite to what you expect.
        self.sign_x = float(self.declare_parameter("sign_x", 1.0).value)
        self.sign_y = float(self.declare_parameter("sign_y", -1.0).value)

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

        self.image_sub = self.create_subscription(
            Image, self.image_topic, self.image_callback, 10
        )

        self.latest_centroid = None
        self.latest_area = 0.0
        self.latest_img_size = None
        self.magenta_seen = False
        self.search_phase = 0.0
        self.task_done = False
        self.center_hold_count = 0
        self.center_hold_needed = 10   # about 0.4s if timer = 0.04

        self.timer = self.create_timer(0.04, self.control_loop)
        self.get_logger().info("Third task node started: magenta search + XY centering.")

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

    def build_twist(self, vx, vy):
        twist = Twist()
        twist.linear.x = float(vx)
        twist.linear.y = float(vy)
        twist.linear.z = 0.0   # keep Z fixed
        twist.angular.x = 0.0
        twist.angular.y = 0.0
        twist.angular.z = 0.0  # keep orientation fixed
        return twist

    def image_callback(self, image_msg: Image):
        try:
            img = self.bridge.imgmsg_to_cv2(image_msg, desired_encoding="bgr8")
        except Exception as e:
            self.get_logger().warn(f"Image conversion failed: {e}")
            return

        self.latest_img_size = (img.shape[1], img.shape[0])

        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

        lower = np.array([self.magenta_h_low, self.magenta_s_low, self.magenta_v_low])
        upper = np.array([self.magenta_h_high, 255, 255])

        mask = cv2.inRange(hsv, lower, upper)

        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
            mask, connectivity=8
        )

        if num_labels <= 1:
            self.magenta_seen = False
            self.latest_centroid = None
            self.latest_area = 0.0
            return

        best_idx = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
        area = float(stats[best_idx, cv2.CC_STAT_AREA])

        if area < self.min_blob_area:
            self.magenta_seen = False
            self.latest_centroid = None
            self.latest_area = area
            return

        cx, cy = centroids[best_idx]
        self.latest_centroid = (float(cx), float(cy))
        self.latest_area = area
        self.magenta_seen = True

    def control_loop(self):
        if self.latest_img_size is None:
            return

        img_w, img_h = self.latest_img_size
        cx_img = img_w * 0.5
        cy_img = img_h * 0.5

        if self.magenta_seen and self.latest_centroid is not None:
            u, v = self.latest_centroid
            ex = u - cx_img
            ey = v - cy_img

            if abs(ex) < self.center_tolerance_px and abs(ey) < self.center_tolerance_px:
                vx = 0.0
                vy = 0.0
                self.center_hold_count += 1
            else:
                self.center_hold_count = 0
                vx = np.clip(
                    self.sign_x * self.kp_xy * ex,
                    -self.max_xy_vel,
                    self.max_xy_vel,
                )
                vy = np.clip(
                    self.sign_y * self.kp_xy * ey,
                    -self.max_xy_vel,
                    self.max_xy_vel,
                )

            twist = self.build_twist(vx, vy)
            self.motion_update_publisher.publish(
                self.generate_velocity_motion_update(twist, self.base_frame)
            )

            self.get_logger().info(
                f"Servo magenta | centroid=({u:.1f}, {v:.1f}) "
                f"err=({ex:.1f}, {ey:.1f}) area={self.latest_area:.0f}"
            )

            if self.center_hold_count >= self.center_hold_needed:
                self.get_logger().info("Magenta centered. Third task complete.")
                self.task_done = True
                zero_twist = self.build_twist(0.0, 0.0)
                self.motion_update_publisher.publish(
                    self.generate_velocity_motion_update(zero_twist, self.base_frame)
                )
            return

        else:
            now = self.get_clock().now()
            elapsed = (now - self.search_state_start).nanoseconds * 1e-9

            # State machine:
            # 0: +y
            # 1: +x
            # 2: -y
            # 3: +x
            # repeat
            vx = 0.0
            vy = self.search_step_vel


            twist = self.build_twist(vx, vy)
            self.motion_update_publisher.publish(
                self.generate_velocity_motion_update(twist, self.base_frame)
            )

            if elapsed >= self.search_move_time:
                self.search_state = (self.search_state + 1) % 4
                self.search_state_start = now

            self.get_logger().info(
                f"Searching zig-zag | state={self.search_state} vx={vx:.3f} vy={vy:.3f}"
            )

class FourthTaskNode(Node):
    def __init__(self):
        super().__init__("fourth_task_node")

        self.bridge = CvBridge()

        self.image_topic = self.declare_parameter(
            "image_topic", "/center_camera/image"
        ).value

        self.magenta_h_low = int(self.declare_parameter("magenta_h_low", 135).value)
        self.magenta_h_high = int(self.declare_parameter("magenta_h_high", 175).value)
        self.magenta_s_low = int(self.declare_parameter("magenta_s_low", 70).value)
        self.magenta_v_low = int(self.declare_parameter("magenta_v_low", 40).value)

        self.image_sub = self.create_subscription(
            Image, self.image_topic, self.image_callback, 10
        )

        self.window_name = "magenta_mask"
        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)

        self.get_logger().info("Fourth task node started: showing magenta mask.")

    def image_callback(self, image_msg: Image):
        try:
            img = self.bridge.imgmsg_to_cv2(image_msg, desired_encoding="bgr8")
        except Exception as e:
            self.get_logger().warn(f"Image conversion failed: {e}")
            return

        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

        lower = np.array([self.magenta_h_low, self.magenta_s_low, self.magenta_v_low])
        upper = np.array([self.magenta_h_high, 255, 255])

        mask = cv2.inRange(hsv, lower, upper)

        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
            mask, connectivity=8
        )
        # Find contours instead of connected components
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        vis = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)

        if len(contours) > 0:
            # take largest contour
            cnt = max(contours, key=cv2.contourArea)
            area = cv2.contourArea(cnt)

            if area > 250.0:
                # fit minimum area rectangle
                rect = cv2.minAreaRect(cnt)
                (cx, cy), (w, h), angle = rect

                # get box corners
                box = cv2.boxPoints(rect)
                box = np.int32(box)

                # draw rectangle
                cv2.drawContours(vis, [box], 0, (0, 255, 0), 2)

                # draw center
                cx_i, cy_i = int(cx), int(cy)
                cv2.circle(vis, (cx_i, cy_i), 6, (0, 0, 255), -1)

                # draw orientation line
                length = 40
                theta = np.deg2rad(angle)
                x2 = int(cx + length * np.cos(theta))
                y2 = int(cy + length * np.sin(theta))
                cv2.line(vis, (cx_i, cy_i), (x2, y2), (255, 0, 0), 2)

                # label
                cv2.putText(
                    vis,
                    f"angle={angle:.1f}",
                    (cx_i + 10, cy_i - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 255, 0),
                    1,
                    cv2.LINE_AA,
                )

        # Convert mask to BGR so we can draw colored annotations on it
        # vis = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)

        # if num_labels > 1:
        #     best_idx = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
        #     area = float(stats[best_idx, cv2.CC_STAT_AREA])

        #     if area >= 250.0:
        #         cx, cy = centroids[best_idx]
        #         cx_i, cy_i = int(round(cx)), int(round(cy))

        #         # draw centroid
        #         cv2.circle(vis, (cx_i, cy_i), 6, (0, 255, 0), -1)
        #         cv2.drawMarker(
        #             vis, (cx_i, cy_i), (0, 255, 0),
        #             markerType=cv2.MARKER_CROSS, markerSize=18, thickness=2
        #         )
        #         cv2.putText(
        #             vis, f"({cx_i},{cy_i})",
        #             (cx_i + 10, cy_i - 10),
        #             cv2.FONT_HERSHEY_SIMPLEX, 0.5,
        #             (0, 255, 0), 1, cv2.LINE_AA
        #         )


        cv2.imshow(self.window_name, vis)
        cv2.waitKey(1)

# def main(args=None):
#     try:
#         rclpy.init(args=args)

#         ref_node = ReferencePoseInitializer()
#         ref_node.send_change_control_mode_req(TargetMode.MODE_CARTESIAN)

#         while rclpy.ok() and not ref_node.goal_reached:
#             rclpy.spin_once(ref_node, timeout_sec=0.1)

#         ref_node.get_logger().info("Starting second code now.")
#         ref_node.destroy_node()

#         search_node = search_find_Node()
#         search_node.send_change_control_mode_req(TargetMode.MODE_CARTESIAN)

#         while rclpy.ok() and not search_node.task_done:
#             rclpy.spin_once(search_node, timeout_sec=0.1)

#         search_node.destroy_node()

#         fourth_node = FourthTaskNode()
#         rclpy.spin(fourth_node)

#     except (KeyboardInterrupt, ExternalShutdownException):
#         pass
#     finally:
#         if "fourth_node" in locals():
#             fourth_node.destroy_node()
#         cv2.destroyAllWindows()
#         if rclpy.ok():
#             rclpy.shutdown()

# if __name__ == "__main__":
#     main(sys.argv)