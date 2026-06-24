#!/usr/bin/env python3

from __future__ import annotations

import sys
import time

import numpy as np
import rclpy
from aic_control_interfaces.msg import MotionUpdate, TargetMode, TrajectoryGenerationMode
from aic_control_interfaces.srv import ChangeTargetMode
from geometry_msgs.msg import Point, Pose, Quaternion, Transform, Twist, Vector3, Wrench
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from rclpy.time import Time
from std_srvs.srv import Trigger
from tf2_ros import Buffer, TransformException, TransformListener
from transforms3d._gohlketransforms import quaternion_multiply, quaternion_slerp
from scipy.spatial.transform import Rotation as Rot

class _Phase:
    INTERP = "interp"
    DESCEND = "descend"
    STABILIZE = "stabilize"
    DONE = "done"


def _clip_norm(vec: np.ndarray, max_norm: float) -> np.ndarray:
    norm = float(np.linalg.norm(vec))
    if norm <= max_norm or norm < 1e-12:
        return vec
    return vec * (max_norm / norm)


NIC_TIP_QUAT_XYZW = np.array([0.180, 0.006, -0.027, 0.983], dtype=float)
SC_TIP_QUAT_XYZW = np.array([-0.161, 0.167, -0.694, -0.681], dtype=float)
NIC_GRIPPER_OFFSET = np.array([-0.000, -0.018, 0.048], dtype=float)
SC_GRIPPER_OFFSET = np.array([-0.001, -0.010, 0.008], dtype=float)
NIC_PORT_Z_OFFSET = 0.133
SC_PORT_Z_OFFSET = 0.017


def _is_nic_module(module_name: str) -> bool:
    return module_name.startswith("nic")


def get_tip_pose_config(module_name: str) -> tuple[np.ndarray, np.ndarray, float]:
    if _is_nic_module(module_name):
        return NIC_TIP_QUAT_XYZW.copy(), NIC_GRIPPER_OFFSET.copy() ,NIC_PORT_Z_OFFSET
    return SC_TIP_QUAT_XYZW.copy(), SC_GRIPPER_OFFSET.copy() , SC_PORT_Z_OFFSET


class insert_task(Node):
    def __init__(self,port_pos,port_quaternion,module_name=None) -> None:
        super().__init__("insert_task")

        # Controller ROS API.
        self.controller_namespace = self.declare_parameter(
            "controller_namespace", "aic_controller"
        ).value
        self.frame_id = self.declare_parameter("frame_id", "base_link").value

        self.pose_commands_topic = f"/{self.controller_namespace}/pose_commands"
        self.change_target_mode_srv = f"/{self.controller_namespace}/change_target_mode"
        self.tare_srv = f"/{self.controller_namespace}/tare_force_torque_sensor"

        # Task configuration (same naming as aic_task_interfaces/msg/Task).
        self.target_module_name = self.declare_parameter(
            "target_module_name", "nic_card_mount_0"
        ).value
        self.module_name = module_name or self.target_module_name
        self.q_tip_g, self.gripper_offset, self.port_z_offset = get_tip_pose_config(self.module_name)
        self.port_name = self.declare_parameter("port_name", "sfp_port_0").value
        self.cable_name = self.declare_parameter("cable_name", "cable_0").value
        self.plug_name = self.declare_parameter("plug_name", "sfp_tip").value

        # CheatCode motion parameters.
        self.z_offset_start = float(self.declare_parameter("z_offset_start", 0.2).value)
        self.z_offset_end = float(self.declare_parameter("z_offset_end", -0.015).value)
        self.interp_duration_s = float(
            self.declare_parameter("interp_duration_s", 3.0).value
        )
        self.descend_speed_m_s = float(
            self.declare_parameter("descend_speed_m_s", 0.01).value
        )
        self.stabilize_time_s = float(
            self.declare_parameter("stabilize_time_s", 3.0).value
        )

        # Velocity controller parameters.
        self.kp_position = float(self.declare_parameter("kp_position", 2.0).value)
        self.kp_orientation = float(self.declare_parameter("kp_orientation", 2.0).value)
        self.max_linear_speed_m_s = float(
            self.declare_parameter("max_linear_speed_m_s", 0.05).value
        )
        self.max_angular_speed_rad_s = float(
            self.declare_parameter("max_angular_speed_rad_s", 0.5).value
        )

        self.tare_on_start = bool(self.declare_parameter("tare_on_start", True).value)

        self.motion_update_publisher = self.create_publisher(
            MotionUpdate, self.pose_commands_topic, 10
        )

        while self.motion_update_publisher.get_subscription_count() == 0:
            self.get_logger().info(
                f"Waiting for subscriber to '{self.pose_commands_topic}'..."
            )
            time.sleep(1.0)

        self.change_target_mode_client = self.create_client(
            ChangeTargetMode, self.change_target_mode_srv
        )
        while not self.change_target_mode_client.wait_for_service():
            self.get_logger().info(
                f"Waiting for service '{self.change_target_mode_srv}'..."
            )
            time.sleep(1.0)

        self.tare_client = self.create_client(Trigger, self.tare_srv)

        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        # Integrator copied from CheatCode.py.
        self._tip_x_error_integrator = 0.0
        self._tip_y_error_integrator = 0.0
        self._max_integrator_windup = 0.05

        self._phase = _Phase.INTERP
        self._phase_started_monotonic = time.monotonic()
        self.port_pos = port_pos
        self.port_quaternion = port_quaternion

        if self.tare_on_start:
            self._tare_force_torque_sensor()

        self._send_change_control_mode_req(TargetMode.MODE_CARTESIAN)

        # Publish commands at 25Hz.
        self.timer = self.create_timer(0.04, self._control_loop)

    def _tare_force_torque_sensor(self) -> None:
        if not self.tare_client.wait_for_service(timeout_sec=2.0):
            self.get_logger().warn(
                f"Service '{self.tare_srv}' not available (evaluation disables taring); continuing."
            )
            return
        req = Trigger.Request()
        future = self.tare_client.call_async(req)
        rclpy.spin_until_future_complete(self, future, timeout_sec=5.0)
        if future.result() is None:
            self.get_logger().warn("FT tare request did not return; continuing.")
            return
        res = future.result()
        if res.success:
            self.get_logger().info("FT sensor tared.")
        else:
            self.get_logger().warn(f"FT tare failed: {res.message}")

    def _send_change_control_mode_req(self, mode: int) -> None:
        req = ChangeTargetMode.Request()
        req.target_mode.mode = int(mode)
        self.get_logger().info(f"Sending request to change target mode to {mode}")
        future = self.change_target_mode_client.call_async(req)
        rclpy.spin_until_future_complete(self, future)
        if future.result() is None:
            self.get_logger().error("Failed to change target mode (no response).")
            return
        if future.result().success:
            self.get_logger().info(f"Successfully changed target mode to {mode}")
        else:
            self.get_logger().error(f"Failed to change target mode to {mode}")
        time.sleep(0.5)

    def _lookup_transform(self, target_frame: str, source_frame: str) -> Transform:
        tf_stamped = self.tf_buffer.lookup_transform(
            target_frame, source_frame, Time()
        )
        return tf_stamped.transform
    def _wxyz_to_xyzw(self, q_wxyz):
        return np.array([q_wxyz[1], q_wxyz[2], q_wxyz[3], q_wxyz[0]], dtype=float)
    def _calc_gripper_pose(
        self,
        *,
        slerp_fraction: float = 1.0,
        position_fraction: float = 1.0,
        z_offset: float = 0.1,
        reset_xy_integrator: bool = False,
    ) -> Pose:
        gripper_tf = self._lookup_transform(self.frame_id, "gripper/tcp")
        gripper_xyz = np.array( [
            gripper_tf.translation.x,
            gripper_tf.translation.y,
            gripper_tf.translation.z,
        ])
        q_gripper = (
            gripper_tf.rotation.w,
            gripper_tf.rotation.x,
            gripper_tf.rotation.y,
            gripper_tf.rotation.z,
        )
        q_tip_g_wxyz = (
            float(self.q_tip_g[3]),
            float(self.q_tip_g[0]),
            float(self.q_tip_g[1]),
            float(self.q_tip_g[2]),
        )
        q_plug = quaternion_multiply(q_gripper, q_tip_g_wxyz)


        scipy_quat = (q_gripper[1], q_gripper[2], q_gripper[3], q_gripper[0])
        rot = Rot.from_quat(scipy_quat)
        plug_xyz = gripper_xyz + rot.apply(self.gripper_offset)
        # self.get_logger().info(f"Gripper XYZ: {gripper_xyz}, Plug XYZ: {plug_xyz}")


        q_plug_inv = (q_plug[0], -q_plug[1], -q_plug[2], -q_plug[3])

        q_diff = quaternion_multiply(self.port_quaternion, q_plug_inv)

        q_gripper_target = quaternion_multiply(q_diff, q_gripper)
        q_gripper_slerp = quaternion_slerp(q_gripper, q_gripper_target, slerp_fraction)

        port_xy = (self.port_pos[0], self.port_pos[1])
        plug_tip_gripper_offset = (
            gripper_xyz[0] - plug_xyz[0],
            gripper_xyz[1] - plug_xyz[1],
            gripper_xyz[2] - plug_xyz[2],
        )

        tip_x_error = port_xy[0] - plug_xyz[0]
        tip_y_error = port_xy[1] - plug_xyz[1]

        if reset_xy_integrator:
            self._tip_x_error_integrator = 0.0
            self._tip_y_error_integrator = 0.0
        else:
            self._tip_x_error_integrator = float(
                np.clip(
                    self._tip_x_error_integrator + tip_x_error,
                    -self._max_integrator_windup,
                    self._max_integrator_windup,
                )
            )
            self._tip_y_error_integrator = float(
                np.clip(
                    self._tip_y_error_integrator + tip_y_error,
                    -self._max_integrator_windup,
                    self._max_integrator_windup,
                )
            )

        i_gain = 0.15
        target_x = port_xy[0] + i_gain * self._tip_x_error_integrator
        target_y = port_xy[1] + i_gain * self._tip_y_error_integrator

        target_z = self.port_z_offset + z_offset - plug_tip_gripper_offset[2]

        blend_xyz = (
            position_fraction * target_x + (1.0 - position_fraction) * gripper_xyz[0],
            position_fraction * target_y + (1.0 - position_fraction) * gripper_xyz[1],
            position_fraction * target_z + (1.0 - position_fraction) * gripper_xyz[2],
        )

        return Pose(
            position=Point(x=blend_xyz[0], y=blend_xyz[1], z=blend_xyz[2]),
            orientation=Quaternion(
                w=float(q_gripper_slerp[0]),
                x=float(q_gripper_slerp[1]),
                y=float(q_gripper_slerp[2]),
                z=float(q_gripper_slerp[3]),
            ),
        )

    def _pose_to_twist(self, target_pose: Pose) -> Twist:
        gripper_tf = self._lookup_transform(self.frame_id, "gripper/tcp")
        cur_p = np.array(
            [
                gripper_tf.translation.x,
                gripper_tf.translation.y,
                gripper_tf.translation.z,
            ],
            dtype=np.float64,
        )
        tgt_p = np.array(
            [target_pose.position.x, target_pose.position.y, target_pose.position.z],
            dtype=np.float64,
        )
        linear_vel = self.kp_position * (tgt_p - cur_p)
        linear_vel = _clip_norm(linear_vel, self.max_linear_speed_m_s)

        q_cur = np.array(
            [
                gripper_tf.rotation.w,
                gripper_tf.rotation.x,
                gripper_tf.rotation.y,
                gripper_tf.rotation.z,
            ],
            dtype=np.float64,
        )
        q_tgt = np.array(
            [
                target_pose.orientation.w,
                target_pose.orientation.x,
                target_pose.orientation.y,
                target_pose.orientation.z,
            ],
            dtype=np.float64,
        )

        q_cur_inv = np.array(
            [q_cur[0], -q_cur[1], -q_cur[2], -q_cur[3]], dtype=np.float64
        )
        q_err = np.array(quaternion_multiply(tuple(q_tgt), tuple(q_cur_inv)), dtype=np.float64)
        if q_err[0] < 0:
            q_err *= -1.0

        w = float(np.clip(q_err[0], -1.0, 1.0))
        angle = 2.0 * float(np.arccos(w))
        if angle > np.pi:
            angle = 2.0 * np.pi - angle

        sin_half = float(np.sqrt(max(1.0 - w * w, 0.0)))
        if sin_half < 1e-6 or angle < 1e-6:
            axis = np.zeros(3, dtype=np.float64)
        else:
            axis = q_err[1:4] / sin_half

        angular_vel = self.kp_orientation * (axis * angle)
        angular_vel = _clip_norm(angular_vel, self.max_angular_speed_rad_s)

        twist = Twist()
        twist.linear.x = float(linear_vel[0])
        twist.linear.y = float(linear_vel[1])
        twist.linear.z = float(linear_vel[2])
        twist.angular.x = float(angular_vel[0])
        twist.angular.y = float(angular_vel[1])
        twist.angular.z = float(angular_vel[2])
        return twist

    def _publish_twist(self, twist: Twist) -> None:
        msg = MotionUpdate()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = self.frame_id
        msg.velocity = twist
        msg.target_stiffness = np.diag([85.0, 85.0, 85.0, 85.0, 85.0, 85.0]).flatten()
        msg.target_damping = np.diag([75.0, 75.0, 75.0, 75.0, 75.0, 75.0]).flatten()
        msg.feedforward_wrench_at_tip = Wrench(
            force=Vector3(x=0.0, y=0.0, z=0.0),
            torque=Vector3(x=0.0, y=0.0, z=0.0),
        )
        msg.wrench_feedback_gains_at_tip = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        msg.trajectory_generation_mode.mode = TrajectoryGenerationMode.MODE_VELOCITY
        self.motion_update_publisher.publish(msg)

    def _zero_twist(self) -> Twist:
        return Twist()

    def _enter_phase(self, phase: str) -> None:
        self._phase = phase
        self._phase_started_monotonic = time.monotonic()

    def _phase_elapsed(self) -> float:
        return time.monotonic() - self._phase_started_monotonic

    def _control_loop(self) -> None:
        if self._phase == _Phase.DONE:
            self._publish_twist(self._zero_twist())
            return

        if self._phase == _Phase.INTERP:
            frac = min(max(self._phase_elapsed() / self.interp_duration_s, 0.0), 1.0)
            try:
                target_pose = self._calc_gripper_pose(
                    slerp_fraction=frac,
                    position_fraction=frac,
                    z_offset=self.z_offset_start,
                    reset_xy_integrator=True,
                )
                twist = self._pose_to_twist(target_pose)
            except TransformException:
                twist = self._zero_twist()
            self._publish_twist(twist)
            if frac >= 1.0:
                self._enter_phase(_Phase.DESCEND)
            return

        if self._phase == _Phase.DESCEND:
            z_offset = self.z_offset_start - self.descend_speed_m_s * self._phase_elapsed()
            if z_offset <= self.z_offset_end:
                self._enter_phase(_Phase.STABILIZE)
                self._publish_twist(self._zero_twist())
                return
            try:
                target_pose = self._calc_gripper_pose(z_offset=z_offset)
                twist = self._pose_to_twist(target_pose)
            except TransformException:
                twist = self._zero_twist()
            self._publish_twist(twist)
            return

        if self._phase == _Phase.STABILIZE:
            self._publish_twist(self._zero_twist())
            if self._phase_elapsed() >= self.stabilize_time_s:
                self._enter_phase(_Phase.DONE)
                self.get_logger().info("Insert task complete.")
            return

        self._publish_twist(self._zero_twist())


def main(args=None) -> None:
    print(
        """
        AIC CheatCode TF Teleop
        ----------------------
        Runs an autonomous TF-based controller (requires `ground_truth:=true` TF frames).
        Publishes cartesian velocity commands to `/aic_controller/pose_commands`.
        """
    )

    node: insert_task | None = None
    try:
        with rclpy.init(args=args):
            node = insert_task()
            rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        if node is not None:
            node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main(sys.argv)
