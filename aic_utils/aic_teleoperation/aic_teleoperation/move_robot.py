#!/usr/bin/env python3

from __future__ import annotations

import argparse
import sys
import time
from collections.abc import Sequence

import numpy as np
import rclpy
from aic_control_interfaces.msg import (
    MotionUpdate,
    TargetMode,
    TrajectoryGenerationMode,
)
from aic_control_interfaces.srv import ChangeTargetMode
from geometry_msgs.msg import Transform, Twist, Vector3, Wrench
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from rclpy.time import Time
from tf2_ros import Buffer, TransformException, TransformListener
from transforms3d._gohlketransforms import quaternion_multiply, quaternion_slerp

DEFAULT_TARGET_POS =  [-0.371, 0.195, 0.329]
DEFAULT_TARGET_QUAT = [-1.000, -0.000, 0.000, -0.000]
TARGET_POS = DEFAULT_TARGET_POS
TARGET_QUAT = DEFAULT_TARGET_QUAT


def _as_vector(values: Sequence[float], size: int, name: str) -> np.ndarray:
    vector = np.asarray(values, dtype=np.float64)
    if vector.shape != (size,):
        raise ValueError(f"{name} must contain exactly {size} values")
    if not np.all(np.isfinite(vector)):
        raise ValueError(f"{name} must contain only finite values")
    return vector


def _normalize_quaternion(quaternion: Sequence[float], name: str) -> np.ndarray:
    quat = _as_vector(quaternion, 4, name)
    norm = float(np.linalg.norm(quat))
    if norm < 1e-12:
        raise ValueError(f"{name} must have a non-zero norm")
    return quat / norm


def interpolate_pose(
    start_position: Sequence[float],
    start_quaternion: Sequence[float],
    goal_position: Sequence[float],
    goal_quaternion: Sequence[float],
    fraction: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Interpolate a Cartesian pose.

    Positions are linearly interpolated. Quaternions use WXYZ order and are
    interpolated along the shortest path using SLERP. Keeping this function
    independent from ROS makes it easy to replace with a custom motion planner.
    """
    start_pos = _as_vector(start_position, 3, "start_position")
    goal_pos = _as_vector(goal_position, 3, "goal_position")
    start_quat = _normalize_quaternion(start_quaternion, "start_quaternion")
    goal_quat = _normalize_quaternion(goal_quaternion, "goal_quaternion")
    blend = float(np.clip(fraction, 0.0, 1.0))

    position = start_pos + blend * (goal_pos - start_pos)
    quaternion = np.asarray(
        quaternion_slerp(start_quat, goal_quat, blend, shortestpath=True),
        dtype=np.float64,
    )
    return position, quaternion


def _clip_norm(vector: np.ndarray, max_norm: float) -> np.ndarray:
    norm = float(np.linalg.norm(vector))
    if norm <= max_norm or norm < 1e-12:
        return vector
    return vector * (max_norm / norm)


class MoveRobot(Node):
    """Move the gripper TCP to a goal pose with Cartesian velocity commands."""

    def __init__(
        self,
        goal_tcp_position: Sequence[float] = TARGET_POS,
        goal_quaternion: Sequence[float] = TARGET_QUAT,
    ) -> None:
        super().__init__("move_robot")

        self.goal_position = _as_vector(
            goal_tcp_position, 3, "goal_tcp_position"
        )
        self.goal_quaternion = _normalize_quaternion(
            goal_quaternion, "goal_quaternion"
        )

        self.controller_namespace = self.declare_parameter(
            "controller_namespace", "aic_controller"
        ).value
        self.frame_id = self.declare_parameter("frame_id", "base_link").value
        self.tcp_frame = self.declare_parameter("tcp_frame", "gripper/tcp").value
        self.interpolation_duration_s = float(
            self.declare_parameter("interpolation_duration_s", 3.0).value
        )
        self.kp_position = float(self.declare_parameter("kp_position", 2.0).value)
        self.kp_orientation = float(
            self.declare_parameter("kp_orientation", 2.0).value
        )
        self.max_linear_speed_m_s = float(
            self.declare_parameter("max_linear_speed_m_s", 0.05).value
        )
        self.max_angular_speed_rad_s = float(
            self.declare_parameter("max_angular_speed_rad_s", 0.5).value
        )
        self.position_tolerance_m = float(
            self.declare_parameter("position_tolerance_m", 0.002).value
        )
        self.orientation_tolerance_rad = float(
            self.declare_parameter("orientation_tolerance_rad", 0.01).value
        )

        if self.interpolation_duration_s <= 0.0:
            raise ValueError("interpolation_duration_s must be positive")
        if self.max_linear_speed_m_s <= 0.0:
            raise ValueError("max_linear_speed_m_s must be positive")
        if self.max_angular_speed_rad_s <= 0.0:
            raise ValueError("max_angular_speed_rad_s must be positive")

        self.pose_commands_topic = f"/{self.controller_namespace}/pose_commands"
        self.change_target_mode_srv = (
            f"/{self.controller_namespace}/change_target_mode"
        )
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
        while not self.change_target_mode_client.wait_for_service(timeout_sec=1.0):
            self.get_logger().info(
                f"Waiting for service '{self.change_target_mode_srv}'..."
            )

        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        self.start_position: np.ndarray | None = None
        self.start_quaternion: np.ndarray | None = None
        self.motion_started_monotonic: float | None = None
        self.goal_reached = False

        self._send_change_control_mode_req(TargetMode.MODE_CARTESIAN)
        self.timer = self.create_timer(0.04, self._control_loop)

    def _send_change_control_mode_req(self, mode: int) -> None:
        request = ChangeTargetMode.Request()
        request.target_mode.mode = int(mode)
        future = self.change_target_mode_client.call_async(request)
        rclpy.spin_until_future_complete(self, future)
        response = future.result()
        if response is None or not response.success:
            raise RuntimeError(f"Failed to change target mode to {mode}")
        self.get_logger().info(f"Changed target mode to {mode}")
        time.sleep(0.5)

    def _lookup_tcp_transform(self) -> Transform:
        transform = self.tf_buffer.lookup_transform(
            self.frame_id, self.tcp_frame, Time()
        )
        return transform.transform

    @staticmethod
    def _transform_to_pose_arrays(
        transform: Transform,
    ) -> tuple[np.ndarray, np.ndarray]:
        position = np.array(
            [
                transform.translation.x,
                transform.translation.y,
                transform.translation.z,
            ],
            dtype=np.float64,
        )
        quaternion = np.array(
            [
                transform.rotation.w,
                transform.rotation.x,
                transform.rotation.y,
                transform.rotation.z,
            ],
            dtype=np.float64,
        )
        return position, _normalize_quaternion(quaternion, "current_quaternion")

    def _planned_pose(self, fraction: float) -> tuple[np.ndarray, np.ndarray]:
        """Return the current plan target; replace this for a custom planner."""
        if self.start_position is None or self.start_quaternion is None:
            raise RuntimeError(
                "Motion plan requested before the start pose was captured"
            )
        return interpolate_pose(
            self.start_position,
            self.start_quaternion,
            self.goal_position,
            self.goal_quaternion,
            fraction,
        )

    def _pose_error_to_twist(
        self,
        current_position: np.ndarray,
        current_quaternion: np.ndarray,
        target_position: np.ndarray,
        target_quaternion: np.ndarray,
    ) -> tuple[Twist, float, float]:
        position_error = target_position - current_position
        linear_velocity = _clip_norm(
            self.kp_position * position_error, self.max_linear_speed_m_s
        )

        current_quaternion_inverse = np.array(
            [
                current_quaternion[0],
                -current_quaternion[1],
                -current_quaternion[2],
                -current_quaternion[3],
            ],
            dtype=np.float64,
        )
        quaternion_error = np.asarray(
            quaternion_multiply(target_quaternion, current_quaternion_inverse),
            dtype=np.float64,
        )
        if quaternion_error[0] < 0.0:
            quaternion_error *= -1.0

        scalar = float(np.clip(quaternion_error[0], -1.0, 1.0))
        angle = 2.0 * float(np.arccos(scalar))
        sin_half_angle = float(np.sqrt(max(1.0 - scalar * scalar, 0.0)))
        if sin_half_angle < 1e-6 or angle < 1e-6:
            rotation_vector = np.zeros(3, dtype=np.float64)
        else:
            rotation_vector = quaternion_error[1:4] * (angle / sin_half_angle)

        angular_velocity = _clip_norm(
            self.kp_orientation * rotation_vector,
            self.max_angular_speed_rad_s,
        )

        twist = Twist()
        twist.linear.x = float(linear_velocity[0])
        twist.linear.y = float(linear_velocity[1])
        twist.linear.z = float(linear_velocity[2])
        twist.angular.x = float(angular_velocity[0])
        twist.angular.y = float(angular_velocity[1])
        twist.angular.z = float(angular_velocity[2])
        return twist, float(np.linalg.norm(position_error)), angle

    def _publish_twist(self, twist: Twist) -> None:
        message = MotionUpdate()
        message.header.stamp = self.get_clock().now().to_msg()
        message.header.frame_id = self.frame_id
        message.velocity = twist
        message.target_stiffness = np.diag([85.0] * 6).flatten()
        message.target_damping = np.diag([75.0] * 6).flatten()
        message.feedforward_wrench_at_tip = Wrench(
            force=Vector3(x=0.0, y=0.0, z=0.0),
            torque=Vector3(x=0.0, y=0.0, z=0.0),
        )
        message.wrench_feedback_gains_at_tip = [0.0] * 6
        message.trajectory_generation_mode.mode = (
            TrajectoryGenerationMode.MODE_VELOCITY
        )
        self.motion_update_publisher.publish(message)

    def _control_loop(self) -> None:
        if self.goal_reached:
            self._publish_twist(Twist())
            return

        try:
            current_position, current_quaternion = self._transform_to_pose_arrays(
                self._lookup_tcp_transform()
            )
        except TransformException as error:
            self.get_logger().warn(
                f"Could not look up TCP transform: {error}",
                throttle_duration_sec=2.0,
            )
            self._publish_twist(Twist())
            return

        if self.start_position is None:
            self.start_position = current_position.copy()
            self.start_quaternion = current_quaternion.copy()
            self.motion_started_monotonic = time.monotonic()
            self.get_logger().info("Captured TCP start pose; beginning interpolation.")

        if self.motion_started_monotonic is None:
            raise RuntimeError("Motion start time was not initialized")

        elapsed = time.monotonic() - self.motion_started_monotonic
        fraction = min(elapsed / self.interpolation_duration_s, 1.0)
        target_position, target_quaternion = self._planned_pose(fraction)
        twist, position_error, orientation_error = self._pose_error_to_twist(
            current_position,
            current_quaternion,
            target_position,
            target_quaternion,
        )

        at_goal = (
            fraction >= 1.0
            and position_error <= self.position_tolerance_m
            and orientation_error <= self.orientation_tolerance_rad
        )
        if at_goal:
            self.goal_reached = True
            self._publish_twist(Twist())
            self.get_logger().info("Goal TCP pose reached.")
            return

        self._publish_twist(twist)


def _parse_arguments(args: Sequence[str]) -> tuple[argparse.Namespace, list[str]]:
    parser = argparse.ArgumentParser(
        description="Move the gripper TCP to a Cartesian goal pose."
    )
    parser.add_argument(
        "--position",
        nargs=3,
        type=float,
        default=TARGET_POS,
        metavar=("X", "Y", "Z"),
        help=f"goal TCP position (default: {TARGET_POS})",
    )
    parser.add_argument(
        "--quaternion",
        nargs=4,
        type=float,
        default=TARGET_QUAT,
        metavar=("W", "X", "Y", "Z"),
        help=f"goal TCP quaternion in WXYZ order (default: {TARGET_QUAT})",
    )
    return parser.parse_known_args(args)


def main(args: Sequence[str] | None = None) -> None:
    command_line_args = list(sys.argv[1:] if args is None else args)
    parsed_args, ros_args = _parse_arguments(command_line_args)

    node: MoveRobot | None = None
    try:
        with rclpy.init(args=[sys.argv[0], *ros_args]):
            node = MoveRobot(parsed_args.position, parsed_args.quaternion)
            rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        if node is not None:
            node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
