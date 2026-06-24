import re
import time

import numpy as np
import rclpy
from aic_control_interfaces.msg import MotionUpdate, TargetMode
from aic_model.policy import (
    GetObservationCallback,
    MoveRobotCallback,
    Policy,
    SendFeedbackCallback,
)
from aic_task_interfaces.msg import Task
from aic_teleoperation.insert import insert_task
from aic_teleoperation.mag_test import Stereo_mag_square
from aic_teleoperation.move2task import get_goal_position_tb, nic_task
from aic_teleoperation.port_finder import Stereo_nic_port
from aic_teleoperation.reference_pose_initializer import (
    ReferencePoseInitializer,
    search_find_Node,
)
from aic_teleoperation.sc_port_finder import Stereo_sc_port
from scipy.spatial.transform import Rotation as Rot


class _MoveRobotPublisher:
    """Publisher-shaped adapter that forwards MotionUpdate messages to aic_model."""

    def __init__(self, move_robot: MoveRobotCallback):
        self._move_robot = move_robot

    def publish(self, motion_update: MotionUpdate) -> None:
        self._move_robot(motion_update=motion_update)

    def get_subscription_count(self) -> int:
        return 1


class NicInsertionPolicy(Policy):
    """Policy API wrapper around the working NIC insertion sequence."""

    def __init__(self, parent_node):
        super().__init__(parent_node)
        self.get_logger().info("NicInsertionPolicy.__init__()")

    def _spin_until(self, node, predicate, feedback, timeout_sec=None):
        start = time.perf_counter()
        last_feedback = 0.0
        while rclpy.ok() and not predicate():
            if timeout_sec is not None and time.perf_counter() - start > timeout_sec:
                node.get_logger().error(f"Timed out while {feedback}.")
                return False
            now = time.perf_counter()
            if now - last_feedback > 2.0:
                self.get_logger().info(feedback)
                last_feedback = now
            rclpy.spin_once(node, timeout_sec=0.1)
        return predicate()

    def _route_motion_updates_through_callback(
        self,
        node,
        move_robot: MoveRobotCallback,
    ) -> None:
        if not hasattr(node, "motion_update_publisher"):
            self.get_logger().warn(
                f"{node.get_name()} has no motion_update_publisher to route."
            )
            return
        node.motion_update_publisher = _MoveRobotPublisher(move_robot)

    def _target_module_name(self, task: Task) -> str:
        if task.target_module_name:
            return task.target_module_name
        self.get_logger().warn("Task has no target_module_name; defaulting to nic_card_0.")
        return "nic_card_0"

    def _is_nic_module(self, module_name: str) -> bool:
        return module_name.startswith("nic")

    def _goal_position_tb(self, module_name: str) -> np.ndarray:
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

    def _board_to_module_rotation(self, module_name: str) -> np.ndarray:
        if re.match(r"sc_port_?(\d+)$", module_name):
            return np.array(
                [
                    [0.0, 1.0, 0.0],
                    [1.0, 0.0, 0.0],
                    [0.0, 0.0, -1.0],
                ]
            )
        return np.diag([-1.0, 1.0, -1.0])

    def _select_nic_port_position(self, stereo_port_node, task: Task):
        if task.port_name == "sfp_port_0":
            return stereo_port_node.port0
        if task.port_name == "sfp_port_1":
            return stereo_port_node.port1

        match = re.search(r"(\d+)$", task.port_name)
        if match is not None:
            port_idx = int(match.group(1))
            if port_idx == 0:
                return stereo_port_node.port0
            if port_idx == 1:
                return stereo_port_node.port1

        self.get_logger().warn(
            f"Unsupported or empty NIC port_name '{task.port_name}'; defaulting to sfp_port_0."
        )
        return stereo_port_node.port0

    def insert_cable(
        self,
        task: Task,
        get_observation: GetObservationCallback,
        move_robot: MoveRobotCallback,
        send_feedback: SendFeedbackCallback,
    ) -> bool:
        self.get_logger().info(f"NicInsertionPolicy.insert_cable() task: {task}")
        total_start = time.perf_counter()
        nodes_to_destroy = []

        try:
            module_name = self._target_module_name(task)
            goal_position_tb = self._goal_position_tb(module_name)
            send_feedback(f"Preparing insertion for {module_name}")

            init_ref_node = ReferencePoseInitializer()
            nodes_to_destroy.append(init_ref_node)
            self._route_motion_updates_through_callback(init_ref_node, move_robot)
            init_ref_node.send_change_control_mode_req(TargetMode.MODE_CARTESIAN)
            if not self._spin_until(
                init_ref_node,
                lambda: init_ref_node.goal_reached,
                "Moving to initial reference pose",
            ):
                return False

            search_node = search_find_Node()
            nodes_to_destroy.append(search_node)
            self._route_motion_updates_through_callback(search_node, move_robot)
            search_node.send_change_control_mode_req(TargetMode.MODE_CARTESIAN)
            if not self._spin_until(
                search_node,
                lambda: search_node.task_done,
                "Searching for task board",
            ):
                return False

            mag_ref_node = Stereo_mag_square()
            nodes_to_destroy.append(mag_ref_node)
            if not self._spin_until(
                mag_ref_node,
                lambda: mag_ref_node.latest_result is not None,
                "Estimating task board pose",
            ):
                return False

            result = mag_ref_node.latest_result
            if result is None:
                self.get_logger().error("Failed to get task board pose.")
                return False

            center = np.asarray(result["center"])
            rz = np.asarray(result["Rz"])
            goal_position = rz @ goal_position_tb + center

            r_mb_t = self._board_to_module_rotation(module_name)
            r_mb = rz @ r_mb_t
            quat = Rot.from_matrix(r_mb).as_quat()
            goal_quaternion_tip = [quat[0], quat[1], quat[2], quat[3]]

            mag_ref_node.destroy_node()
            nodes_to_destroy.remove(mag_ref_node)

            move_to_port_node = nic_task(goal_position, goal_quaternion_tip, module_name)
            nodes_to_destroy.append(move_to_port_node)
            self._route_motion_updates_through_callback(move_to_port_node, move_robot)
            if not self._spin_until(
                move_to_port_node,
                lambda: move_to_port_node.goal_reached,
                "Moving plug above target port",
            ):
                return False

            if self._is_nic_module(module_name):
                stereo_port_node = Stereo_nic_port()
                nodes_to_destroy.append(stereo_port_node)
                if not self._spin_until(
                    stereo_port_node,
                    lambda: stereo_port_node.card_orientation is not None,
                    "Estimating NIC port pose",
                ):
                    return False

                if (
                    stereo_port_node.port1 is None
                    or stereo_port_node.card_orientation is None
                ):
                    self.get_logger().error("Failed to get NIC port pose.")
                    return False

                q_xyzw = stereo_port_node.card_orientation
                port_pos = self._select_nic_port_position(stereo_port_node, task)
                if port_pos is None:
                    self.get_logger().error(
                        f"Failed to get requested NIC port pose for '{task.port_name}'."
                    )
                    return False
            else:
                stereo_port_node = Stereo_sc_port()
                nodes_to_destroy.append(stereo_port_node)
                if not self._spin_until(
                    stereo_port_node,
                    lambda: stereo_port_node.port is not None,
                    "Estimating SC port pose",
                ):
                    return False

                if stereo_port_node.port is None:
                    self.get_logger().error("Failed to get SC port pose.")
                    return False

                q_xyzw = goal_quaternion_tip
                port_pos = stereo_port_node.port

            q_wxyz = (q_xyzw[3], q_xyzw[0], q_xyzw[1], q_xyzw[2])
            insert_node = insert_task(port_pos, q_wxyz, module_name)
            nodes_to_destroy.append(insert_node)
            self._route_motion_updates_through_callback(insert_node, move_robot)

            send_feedback("Inserting cable")
            if not self._spin_until(
                insert_node,
                lambda: insert_node._phase == "done",
                "Running insertion controller",
            ):
                return False

            self.get_logger().info("NicInsertionPolicy.insert_cable() exiting...")
            return True

        except Exception as exc:
            self.get_logger().error(f"NIC insertion policy failed: {exc}")
            return False

        finally:
            for node in reversed(nodes_to_destroy):
                try:
                    node.destroy_node()
                except Exception as exc:
                    self.get_logger().warn(f"Failed to destroy node: {exc}")
            total_time = time.perf_counter() - total_start
            self.get_logger().info(f"[TIME] TOTAL EXECUTION TIME: {total_time:.3f} sec")
