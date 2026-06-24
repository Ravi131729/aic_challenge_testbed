#!/usr/bin/env python3

import cv2
import numpy as np
import rclpy
import tf2_ros

from cv_bridge import CvBridge
from rclpy.node import Node
from sensor_msgs.msg import CameraInfo, Image


class BoardPoseEstimator(Node):
    def __init__(self):
        super().__init__("board_pose_estimator")

        self.bridge = CvBridge()
        self.center_info = None
        self.done = False

        self.create_subscription(
            CameraInfo,
            "/center_camera/camera_info",
            self.center_info_cb,
            10,
        )

        self.image_sub = self.create_subscription(
            Image,
            "/center_camera/image",
            self.image_cb,
            10,
        )

        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        self.orb = cv2.ORB_create(nfeatures=1000)

        self.ref_img = self.load_reference_image()
        self.ref_kp = None
        self.ref_des = None
        self.ref_mask = None

        if self.ref_img is not None:
            ref_gray = cv2.cvtColor(self.ref_img, cv2.COLOR_BGR2GRAY)
            _, self.ref_mask = self.end_eff_mask(self.ref_img)

            self.ref_kp, self.ref_des = self.orb.detectAndCompute(ref_gray, self.ref_mask)

            self.get_logger().info(
                f"Reference image loaded: {len(self.ref_kp) if self.ref_kp else 0} keypoints"
            )

    def center_info_cb(self, msg):
        self.center_info = msg
        self.K = np.array(msg.k, dtype=np.float64).reshape(3, 3)
        self.dist = np.array(msg.d, dtype=np.float64).reshape(-1, 1)
        self.get_logger().info(f"Camera matrix K:\n{self.K}")
    def compute_pose_and_print(self, ref_pts, frame_pts):
        # Homography from reference image to live frame
        H, inliers = cv2.findHomography(ref_pts, frame_pts, cv2.RANSAC, 5.0)

        if H is None:
            self.get_logger().warn("Homography failed.")
            return

        # Use 4 reference corners of the planar board
        h_ref, w_ref = self.ref_img.shape[:2]
        ref_corners = np.array([
            [0, 0],
            [w_ref - 1, 0],
            [w_ref - 1, h_ref - 1],
            [0, h_ref - 1]
        ], dtype=np.float32).reshape(-1, 1, 2)

        img_corners = cv2.perspectiveTransform(ref_corners, H)

        # Real object points on the board.
        # Replace these with your real board dimensions in meters if you know them.
        board_w = 1.0
        board_h = 1.0
        obj_pts = np.array([
            [0.0, 0.0, 0.0],
            [board_w, 0.0, 0.0],
            [board_w, board_h, 0.0],
            [0.0, board_h, 0.0]
        ], dtype=np.float64)

        img_pts = img_corners.reshape(-1, 2).astype(np.float64)

        success, rvec, tvec = cv2.solvePnP(
            obj_pts,
            img_pts,
            self.K,
            self.dist,
            flags=cv2.SOLVEPNP_ITERATIVE
        )

        if not success:
            self.get_logger().warn("solvePnP failed.")
            return

        R, _ = cv2.Rodrigues(rvec)
        angle = np.linalg.norm(rvec) * 180.0 / np.pi

        print("\n=== POSE ESTIMATE ===")
        print("rvec:\n", rvec)
        print("tvec:\n", tvec)
        print("Rotation matrix R:\n", R)
        print(f"Rotation angle (degrees): {angle:.2f}")
        print("=====================\n")

        return rvec, tvec, R
    def load_reference_image(self):
        ref_img_path = "ref_board_90.png"
        ref_img = cv2.imread(ref_img_path)
        if ref_img is None:
            self.get_logger().error(f"Failed to load reference image from {ref_img_path}")
            return None
        return ref_img

    def end_eff_mask(self, img):
        pts = np.array([
            [297, 1023],
            [370, 822],
            [493, 810],
            [498, 776],
            [532, 774],
            [537, 628],
            [608, 626],
            [614, 778],
            [653, 778],
            [658, 813],
            [770, 822],
            [861, 1023]
        ], dtype=np.int32)

        mask = np.zeros(img.shape[:2], dtype=np.uint8)
        cv2.fillPoly(mask, [pts], 255)

        inv_mask = cv2.bitwise_not(mask)  # outside polygon = white
        return img, inv_mask

    def image_cb(self, center_img_msg):
        if self.done:
            return

        if self.center_info is None or self.ref_img is None or self.ref_des is None:
            return

        self.done = True  # process only once

        frame = self.bridge.imgmsg_to_cv2(center_img_msg, desired_encoding="bgr8")
        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Mask to keep only outside the polygon
        _, frame_mask = self.end_eff_mask(frame)

        # Detect ORB only outside the polygon
        frame_kp, frame_des = self.orb.detectAndCompute(frame_gray, frame_mask)

        if frame_des is None or frame_kp is None or len(frame_kp) == 0:
            self.get_logger().warn("No features found outside mask.")
            self.destroy_subscription(self.image_sub)
            return

        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        matches = bf.match(self.ref_des, frame_des)
        matches = sorted(matches, key=lambda x: x.distance)
        good_matches = matches[:50]

        matched_vis = cv2.drawMatches(
            self.ref_img,
            self.ref_kp,
            frame,
            frame_kp,
            good_matches,
            None,
            flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS,
        )
        ref_pts = np.float32([self.ref_kp[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
        frame_pts = np.float32([frame_kp[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

        self.compute_pose_and_print(ref_pts, frame_pts)

        cv2.imshow("ORB Feature Matches", matched_vis)
        cv2.waitKey(0)  # wait until key press
        cv2.destroyAllWindows()

        self.get_logger().info(f"Matches found: {len(good_matches)}")

        # Stop receiving more images
        self.destroy_subscription(self.image_sub)
        rclpy.shutdown()


def main():
    rclpy.init()
    node = BoardPoseEstimator()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        if rclpy.ok():
            rclpy.shutdown()
        node.destroy_node()


if __name__ == "__main__":
    main()