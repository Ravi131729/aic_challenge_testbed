import rclpy
from rclpy.node import Node

from sensor_msgs.msg import Image, CameraInfo
from cv_bridge import CvBridge

import cv2
import numpy as np

import tf2_ros

from message_filters import Subscriber, ApproximateTimeSynchronizer


class Stereo_mag_square(Node):
    def __init__(self):
        super().__init__("stereo_mag_square")

        self.bridge = CvBridge()

        self.left_info = None
        self.right_info = None

        self.create_subscription(CameraInfo, "/left_camera/camera_info", self.left_info_cb, 10)
        self.create_subscription(CameraInfo, "/right_camera/camera_info", self.right_info_cb, 10)

        self.left_sub = Subscriber(self, Image, "/left_camera/image")
        self.right_sub = Subscriber(self, Image, "/right_camera/image")

        self.sync = ApproximateTimeSynchronizer(
            [self.left_sub, self.right_sub],
            queue_size=10,
            slop=0.1
        )
        self.sync.registerCallback(self.image_cb)

        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)
        self.selected_corner_idx = None
        self.ref_vector = np.array([-1, -1, 0]) / np.linalg.norm([-1, -1, 0])
        self.latest_result = None
    def left_info_cb(self, msg):
        self.left_info = msg

    def right_info_cb(self, msg):
        self.right_info = msg



    def detect_magenta_corners(self, img, name):
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

        lower = np.array([140, 80, 80])
        upper = np.array([170, 255, 255])

        mask = cv2.inRange(hsv, lower, upper)

        kernel = np.ones((7, 7), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        self.get_logger().info(f"{name}: found {len(contours)} contours")

        if not contours:
            cv2.imwrite(f"/tmp/{name}_mask.png", mask)
            return None

        largest = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(largest)
        self.get_logger().info(f"{name}: largest area = {area}")

        if area < 300:
            cv2.imwrite(f"/tmp/{name}_mask.png", mask)
            return None

        rect = cv2.minAreaRect(largest)
        box = cv2.boxPoints(rect)
        box = box.astype(np.float32)
        overlay = img.copy()

        # Draw actual contour (WHITE)
        cv2.drawContours(overlay, [largest], -1, (255, 255, 255), 2)

        # Draw minAreaRect box (GREEN)
        # cv2.drawContours(overlay, [box.astype(np.int32)], -1, (0, 255, 0), 2)

        # Draw actual contour corners (BLUE - from approxPolyDP)
        epsilon = 0.005 * cv2.arcLength(largest, True)
        approx = cv2.approxPolyDP(largest, epsilon, True)

        for p in approx:
            x, y = p[0]
            cv2.circle(overlay, (int(x), int(y)), 6, (255, 0, 0), -1)

        self.get_logger().info(f"{name}: approx corners = {len(approx)}")

        # # Draw minAreaRect corners (RED)

        outer_corners = box
        i = 0
        for p in box:
          dist = np.linalg.norm(approx - p, axis=2)
          outer_corners[i] = approx[np.argmin(dist, axis=0)]
          i = i + 1

        for p in outer_corners:
            cv2.circle(overlay, (int(p[0]), int(p[1])), 6, (0, 0, 255), -1)

        approx_pts = approx.reshape(-1, 2).astype(np.float32)
        radius = 80.0

        counts = []
        for i, corner in enumerate(outer_corners):
            dists = np.linalg.norm(approx_pts - corner, axis=1)
            count = np.sum(dists <= radius)
            counts.append(count)

        best_idx = int(np.argmax(counts))


        # draw selected corner (yellow)
        selected = outer_corners[best_idx]
        selected = outer_corners[best_idx]

        max_dist = -1
        opp_idx = -1

        for i, corner in enumerate(outer_corners):
            if i == best_idx:
                continue

            dist = np.linalg.norm(corner - selected)
            if dist > max_dist:
                max_dist = dist
                opp_idx = i
        cv2.circle(overlay, (int(selected[0]), int(selected[1])), 10, (0, 255, 255), -1)
        cv2.circle(overlay, (int(outer_corners[opp_idx][0]), int(outer_corners[opp_idx][1])), 10, (255, 255, 0), -1)

        cv2.imwrite(f"/tmp/{name}_overlay.png", overlay)
        two_corners = np.array([outer_corners[best_idx], outer_corners[opp_idx]])






        return two_corners


    def order_points(self, pts):
        rect = np.zeros((4, 2), dtype="float32")

        s = pts.sum(axis=1)
        rect[0] = pts[np.argmin(s)]      # top-left
        rect[2] = pts[np.argmax(s)]      # bottom-right

        diff = np.diff(pts, axis=1)
        rect[1] = pts[np.argmin(diff)]   # top-right
        rect[3] = pts[np.argmax(diff)]   # bottom-left

        return rect

    def ray(self, u, v, K):
        fx, fy, cx, cy = K[0], K[4], K[2], K[5]

        d = np.array([
            (u - cx) / fx,
            (v - cy) / fy,
            1.0
        ])

        return d / np.linalg.norm(d)

    def tf_to_pose(self, tf):
        t = tf.transform.translation
        q = tf.transform.rotation

        # rotation matrix
        x, y, z, w = q.x, q.y, q.z, q.w

        R = np.array([
            [1-2*y*y-2*z*z, 2*x*y-2*z*w, 2*x*z+2*y*w],
            [2*x*y+2*z*w, 1-2*x*x-2*z*z, 2*y*z-2*x*w],
            [2*x*z-2*y*w, 2*y*z+2*x*w, 1-2*x*x-2*y*y]
        ])

        t = np.array([t.x, t.y, t.z])

        return R, t

    def triangulate(self, o1, d1, o2, d2):
        d1 = d1 / np.linalg.norm(d1)
        d2 = d2 / np.linalg.norm(d2)

        r = o1 - o2

        a = np.dot(d1, d1)
        b = np.dot(d1, d2)
        c = np.dot(d2, d2)
        d = np.dot(d1, r)
        e = np.dot(d2, r)

        denom = a*c - b*b
        if abs(denom) < 1e-6:
            return None

        s = (b*e - c*d) / denom
        t = (a*e - b*d) / denom

        p1 = o1 + s*d1
        p2 = o2 + t*d2

        return 0.5 * (p1 + p2)
    # def image_cb(self, left_msg, right_msg):
    #     self.get_logger().info("image_cb called")
    #     if self.left_info is None or self.right_info is None:
    #         return

    #     left = self.bridge.imgmsg_to_cv2(left_msg, "bgr8")
    #     right = self.bridge.imgmsg_to_cv2(right_msg, "bgr8")



    #     cL = self.detect_magenta_corners(left, "left")
    #     cR = self.detect_magenta_corners(right, "right")

    #     if cL is None or cR is None:
    #         return

    #     # cL = self.order_points(cL)
    #     # cR = self.order_points(cR)
    #     try:
    #         tfL = self.tf_buffer.lookup_transform("base_link", "left_camera/optical", rclpy.time.Time())
    #         tfR = self.tf_buffer.lookup_transform("base_link", "right_camera/optical", rclpy.time.Time())
    #     except Exception as e:
    #         self.get_logger().warn(f"TF lookup failed: {e}")
    #         return

    #     RL, oL = self.tf_to_pose(tfL)
    #     RR, oR = self.tf_to_pose(tfR)

    #     points_3d = []

    #     for i in range(2):
    #         uL, vL = cL[i]
    #         uR, vR = cR[i]

    #         dL = self.ray(uL, vL, self.left_info.k)
    #         dR = self.ray(uR, vR, self.right_info.k)

    #         dL = RL @ dL
    #         dR = RR @ dR

    #         p = self.triangulate(oL, dL, oR, dR)
    #         if p is None:
    #             return

    #         points_3d.append(p)

    #     points_3d = np.array(points_3d)
    #     center = np.mean(points_3d, axis=0)

    #     diag_dist = np.linalg.norm(points_3d[0] - points_3d[1])

    #     diag_vec = points_3d[1] - points_3d[0]
    #     diag_vec = diag_vec / np.linalg.norm(diag_vec)

    #     normal = np.array([0, 0, 1])  # important!

    #     dot = np.dot(diag_vec, self.ref_vector)
    #     cross = np.cross(self.ref_vector, diag_vec)
    #     print(f"dot: {dot}, cross: {cross}")

    #     angle = np.arctan2(np.dot(cross, normal), dot)



    #     self.get_logger().info(
    #         f"Square center in base_link: x={center[0]:.3f}, y={center[1]:.3f}, z={center[2]:.3f}"
    #     )

    #     self.get_logger().info(f"Corner 1: x={points_3d[0][0]:.3f}, y={points_3d[0][1]:.3f}, z={points_3d[0][2]:.3f}")
    #     self.get_logger().info(f"Corner 2: x={points_3d[1][0]:.3f}, y={points_3d[1][1]:.3f}, z={points_3d[1][2]:.3f}")
    #     self.get_logger().info(f"diag_dist: {diag_dist:.3f}m")

    #     self.get_logger().info(f"angle to ref vector: {angle:.2f} radians, which is {angle*180.0/np.pi:.2f} degrees")

    def rotation_z(self, angle):
        c = np.cos(angle)
        s = np.sin(angle)
        return np.array([
            [ c, -s, 0.0],
            [ s,  c, 0.0],
            [0.0, 0.0, 1.0]
        ])

    def compute_square_pose(self, left_msg, right_msg):
        """
        Returns:
            Rz: 3x3 rotation matrix
            center: 3D center in base_link
            rotated_center: Rz @ center
            angle: rotation around z
        """
        if self.left_info is None or self.right_info is None:
            return None

        left = self.bridge.imgmsg_to_cv2(left_msg, "bgr8")
        right = self.bridge.imgmsg_to_cv2(right_msg, "bgr8")

        cL = self.detect_magenta_corners(left, "left")
        cR = self.detect_magenta_corners(right, "right")

        if cL is None or cR is None:
            return None

        try:
            tfL = self.tf_buffer.lookup_transform(
                "base_link", "left_camera/optical", rclpy.time.Time()
            )
            tfR = self.tf_buffer.lookup_transform(
                "base_link", "right_camera/optical", rclpy.time.Time()
            )
        except Exception as e:
            self.get_logger().warn(f"TF lookup failed: {e}")
            return None

        RL, oL = self.tf_to_pose(tfL)
        RR, oR = self.tf_to_pose(tfR)

        points_3d = []
        for i in range(2):
            uL, vL = cL[i]
            uR, vR = cR[i]

            dL = self.ray(uL, vL, self.left_info.k)
            dR = self.ray(uR, vR, self.right_info.k)

            dL = RL @ dL
            dR = RR @ dR

            p = self.triangulate(oL, dL, oR, dR)
            if p is None:
                return None

            points_3d.append(p)

        points_3d = np.array(points_3d)
        center = np.mean(points_3d, axis=0)

        diag_vec = points_3d[1] - points_3d[0]
        diag_vec = diag_vec / np.linalg.norm(diag_vec)

        ref = self.ref_vector
        dot = np.dot(diag_vec, ref)
        cross = np.cross(ref, diag_vec)

        normal = np.array([0, 0, 1.0])
        angle = np.arctan2(np.dot(cross, normal), dot)

        Rz = self.rotation_z(angle)
        rotated_center = Rz @ center

        return {
            "Rz": Rz,
            "center": center,
            "angle": angle,
            "points_3d": points_3d,
        }

    def image_cb(self, left_msg, right_msg):
        self.get_logger().info("image_cb called")

        result = self.compute_square_pose(left_msg, right_msg)
        if result is None:
            return

        self.latest_result = result

        center = result["center"]
        angle = result["angle"]
        Rz = result["Rz"]

        self.get_logger().info(
            f"center = {center}, angle = {angle:.3f} rad, Rz = \n{Rz}"
        )

# def main():
#     rclpy.init()
#     node = Stereo_mag_square()
#     rclpy.spin(node)
#     rclpy.shutdown()