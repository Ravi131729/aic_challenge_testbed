import rclpy
from rclpy.node import Node

from sensor_msgs.msg import Image, CameraInfo
from cv_bridge import CvBridge

import cv2
import numpy as np

import tf2_ros

from message_filters import Subscriber, ApproximateTimeSynchronizer


class StereoSkyblue(Node):
    def __init__(self):
        super().__init__("stereo_skyblue")

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

    def left_info_cb(self, msg):
        self.left_info = msg

    def right_info_cb(self, msg):
        self.right_info = msg

    def detect(self, img, name):
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

        lower = np.array([85, 50, 50])
        upper = np.array([110, 255, 255])

        mask = cv2.inRange(hsv, lower, upper)

        # Save mask image
        cv2.imwrite(f"/tmp/{name}_mask.png", mask)

        # Also overlay mask on original for easier debugging
        overlay = img.copy()
        overlay[mask > 0] = [0, 0, 255]  # mark detected pixels in red
        cv2.imwrite(f"/tmp/{name}_overlay.png", overlay)

        M = cv2.moments(mask)
        if M["m00"] == 0:
            return None

        u = M["m10"] / M["m00"]
        v = M["m01"] / M["m00"]

        return np.array([u, v])

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

    def image_cb(self, left_msg, right_msg):
        if self.left_info is None or self.right_info is None:
            return

        left = self.bridge.imgmsg_to_cv2(left_msg, "bgr8")
        right = self.bridge.imgmsg_to_cv2(right_msg, "bgr8")

        # pL = self.detect(left)
        # pR = self.detect(right)
        pL = self.detect(left, "left")
        pR = self.detect(right, "right")

        if pL is None or pR is None:
            return

        uL, vL = pL
        uR, vR = pR

        dL = self.ray(uL, vL, self.left_info.k)
        dR = self.ray(uR, vR, self.right_info.k)

        try:
            tfL = self.tf_buffer.lookup_transform("base_link", "left_camera/optical", rclpy.time.Time())
            tfR = self.tf_buffer.lookup_transform("base_link", "right_camera/optical", rclpy.time.Time())
        except:
            return

        RL, oL = self.tf_to_pose(tfL)
        RR, oR = self.tf_to_pose(tfR)

        dL = RL @ dL
        dR = RR @ dR

        point = self.triangulate(oL, dL, oR, dR)

        if point is None:
            return

        self.get_logger().info(
            f"3D position in base_link: x={point[0]:.3f}, y={point[1]:.3f}, z={point[2]:.3f}"
        )


def main():
    rclpy.init()
    node = StereoSkyblue()
    rclpy.spin(node)
    rclpy.shutdown()