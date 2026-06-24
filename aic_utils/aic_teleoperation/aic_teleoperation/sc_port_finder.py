import rclpy
from rclpy.node import Node

from sensor_msgs.msg import Image, CameraInfo
from cv_bridge import CvBridge

import cv2
import numpy as np

import tf2_ros

from message_filters import Subscriber, ApproximateTimeSynchronizer


SKYBLUE_LOWER = np.array([85, 50, 50])
SKYBLUE_UPPER = np.array([110, 255, 255])

SIDE_CONFIGS = {
    "left": {
        "theta_deg": 65,
        "band_width": 120,
        "band_length": 700,
        "band_center": (850, 420),
    },
    "right": {
        "theta_deg": 120,
        "band_width": 120,
        "band_length": 900,
        "band_center": (300, 400),
    },
}

LINE_COLOR = (0, 0, 255)
LINE_THICKNESS = 2


class Stereo_sc_port(Node):
    def __init__(self):
        super().__init__("stereo_sc_port")

        self.bridge = CvBridge()

        self.left_info = None
        self.right_info = None
        self.port = None

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

    def crop(self, img, side):
        config = SIDE_CONFIGS[side]

        height, width = img.shape[:2]
        band_center = config["band_center"]
        center_x = width / 2.0 if band_center[0] is None else band_center[0]
        center_y = height / 2.0 if band_center[1] is None else band_center[1]
        center = np.array([center_x, center_y], dtype=np.float32)

        theta = np.deg2rad(config["theta_deg"])
        direction = np.array([np.cos(theta), np.sin(theta)], dtype=np.float32)
        normal = np.array([-np.sin(theta), np.cos(theta)], dtype=np.float32)

        half_length = config["band_length"] / 2.0
        half_band = config["band_width"] / 2.0

        corner1 = center - direction * half_length + normal * half_band
        corner2 = center + direction * half_length + normal * half_band
        corner3 = center + direction * half_length - normal * half_band
        corner4 = center - direction * half_length - normal * half_band

        crop_polygon = np.array([corner1, corner2, corner3, corner4], dtype=np.int32)

        mask = np.zeros((height, width), dtype=np.uint8)
        cv2.fillPoly(mask, [crop_polygon], 255)

        result = cv2.bitwise_and(img, img, mask=mask)
        corners = [corner1, corner2, corner3, corner4]
        for start, end in zip(corners, corners[1:] + corners[:1]):
            cv2.line(
                result,
                tuple(start.astype(int)),
                tuple(end.astype(int)),
                LINE_COLOR,
                LINE_THICKNESS,
            )

        cv2.imwrite(f"/tmp/{side}_crop.png", result)
        return result

    def mask_port(self, img, side=None):
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, SKYBLUE_LOWER, SKYBLUE_UPPER)
        masked = cv2.bitwise_and(img, img, mask=mask)

        if side is not None:
            overlay = img.copy()
            overlay[mask > 0] = (0, 0, 255)
            cv2.imwrite(f"/tmp/{side}_slot_mask.png", mask)
            cv2.imwrite(f"/tmp/{side}_slot_overlay.png", overlay)
            cv2.imwrite(f"/tmp/{side}_slot_masked.png", masked)

        return masked, mask

    def mask_ports(self, left_img, right_img):
        left_masked, left_mask = self.mask_port(left_img, "left")
        right_masked, right_mask = self.mask_port(right_img, "right")
        return left_masked, right_masked, left_mask, right_mask

    def detect(self, mask, name, debug_img=None):
        if mask.ndim == 3:
            mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)

        M = cv2.moments(mask)
        if M["m00"] == 0:
            self.get_logger().warn(f"No slot found in {name} mask.")
            return None

        slot = np.array([M["m10"] / M["m00"], M["m01"] / M["m00"]], dtype=np.float64)

        if debug_img is not None:
            debug = debug_img.copy()
            center = tuple(np.round(slot).astype(int))
            cv2.circle(debug, center, 6, (0, 0, 255), -1)
            cv2.putText(
                debug,
                "slot",
                (center[0] + 8, center[1] - 8),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 0, 255),
                1,
                cv2.LINE_AA,
            )
            cv2.imwrite(f"/tmp/{name}_slot_centroids.png", debug)

        return slot

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

    def reset_port(self):
        self.port = None

    def image_cb(self, left_msg, right_msg):
        if self.left_info is None or self.right_info is None:
            return

        left = self.bridge.imgmsg_to_cv2(left_msg, "bgr8")
        right = self.bridge.imgmsg_to_cv2(right_msg, "bgr8")
        left_crop = self.crop(left, "left")
        right_crop = self.crop(right, "right")
        left_masked, right_masked, left_mask, right_mask = self.mask_ports(left_crop, right_crop)

        left_slot = self.detect(left_mask, "left", left_masked)
        right_slot = self.detect(right_mask, "right", right_masked)

        if left_slot is None or right_slot is None:
            self.reset_port()
            return

        try:
            tfL = self.tf_buffer.lookup_transform("base_link", "left_camera/optical", rclpy.time.Time())
            tfR = self.tf_buffer.lookup_transform("base_link", "right_camera/optical", rclpy.time.Time())
        except:
            self.reset_port()
            return

        RL, oL = self.tf_to_pose(tfL)
        RR, oR = self.tf_to_pose(tfR)

        dL = self.ray(left_slot[0], left_slot[1], self.left_info.k)
        dR = self.ray(right_slot[0], right_slot[1], self.right_info.k)

        dL = RL @ dL
        dR = RR @ dR

        self.port = self.triangulate(oL, dL, oR, dR)
        if self.port is None:
            return

        self.get_logger().info(
            f"port in base_link: x={self.port[0]:.3f}, y={self.port[1]:.3f}, z={self.port[2]:.3f}"
        )

    def get_port(self):
        return self.port


def main():
    rclpy.init()
    node = Stereo_sc_port()
    rclpy.spin(node)
    rclpy.shutdown()
