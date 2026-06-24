from pathlib import Path

import rclpy
from rclpy.node import Node

from sensor_msgs.msg import Image, CameraInfo
from cv_bridge import CvBridge

import cv2
import numpy as np
import torch
import torch.nn as nn
from scipy.spatial.transform import Rotation as Rot

import tf2_ros

from message_filters import Subscriber, ApproximateTimeSynchronizer


DEFAULT_CHECKPOINT = Path("best.pt")
SLOT_THRESHOLD = 0.5

SIDE_CONFIGS = {
    "left": {
        "theta_deg": 58,
        "band_width": 120,
        "band_length": 700,
        "band_center": (680, 600),
    },
    "right": {
        "theta_deg": 110,
        "band_width": 120,
        "band_length": 700,
        "band_center": (450, 600),
    },
}

LINE_COLOR = (0, 0, 255)
LINE_THICKNESS = 2


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class UNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=1, base_channels=32):
        super().__init__()
        self.enc1 = ConvBlock(in_channels, base_channels)
        self.pool1 = nn.MaxPool2d(2)
        self.enc2 = ConvBlock(base_channels, base_channels * 2)
        self.pool2 = nn.MaxPool2d(2)
        self.enc3 = ConvBlock(base_channels * 2, base_channels * 4)
        self.pool3 = nn.MaxPool2d(2)
        self.enc4 = ConvBlock(base_channels * 4, base_channels * 8)
        self.pool4 = nn.MaxPool2d(2)

        self.bottleneck = ConvBlock(base_channels * 8, base_channels * 16)

        self.up4 = nn.ConvTranspose2d(
            base_channels * 16, base_channels * 8, kernel_size=2, stride=2
        )
        self.dec4 = ConvBlock(base_channels * 16, base_channels * 8)
        self.up3 = nn.ConvTranspose2d(
            base_channels * 8, base_channels * 4, kernel_size=2, stride=2
        )
        self.dec3 = ConvBlock(base_channels * 8, base_channels * 4)
        self.up2 = nn.ConvTranspose2d(
            base_channels * 4, base_channels * 2, kernel_size=2, stride=2
        )
        self.dec2 = ConvBlock(base_channels * 4, base_channels * 2)
        self.up1 = nn.ConvTranspose2d(
            base_channels * 2, base_channels, kernel_size=2, stride=2
        )
        self.dec1 = ConvBlock(base_channels * 2, base_channels)

        self.out = nn.Conv2d(base_channels, out_channels, kernel_size=1)

    def forward(self, x):
        enc1 = self.enc1(x)
        enc2 = self.enc2(self.pool1(enc1))
        enc3 = self.enc3(self.pool2(enc2))
        enc4 = self.enc4(self.pool3(enc3))

        x = self.bottleneck(self.pool4(enc4))

        x = self.up4(x)
        x = self.dec4(torch.cat([x, enc4], dim=1))
        x = self.up3(x)
        x = self.dec3(torch.cat([x, enc3], dim=1))
        x = self.up2(x)
        x = self.dec2(torch.cat([x, enc2], dim=1))
        x = self.up1(x)
        x = self.dec1(torch.cat([x, enc1], dim=1))
        return self.out(x)


class Stereo_nic_port(Node):
    def __init__(self):
        super().__init__("stereo_nic_port")

        self.bridge = CvBridge()

        self.left_info = None
        self.right_info = None
        self.slot_model = None
        self.slot_image_size = None
        self.slot_device = None
        self.port0 = None
        self.port1 = None
        self.card_orientation = None

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

    def load_slot_model(self, checkpoint_path=DEFAULT_CHECKPOINT, device=None):
        checkpoint_path = Path(checkpoint_path)
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"

        checkpoint = torch.load(checkpoint_path, map_location=device)
        base_channels = checkpoint.get("base_channels", 32)
        model = UNet(base_channels=base_channels)
        model.load_state_dict(checkpoint["model"])
        model.to(device)
        model.eval()

        image_size = tuple(checkpoint.get("image_size", (576, 512)))
        return model, image_size, device

    def get_slot_model(self):
        if self.slot_model is None:
            self.slot_model, self.slot_image_size, self.slot_device = self.load_slot_model()
        return self.slot_model, self.slot_image_size, self.slot_device

    @torch.no_grad()
    def mask_port(self, img, side=None, threshold=SLOT_THRESHOLD):
        model, image_size, device = self.get_slot_model()

        original_height, original_width = img.shape[:2]
        image_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        resized = cv2.resize(image_rgb, image_size, interpolation=cv2.INTER_AREA)

        tensor = torch.from_numpy(resized.astype(np.float32) / 255.0)
        tensor = tensor.permute(2, 0, 1).unsqueeze(0).to(device)

        logits = model(tensor)
        probability = torch.sigmoid(logits)[0, 0].cpu().numpy()
        probability = cv2.resize(
            probability,
            (original_width, original_height),
            interpolation=cv2.INTER_LINEAR,
        )

        mask = (probability >= threshold).astype(np.uint8) * 255
        masked = cv2.bitwise_and(img, img, mask=mask)

        if side is not None:
            overlay = img.copy()
            overlay[mask > 0] = (0, 255, 0)
            overlay = cv2.addWeighted(img, 0.55, overlay, 0.45, 0)
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

        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
            mask, connectivity=8
        )
        components = []
        for label in range(1, num_labels):
            area = stats[label, cv2.CC_STAT_AREA]
            if area <= 0:
                continue
            components.append((area, centroids[label]))

        if len(components) < 2:
            self.get_logger().warn(f"Expected 2 slots in {name} mask, found {len(components)}.")
            return None

        components = sorted(components, key=lambda item: item[0], reverse=True)[:2]
        points = sorted(
            [np.array(centroid, dtype=np.float64) for _, centroid in components],
            key=lambda point: point[1],
        )
        slots = {
            "upper": points[0],
            "lower": points[1],
        }

        if debug_img is not None:
            debug = debug_img.copy()
            for label, point in slots.items():
                center = tuple(np.round(point).astype(int))
                cv2.circle(debug, center, 6, (0, 0, 255), -1)
                cv2.putText(
                    debug,
                    label,
                    (center[0] + 8, center[1] - 8),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 0, 255),
                    1,
                    cv2.LINE_AA,
                )
            cv2.imwrite(f"/tmp/{name}_slot_centroids.png", debug)

        return slots

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

    def reset_ports(self):
        self.port0 = None
        self.port1 = None
        self.card_orientation = None

    def compute_card_orientation(self, port0, port1):
        a = np.array(port0, dtype=float) - np.array(port1, dtype=float)
        c = np.array([0.0, 0.0, -1.0], dtype=float)

        a = a - np.dot(a, c) * c
        a_norm = np.linalg.norm(a)
        if a_norm < 1e-9:
            return None
        a = a / a_norm

        b = np.cross(c, a)
        b_norm = np.linalg.norm(b)
        if b_norm < 1e-9:
            return None
        b = b / b_norm

        R = np.column_stack((a, b, c))
        quat = Rot.from_matrix(R).as_quat()
        return np.array([quat[0], quat[1], quat[2], quat[3]], dtype=float)

    def image_cb(self, left_msg, right_msg):
        if self.left_info is None or self.right_info is None:
            return

        left = self.bridge.imgmsg_to_cv2(left_msg, "bgr8")
        right = self.bridge.imgmsg_to_cv2(right_msg, "bgr8")
        left_crop = self.crop(left, "left")
        right_crop = self.crop(right, "right")
        left_masked, right_masked, left_mask, right_mask = self.mask_ports(left_crop, right_crop)

        left_slots = self.detect(left_mask, "left", left_masked)
        right_slots = self.detect(right_mask, "right", right_masked)

        if left_slots is None or right_slots is None:
            self.reset_ports()
            return

        try:
            tfL = self.tf_buffer.lookup_transform("base_link", "left_camera/optical", rclpy.time.Time())
            tfR = self.tf_buffer.lookup_transform("base_link", "right_camera/optical", rclpy.time.Time())
        except:
            self.reset_ports()
            return

        RL, oL = self.tf_to_pose(tfL)
        RR, oR = self.tf_to_pose(tfR)

        point_pairs = [
            ("port0", left_slots["lower"], right_slots["upper"]),
            ("port1", left_slots["upper"], right_slots["lower"]),
        ]
        ports = {"port0": None, "port1": None}
        for point_name, left_point, right_point in point_pairs:
            dL = self.ray(left_point[0], left_point[1], self.left_info.k)
            dR = self.ray(right_point[0], right_point[1], self.right_info.k)

            dL = RL @ dL
            dR = RR @ dR

            point = self.triangulate(oL, dL, oR, dR)
            if point is None:
                continue

            ports[point_name] = point
            self.get_logger().info(
                f"{point_name} in base_link: x={point[0]:.3f}, y={point[1]:.3f}, z={point[2]:.3f}"
            )

        self.port0 = ports["port0"]
        self.port1 = ports["port1"]
        if self.port0 is None or self.port1 is None:
            self.card_orientation = None
            return

        self.card_orientation = self.compute_card_orientation(self.port0, self.port1)
        if self.card_orientation is not None:
            self.get_logger().info(
                "card orientation quat xyzw: "
                f"x={self.card_orientation[0]:.3f}, y={self.card_orientation[1]:.3f}, "
                f"z={self.card_orientation[2]:.3f}, w={self.card_orientation[3]:.3f}"
            )

    def get_ports(self):
        return self.port0, self.port1

    def get_card_pose(self):
        return self.port0, self.port1, self.card_orientation


# def main():
#     rclpy.init()
#     node = Stereo_nic_port()
#     rclpy.spin(node)
#     rclpy.shutdown()
