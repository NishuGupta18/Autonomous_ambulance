# Copyright 2024-2026 NXP
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import CompressedImage
from std_msgs.msg import String
import cv2
import numpy as np
import time

# HINT: If you want to use pyzbar for QR code detection, you can install it using:
# pip install pyzbar
# And uncomment/import it here:
# try:
#     from pyzbar import pyzbar
# except ImportError:
#     pyzbar = None

class QRDetector(Node):
    """
    ROS 2 Node that processes raw camera images to scan for QR codes.
    It publishes the detected QR code payload on the `/qr_detection` topic.
    """
    def __init__(self):
        super().__init__('qr_detector')

        # Subscription for camera images.
        self.subscription_camera = self.create_subscription(
            CompressedImage,
            '/camera/image_raw/compressed',
            self.camera_image_callback,
            10)

        # Publisher for QR code detection results.
        self.publisher_qr = self.create_publisher(
            String,
            '/qr_detection',
            10)

        self.get_logger().info("QR Detector Node started. Waiting for images...")

        # --- Persistent QR detector (reuse across frames) ---
        self.qr_detector = cv2.QRCodeDetector()

        # --- Frame skip for performance ---
        self.frame_count = 0
        self.PROCESS_EVERY_N = 2       # Process 1 in 2 frames

        # --- Debounce: don't re-publish same QR within N seconds ---
        self.last_published_qr = None
        self.last_published_time = 0.0
        self.DEBOUNCE_SECONDS = 2.0    # Must be shorter than line follower's 3s debounce

    def camera_image_callback(self, message):
        """Processes incoming camera frames to detect QR codes."""
        # Frame skip for performance
        self.frame_count += 1
        if self.frame_count % self.PROCESS_EVERY_N != 0:
            return

        # Convert compressed image message to OpenCV format
        np_arr = np.frombuffer(message.data, np.uint8)
        image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        if image is None:
            return

        qr_data = self.detect_qr_code(image)

        if qr_data is not None:
            current_time = time.time()
            # Debounce — don't re-publish same QR within DEBOUNCE_SECONDS
            if (qr_data == self.last_published_qr and
                    (current_time - self.last_published_time) < self.DEBOUNCE_SECONDS):
                return

            self.last_published_qr = qr_data
            self.last_published_time = current_time

            # Publish the decoded QR payload
            msg = String()
            msg.data = qr_data
            self.publisher_qr.publish(msg)
            self.get_logger().info(f"Published QR Data: {qr_data}")

    def detect_qr_code(self, image):
        """
        Detect and decode QR code in the image.
        Uses multiple strategies for robustness:
        1. Direct detection on original image
        2. Detection on pre-processed (grayscale + contrast enhanced) image
        3. Detection on sharpened image (for blurry/distant QR)
        """
        # --- Strategy 1: Direct detection (fastest, works when QR is clean & frontal) ---
        result = self._try_opencv_detect(image)
        if result is not None:
            return result

        # --- Strategy 2: Grayscale + CLAHE contrast enhancement ---
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(gray)
        # Convert back to 3-channel for QRCodeDetector (some versions require it)
        enhanced_bgr = cv2.cvtColor(enhanced, cv2.COLOR_GRAY2BGR)

        result = self._try_opencv_detect(enhanced_bgr)
        if result is not None:
            return result

        # --- Strategy 3: Sharpen for distant/blurry QR ---
        kernel = np.array([[-1, -1, -1],
                           [-1,  9, -1],
                           [-1, -1, -1]])
        sharpened = cv2.filter2D(image, -1, kernel)

        result = self._try_opencv_detect(sharpened)
        if result is not None:
            return result

        return None

    def _try_opencv_detect(self, image):
        """Attempt QR detection using the persistent OpenCV detector."""
        try:
            data, bbox, _ = self.qr_detector.detectAndDecode(image)
            if bbox is not None and data != "":
                return data
        except Exception as e:
            self.get_logger().debug(f"OpenCV QR Detection failed: {e}")
        return None

def main(args=None):
    rclpy.init(args=args)
    node = QRDetector()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()