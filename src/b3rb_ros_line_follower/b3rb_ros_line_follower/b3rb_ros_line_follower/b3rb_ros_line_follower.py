# Copyright 2024-2026 NXP
# Copyright 2016 Open Source Robotics Foundation, Inc.

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, QoSReliabilityPolicy
from rclpy.duration import Duration
import math
import numpy as np
from sensor_msgs.msg import Joy, LaserScan
from std_msgs.msg import String
from synapse_msgs.msg import EdgeVectors, ServerCommunication

QOS_PROFILE_DEFAULT = 10

# Navigation Motion States
STATE_LANE_FOLLOW = 'LANE_FOLLOW'
STATE_SIGN_MANEUVER = 'SIGN_MANEUVER'
STATE_DEAD_RECKON = 'DEAD_RECKON'
STATE_TURN_AWAY = 'TURN_AWAY'
STATE_PASS_OBSTACLE = 'PASS_OBSTACLE'
STATE_RETURN_TO_LANE = 'RETURN_TO_LANE'
STATE_EMERGENCY_STOP = 'EMERGENCY_STOP'

# Mission Protocol States
MISSION_NAVIGATING = 'NAVIGATING'
MISSION_WAITING_SERVER_DEST = 'WAITING_SERVER_DEST'
MISSION_PARKING = 'PARKING'
MISSION_COMPLETE = 'MISSION_COMPLETE'

TARGET_QR_ALIASES = {
    "A": {"A", "PATIENT_1"},
    "B": {"B", "PATIENT_2"},
    "C": {"C", "PATIENT_3"},
    "X": {"X", "HOSPITAL_1"},
    "Y": {"Y", "HOSPITAL_2"},
    "Z": {"Z", "HOSPITAL_3"},
    "PATIENT_1": {"A", "PATIENT_1"},
    "PATIENT_2": {"B", "PATIENT_2"},
    "PATIENT_3": {"C", "PATIENT_3"},
    "HOSPITAL_1": {"X", "HOSPITAL_1"},
    "HOSPITAL_2": {"Y", "HOSPITAL_2"},
    "HOSPITAL_3": {"Z", "HOSPITAL_3"},
}


class LineTracker:
    """Validates incoming edge vectors for spatial continuity."""
    def __init__(self, history_size=5):
        self.history_size = history_size
        self.position_history = []
        self.heading_history = []

    def validate_and_update(self, near_x, far_x):
        heading = far_x - near_x
        if len(self.position_history) > 0:
            last_pos = self.position_history[-1]
            last_heading = self.heading_history[-1]

            if abs(near_x - last_pos) > 160.0 or abs(heading - last_heading) > 180.0:
                return False

        self.position_history.append(near_x)
        self.heading_history.append(heading)
        if len(self.position_history) > self.history_size:
            self.position_history.pop(0)
            self.heading_history.pop(0)
        return True


class LineFollower(Node):
    def __init__(self):
        super().__init__('line_follower')

        # Subscriptions
        self.subscription_vectors = self.create_subscription(
            EdgeVectors, '/edge_vectors', self.edge_vectors_callback, QOS_PROFILE_DEFAULT)

        qos_lidar = QoSProfile(depth=10)
        qos_lidar.reliability = QoSReliabilityPolicy.BEST_EFFORT
        self.subscription_lidar = self.create_subscription(
            LaserScan, '/scan', self.lidar_callback, qos_lidar)

        self.subscription_server = self.create_subscription(
            ServerCommunication, '/ServerCommunication', self.server_communication_callback, QOS_PROFILE_DEFAULT)

        self.subscription_qr = self.create_subscription(
            String, '/qr_detection', self.qr_detection_callback, QOS_PROFILE_DEFAULT)

        self.subscription_signs = self.create_subscription(
            String, '/sign_board_detection', self.sign_board_callback, QOS_PROFILE_DEFAULT)

        # Publishers
        self.publisher_joy = self.create_publisher(Joy, '/cerebri/in/joy', QOS_PROFILE_DEFAULT)
        self.publisher_server = self.create_publisher(ServerCommunication, '/ServerCommunication', QOS_PROFILE_DEFAULT)

        # Mission Variables
        self.current_target = "A"               
        self.mission_status = MISSION_NAVIGATING
        self.uid_counter = 10
        self.has_arrived_at_target = False
        self.depart_hold_until = None

        # Dedicated QR Arrival Stop State
        self.qr_stop_active = False
        self.qr_stop_start_time = None
        self.qr_server_notified = False

        # Hospital Proximity Memory
        self.hospital_proximity_confirmed = False
        self.hospital_proximity_count = 0
        self.hospital_proximity_required = 3

        # Tracker Instance
        self.tracker = LineTracker(history_size=5)

        # Navigation State & LiDAR Sign Trigger Flags
        self.navigation_state = STATE_LANE_FOLLOW
        self.pending_sign_command = None
        self.pending_sign_timestamp = None
        self.sign_maneuver_start_time = None
        self.dead_reckon_start_time = None
        self.sign_override_steering = 0.0
        self.last_valid_turn_dir = 1.0

        # LiDAR Trigger Flags for Passing Signboard
        self.sign_seen = False
        self.sign_close_confirmed = False

        # Speeds & Steering Commands
        self.lane_speed_command = 0.40
        self.lane_turn_command = 0.0
        self.avoidance_speed_command = 0.0
        self.avoidance_turn_command = 0.0

        # LiDAR Distances
        self.front_distance = float('inf')
        self.left_front_distance = float('inf')
        self.right_front_distance = float('inf')
        self.left_side_distance = float('inf')
        self.right_side_distance = float('inf')

        self.filtered_lane_turn = 0.0
        self.last_lane_time = None

        # Parameters
        self.declare_parameter('lane_kp', 1.20)
        self.declare_parameter('heading_gain', 1.40)
        self.declare_parameter('lane_width_fraction', 0.50)
        self.declare_parameter('lane_speed', 0.40)
        self.declare_parameter('sharp_curve_speed', 0.20)
        self.declare_parameter('maximum_steering', 0.42)
        self.declare_parameter('steering_smoothing', 0.25)
        self.declare_parameter('lane_timeout', 0.40)
        self.declare_parameter('lane_search_speed', 0.18)
        self.declare_parameter('sign_validity_window', 8.0)
        self.declare_parameter('sign_hold_duration', 1.3)          
        self.declare_parameter('dead_reckon_duration', 0.7)        
        self.declare_parameter('sign_turn_magnitude', 0.38)
        self.declare_parameter('target_qr_hold_duration', 2.0)     # 2-second stop delay at target QR
        self.declare_parameter('server_depart_hold', 1.0)
        self.declare_parameter('obstacle_front_stop', 0.85)
        self.declare_parameter('obstacle_front_clear', 1.20)
        self.declare_parameter('obstacle_emergency_stop', 0.35)
        self.declare_parameter('obstacle_side_clear', 0.65)
        self.declare_parameter('obstacle_avoid_speed', 0.18)
        self.declare_parameter('obstacle_merge_speed', 0.25)

        self.control_timer = self.create_timer(0.1, self.publish_drive_commands)
        self.get_logger().info('NXP Buggy Controller Active. Unconditional Target QR Stop Engaged.')

    def get_next_uid(self):
        self.uid_counter += 1
        return self.uid_counter

    def extract_qr_location(self, qr_text):
        cleaned = qr_text.strip().upper()
        for char in "{}":
            cleaned = cleaned.replace(char, "")
        if ":" in cleaned:
            key, value = cleaned.split(":", 1)
            if key.strip() == "LOC":
                return value.strip()
        return cleaned

    def qr_matches_current_target(self, qr_text):
        qr_location = self.extract_qr_location(qr_text)
        target_aliases = TARGET_QR_ALIASES.get(self.current_target, {self.current_target})
        return qr_location in target_aliases

    # -------------------------------------------------------------------------
    # 1. Signboard Callback
    # -------------------------------------------------------------------------
    def sign_board_callback(self, message):
        raw_data = message.data.strip().upper()
        now = self.get_clock().now()

        if ":" in raw_data:
            detected_target, direction = raw_data.split(":", 1)
        else:
            detected_target, direction = raw_data, "LEFT"

        target_aliases = TARGET_QR_ALIASES.get(self.current_target, {self.current_target})
        if detected_target in target_aliases:
            sign_steer_mag = float(self.get_parameter('sign_turn_magnitude').value)

            if direction == "LEFT":
                self.pending_sign_command = 'TURN_LEFT'
                self.sign_override_steering = sign_steer_mag
                self.last_valid_turn_dir = 1.0
            elif direction == "RIGHT":
                self.pending_sign_command = 'TURN_RIGHT'
                self.sign_override_steering = -sign_steer_mag
                self.last_valid_turn_dir = -1.0
            elif direction == "STRAIGHT":
                self.pending_sign_command = 'GO_STRAIGHT'
                self.sign_override_steering = 0.0
                self.last_valid_turn_dir = 0.0

            self.pending_sign_timestamp = now
            self.sign_seen = True
            self.sign_close_confirmed = False
            self.get_logger().info(
                f"STORED SIGN DIRECTIVE [{detected_target}:{direction}] -> Override Steer: {self.sign_override_steering:.2f}"
            )

    # -------------------------------------------------------------------------
    # 2. QR Code Callback: UNCONDITIONAL STOP ON MATCH
    # -------------------------------------------------------------------------
    def qr_detection_callback(self, message):
        scanned_qr = message.data.strip().upper()

        if self.qr_stop_active or self.has_arrived_at_target:
            return

        current_aliases = TARGET_QR_ALIASES.get(self.current_target, {self.current_target})
        is_hospital = any(alias.startswith("HOSPITAL") or alias in {"X", "Y", "Z"} for alias in current_aliases)
        if is_hospital and not self.hospital_proximity_confirmed:
            return

        if scanned_qr and self.qr_matches_current_target(scanned_qr):
            if self.depart_hold_until is not None:
                return

            # UNCONDITIONAL IMMEDIATE STOP ON TARGET QR MATCH
            self.qr_stop_active = True
            self.qr_stop_start_time = self.get_clock().now()
            self.qr_server_notified = False
            self.has_arrived_at_target = True
            self.mission_status = MISSION_WAITING_SERVER_DEST

            self.lane_speed_command = 0.0
            self.lane_turn_command = 0.0
            self.get_logger().info(f"MATCHED TARGET QR [{scanned_qr}]! FULL STOP INSTANTIATED FOR 2-SECOND DELAY...")

    # -------------------------------------------------------------------------
    # 3. Server Communication Callback
    # -------------------------------------------------------------------------
    def server_communication_callback(self, message):
        if message.dest == 1:
            server_msg = message.msg.strip().upper()
            self.get_logger().info(f"Server Msg -> uid: {message.uid}, ack: {message.ack}, msg: '{server_msg}'")

            if message.ack == 1 and self.mission_status == MISSION_WAITING_SERVER_DEST:
                return

            if message.ack == 0:
                ack_reply = ServerCommunication()
                ack_reply.src = 1
                ack_reply.dest = 2
                ack_reply.uid = message.uid
                ack_reply.ack = 1
                ack_reply.msg = ""
                self.publisher_server.publish(ack_reply)

                if server_msg == "OK":
                    if self.current_target == "PARKED":
                        self.mission_status = MISSION_COMPLETE
                        self.get_logger().info("RUN COMPLETE!")
                    else:
                        self.current_target = "PARKED"
                        self.mission_status = MISSION_PARKING
                elif len(server_msg) > 0 and server_msg != "INVALID":
                    self.current_target = server_msg
                    self.has_arrived_at_target = False
                    self.qr_stop_active = False
                    self.qr_server_notified = False
                    self.depart_hold_until = self.get_clock().now() + Duration(seconds=float(self.get_parameter('server_depart_hold').value))
                    self.hospital_proximity_confirmed = False
                    self.mission_status = MISSION_NAVIGATING
                    self.navigation_state = STATE_LANE_FOLLOW
                    self.pending_sign_command = None
                    self.sign_seen = False
                    self.sign_close_confirmed = False
                    self.get_logger().info(f"UPDATED NEXT TARGET FROM SERVER -> [{self.current_target}]")

    def notify_server_arrival(self, target_label):
        msg = ServerCommunication()
        msg.src = 1
        msg.dest = 2
        msg.uid = self.get_next_uid()
        msg.ack = 0
        msg.msg = target_label
        self.publisher_server.publish(msg)

    def update_hospital_proximity(self, ranges):
        def sector_span_count(start_deg, end_deg, threshold):
            total = len(ranges)
            if total == 0:
                return 0
            start = int((start_deg % 360.0) / 360.0 * total)
            end = int((end_deg % 360.0) / 360.0 * total)
            if start <= end:
                sector = ranges[start:end + 1]
            else:
                sector = np.concatenate((ranges[start:], ranges[:end + 1]))
            return int(np.count_nonzero(sector < threshold))

        side_clear = float(self.get_parameter('obstacle_side_clear').value)
        left_wall = sector_span_count(75.0, 110.0, side_clear)
        right_wall = sector_span_count(250.0, 285.0, side_clear)
        
        side_wall_present = left_wall >= 6 or right_wall >= 6

        if side_wall_present:
            self.hospital_proximity_count = min(self.hospital_proximity_required, self.hospital_proximity_count + 1)
        else:
            self.hospital_proximity_count = max(0, self.hospital_proximity_count - 1)

        self.hospital_proximity_confirmed = (self.hospital_proximity_count >= self.hospital_proximity_required)

    # -------------------------------------------------------------------------
    # 4. Tangent Vector Tracking
    # -------------------------------------------------------------------------
    def edge_vectors_callback(self, message):
        if self.navigation_state in (STATE_SIGN_MANEUVER, STATE_DEAD_RECKON) or self.qr_stop_active:
            return

        if message.image_width <= 0 or message.vector_count == 0:
            return
        now = self.get_clock().now()

        image_center = message.image_width / 2.0
        vehicle_center = image_center
        detected = []

        for ros_vector in [message.vector_1, message.vector_2][:message.vector_count]:
            top_point = min(ros_vector, key=lambda p: p.y)
            bottom_point = max(ros_vector, key=lambda p: p.y)
            detected.append({'top_x': float(top_point.x), 'bottom_x': float(bottom_point.x)})

        detected.sort(key=lambda v: v['bottom_x'])
        both = (len(detected) == 2)

        if both:
            near_x = (detected[0]['bottom_x'] + detected[1]['bottom_x']) / 2.0
            far_x = (detected[0]['top_x'] + detected[1]['top_x']) / 2.0
        else:
            boundary = detected[0]
            near_x = boundary['bottom_x']
            far_x = boundary['top_x']

        if not self.tracker.validate_and_update(near_x, far_x):
            return

        self.last_lane_time = now

        if both:
            lateral_error = (near_x - vehicle_center) / max(image_center, 1.0)
        else:
            lateral_error = 0.0

        dx = far_x - near_x
        dy = max(1.0, float(message.image_height))
        tangent_angle = math.atan2(dx, dy)

        kp = float(self.get_parameter('lane_kp').value)
        heading_gain = float(self.get_parameter('heading_gain').value)
        raw_turn = -1.0 * (kp * lateral_error + heading_gain * tangent_angle)

        max_steer = abs(float(self.get_parameter('maximum_steering').value))
        raw_turn = max(-max_steer, min(max_steer, raw_turn))

        smoothing = float(self.get_parameter('steering_smoothing').value)
        self.filtered_lane_turn = smoothing * raw_turn + (1.0 - smoothing) * self.filtered_lane_turn
        self.lane_turn_command = float(self.filtered_lane_turn)

        if abs(tangent_angle) > 0.15:
            self.last_valid_turn_dir = 1.0 if tangent_angle < 0 else -1.0

        if abs(tangent_angle) > 0.25:
            self.lane_speed_command = float(self.get_parameter('sharp_curve_speed').value)
        else:
            self.lane_speed_command = float(self.get_parameter('lane_speed').value)

    # -------------------------------------------------------------------------
    # 5. Drive Command Loop (Absolute QR Priority)
    # -------------------------------------------------------------------------
    def publish_drive_commands(self):
        now = self.get_clock().now()
        speed = 0.0
        turn = 0.0

        # --- ABSOLUTE TOP PRIORITY: QR STOP OVERRIDE ---
        if self.qr_stop_active:
            hold_duration = float(self.get_parameter('target_qr_hold_duration').value)
            elapsed = (now - self.qr_stop_start_time).nanoseconds / 1e9

            speed = 0.0
            turn = 0.0

            if elapsed >= hold_duration and not self.qr_server_notified:
                self.qr_server_notified = True
                self.notify_server_arrival(self.current_target)
                self.get_logger().info(f"2-Second Hold Complete! Server Notified of Arrival at [{self.current_target}].")

        elif self.mission_status in (MISSION_WAITING_SERVER_DEST, MISSION_COMPLETE):
            speed = 0.0
            turn = 0.0

        elif self.depart_hold_until is not None:
            if now < self.depart_hold_until:
                speed = 0.0
                turn = 0.0
            else:
                self.depart_hold_until = None

        # --- SIGNBOARD TURN MANEUVER ---
        elif self.navigation_state == STATE_SIGN_MANEUVER:
            hold_duration = float(self.get_parameter('sign_hold_duration').value)
            elapsed = (now - self.sign_maneuver_start_time).nanoseconds / 1e9

            if elapsed < hold_duration:
                speed = float(self.get_parameter('sharp_curve_speed').value)
                turn = self.sign_override_steering
            else:
                self.dead_reckon_start_time = now
                self.navigation_state = STATE_DEAD_RECKON
                self.get_logger().info("Turn execution finished. Dead-reckoning straight into new branch...")

        # --- DEAD RECKONING ---
        elif self.navigation_state == STATE_DEAD_RECKON:
            dr_duration = float(self.get_parameter('dead_reckon_duration').value)
            elapsed = (now - self.dead_reckon_start_time).nanoseconds / 1e9

            if elapsed < dr_duration:
                speed = float(self.get_parameter('lane_speed').value)
                turn = 0.0
            else:
                self.pending_sign_command = None
                self.pending_sign_timestamp = None
                self.sign_maneuver_start_time = None
                self.dead_reckon_start_time = None
                self.sign_seen = False
                self.sign_close_confirmed = False
                self.navigation_state = STATE_LANE_FOLLOW
                self.get_logger().info("Seated in new lane. Re-enabling Lane Follower.")

        # --- STANDARD LANE FOLLOW ---
        elif self.navigation_state == STATE_LANE_FOLLOW:
            if self.last_lane_time is not None:
                lane_age = (now - self.last_lane_time).nanoseconds / 1e9
                lane_timeout = float(self.get_parameter('lane_timeout').value)

                if lane_age <= lane_timeout:
                    speed = self.lane_speed_command
                    turn = self.lane_turn_command
                else:
                    speed = float(self.get_parameter('lane_search_speed').value)
                    turn = self.last_valid_turn_dir * 0.22
            else:
                speed = float(self.get_parameter('lane_search_speed').value)
                turn = self.last_valid_turn_dir * 0.22

        # --- OBSTACLE AVOIDANCE ---
        elif self.navigation_state in (STATE_TURN_AWAY, STATE_PASS_OBSTACLE, STATE_RETURN_TO_LANE):
            speed = self.avoidance_speed_command
            turn = self.avoidance_turn_command

        max_steer = abs(float(self.get_parameter('maximum_steering').value))
        turn = max(-max_steer, min(max_steer, float(turn)))
        speed = max(0.0, min(1.0, float(speed)))

        msg = Joy()
        msg.buttons = [1, 0, 0, 0, 0, 0, 0, 1]
        msg.axes = [0.0, float(speed), 0.0, float(turn)]
        self.publisher_joy.publish(msg)

    # -------------------------------------------------------------------------
    # 6. LiDAR Callback: Robust Front Sector Obstacle Avoidance
    # -------------------------------------------------------------------------
    def lidar_callback(self, message):
        if not message.ranges or self.qr_stop_active:
            return

        ranges = np.asarray(message.ranges, dtype=np.float32)
        ranges = np.where(np.isfinite(ranges), ranges, np.inf)

        self.update_hospital_proximity(ranges)

        def sector_min(start_deg, end_deg):
            total = len(ranges)
            if total == 0:
                return float('inf')
            start = int((start_deg % 360.0) / 360.0 * total)
            end = int((end_deg % 360.0) / 360.0 * total)
            if start <= end:
                sector = ranges[start:end + 1]
            else:
                sector = np.concatenate((ranges[start:], ranges[:end + 1]))
            return float(np.min(sector)) if sector.size > 0 else float('inf')

        # Precise center-front scanning sector (350° to 10°) for real road obstacles
        front = min(sector_min(350.0, 360.0), sector_min(0.0, 10.0))
        left_front = sector_min(10.0, 45.0)
        right_front = sector_min(315.0, 350.0)
        left_side = sector_min(75.0, 110.0)
        right_side = sector_min(250.0, 285.0)

        min_sign_dist = min(front, left_front, right_front)

        # --- LIDAR SIGNBOARD PASS TRIGGER ---
        now = self.get_clock().now()
        validity_window = float(self.get_parameter('sign_validity_window').value)
        has_valid_pending_sign = (
            self.pending_sign_command is not None and
            self.pending_sign_timestamp is not None and
            ((now - self.pending_sign_timestamp).nanoseconds / 1e9) < validity_window
        )

        if self.navigation_state == STATE_LANE_FOLLOW and (self.sign_seen or has_valid_pending_sign):
            if min_sign_dist < 1.10:
                self.sign_close_confirmed = True

            elif min_sign_dist > 1.25 and self.sign_close_confirmed:
                self.sign_maneuver_start_time = now
                self.navigation_state = STATE_SIGN_MANEUVER
                self.get_logger().info(
                    f"LIDAR PASSED SIGNBOARD -> Executing Stored Turn [{self.pending_sign_command}] with Steer: {self.sign_override_steering:.2f}"
                )

        self.front_distance = front
        self.left_front_distance = left_front
        self.right_front_distance = right_front
        self.left_side_distance = left_side
        self.right_side_distance = right_side

        stop_dist = float(self.get_parameter('obstacle_front_stop').value)
        clear_dist = float(self.get_parameter('obstacle_front_clear').value)
        emergency_dist = float(self.get_parameter('obstacle_emergency_stop').value)
        side_clear = float(self.get_parameter('obstacle_side_clear').value)
        avoid_speed = float(self.get_parameter('obstacle_avoid_speed').value)
        merge_speed = float(self.get_parameter('obstacle_merge_speed').value)

        # EMERGENCY STOP
        if front <= emergency_dist:
            self.navigation_state = STATE_EMERGENCY_STOP
            self.avoidance_speed_command = 0.0
            self.avoidance_turn_command = 0.0
            return

        if self.navigation_state == STATE_EMERGENCY_STOP:
            if front >= clear_dist:
                self.navigation_state = STATE_LANE_FOLLOW
            return

        # OBSTACLE EVASION
        if (
            front < stop_dist
            and self.navigation_state not in (STATE_SIGN_MANEUVER, STATE_DEAD_RECKON)
        ):
            self.navigation_state = STATE_TURN_AWAY
            self.avoidance_speed_command = avoid_speed

            left_blocked = min(front, left_front) < side_clear
            right_blocked = min(front, right_front) < side_clear

            if left_blocked and not right_blocked:
                self.avoidance_turn_command = -0.35
            elif right_blocked and not left_blocked:
                self.avoidance_turn_command = 0.35
            else:
                self.avoidance_turn_command = 0.35 if right_front > left_front else -0.35
            return

        if self.navigation_state == STATE_TURN_AWAY:
            self.avoidance_speed_command = avoid_speed
            if front >= clear_dist:
                self.navigation_state = STATE_RETURN_TO_LANE
                self.avoidance_speed_command = merge_speed
                self.avoidance_turn_command = 0.0
            return

        if self.navigation_state == STATE_RETURN_TO_LANE:
            self.avoidance_speed_command = merge_speed
            if front >= clear_dist:
                if left_side < side_clear and right_side >= side_clear:
                    self.avoidance_turn_command = -0.20
                elif right_side < side_clear and left_side >= side_clear:
                    self.avoidance_turn_command = 0.20
                else:
                    self.avoidance_turn_command = 0.0
                    self.navigation_state = STATE_LANE_FOLLOW


def main(args=None):
    rclpy.init(args=args)
    node = LineFollower()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()