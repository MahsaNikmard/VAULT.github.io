#!/usr/bin/env python3
"""
webots_controller.py - Webots to ROS 2 bridge for the VAULT simulation
======================================================================
Replaces the physical TurtleBot4 + tb4_bridge for simulation.
Integrates PD control directly so no separate pd_controller process is needed.
Also records the ground-truth collision metrics reported in the paper
(collision count, time to first collision, distance to first collision) to
VAULT_METRICS_DIR when that variable is set.

Publishes:
    /robot2/oakd/rgb/preview/image_raw   (sensor_msgs/Image, bgr8)
    /odom                                (nav_msgs/Odometry)
    /tf                                  (odom → base_link)
    /tf_static                           (base_link → camera frames)

Subscribes:
    /robot2/waypoint                     (std_msgs/Float32MultiArray)
    /topoplan/reached_goal               (std_msgs/Bool)

Usage:
    The simulation is normally started with one command, which launches Webots,
    this bridge, and the navigation node together:

        bash simulation/run_sim.sh --gpu

    To run the pieces by hand instead, from the repository root:

        # Open Webots with the simulation arena
        webots simulation/worlds/multi_robot.wbt

        # Start this bridge (connects to the Webots extern robot)
        WEBOTS_HOME=/usr/local/webots python3 simulation/src/webots_controller.py \\
            --robot turtlebot4 --control vfh --config-dir simulation/config

        # Start the navigation node
        python3 deployment/src/explore_vfh.py --robot turtlebot4 --model nomad \\
            --config-dir simulation/config
"""

import argparse
import atexit
import json
import math
import os
import sys
import time
from pathlib import Path
from typing import Optional, Tuple

# ── Webots controller ───────────────────────────────────────────────────────
WEBOTS_HOME = os.environ.get("WEBOTS_HOME", "/usr/local/webots")
os.environ["WEBOTS_HOME"] = WEBOTS_HOME
_wpy = os.path.join(WEBOTS_HOME, "lib", "controller", "python")
if _wpy not in sys.path:
    sys.path.insert(0, _wpy)

from controller import Robot, Supervisor

# ── ROS 2 ───────────────────────────────────────────────────────────────────
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy
from geometry_msgs.msg import TransformStamped
from nav_msgs.msg import Odometry
from sensor_msgs.msg import Image, CameraInfo
from std_msgs.msg import Bool, Float32MultiArray
from tf2_ros import TransformBroadcaster, StaticTransformBroadcaster
import yaml

# ── TurtleBot4 physical parameters ────────────────────────────────────────
WHEEL_RADIUS = 0.036
WHEEL_DISTANCE = 0.233
MAX_MOTOR_SPEED = 10.0
CAM_HFOV = 1.55334  # 89°


def _clip_angle(a: float) -> float:
    """Wrap angle to (−π, π]."""
    return (a + math.pi) % (2 * math.pi) - math.pi


_REPO_ROOT = Path(__file__).resolve().parents[2]


def _resolve_config_dir() -> Path:
    """Extract --config-dir early so module-level config loading works."""
    _p = argparse.ArgumentParser(add_help=False)
    _p.add_argument("--config-dir", type=str, default="simulation/config")
    _args, _ = _p.parse_known_args()
    cd = Path(_args.config_dir)
    if not cd.is_absolute():
        cd = _REPO_ROOT / cd
    return cd


_CONFIG_DIR = _resolve_config_dir()

with open(_CONFIG_DIR / "robot.yaml", "r") as _f:
    _ROBOT_CFG = yaml.safe_load(_f)

MAX_V: float = _ROBOT_CFG["max_v"]
MAX_W: float = _ROBOT_CFG["max_w"]
RATE: int = _ROBOT_CFG["frame_rate"]
DT: float = 1.0 / RATE
EPS: float = 1e-8

WAYPOINT_TIMEOUT: float = 1.0
MAX_DISTANCE: float = 100.0   # metres — triggers goal-reached
MAX_TIME: float = 30000.0     # seconds
FLOOR_CONTACT_Z: float = 0.02  # metres — contact points below this are ground, ignored


# ═════════════════════════════════════════════════════════════════════════════
class WebotsController(Node):
    """Webots bridge with integrated PD waypoint control."""

    def __init__(self, robot: Robot, args: argparse.Namespace):
        super().__init__("webots_controller")
        self._robot = robot
        self._dt_ms = int(robot.getBasicTimeStep())
        self.controller_type = args.control

        # ── Motors ──────────────────────────────────────────────────
        self._lm = robot.getDevice("left_wheel_motor")
        self._rm = robot.getDevice("right_wheel_motor")
        self._lm.setPosition(float("inf"))
        self._rm.setPosition(float("inf"))
        self._lm.setVelocity(0.0)
        self._rm.setVelocity(0.0)

        # ── Encoders ───────────────────────────────────────────────
        self._le = robot.getDevice("left_wheel_sensor")
        self._re = robot.getDevice("right_wheel_sensor")
        self._le.enable(self._dt_ms)
        self._re.enable(self._dt_ms)

        # ── Camera ─────────────────────────────────────────────────
        self._cam = robot.getDevice("oakd_rgb")
        self._cam.enable(self._dt_ms)
        self._cam_w = self._cam.getWidth()
        self._cam_h = self._cam.getHeight()

        # ── Enable optional sensors ────────────────────────────────
        for name in ["gps", "imu", "gyro", "accelerometer"]:
            d = robot.getDevice(name)
            if d:
                d.enable(self._dt_ms)

        # ── Odometry state ─────────────────────────────────────────
        self._x = 0.0
        self._y = 0.0
        self._theta = 0.0
        self._pl = 0.0
        self._pr = 0.0
        self._first = True

        # ── Velocity commands (set by PD control each tick) ────────
        self._v = 0.0
        self._w = 0.0

        # ── Waypoint / goal state ──────────────────────────────────
        self._waypoint: Optional[np.ndarray] = None
        self._last_wp_time: float = 0.0
        self._reached_goal: bool = False

        # ── Distance & time tracking ───────────────────────────────
        self._total_distance: float = 0.0
        self._last_odom_x: Optional[float] = None
        self._last_odom_y: Optional[float] = None
        self._start_time: float = time.time()

        # ── Ground-truth collision tracking (Supervisor) ───────────
        # A collision is a real physics contact between the robot body and an
        # obstacle (or wall) bounding box, read from the simulator. The robot's
        # own bounding cylinder sits above the floor, so its contacts (excluding
        # the wheel sub-solids) are only ever with obstacles, not the ground.
        self._gt_enabled = hasattr(robot, "getSelf")
        self._self_node = robot.getSelf() if self._gt_enabled else None
        self._gt_last_pos: Optional[Tuple[float, float]] = None
        self._gt_distance: float = 0.0
        self._gt_collisions: int = 0
        self._gt_first_collision_t: Optional[float] = None
        self._gt_first_collision_d: Optional[float] = None
        self._gt_in_collision: bool = False
        self._gt_saved: bool = False
        if self._gt_enabled:
            self.get_logger().info("Ground-truth collision tracking on (physics contacts)")
            atexit.register(self._save_ground_truth)

        # ── ROS publishers ─────────────────────────────────────────
        self._img_pub = self.create_publisher(
            Image, "/robot2/oakd/rgb/preview/image_raw", 10
        )
        self._ci_pub = self.create_publisher(
            CameraInfo, "/robot2/oakd/rgb/preview/camera_info", 10
        )
        self._odom_pub = self.create_publisher(Odometry, "/odom", 10)
        self._tf_bc = TransformBroadcaster(self)
        self._stf_bc = StaticTransformBroadcaster(self)

        # ── ROS subscribers ────────────────────────────────────────
        self.create_subscription(
            Float32MultiArray, "/robot2/waypoint", self._waypoint_cb, 1
        )
        self.create_subscription(
            Bool, "/topoplan/reached_goal", self._goal_cb, 1
        )
        odom_qos = QoSProfile(depth=1, reliability=ReliabilityPolicy.RELIABLE)
        self.create_subscription(Odometry, "/odom", self._odom_distance_cb, odom_qos)

        # ── Camera intrinsics ──────────────────────────────────────
        fx = (self._cam_w / 2.0) / math.tan(CAM_HFOV / 2.0)
        ci = CameraInfo()
        ci.width = self._cam_w
        ci.height = self._cam_h
        ci.distortion_model = "plumb_bob"
        ci.d = [0.0] * 5
        ci.k = [fx, 0.0, self._cam_w / 2.0,
                0.0, fx, self._cam_h / 2.0,
                0.0, 0.0, 1.0]
        ci.r = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]
        ci.p = [fx, 0.0, self._cam_w / 2.0, 0.0,
                0.0, fx, self._cam_h / 2.0, 0.0,
                0.0, 0.0, 1.0, 0.0]
        self._ci = ci

        # ── Static TFs ────────────────────────────────────────────
        self._pub_static_tf()

        self.get_logger().info(
            f"Webots controller ready — camera {self._cam_w}x{self._cam_h}, "
            f"timestep {self._dt_ms}ms, control={self.controller_type}, "
            f"max_v={MAX_V}, max_w={MAX_W}, rate={RATE}Hz"
        )

    # ────────────────────────────────────────────────────────────────
    # ROS callbacks
    # ────────────────────────────────────────────────────────────────
    def _waypoint_cb(self, msg: Float32MultiArray) -> None:
        self._waypoint = np.asarray(msg.data, dtype=float)
        self._last_wp_time = time.time()
        self.get_logger().debug(f"Waypoint received: {self._waypoint.tolist()}")

    def _goal_cb(self, msg: Bool) -> None:
        self._reached_goal = msg.data
        if self._reached_goal:
            elapsed = time.time() - self._start_time
            self.get_logger().info(
                f"Goal reached — distance: {self._total_distance:.3f} m, "
                f"time: {elapsed:.1f} s"
            )

    def _odom_distance_cb(self, msg: Odometry) -> None:
        """Accumulate actual distance traveled from published odometry."""
        x = msg.pose.pose.position.x
        y = msg.pose.pose.position.y
        if self._last_odom_x is not None:
            dx = x - self._last_odom_x
            dy = y - self._last_odom_y
            self._total_distance += math.sqrt(dx * dx + dy * dy)
            if self._total_distance > MAX_DISTANCE:
                self._goal_cb(Bool(data=True))
        self._last_odom_x = x
        self._last_odom_y = y

    # ────────────────────────────────────────────────────────────────
    # PD control
    # ────────────────────────────────────────────────────────────────
    def _waypoint_valid(self) -> bool:
        return (
            self._waypoint is not None
            and (time.time() - self._last_wp_time) < WAYPOINT_TIMEOUT
        )

    def _pd_control(self, wp: np.ndarray) -> Tuple[float, float]:
        """Compute (v, w) from a 2-D or 4-D waypoint."""
        if wp.size == 2:
            dx, dy = wp
            use_heading = False
        elif wp.size == 4:
            dx, dy, hx, hy = wp
            use_heading = abs(dx) < EPS and abs(dy) < EPS
        else:
            raise ValueError(f"Waypoint must be 2-D or 4-D, got size={wp.size}")

        # Backward waypoint (dx < 0) → reverse straight (VFH* recovery)
        if dx < -EPS and not use_heading:
            v = float(np.clip(dx / DT, -MAX_V, 0.0))
            return v, 0.0

        if use_heading:
            v = 0.0
            desired_yaw = math.atan2(hy, hx)
        elif abs(dx) < EPS:
            v = 0.0
            desired_yaw = math.copysign(math.pi / 2, dy)
        else:
            v = dx / DT
            desired_yaw = math.atan(dy / dx)

        if self.controller_type != "nomad":
            if abs(desired_yaw) > math.radians(30):
                v = 0.0

        w = _clip_angle(desired_yaw) / DT
        return float(np.clip(v, 0.0, MAX_V)), float(np.clip(w, -MAX_W, MAX_W))

    # ────────────────────────────────────────────────────────────────
    # Static TFs
    # ────────────────────────────────────────────────────────────────
    def _pub_static_tf(self):
        now = self.get_clock().now().to_msg()
        tfs = []
        # base_link → camera
        t = TransformStamped()
        t.header.stamp = now
        t.header.frame_id = "base_link"
        t.child_frame_id = "oakd_rgb_camera_frame"
        t.transform.translation.x = 0.075
        t.transform.translation.z = 0.194
        t.transform.rotation.w = 1.0
        tfs.append(t)
        # camera → optical
        t2 = TransformStamped()
        t2.header.stamp = now
        t2.header.frame_id = "oakd_rgb_camera_frame"
        t2.child_frame_id = "oakd_rgb_camera_optical_frame"
        t2.transform.rotation.x = -0.5
        t2.transform.rotation.y = 0.5
        t2.transform.rotation.z = -0.5
        t2.transform.rotation.w = 0.5
        tfs.append(t2)
        self._stf_bc.sendTransform(tfs)

    # ────────────────────────────────────────────────────────────────
    # Ground-truth collision tracking
    # ────────────────────────────────────────────────────────────────
    def _body_in_contact(self) -> bool:
        """True when the robot body bounding cylinder is touching an obstacle.

        getContactPoints(includeDescendants=False) returns contacts of the
        robot's own bounding object only, not the wheel sub-solids. The body
        cylinder is raised off the floor, so any such contact is with an
        obstacle or a wall, i.e. a real collision. A small height guard ignores
        any stray ground contact."""
        try:
            contacts = self._self_node.getContactPoints(False)
        except Exception:
            return False
        for c in contacts:
            if c.point[2] > FLOOR_CONTACT_Z:
                return True
        return False

    def _update_ground_truth(self) -> None:
        """Update true distance travelled and the physics-based collision count."""
        if not self._gt_enabled:
            return
        pos = self._self_node.getPosition()
        x, y = pos[0], pos[1]

        # True distance travelled
        if self._gt_last_pos is not None:
            dx = x - self._gt_last_pos[0]
            dy = y - self._gt_last_pos[1]
            self._gt_distance += math.sqrt(dx * dx + dy * dy)
        self._gt_last_pos = (x, y)

        # Collision = real contact between the robot body and an obstacle box.
        # Edge-triggered (debounced) so each separate bump is one event.
        if self._body_in_contact():
            if not self._gt_in_collision:
                self._gt_collisions += 1
                self._gt_in_collision = True
                if self._gt_first_collision_t is None:
                    self._gt_first_collision_t = time.time() - self._start_time
                    self._gt_first_collision_d = self._gt_distance
        else:
            self._gt_in_collision = False

    def _save_ground_truth(self, force: bool = False) -> None:
        """Write ground-truth safety metrics to VAULT_METRICS_DIR/ground_truth.json.

        Called periodically (force=True) so the latest snapshot survives even a
        hard kill, and once on shutdown."""
        if not self._gt_enabled:
            return
        if self._gt_saved and not force:
            return
        out_dir = os.environ.get("VAULT_METRICS_DIR")
        if not out_dir:
            return
        data = {
            "traversed_path_m": self._gt_distance,
            "collisions": self._gt_collisions,
            "time_to_first_collision_s": self._gt_first_collision_t,
            "distance_to_first_collision_m": self._gt_first_collision_d,
            "duration_s": time.time() - self._start_time,
        }
        try:
            Path(out_dir).mkdir(parents=True, exist_ok=True)
            with open(Path(out_dir) / "ground_truth.json", "w") as f:
                json.dump(data, f, indent=2, default=str)
            self._gt_saved = True
            self.get_logger().info(f"Ground-truth metrics saved to {out_dir}/ground_truth.json")
        except Exception as e:  # pragma: no cover - defensive
            self.get_logger().warn(f"Ground-truth save failed: {e}")

    # ────────────────────────────────────────────────────────────────
    # Main step — called every simulation tick
    # ────────────────────────────────────────────────────────────────
    def step(self):
        """One simulation tick: PD control → motors → sensors → publish."""

        # ── Ground-truth collision/clearance update ─────────────────
        self._update_ground_truth()

        # ── Check time limit ───────────────────────────────────────
        elapsed = time.time() - self._start_time
        if elapsed > MAX_TIME:
            self._goal_cb(Bool(data=True))

        # ── PD control: waypoint → (v, w) ─────────────────────────
        if self._reached_goal:
            self._v, self._w = 0.0, 0.0
        elif self._waypoint_valid():
            self._v, self._w = self._pd_control(self._waypoint)
            self.get_logger().debug(f"PD → v={self._v:.3f}  w={self._w:.3f}")
        else:
            self._v, self._w = 0.0, 0.0

        # ── Drive motors ───────────────────────────────────────────
        ls = (self._v - self._w * WHEEL_DISTANCE / 2) / WHEEL_RADIUS
        rs = (self._v + self._w * WHEEL_DISTANCE / 2) / WHEEL_RADIUS
        self._lm.setVelocity(max(-MAX_MOTOR_SPEED, min(MAX_MOTOR_SPEED, ls)))
        self._rm.setVelocity(max(-MAX_MOTOR_SPEED, min(MAX_MOTOR_SPEED, rs)))

        # ── Odometry ───────────────────────────────────────────────
        lp = self._le.getValue()
        rp = self._re.getValue()
        if self._first:
            self._pl, self._pr, self._first = lp, rp, False

        dl = (lp - self._pl) * WHEEL_RADIUS
        dr = (rp - self._pr) * WHEEL_RADIUS
        self._pl, self._pr = lp, rp

        dc = (dl + dr) / 2
        dth = (dr - dl) / WHEEL_DISTANCE
        self._x += dc * math.cos(self._theta + dth / 2)
        self._y += dc * math.sin(self._theta + dth / 2)
        self._theta += dth

        dt_s = self._dt_ms / 1000.0
        now = self.get_clock().now().to_msg()
        cy = math.cos(self._theta / 2)
        sy = math.sin(self._theta / 2)

        odom = Odometry()
        odom.header.stamp = now
        odom.header.frame_id = "odom"
        odom.child_frame_id = "base_link"
        odom.pose.pose.position.x = self._x
        odom.pose.pose.position.y = self._y
        odom.pose.pose.orientation.z = sy
        odom.pose.pose.orientation.w = cy
        odom.twist.twist.linear.x = dc / dt_s if dt_s > 0 else 0.0
        odom.twist.twist.angular.z = dth / dt_s if dt_s > 0 else 0.0
        self._odom_pub.publish(odom)

        tf = TransformStamped()
        tf.header.stamp = now
        tf.header.frame_id = "odom"
        tf.child_frame_id = "base_link"
        tf.transform.translation.x = self._x
        tf.transform.translation.y = self._y
        tf.transform.rotation.z = sy
        tf.transform.rotation.w = cy
        self._tf_bc.sendTransform(tf)

        # ── Camera ─────────────────────────────────────────────────
        raw = self._cam.getImage()
        if raw:
            bgra = np.frombuffer(raw, np.uint8).reshape(self._cam_h, self._cam_w, 4)
            bgr = np.ascontiguousarray(bgra[:, :, :3])

            img = Image()
            img.header.stamp = now
            img.header.frame_id = "oakd_rgb_camera_optical_frame"
            img.width = self._cam_w
            img.height = self._cam_h
            img.encoding = "bgr8"
            img.step = self._cam_w * 3
            img.data = bgr.tobytes()
            self._img_pub.publish(img)

            self._ci.header = img.header
            self._ci_pub.publish(self._ci)


# ═════════════════════════════════════════════════════════════════════════════
def main():
    parser = argparse.ArgumentParser("Webots controller with integrated PD control")
    parser.add_argument(
        "--control", type=str, default="vfh",
        choices=["nomad", "vfh", "care"],
        help="Waypoint control mode (nomad: smooth, vfh/care: rotate-in-place if yaw > 30°)",
    )
    parser.add_argument(
        "--robot", type=str, default="turtlebot4",
        choices=["locobot", "robomaster", "turtlebot4"],
    )
    parser.add_argument(
        "--config-dir", type=str, default="simulation/config",
        help="Directory containing robot.yaml",
    )
    args, _ = parser.parse_known_args()

    # Supervisor (the simulation robot has supervisor TRUE) so ground-truth
    # collision metrics can read the true robot pose and obstacle layout.
    robot = Supervisor()
    ts = int(robot.getBasicTimeStep())

    rclpy.init()
    node = WebotsController(robot, args)
    node.get_logger().info("Running — Ctrl+C to stop")

    step_count = 0
    try:
        while robot.step(ts) != -1:
            rclpy.spin_once(node, timeout_sec=0)
            node.step()
            step_count += 1
            if step_count % 100 == 0:
                node.get_logger().info(
                    f"step={step_count}  v={node._v:.3f}  w={node._w:.3f}  "
                    f"dist={node._gt_distance:.2f}m  "
                    f"collisions={node._gt_collisions}"
                )
                node._save_ground_truth(force=True)   # snapshot survives a hard kill
    except KeyboardInterrupt:
        pass
    finally:
        node._save_ground_truth()
        node.get_logger().info(
            f"Exiting after {step_count} steps — ground-truth distance "
            f"{node._gt_distance:.2f} m, collisions {node._gt_collisions}"
        )
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
