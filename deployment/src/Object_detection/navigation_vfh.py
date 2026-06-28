"""
navigation_vfh.py – Goal-oriented VFH* navigation with state-machine control.

State machine
-------------
    EXPLORE  (default)
        NoMaD trajectory bins → VFH* → waypoint
        Transition → NAV_GOAL  when YOLO detects a target-class object
                                (first detection triggers the switch)

    NAV_GOAL
        YOLO detection bins → VFH* safety validation → direction waypoint
        The detected-object bins replace NoMaD as VFH*'s reference_bins.
        VFH* finds the safest reachable direction toward the object.
        Transition → REACHED   when any goal bin depth < safety_threshold
        Transition → EXPLORE   when no detection for `goal_timeout_frames`
                                consecutive timer ticks (configurable via
                                --goal-timeout-frames, default 10)

    REACHED  (terminal)
        Publish zero velocity. Node stays here until ROS shutdown.

Depth visualisation
-------------------
    Publishes VFH* per-bin distance markers to /nav_vfh/depth_markers
    (same DepthMarkerPublisher used in explore_vfh).

Usage
-----
    python navigation_vfh.py \\
        --robot turtlebot4 \\
        --yolo-weights /path/to/yolov8n.pt \\
        --yolo-conf 0.3 \\
        --yolo-classes 0 \\
        --goal-timeout-frames 10
"""

from __future__ import annotations

import argparse
import math
import os
import sys
from collections import deque
from enum import Enum
from pathlib import Path
from typing import Deque, List, Optional, Tuple

# ── DA2 metric depth package path ─────────────────────────────────────────────
_DA2_METRIC = str(Path(__file__).resolve().parents[3] / "Depth-Anything-V2" / "metric_depth")
if _DA2_METRIC not in sys.path:
    sys.path.insert(0, _DA2_METRIC)

# ── Deployment src on path ────────────────────────────────────────────────────
_SRC = str(Path(__file__).resolve().parents[1])
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

import cv2
import numpy as np
import rclpy
from cv_bridge import CvBridge
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import Image
from std_msgs.msg import Bool, Float32MultiArray
import torch
import yaml
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler

from utils import msg_to_pil, to_numpy, transform_images, load_model
from vint_train.training.train_utils import get_action
from depth_anything_v2.dpt import DepthAnythingV2

from VfhPlus import defaults as D
from vfhstar_nav import VFHStar
from VfhPlus.depth_processing import (
    compute_distance_vector, TemporalAggregator, pad_distance_vector,
)
from VfhPlus.nomad_vector import waypoint_to_reference, generate_direction_waypoints
from VfhPlus.depth_markers import DepthMarkerPublisher, BinRayMarkerPublisher

from Object_detection.Object_detection import (
    load_yolo_model, detect_objects_with_confidence,
)

# ── Config paths ──────────────────────────────────────────────────────────────
THIS_DIR = Path.cwd()


def _parse_config_dir() -> Path:
    import argparse as _ap
    _p = _ap.ArgumentParser(add_help=False)
    _p.add_argument("--config-dir", type=str, default="deployment/config")
    _args, _ = _p.parse_known_args()
    cd = Path(_args.config_dir)
    if not cd.is_absolute():
        cd = THIS_DIR / cd
    return cd


_CONFIG_DIR       = _parse_config_dir()
ROBOT_CONFIG_PATH = _CONFIG_DIR / "robot.yaml"
MODEL_CONFIG_PATH = THIS_DIR / "deployment/config/models.yaml"
VFH_CONFIG_PATH   = _CONFIG_DIR / "vfh.yaml"

with open(ROBOT_CONFIG_PATH) as f:
    ROBOT_CONF = yaml.safe_load(f)
MAX_V = ROBOT_CONF["max_v"]
MAX_W = ROBOT_CONF["max_w"]
RATE  = ROBOT_CONF["frame_rate"]

with open(VFH_CONFIG_PATH) as f:
    VFH_CONF = yaml.safe_load(f)


# ═════════════════════════════════════════════════════════════════════════════
# Navigation state machine
# ═════════════════════════════════════════════════════════════════════════════

class NavState(Enum):
    EXPLORE  = "EXPLORE"   # NoMaD-driven free exploration
    NAV_GOAL = "NAV_GOAL"  # Navigate toward YOLO-detected object
    REACHED  = "REACHED"   # Object within safety_threshold — stop


# ═════════════════════════════════════════════════════════════════════════════
# NavigationVFHNode
# ═════════════════════════════════════════════════════════════════════════════

def _load_nomad_model(model_name: str, device: torch.device):
    with open(MODEL_CONFIG_PATH) as f:
        model_paths = yaml.safe_load(f)
    model_config_path = model_paths[model_name]["config_path"]
    with open(model_config_path) as f:
        model_params = yaml.safe_load(f)
    ckpt_path = model_paths[model_name]["ckpt_path"]
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Model weights not found: {ckpt_path}")
    model = load_model(ckpt_path, model_params, device)
    return model.to(device).eval(), model_params


class NavigationVFHNode(Node):
    """Goal-oriented VFH* navigation with EXPLORE / NAV_GOAL / REACHED states.

    EXPLORE  – NoMaD bins → VFH* → NoMaD waypoint (free roaming)
    NAV_GOAL – YOLO bins  → VFH* → direction waypoint (approach object)
    REACHED  – zero velocity (terminal)
    """

    def __init__(self, args: argparse.Namespace) -> None:
        super().__init__("navigation_vfh")
        self.args = args

        # ── Device ───────────────────────────────────────────────────────────
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.get_logger().info(f"Device: {self.device}")

        # ── NoMaD ────────────────────────────────────────────────────────────
        self.model, self.model_params = _load_nomad_model(args.model, self.device)
        self.context_size: int = self.model_params["context_size"]
        self.last_ctx_time = self.get_clock().now()
        self.ctx_dt = 0.1
        self.noise_scheduler = DDPMScheduler(
            num_train_timesteps=self.model_params["num_diffusion_iters"],
            beta_schedule="squaredcos_cap_v2",
            clip_sample=True,
            prediction_type="epsilon",
        )
        self.context_queue: Deque = deque(maxlen=self.context_size + 1)
        self.bridge = CvBridge()

        # ── Robot topics ─────────────────────────────────────────────────────
        if args.robot == "turtlebot4":
            image_topic           = "/robot2/oakd/rgb/preview/image_raw"
            waypoint_topic        = "/robot2/waypoint"
            sampled_actions_topic = "/robot2/sampled_actions"
            trajectory_viz_topic  = "/robot2/trajectory_viz"
            self.DIM = (320, 200)
        elif args.robot == "locobot":
            image_topic           = "/robot1/camera/image"
            waypoint_topic        = "/robot1/waypoint"
            sampled_actions_topic = "/robot1/sampled_actions"
            trajectory_viz_topic  = "/robot1/trajectory_viz"
            self.DIM = (320, 240)
        else:
            raise ValueError(f"Unknown robot: {args.robot}")

        # ── VFH* parameters ──────────────────────────────────────────────────
        self.vfh_num_bins      = VFH_CONF.get("num_bins",          D.NUM_BINS)
        self.vfh_fov_deg       = VFH_CONF.get("fov_deg",           D.FOV_DEG)
        self.vfh_max_range     = VFH_CONF.get("max_sensing_range",  D.MAX_RANGE)
        self.vfh_v_margin      = VFH_CONF.get("vertical_margin",    D.VERTICAL_MARGIN)
        self.vfh_floor_margin  = VFH_CONF.get("floor_margin",       D.FLOOR_MARGIN)
        self.vfh_safety_margin = VFH_CONF.get("safety_margin",      D.SAFETY_MARGIN)
        self.vfh_depth_scale   = VFH_CONF.get("depth_scale",        D.DEPTH_SCALE)
        self.vfh_speed_red     = VFH_CONF.get("speed_reduction",    D.SPEED_REDUCTION)
        self.vfh_num_wps       = VFH_CONF.get("num_vfh_waypoints",  D.NUM_VFH_WAYPOINTS)
        self.vfh_wp_idx        = VFH_CONF.get("vfh_waypoint_index", D.VFH_WAYPOINT_INDEX)
        self.fov_padding_bins  = VFH_CONF.get("fov_padding_bins",   D.FOV_PADDING_BINS)

        self.vfh_total_bins  = self.vfh_num_bins + 2 * self.fov_padding_bins
        bin_width_deg        = self.vfh_fov_deg / self.vfh_num_bins
        self.vfh_virtual_fov = bin_width_deg * self.vfh_total_bins

        self.vfh = VFHStar(
            num_bins                = self.vfh_total_bins,
            fov_deg                 = self.vfh_virtual_fov,
            safety_threshold        = VFH_CONF.get("safety_threshold",        D.SAFETY_THRESHOLD),
            s_max                   = VFH_CONF.get("s_max",                    D.S_MAX),
            mu1                     = VFH_CONF.get("mu1",                      D.MU1),
            mu2                     = VFH_CONF.get("mu2",                      D.MU2),
            mu3                     = VFH_CONF.get("mu3",                      D.MU3),
            robot_radius            = VFH_CONF.get("robot_radius",             D.ROBOT_RADIUS),
            recovery_reverse_cycles = VFH_CONF.get("recovery_reverse_cycles", D.RECOVERY_REVERSE_CYCLES),
            recovery_turn_cycles    = VFH_CONF.get("recovery_turn_cycles",     D.RECOVERY_TURN_CYCLES),
            fov_padding_bins        = self.fov_padding_bins,
        )

        # ── DA2 depth model ──────────────────────────────────────────────────
        intrinsics_path = self._intrinsics_path()
        if not os.path.exists(intrinsics_path):
            raise FileNotFoundError(f"Intrinsics not found: {intrinsics_path}")
        self.K = np.load(intrinsics_path)

        da2_cfg = {
            "vits": {"encoder": "vits", "features": 64,  "out_channels": [48,  96,  192,  384]},
            "vitb": {"encoder": "vitb", "features": 128, "out_channels": [96,  192, 384,  768]},
            "vitl": {"encoder": "vitl", "features": 256, "out_channels": [256, 512, 1024, 1024]},
        }
        enc     = VFH_CONF.get("depth_encoder", "vits")
        weights = VFH_CONF.get("depth_weights", "")
        if not os.path.isabs(weights):
            weights = str(THIS_DIR / weights)
        self.depth_model = DepthAnythingV2(
            **{**da2_cfg[enc], "max_depth": VFH_CONF.get("depth_max_depth", 20)}
        )
        self.depth_model.load_state_dict(
            torch.load(weights, map_location="cpu", weights_only=True)
        )
        self.depth_model = self.depth_model.to(self.device).eval()

        # ── YOLO model ───────────────────────────────────────────────────────
        yolo_weights = args.yolo_weights
        if not os.path.isabs(yolo_weights):
            yolo_weights = str(THIS_DIR / yolo_weights)
        self.yolo_model, _ = load_yolo_model(yolo_weights, device=str(self.device))
        self.yolo_conf_threshold = args.yolo_conf
        self.yolo_classes        = args.yolo_classes  # None = all classes

        # ── Shared state ─────────────────────────────────────────────────────
        self.distance_vector = pad_distance_vector(
            np.full(self.vfh_num_bins, np.inf),
            padding_bins=self.fov_padding_bins,
        )
        self.temporal_agg = TemporalAggregator(
            num_bins        = self.vfh_num_bins,
            window_size     = VFH_CONF.get("temporal_window",  D.TEMPORAL_WINDOW),
            danger_threshold= VFH_CONF.get("safety_threshold", D.SAFETY_THRESHOLD),
        )
        self.current_waypoint    = np.zeros(2)
        self._new_depth_available = False

        # Latest YOLO results — updated in _image_cb at camera rate.
        # _goal_stale counts consecutive no-detection image frames and drives
        # both the memory-clear (EXPLORE only, threshold _goal_max_stale) and
        # the NAV_GOAL → EXPLORE timeout (threshold goal_timeout_frames).
        # _goal_seen_last_image is the per-image live-detection flag used by
        # the debug log to distinguish live detections from cached memory.
        self._goal_bins:             List[int]   = []
        self._goal_confs:            List[float] = []
        self._goal_stale:            int  = 0
        self._goal_max_stale:        int  = args.goal_stale_frames
        self._goal_seen_last_image:  bool = False

        # ── State machine ─────────────────────────────────────────────────────
        self.state: NavState = NavState.EXPLORE
        self.goal_timeout_frames: int  = args.goal_timeout_frames
        # Separate arrival threshold (smaller than VFH* safety_threshold so the
        # robot actually approaches before stopping).
        self.goal_reach_distance: float = args.goal_reach_distance

        # ── Depth markers (RViz visualisation) ───────────────────────────────
        self.depth_marker_pub = DepthMarkerPublisher(
            node             = self,
            topic            = "/vfh/depth_markers",
            num_bins         = self.vfh_total_bins,
            fov_deg          = self.vfh_virtual_fov,
            max_range        = self.vfh_max_range,
            safety_threshold = VFH_CONF.get("safety_threshold", D.SAFETY_THRESHOLD),
        )


        # ── Bin-ray marker publishers (RViz) ──────────────────────────────
        # Detected-object ray (white) — YOLO detections, regardless of state
        self.detected_ray_pub = BinRayMarkerPublisher(
            node      = self,
            topic     = "/vfh/detected_objects_ray",
            num_bins  = self.vfh_total_bins,
            fov_deg   = self.vfh_virtual_fov,
            color     = (1.0, 1.0, 1.0, 1.0),
            marker_ns = "detected_objects",
        )
        # NoMaD reference-bin rays (yellow) — all NoMaD trajectory bins
        self.nomad_ray_pub = BinRayMarkerPublisher(
            node      = self,
            topic     = "/vfh/nomad_reference_bins",
            num_bins  = self.vfh_total_bins,
            fov_deg   = self.vfh_virtual_fov,
            color     = (1.0, 0.95, 0.0, 0.9),
            marker_ns = "nomad_refs",
        )
        # Goal reference-bin rays (orange) — published only in NAV_GOAL
        self.goal_ray_pub = BinRayMarkerPublisher(
            node      = self,
            topic     = "/vfh/goal_reference_bins",
            num_bins  = self.vfh_total_bins,
            fov_deg   = self.vfh_virtual_fov,
            color     = (1.0, 0.45, 0.0, 1.0),
            marker_ns = "goal_refs",
        )
        # Chosen (VFH*-selected) bin ray (bright green) — the waypoint direction
        self.chosen_ray_pub = BinRayMarkerPublisher(
            node       = self,
            topic      = "/vfh/chosen_bin",
            num_bins   = self.vfh_total_bins,
            fov_deg    = self.vfh_virtual_fov,
            color      = (0.0, 1.0, 0.2, 1.0),
            marker_ns  = "chosen_bin",
            point_size = 0.09,
        )






        # ── ROS topics ────────────────────────────────────────────────────────
        img_qos = QoSProfile(depth=1, reliability=ReliabilityPolicy.BEST_EFFORT)
        self.create_subscription(Image, image_topic, self._image_cb, img_qos)
        self.waypoint_pub        = self.create_publisher(Float32MultiArray, waypoint_topic,        1)
        self.sampled_actions_pub = self.create_publisher(Float32MultiArray, sampled_actions_topic, 1)
        self.viz_pub             = self.create_publisher(Image, trajectory_viz_topic,              1)
        self.reached_goal_pub    = self.create_publisher(Bool, "/topoplan/reached_goal",           1)

        self.create_timer(1.0 / RATE, self._timer_cb)
        self.get_logger().info(
            f"NavigationVFHNode ready — robot={args.robot}, "
            f"state={self.state.value}, "
            f"goal_timeout={self.goal_timeout_frames} frames"
        )

    # ── Helpers ──────────────────────────────────────────────────────────────

    def _intrinsics_path(self) -> str:
        p = ROBOT_CONF.get("intrinsics_path", "")
        if not p:
            p = f"intrinsic/{self.args.robot}/intrinsics.npy"
        if not os.path.isabs(p):
            p = str(THIS_DIR / p)
        return p

    def _publish_stop(self) -> None:
        wp_msg = Float32MultiArray()
        wp_msg.data = [0.0, 0.0]
        self.waypoint_pub.publish(wp_msg)
        self.current_waypoint = np.zeros(2)
        self.reached_goal_pub.publish(Bool(data=True))

    def _republish_waypoint(self) -> None:
        if np.linalg.norm(self.current_waypoint[:2]) > 1e-3:
            msg = Float32MultiArray()
            msg.data = [float(x) for x in self.current_waypoint]
            self.waypoint_pub.publish(msg)

    # ── Image callback: depth + YOLO ─────────────────────────────────────────

    def _image_cb(self, msg: Image) -> None:
        now = self.get_clock().now()
        if (now - self.last_ctx_time).nanoseconds < self.ctx_dt * 1e9:
            return
        self.context_queue.append(msg_to_pil(msg))
        self.last_ctx_time = now

        # Do NOT pass desired_encoding — cv_bridge's C++ color conversion
        # segfaults when the compiled NumPy ABI differs from the active one.
        # Decode the raw frame and handle the color space in Python instead.
        frame = self.bridge.imgmsg_to_cv2(msg)
        enc = msg.encoding.lower().replace("-", "")
        if enc in ("rgb8", "rgb"):
            bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        elif enc in ("bgr8", "bgr"):
            bgr = frame
        else:
            bgr = frame  # best-effort; DA2 and YOLO assume BGR

        # DA2 metric depth (expects BGR)
        with torch.no_grad():
            depth_map = self.depth_model.infer_image(bgr)

        raw_dv = compute_distance_vector(
            depth_map, self.K,
            num_bins        = self.vfh_num_bins,
            fov_deg         = self.vfh_fov_deg,
            max_range       = self.vfh_max_range,
            vertical_margin = self.vfh_v_margin,
            floor_margin    = self.vfh_floor_margin,
            safety_margin   = self.vfh_safety_margin,
            depth_scale     = self.vfh_depth_scale,
        )
        smoothed = self.temporal_agg.update(raw_dv)
        self.distance_vector      = pad_distance_vector(smoothed, self.fov_padding_bins)
        self._new_depth_available = True

        # YOLO detection — image pixels span the REAL camera FOV (vfh_num_bins),
        # not the padded virtual FOV. Map into the real-bin range and shift by
        # fov_padding_bins so the returned indices line up with distance_vector.
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        raw_bins, goal_confs = detect_objects_with_confidence(
            rgb,
            self.yolo_model,
            num_bins       = self.vfh_num_bins,
            fov_deg        = self.vfh_fov_deg,
            conf_threshold = self.yolo_conf_threshold,
            classes        = self.yolo_classes,
        )
        goal_bins = [b + self.fov_padding_bins for b in raw_bins]
        if raw_bins:
            self.get_logger().info(
                f"[DBG/YOLO] raw_bins={raw_bins} (real-FOV space) "
                f"→ padded_bins={goal_bins} "
                f"(shift=+{self.fov_padding_bins}, valid range "
                f"{self.fov_padding_bins}..{self.fov_padding_bins + self.vfh_num_bins - 1})"
            )
        if goal_bins:
            self._goal_bins             = goal_bins
            self._goal_confs            = goal_confs
            self._goal_stale            = 0
            self._goal_seen_last_image  = True
        else:
            self._goal_stale           += 1
            self._goal_seen_last_image  = False
            # Keep last-known goal bins while NAV_GOAL is active so brief YOLO
            # misses don't hand the wheel back to NoMaD exploration bins.
            # The NAV_GOAL → EXPLORE timeout in _timer_cb is authoritative and
            # fires on the _goal_stale counter, not on bin presence.
            if self._goal_stale >= self._goal_max_stale and self.state != NavState.NAV_GOAL:
                self._goal_bins  = []
                self._goal_confs = []
        if goal_bins:
            self.get_logger().info(
                f"[YOLO] {len(goal_bins)} object(s) → "
                f"bins={goal_bins}, confs={[f'{c:.2f}' for c in goal_confs]}"
            )

    # ── Timer callback: state machine + VFH* + publish ───────────────────────

    def _timer_cb(self) -> None:
        # Re-broadcast last waypoint so PD controller doesn't time-out
        self._republish_waypoint()

        # REACHED is terminal — keep publishing stop
        if self.state == NavState.REACHED:
            self._publish_stop()
            return

        if len(self.context_queue) <= self.context_size:
            return
        if not self._new_depth_available:
            return
        self._new_depth_available = False

        goal_bins  = self._goal_bins  if self._goal_bins  else None
        goal_confs = self._goal_confs if self._goal_bins  else None

        # ── State transitions ─────────────────────────────────────────────
        prev_state = self.state

        # Enter NAV_GOAL on first detection (EXPLORE → NAV_GOAL).
        if self.state == NavState.EXPLORE and goal_bins:
            self.state = NavState.NAV_GOAL
            self.get_logger().info(
                f"[NavSM] EXPLORE → NAV_GOAL  (detected bins={goal_bins})"
            )

        # NAV_GOAL body — runs on the transition tick too, so the REACHED
        # check fires immediately if the object is already within reach.
        if self.state == NavState.NAV_GOAL:
            # Timeout first: fire off _goal_stale (YOLO-miss counter) so the
            # latched memory in _goal_bins cannot suppress the transition.
            if self._goal_stale >= self.goal_timeout_frames:
                self.state = NavState.EXPLORE
                self._goal_bins  = []
                self._goal_confs = []
                self._goal_stale = 0
                self.get_logger().info(
                    f"[NavSM] NAV_GOAL → EXPLORE  "
                    f"(goal lost for {self.goal_timeout_frames} frames)"
                )
            elif goal_bins:
                # Goal-reached: object within goal_reach_distance.
                # Padding bins are filled with 0.0 by design (synthetic blocked
                # boundary) — skip them to avoid a false REACHED trigger.
                valid_lo = self.fov_padding_bins
                valid_hi = self.fov_padding_bins + self.vfh_num_bins - 1
                for gb in goal_bins:
                    if gb < valid_lo or gb > valid_hi:
                        self.get_logger().warn(
                            f"[NavSM] goal bin={gb} is padding "
                            f"(valid FOV: {valid_lo}–{valid_hi}) — skipping REACHED check"
                        )
                        continue
                    actual_dist = self.distance_vector[gb]
                    self.get_logger().info(
                        f"[NavSM] NAV_GOAL  bin={gb}  dist={actual_dist:.2f} m  "
                        f"goal_reach={self.goal_reach_distance:.2f} m"
                    )
                    if actual_dist < self.goal_reach_distance:
                        self.state = NavState.REACHED
                        self.get_logger().info(
                            f"[NavSM] NAV_GOAL → REACHED  "
                            f"(bin={gb}, dist={actual_dist:.2f} m "
                            f"< goal_reach={self.goal_reach_distance:.2f} m)"
                        )
                        self._publish_stop()
                        return

        # Reset temporal aggregator only when leaving NAV_GOAL / entering
        # NAV_GOAL from EXPLORE; skip the no-op reset on the REACHED transition
        # (we already returned above).
        if prev_state != self.state:
            self.temporal_agg.reset()

        # ── NoMaD inference (runs in both EXPLORE and NAV_GOAL) ──────────
        obs_imgs = transform_images(
            list(self.context_queue), self.model_params["image_size"], center_crop=False
        ).to(self.device)
        fake_goal = torch.randn(
            (1, 3, *self.model_params["image_size"]), device=self.device
        )
        mask = torch.ones(1, device=self.device, dtype=torch.long)

        with torch.no_grad():
            obs_cond = self.model(
                "vision_encoder",
                obs_img=obs_imgs,
                goal_img=fake_goal,
                input_goal_mask=mask,
            )
            rep = (
                (lambda x: x.repeat(self.args.num_samples, 1))
                if obs_cond.ndim == 2
                else (lambda x: x.repeat(self.args.num_samples, 1, 1))
            )
            obs_cond = rep(obs_cond)

            naction = torch.randn(
                (self.args.num_samples, self.model_params["len_traj_pred"], 2),
                device=self.device,
            )
            self.noise_scheduler.set_timesteps(self.model_params["num_diffusion_iters"])
            for k in self.noise_scheduler.timesteps:
                noise_pred = self.model(
                    "noise_pred_net", sample=naction, timestep=k, global_cond=obs_cond
                )
                naction = self.noise_scheduler.step(noise_pred, k, naction).prev_sample

        traj_batch = to_numpy(get_action(naction))   # (S, L, 2)

        # Convert all NoMaD trajectories to bins (always needed for EXPLORE
        # and for viz in NAV_GOAL)
        all_ref_bins: List[int] = []
        all_ref_wps             = []
        for i in range(len(traj_batch)):
            wp = traj_batch[i][self.args.waypoint]
            _, rb = waypoint_to_reference(
                float(wp[0]), float(wp[1]),
                num_bins=self.vfh_total_bins,
                fov_deg=self.vfh_virtual_fov,
            )
            all_ref_bins.append(rb)
            all_ref_wps.append(wp)

        # ── Select reference_bins and prepare dist_for_vfh ──────────────
        # VFH* internally clamps any reference bin that falls in the padding
        # zone to `fov_padding_bins` (left) or `vfh_total_bins-1-fov_padding_bins`
        # (right).  We must apply the SAME clamping here so that:
        #   (a) reference_bins we pass are exactly what VFH* will use, and
        #   (b) the inf-masking targets the bin VFH* actually checks.
        valid_lo = self.fov_padding_bins
        valid_hi = self.fov_padding_bins + self.vfh_num_bins - 1

        if self.state == NavState.NAV_GOAL and goal_bins:
            # Clamp YOLO bins into the valid (camera-visible) FOV range
            clamped_goal_bins = [max(valid_lo, min(gb, valid_hi)) for gb in goal_bins]
            reference_bins = clamped_goal_bins
            # Mask the person's bins as clear — VFH* must not treat the
            # detected object as an obstacle to avoid.
            dist_for_vfh = self.distance_vector.copy()
            for gb in clamped_goal_bins:
                dist_for_vfh[gb] = np.inf
        else:
            # EXPLORE or NAV_GOAL with lost detection → NoMaD drives direction
            reference_bins = all_ref_bins
            dist_for_vfh = self.distance_vector

        best_bin, best_angle, was_modified = self.vfh.compute(
            dist_for_vfh, reference_bins
        )

        # chosen_idx — closest NoMaD trajectory index (needed for viz in both
        # states).  chosen_nomad_wp is only meaningful in EXPLORE; in NAV_GOAL
        # the NoMaD direction has no relation to the goal, so we skip it.
        if best_bin in all_ref_bins:
            chosen_idx = all_ref_bins.index(best_bin)
        else:
            chosen_idx = int(np.argmin([abs(b - best_bin) for b in all_ref_bins]))
        chosen_nomad_wp = (
            all_ref_wps[chosen_idx] if self.state == NavState.EXPLORE else None
        )

        # Flush temporal aggregator after a full recovery cycle
        if self.vfh.recovery_just_completed:
            self.temporal_agg.reset()
            self.vfh.recovery_just_completed = False
            self.get_logger().info("[NavVFH*] Flushed temporal aggregator after recovery")

        # ── Depth markers ─────────────────────────────────────────────────
        # In NAV_GOAL the true reference is the clamped goal bin (passed to
        # VFH*); in EXPLORE it's the NoMaD-trajectory bin VFH* selected.
        ref_for_viz = (
            reference_bins[0]
            if self.state == NavState.NAV_GOAL and goal_bins
            else all_ref_bins[chosen_idx]
        )
        self.depth_marker_pub.publish(
            self.distance_vector,
            selected_bin  = best_bin,
            reference_bin = ref_for_viz,
        )

        # ── Bin-ray markers (RViz) ────────────────────────────────────────
        # Detected objects (white): always show YOLO bins if any
        self.detected_ray_pub.publish(self._goal_bins or [])
        # NoMaD reference bins (yellow): always show the NoMaD trajectory fan
        self.nomad_ray_pub.publish(all_ref_bins)
        # Goal reference bins (orange): only while NAV_GOAL is actually steering
        if self.state == NavState.NAV_GOAL and goal_bins:
            self.goal_ray_pub.publish(reference_bins)
        else:
            self.goal_ray_pub.publish([])
        # Chosen bin (green): the direction the waypoint actually points to
        self.chosen_ray_pub.publish([best_bin])

        # ── DEBUG: reference/waypoint tracing ─────────────────────────────
        self.get_logger().info(
            f"[DBG] state={self.state.value}  "
            f"cached_goal_bins={self._goal_bins}  "
            f"yolo_hit_last_frame={self._goal_seen_last_image}  "
            f"nomad_bins={all_ref_bins}  "
            f"ref_bins→VFH*={reference_bins}  "
            f"best_bin={best_bin}  was_modified={was_modified}  "
            f"goal_stale={self._goal_stale}/"
            f"(mem_clear={self._goal_max_stale}, "
            f"timeout={self.goal_timeout_frames})"
        )

        # ── Waypoint generation ───────────────────────────────────────────
        final_wp = self._make_waypoint(
            best_bin, best_angle, was_modified, chosen_nomad_wp,
        )
        self.current_waypoint = final_wp

        self.get_logger().info(
            f"[NavVFH*] state={self.state.value}  modified={was_modified}  "
            f"bin={best_bin}  angle={math.degrees(best_angle):.1f}°  "
            f"wp={final_wp[:2]}"
        )

        self._publish(traj_batch, final_wp, was_modified, chosen_idx)

    # ── Waypoint construction ─────────────────────────────────────────────────

    def _make_waypoint(
        self,
        best_bin: int,
        best_angle: float,
        was_modified: bool,
        chosen_nomad_wp: Optional[np.ndarray],
    ) -> np.ndarray:
        """Return the 2-D (or 4-D) waypoint to send to the PD controller.

        EXPLORE  + unmodified → use NoMaD waypoint directly (preserves speed)
        EXPLORE  + modified   → direction waypoint from VFH* (obstacle avoidance)
        NAV_GOAL              → always a direction waypoint toward best_angle
                                (no valid NoMaD trajectory in goal direction)
        TURN phase            → 4-D heading waypoint [0, 0, cos, sin]
        """
        # Recovery TURN: send heading command (v=0, pure rotation)
        if was_modified and self.vfh._recovery_phase == self.vfh._PHASE_TURN:
            hx = math.cos(best_angle)
            hy = math.sin(best_angle)
            self.get_logger().info(
                f"[NavVFH*] TURN phase: rotating toward {math.degrees(best_angle):.1f}°"
            )
            return np.array([0.0, 0.0, hx, hy])

        if self.state == NavState.EXPLORE and not was_modified:
            # VFH* fast-path confirmed a NoMaD direction — use its waypoint
            return chosen_nomad_wp

        # All other cases: generate a fresh direction waypoint.
        # In NAV_GOAL the NoMaD trajectory has no relation to the goal direction,
        # so use MAX_V as the base speed rather than inheriting NoMaD's magnitude.
        if self.state == NavState.NAV_GOAL:
            magnitude = MAX_V
        else:
            magnitude = np.linalg.norm(chosen_nomad_wp)
            if magnitude < 1e-3:
                magnitude = MAX_V
        wps = generate_direction_waypoints(
            best_angle,
            max_magnitude = magnitude * self.vfh_speed_red,
            num_waypoints = self.vfh_num_wps,
        )
        return wps[min(self.vfh_wp_idx, len(wps) - 1)]

    # ── Publishing ────────────────────────────────────────────────────────────

    def _publish(
        self,
        traj_batch: np.ndarray,
        final_wp: np.ndarray,
        vfh_active: bool,
        selected_idx: int,
    ) -> None:
        # Sampled trajectories for downstream visualisation
        sa_msg = Float32MultiArray()
        sa_msg.data = [0.0] + [float(x) for x in traj_batch.flatten()]
        self.sampled_actions_pub.publish(sa_msg)

        # Waypoint to PD controller (2-D or 4-D)
        wp_msg = Float32MultiArray()
        wp_msg.data = [float(x) for x in final_wp]
        self.waypoint_pub.publish(wp_msg)
        self.get_logger().info(
            f"[DBG/WP] PUBLISH state={self.state.value} "
            f"final_wp={[round(float(x), 3) for x in final_wp]} "
            f"(len={len(final_wp)})"
        )

        # Trajectory overlay image
        self._publish_viz(traj_batch, vfh_active, selected_idx)

    def _publish_viz(
        self,
        traj_batch: np.ndarray,
        vfh_active: bool,
        selected_idx: int,
    ) -> None:
        frame        = np.array(self.context_queue[-1])
        img_h, img_w = frame.shape[:2]
        viz          = frame.copy()
        cx, cy       = img_w // 2, int(img_h * 0.95)
        ppm          = 3.0

        # Origin cross-hair
        cv2.line(viz, (cx - 10, cy), (cx + 10, cy), (255, 0, 0), 2)
        cv2.line(viz, (cx, cy - 10), (cx, cy + 10), (255, 0, 0), 2)

        # NoMaD trajectory overlays
        for i, traj in enumerate(traj_batch):
            pts = [(cx, cy)]
            ax, ay = 0.0, 0.0
            for dx, dy in traj:
                ax += dx; ay += dy
                pts.append((int(cx - ay * ppm), int(cy - ax * ppm)))
            if len(pts) >= 2:
                color = (
                    ((0, 165, 255) if vfh_active else (0, 255, 0))
                    if i == selected_idx
                    else ((0, 100, 200) if vfh_active else (255, 200, 0))
                )
                cv2.polylines(viz, [np.array(pts, np.int32)], False, color, 2)

        # Goal-bin direction arrows (cyan, length ∝ confidence)
        if self._goal_bins:
            fov_rad   = math.radians(self.vfh_virtual_fov)
            bin_width = fov_rad / self.vfh_total_bins
            for gb, gc in zip(self._goal_bins, self._goal_confs):
                angle  = fov_rad / 2 - (gb + 0.5) * bin_width
                length = int(60 * gc)
                ex = int(cx - math.sin(angle) * length)
                ey = int(cy - math.cos(angle) * length)
                cv2.arrowedLine(viz, (cx, cy), (ex, ey), (0, 255, 255), 2, tipLength=0.3)

        # State label
        state_colors = {
            NavState.EXPLORE:  (200, 200, 200),
            NavState.NAV_GOAL: (0, 255, 255),
            NavState.REACHED:  (0, 255, 0),
        }
        label = f"NAV:{self.state.value}"
        if self.state == NavState.NAV_GOAL and self._goal_stale > 0:
            label += f" (lost {self._goal_stale}/{self.goal_timeout_frames})"
        cv2.putText(
            viz, label, (5, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
            state_colors[self.state], 1,
        )

        img_msg = self.bridge.cv2_to_imgmsg(viz, encoding="rgb8")
        img_msg.header.stamp = self.get_clock().now().to_msg()
        self.viz_pub.publish(img_msg)


# ═════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser("NavigationVFH — goal-oriented object navigation")
    parser.add_argument("--model",       "-m", default="nomad")
    parser.add_argument("--waypoint",    "-w", type=int, default=2)
    parser.add_argument("--num-samples", "-n", type=int, default=8)
    parser.add_argument("--robot",       type=str, default="turtlebot4",
                        choices=["locobot", "turtlebot4"])
    parser.add_argument("--config-dir",  type=str, default="deployment/config")
    parser.add_argument("--yolo-weights",  type=str, required=True,
                        help="Path to YOLO .pt weights file")
    parser.add_argument("--yolo-conf",     type=float, default=0.25,
                        help="YOLO confidence threshold (default: 0.25)")
    parser.add_argument("--yolo-classes",  type=int, nargs="*", default=None,
                        help="YOLO class IDs to detect (default: all classes)")
    parser.add_argument("--goal-timeout-frames", type=int, default=3,
                        help="Timer ticks without detection before reverting to EXPLORE "
                             "(default: 3)")
    parser.add_argument("--goal-reach-distance", type=float, default=0.000001,
                        help="Depth (m) at which the goal object is considered reached "
                             "(default: 0.5); must be smaller than VFH* safety_threshold")
    parser.add_argument("--goal-stale-frames", type=int, default=5,
                        help="Consecutive image frames without detection before clearing "
                             "goal bins (default: 3); prevents race-condition blanking")
    args = parser.parse_args()

    rclpy.init()
    node = NavigationVFHNode(args)
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
