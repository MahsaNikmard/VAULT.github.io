
import math
import pathlib
from typing import Optional

import numpy as np
import yaml



# ── YOLO object detection ─────────────────────────────────────────────────────

def load_yolo_model(model_path: str, device: str | None = None, offline: bool = True):
    """
    Load a YOLO model from a weights file using the ultralytics library.

    Parameters
    ----------
    model_path : str
        Absolute path to a local YOLO weights file (e.g. /path/to/yolov8n.pt).
        When offline=True the file must already exist on disk; no download is
        attempted and a clear FileNotFoundError is raised if it is missing.
    device : str or None
        'cuda', 'cpu', or None to auto-select (prefers CUDA when available).
    offline : bool
        When True (default) the YOLO_OFFLINE env-var is set before importing
        ultralytics, preventing any network access even if the model name looks
        like a remotely fetchable name.

    Returns
    -------
    model : ultralytics.YOLO
        Loaded YOLO model ready for inference.
    device : str
        The device string actually used.
    """
    import os
    import torch

    if offline:
        os.environ["YOLO_OFFLINE"] = "true"
    model_path = "/home/mahsa/yolov8n.pt"
    # Fail fast with a clear message if the file does not exist locally.
    if not pathlib.Path(model_path).is_file():
        raise FileNotFoundError(
            f"YOLO weights not found: '{model_path}'\n"
            "Copy a .pt file to that path before starting perception.\n"
            "On a machine with internet access run:\n"
            "  python3 -c \"from ultralytics import YOLO; YOLO('yolov8n.pt')\"\n"
            "then copy ~/.config/Ultralytics/... or the downloaded yolov8n.pt here."
        )

    from ultralytics import YOLO

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    model = YOLO(model_path)
    model.to(device)
    return model, device


def detect_objects_per_bin(
    image: np.ndarray,
    model,
    n_bins: int,
    fov_deg: float,
    conf: float = 0.5,
    classes: list[int] | None = None,
) -> np.ndarray:
    """
    Run YOLO inference on the full image and map detections to N angular bins.

    Each detected bounding box is projected onto the horizontal axis of the
    image.  Every bin whose horizontal span overlaps the box is set to 1.0.
    This preserves the same N-element binary vector contract as the old
    color-detection pipeline so all downstream nodes (mission, navigation,
    controller) remain unchanged.

    Bin ordering matches image_to_polar_histogram:
        bin 0 = leftmost strip, bin N-1 = rightmost.

    Parameters
    ----------
    image : np.ndarray
        RGB image, shape (H, W, 3), uint8.  Converted to BGR internally before
        YOLO inference (ultralytics uses OpenCV / BGR convention).
    model : ultralytics.YOLO
        Loaded YOLO model.
    n_bins : int
        Number of angular bins (must match N in robot.yml).
    fov_deg : float
        Horizontal field-of-view in degrees (used only to keep API symmetric
        with image_to_polar_histogram; bin mapping uses pixel coordinates).
    conf : float
        Minimum confidence threshold for a detection to be accepted.
    classes : list of int or None
        YOLO class IDs to retain (e.g. [0] for 'person').  None = all classes.

    Returns
    -------
    detection_vec : np.ndarray, shape (N,)
        Binary detection vector — 1.0 if an object is detected in the bin,
        0.0 otherwise.
    """
    W = image.shape[1]
    bin_width_px = W / n_bins
    detection_vec = np.zeros(n_bins, dtype=np.float32)

    # Ultralytics YOLO expects BGR (OpenCV convention); image arrives as RGB.
    bgr = image[:, :, ::-1]

    results = model(
        bgr,
        conf=conf,
        classes=classes if classes else None,
        verbose=False,
    )

    if results and results[0].boxes is not None and len(results[0].boxes):
        boxes_xyxy = results[0].boxes.xyxy.cpu().numpy()
        confs      = results[0].boxes.conf.cpu().numpy()
        cls_ids    = results[0].boxes.cls.cpu().numpy().astype(int)
        for (x1, _y1, x2, _y2), score, cid in zip(boxes_xyxy, confs, cls_ids):
            bin_lo = max(0,          int(x1 / bin_width_px))
            bin_hi = min(n_bins - 1, int(x2 / bin_width_px))
            detection_vec[bin_lo : bin_hi + 1] = 1.0
            print(
                f"YOLO detection: class={cid} conf={score:.2f}  "
                f"x=[{x1:.0f},{x2:.0f}px] → bins [{bin_lo},{bin_hi}]"
            )
    else:
        print("YOLO: no detections this frame")

    return detection_vec


def detect_objects_with_confidence(
    image: np.ndarray,
    model,
    num_bins: int,
    fov_deg: float,
    conf_threshold: float = 0.25,
    classes: list[int] | None = None,
) -> tuple[list[int], list[float]]:
    """Run YOLO and return one (center_bin, confidence) pair per detection.

    Unlike detect_objects_per_bin, this returns individual detections so the
    VFH* cost function can weight each goal bin by detection confidence.

    Bin ordering matches VFH* convention:
        bin 0 = leftmost (left edge of image), bin N-1 = rightmost.

    Parameters
    ----------
    image : np.ndarray
        RGB image (H, W, 3) uint8.
    model : ultralytics.YOLO
        Loaded YOLO model.
    num_bins : int
        Total bin count passed to VFH* (including FOV padding bins).
    fov_deg : float
        Horizontal FOV in degrees (kept for API symmetry; bin mapping uses pixels).
    conf_threshold : float
        Minimum confidence to accept a detection.
    classes : list of int or None
        YOLO class IDs to retain.  None = all classes.

    Returns
    -------
    goal_bins : list of int
        Center-bin index of each accepted detection.
    goal_confidences : list of float
        Corresponding YOLO confidence scores in (0, 1].
    """
    W = image.shape[1]
    bin_width_px = W / num_bins

    bgr = image[:, :, ::-1]
    results = model(bgr, conf=conf_threshold, classes=classes, verbose=False)

    goal_bins: list[int] = []
    goal_confidences: list[float] = []

    if results and results[0].boxes is not None and len(results[0].boxes):
        boxes_xyxy = results[0].boxes.xyxy.cpu().numpy()
        confs       = results[0].boxes.conf.cpu().numpy()
        cls_ids     = results[0].boxes.cls.cpu().numpy().astype(int)

        for (x1, _y1, x2, _y2), score, cid in zip(boxes_xyxy, confs, cls_ids):
            u_center = (x1 + x2) / 2.0
            bin_idx  = int(u_center / bin_width_px)
            bin_idx  = max(0, min(bin_idx, num_bins - 1))
            goal_bins.append(bin_idx)
            goal_confidences.append(float(score))
            print(
                f"[YOLO] class={cid} conf={score:.2f}  "
                f"x=[{x1:.0f},{x2:.0f}px]  center_bin={bin_idx}"
            )
    else:
        print("[YOLO] no detections this frame")

    return goal_bins, goal_confidences


# ── ROS2 node ─────────────────────────────────────────────────────────────────

def _load_config() -> dict:
    cfg_path = pathlib.Path(__file__).parents[1] / "config" / "robot.yml"
    with open(cfg_path) as f:
        return yaml.safe_load(f)





#------------------------------Rviz node

try:
    from rclpy.node import Node as _ROSNode
    from rclpy.time import Time
    from visualization_msgs.msg import Marker, MarkerArray
    from geometry_msgs.msg import Point, Vector3
    from std_msgs.msg import ColorRGBA
    from builtin_interfaces.msg import Duration
    _ROS_AVAILABLE = True
except ImportError:
    _ROS_AVAILABLE = False


class ObjectMarkerPublisher:

    def __init__(
        self,
        node,
        topic: str = "/vfh/detected_objects_markers",
        num_bins: int = 72,
        fov_deg: float = 90.0,
        frame_id: str = "base_link",
    ) -> None:
        self._node = node
        self._pub = node.create_publisher(MarkerArray, topic, 10)
        self.num_bins = num_bins
        self.fov_rad = math.radians(fov_deg)
        self.frame_id = frame_id

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def publish(
        self,
        yolo_vector: np.ndarray,
        stamp: Optional[Time] = None,
        selected_bin: Optional[int] = None,
        detected_bin: Optional[int] = None,
    ) -> None:
        """Build and publish the marker array.

        Parameters
        ----------
        yolo_vector : ndarray (num_bins,)
            Per-bin minimum obstacle distance.
        stamp : rclpy.time.Time, optional
            Header timestamp; defaults to ``node.get_clock().now()``.
        selected_bin : int, optional
            Bin chosen by VFH* (drawn in cyan).
        detected_bin : int, optional
            Bin requested by NoMaD (drawn in magenta).
        """
        if stamp is None:
            stamp = self._node.get_clock().now()

        ma = MarkerArray()
        ts = stamp.to_msg()

        half_fov = self.fov_rad / 2.0
        bin_width = self.fov_rad / self.num_bins
        lifetime = Duration(sec=0, nanosec=int(0.5e9))  # 500 ms

        # ── Distance arrows ──────────────────────────────────────────────
        for i in range(self.num_bins):
            # Angle in base_link: 0 = forward (+x), positive = left (+y)
            # Bin 0 = leftmost (most positive angle), bin N-1 = rightmost
            angle = half_fov - (i + 0.5) * bin_width

            detec = yolo_vector[i]
            is_obj = detec < 0.5  # no detection in this bin
            plot_dist = 1

            m = Marker()
            m.header.frame_id = self.frame_id
            m.header.stamp = ts
            m.ns = "detection_vector"
            m.id = i
            m.type = Marker.ARROW
            m.action = Marker.ADD
            m.lifetime = lifetime

            # Arrow from origin along the bin direction
            start = Point(x=0.0, y=0.0, z=0.1)  # slight z lift
            end = Point(
                x=plot_dist * math.cos(angle),
                y=plot_dist * math.sin(angle),
                z=0.1,
            )
            m.points = [start, end]

            # Shaft / head thickness
            m.scale = Vector3(x=0.02, y=0.04, z=0.04)

            # Colour: override for selected / reference bins
            if i == selected_bin:
                color = ColorRGBA(r=0.0, g=1.0, b=1.0, a=1.0)  # cyan
            elif i == detected_bin:
                color = ColorRGBA(r=1.0, g=0.0, b=1.0, a=0.9)  # magenta
            elif is_obj:
                color = ColorRGBA(r=0.5, g=0.5, b=0.5, a=0.2)  # grey ghost
            else:
                color = self._distance_color()
            m.color = color

            ma.markers.append(m)

        # # ── Safety ring ──────────────────────────────────────────────────
        # if self.show_safety_ring:
        #     n_ring = self.num_bins * 2  # denser ring
        #     ring_bin_width = self.fov_rad / n_ring
        #     for j in range(n_ring):
        #         angle = half_fov - (j + 0.5) * ring_bin_width
        #         m = Marker()
        #         m.header.frame_id = self.frame_id
        #         m.header.stamp = ts
        #         m.ns = "safety_ring"
        #         m.id = j
        #         m.type = Marker.SPHERE
        #         m.action = Marker.ADD
        #         m.lifetime = lifetime
        #         m.pose.position.x = self.safety_threshold * math.cos(angle)
        #         m.pose.position.y = self.safety_threshold * math.sin(angle)
        #         m.pose.position.z = 0.1
        #         m.scale = Vector3(x=0.03, y=0.03, z=0.03)
        #         m.color = ColorRGBA(r=1.0, g=0.0, b=0.0, a=0.35)
        #         ma.markers.append(m)

        self._pub.publish(ma)

    # ------------------------------------------------------------------
    # Colour helpers
    # ------------------------------------------------------------------
    def _distance_color(self) -> ColorRGBA:
        """Green (far) → yellow → red (close) colour ramp."""
        # t = max(0.0, min(1.0, dist / self.max_range))  # 0=close, 1=far
        s = 1
        r = 1.0 - s
        g = 1.0

        return ColorRGBA(r=r, g=g, b=0.0, a=0.9)
