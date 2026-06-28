"""
VfhPlus/depth_markers.py – RViz MarkerArray publisher for distance vectors.

Publishes the per-bin minimum-distance vector as a fan of arrow / cylinder
markers in the ``base_link`` frame so RViz can visualise what the depth
pipeline sees at every inference step.

Marker layout (top-down, base_link frame)
-----------------------------------------
- Each angular bin is drawn as an **arrow** originating at the robot.
- Arrow length = measured distance for that bin (capped at ``max_range``).
- Colour ramp: green (far / safe) → yellow → red (close / dangerous).
- Bins at ``inf`` (no obstacle detected) are drawn as short transparent
  arrows to keep the fan shape visible.
- An optional **safety ring** of thin red cylinders shows the
  ``safety_threshold`` radius.

The publisher is frame-rate independent: call :meth:`publish` from whatever
callback drives your depth inference.
"""

from __future__ import annotations

import math
from typing import Optional

import numpy as np

from rclpy.node import Node
from rclpy.time import Time
from visualization_msgs.msg import Marker, MarkerArray
from std_msgs.msg import ColorRGBA
from geometry_msgs.msg import Point, Vector3
from builtin_interfaces.msg import Duration

from VfhPlus.defaults import NUM_BINS, FOV_DEG, MAX_RANGE, SAFETY_THRESHOLD


class DepthMarkerPublisher:
    """Publishes a ``visualization_msgs/MarkerArray`` representing the
    distance vector as a fan of arrows in ``base_link``.

    Parameters
    ----------
    node : rclpy.node.Node
        Parent ROS node (used to create the publisher and read the clock).
    topic : str
        Topic name for the MarkerArray.
    num_bins : int
        Number of angular bins (must match the distance vector).
    fov_deg : float
        Horizontal FOV in degrees.
    max_range : float
        Maximum plotting distance for arrows (metres).
    safety_threshold : float
        Distance at which colour turns fully red.
    frame_id : str
        TF frame for the markers.
    show_safety_ring : bool
        Whether to draw a ring of small markers at ``safety_threshold``.
    """

    def __init__(
        self,
        node: Node,
        topic: str = "/vfh/depth_markers",
        num_bins: int = NUM_BINS,
        fov_deg: float = FOV_DEG,
        max_range: float = MAX_RANGE,
        safety_threshold: float = SAFETY_THRESHOLD,
        frame_id: str = "base_link",
        show_safety_ring: bool = True,
    ) -> None:
        self._node = node
        self._pub = node.create_publisher(MarkerArray, topic, 10)
        self.num_bins = num_bins
        self.fov_rad = math.radians(fov_deg)
        self.max_range = max_range
        self.safety_threshold = safety_threshold
        self.frame_id = frame_id
        self.show_safety_ring = show_safety_ring

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def publish(
        self,
        distance_vector: np.ndarray,
        stamp: Optional[Time] = None,
        selected_bin: Optional[int] = None,
        reference_bin: Optional[int] = None,
    ) -> None:
        """Build and publish the marker array.

        Parameters
        ----------
        distance_vector : ndarray (num_bins,)
            Per-bin minimum obstacle distance.
        stamp : rclpy.time.Time, optional
            Header timestamp; defaults to ``node.get_clock().now()``.
        selected_bin : int, optional
            Bin chosen by VFH* (drawn in cyan).
        reference_bin : int, optional
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

            dist = distance_vector[i]
            is_inf = np.isinf(dist)
            plot_dist = min(dist, self.max_range) if not is_inf else self.max_range * 0.15

            m = Marker()
            m.header.frame_id = self.frame_id
            m.header.stamp = ts
            m.ns = "depth_vector"
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
            elif i == reference_bin:
                color = ColorRGBA(r=1.0, g=0.0, b=1.0, a=0.9)  # magenta
            elif is_inf:
                color = ColorRGBA(r=0.5, g=0.5, b=0.5, a=0.2)  # grey ghost
            else:
                color = self._distance_color(dist)
            m.color = color

            ma.markers.append(m)

        # ── Safety ring ──────────────────────────────────────────────────
        if self.show_safety_ring:
            n_ring = self.num_bins * 2  # denser ring
            ring_bin_width = self.fov_rad / n_ring
            for j in range(n_ring):
                angle = half_fov - (j + 0.5) * ring_bin_width
                m = Marker()
                m.header.frame_id = self.frame_id
                m.header.stamp = ts
                m.ns = "safety_ring"
                m.id = j
                m.type = Marker.SPHERE
                m.action = Marker.ADD
                m.lifetime = lifetime
                m.pose.position.x = self.safety_threshold * math.cos(angle)
                m.pose.position.y = self.safety_threshold * math.sin(angle)
                m.pose.position.z = 0.1
                m.scale = Vector3(x=0.03, y=0.03, z=0.03)
                m.color = ColorRGBA(r=1.0, g=0.0, b=0.0, a=0.35)
                ma.markers.append(m)

        self._pub.publish(ma)

    # ------------------------------------------------------------------
    # Colour helpers
    # ------------------------------------------------------------------
    def _distance_color(self, dist: float) -> ColorRGBA:
        """Green (far) → yellow → red (close) colour ramp."""
        t = max(0.0, min(1.0, dist / self.max_range))  # 0=close, 1=far
        if t > 0.5:
            # far half: green → yellow
            s = (t - 0.5) * 2.0
            r = 1.0 - s
            g = 1.0
        else:
            # close half: yellow → red
            s = t * 2.0
            r = 1.0
            g = s
        return ColorRGBA(r=r, g=g, b=0.0, a=0.9)


class BinRayMarkerPublisher:
    """Publish a MarkerArray of colored point-rays along selected bin directions.

    Each call draws one SPHERE_LIST per listed bin (``num_points`` spheres
    evenly spaced from the robot origin to ``ray_length`` along that bin's
    angle).  Useful for visualising sparse bin sets: NoMaD reference bins,
    YOLO goal bins, or the single chosen VFH* output.
    """

    def __init__(
        self,
        node: Node,
        topic: str,
        num_bins: int,
        fov_deg: float,
        color: tuple,                 # (r, g, b, a) in [0, 1]
        frame_id: str = "base_link",
        num_points: int = 10,
        ray_length: float = 2.0,
        point_size: float = 0.06,
        marker_ns: str = "bin_rays",
    ) -> None:
        self._node = node
        self._pub = node.create_publisher(MarkerArray, topic, 10)
        self.num_bins = num_bins
        self.fov_rad = math.radians(fov_deg)
        self.frame_id = frame_id
        self.color = ColorRGBA(r=color[0], g=color[1], b=color[2], a=color[3])
        self.num_points = max(1, num_points)
        self.ray_length = ray_length
        self.point_size = point_size
        self.marker_ns = marker_ns

    def publish(self, bin_indices, stamp: Optional[Time] = None) -> None:
        if stamp is None:
            stamp = self._node.get_clock().now()
        ts = stamp.to_msg()

        half_fov = self.fov_rad / 2.0
        bin_width = self.fov_rad / self.num_bins
        lifetime = Duration(sec=0, nanosec=int(0.5e9))

        ma = MarkerArray()

        # Always emit a DELETEALL so stale rays from the previous call
        # disappear when the set of bins shrinks.
        clear = Marker()
        clear.header.frame_id = self.frame_id
        clear.header.stamp = ts
        clear.ns = self.marker_ns
        clear.action = Marker.DELETEALL
        ma.markers.append(clear)

        step = self.ray_length / self.num_points
        for idx, b in enumerate(bin_indices):
            angle = half_fov - (b + 0.5) * bin_width
            m = Marker()
            m.header.frame_id = self.frame_id
            m.header.stamp = ts
            m.ns = self.marker_ns
            m.id = idx
            m.type = Marker.SPHERE_LIST
            m.action = Marker.ADD
            m.lifetime = lifetime
            m.scale = Vector3(x=self.point_size, y=self.point_size, z=self.point_size)
            m.color = self.color
            pts = []
            for k in range(1, self.num_points + 1):
                r = step * k
                pts.append(Point(
                    x=r * math.cos(angle),
                    y=r * math.sin(angle),
                    z=0.1,
                ))
            m.points = pts
            ma.markers.append(m)

        self._pub.publish(ma)
