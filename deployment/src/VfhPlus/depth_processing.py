"""
VfhPlus/depth_processing.py – Metric depth map → angular distance vector.

Takes the metric depth map from Depth Anything V2 (in metres) and camera
intrinsics, back-projects each pixel into the camera frame, and produces a
1-D distance vector where each element is the minimum obstacle distance in
the corresponding angular bin of the camera's horizontal FOV.

A :class:`TemporalAggregator` smooths per-frame vectors across a sliding
window, computing per-direction *certainty* and *danger* scores to reduce
oscillations caused by perception noise, robot rotation, and actuation
effects.

Coordinate conventions (camera frame)
--------------------------------------
    X → right
    Y → down
    Z → forward (depth)

The horizontal angle of a point from the forward axis is computed as
``atan2(-X, Z)`` so that positive angles correspond to the robot's left
side, matching the NoMaD waypoint convention ``(dx=forward, dy=left)``.
"""

from __future__ import annotations

import math
from collections import deque

import numpy as np

from VfhPlus.defaults import (
    NUM_BINS, FOV_DEG, MAX_RANGE, VERTICAL_MARGIN,
    FLOOR_MARGIN, SAFETY_MARGIN, DEPTH_SCALE, TEMPORAL_WINDOW,
    SAFETY_THRESHOLD, FOV_PADDING_BINS,
)


# ─────────────────────────────────────────────────────────────────────────────
# FOV padding – add virtual blocked bins to each side of the distance vector
# ─────────────────────────────────────────────────────────────────────────────
def pad_distance_vector(
    distance_vector: np.ndarray,
    padding_bins: int = FOV_PADDING_BINS,
    fill_value: float = 0.0,
) -> np.ndarray:
    """Extend a distance vector with virtual blocked bins on each side.

    The camera has a limited FOV, so VFH* has no information beyond the
    edges.  Adding ``padding_bins`` per side – filled with a value below the
    safety threshold – creates virtual obstacles that prevent the planner
    from choosing directions near the FOV boundary.

    Parameters
    ----------
    distance_vector : ndarray of shape ``(num_bins,)``
        Original per-bin minimum-distance vector from the depth pipeline.
    padding_bins : int
        Number of blocked bins to prepend **and** append.
    fill_value : float
        Distance assigned to padding bins.  Must be below the safety
        threshold so VFH* treats them as blocked.  ``0.0`` is a safe
        default (always below any positive threshold).

    Returns
    -------
    padded : ndarray of shape ``(num_bins + 2 * padding_bins,)``
        Padded distance vector.
    """
    if padding_bins <= 0:
        return distance_vector
    pad = np.full(padding_bins, fill_value, dtype=distance_vector.dtype)
    print(f"Padding distance vector with {padding_bins} bins on each side, fill_value={fill_value}")
    return np.concatenate([pad, distance_vector, pad])


# ─────────────────────────────────────────────────────────────────────────────
# Single-frame depth map → distance vector
# ─────────────────────────────────────────────────────────────────────────────
def compute_distance_vector(
    depth_map: np.ndarray,
    K: np.ndarray,
    *,
    num_bins: int = NUM_BINS,
    fov_deg: float = FOV_DEG,
    max_range: float = MAX_RANGE,
    vertical_margin: float = VERTICAL_MARGIN,
    floor_margin: float = FLOOR_MARGIN,
    safety_margin: float = SAFETY_MARGIN,
    depth_scale: float = DEPTH_SCALE,
) -> np.ndarray:
    """Convert a metric depth map to a per-bin minimum-distance vector.

    Parameters
    ----------
    depth_map : ndarray (H, W)
        Per-pixel depth in metres, as returned by DA2 metric
        ``model.infer_image()``.
    K : ndarray (3, 3)
        Camera intrinsic matrix ``[[fx, 0, cx], [0, fy, cy], [0, 0, 1]]``.
    num_bins : int
        Number of angular bins spanning the FOV.
    fov_deg : float
        Horizontal field of view in degrees.
    max_range : float
        Points farther than this (metres) are discarded (``τ_z``).
    vertical_margin : float
        Ceiling exclusion margin.  Points with ``Y < -vertical_margin``
        are excluded (``ε``).
    floor_margin : float
        Floor exclusion margin (metres, camera Y-down frame).  Points with
        ``Y > floor_margin`` are excluded.  TurtleBot4 OAK-D is ~0.32 m
        above floor; default 0.22 m detects obstacles ≥ 10 cm.
    safety_margin : float
        Conservative clearance subtracted from depth before range computation.
    depth_scale : float
        Multiplicative correction applied to the raw depth map before any
        processing.  Values < 1.0 shrink over-estimated depths;
        values > 1.0 expand under-estimated depths.

    Returns
    -------
    distance_vector : ndarray of shape ``(num_bins,)``
        Minimum obstacle range per bin.  Bins without obstacles are ``np.inf``.
    """
    fov_rad = math.radians(fov_deg)
    half_fov = fov_rad / 2.0
    bin_width = fov_rad / num_bins

    H, W = depth_map.shape
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]

    # --- Apply depth scale correction ---
    scaled_depth = depth_map.astype(np.float64) * depth_scale

    # --- Pixel grid → camera-frame coordinates ---
    u = np.arange(W, dtype=np.float64)
    v = np.arange(H, dtype=np.float64)
    uu, vv = np.meshgrid(u, v)  # (H, W)

    Z = scaled_depth.ravel()
    X = ((uu.ravel() - cx) * Z) / fx   # right-positive
    Y = ((vv.ravel() - cy) * Z) / fy   # down-positive

    # --- Filter (ceiling, floor, range) ---
    mask = (Z > 0) & (Z <= max_range) & (Y >= -vertical_margin) & (Y <= floor_margin)
    n_total = int((Z > 0).sum())
    n_kept = int(mask.sum())
    n_floor = int(((Z > 0) & (Y > floor_margin)).sum())
    print(f"[depth] points: total={n_total}  kept={n_kept}  floor_excluded={n_floor}")
    X_f = X[mask]
    Z_f = Z[mask]

    distance_vector = np.full(num_bins, np.inf, dtype=np.float64)

    if len(X_f) == 0:
        return distance_vector

    # --- Horizontal angle (+=left, camera convention) ---
    # Use actual Z so that off-centre obstacles land in the correct angular bin.
    angles = np.arctan2(-X_f, Z_f)

    # --- Range (Euclidean distance in XZ plane, safety margin subtracted) ---
    # Subtract safety_margin from the actual Euclidean range rather than from
    # the Z component alone.  Subtracting from Z distorts angles and creates
    # a non-uniform scale error (larger for side bins than for forward bins).
    actual_ranges = np.sqrt(X_f ** 2 + Z_f ** 2)
    ranges = np.maximum(actual_ranges - safety_margin, 1e-3)

    # --- Bin each point (bin 0 = leftmost, bin N-1 = rightmost) ---
    bin_indices = ((half_fov - angles) / bin_width).astype(int)
    valid = (bin_indices >= 0) & (bin_indices < num_bins)
    bin_indices = bin_indices[valid]
    ranges = ranges[valid]

    # --- Minimum range per bin (vectorised) ---
    np.minimum.at(distance_vector, bin_indices, ranges)
    print(f"This is distance vector {distance_vector}")

    return distance_vector


# ─────────────────────────────────────────────────────────────────────────────
# Temporal aggregation
# ─────────────────────────────────────────────────────────────────────────────
class TemporalAggregator:
    """Sliding-window filter that smooths per-frame distance vectors.

    Maintains a history of recent distance vectors and computes per-bin
    *certainty* (observation consistency) and *danger* (proximity) scores.
    The filtered output is more stable than any single frame, reducing
    oscillations caused by perception noise, robot rotation, and actuation
    effects.

    Parameters
    ----------
    num_bins : int
        Number of angular bins (must match ``compute_distance_vector``).
    window_size : int
        Number of recent frames kept in the sliding window.
    danger_threshold : float
        Distance (metres) below which a bin is considered *dangerous*.
    """

    def __init__(
        self,
        num_bins: int = NUM_BINS,
        window_size: int = TEMPORAL_WINDOW,
        danger_threshold: float = SAFETY_THRESHOLD,
    ) -> None:
        self.num_bins = num_bins
        self.window_size = window_size
        self.danger_threshold = danger_threshold
        self._history: deque[np.ndarray] = deque(maxlen=window_size)

    def update(self, distance_vector: np.ndarray) -> np.ndarray:
        """Append a new frame and return the temporally filtered vector."""
        self._history.append(distance_vector.copy())
        return self.get_filtered()

    def get_filtered(self) -> np.ndarray:
        """Return the current filtered distance vector without adding a frame."""
        if not self._history:
            return np.full(self.num_bins, np.inf, dtype=np.float64)

        stacked = np.array(list(self._history))  # (T, num_bins)

        # Per-bin minimum across the temporal window (conservative / safety-first).
        # Using median would suppress a real obstacle that only appears in 1-2 frames,
        # since median(inf, inf, inf, inf, 0.06) = inf → not blocked.
        # np.nanmin treats inf normally but ignores NaN; here we use np.min and handle
        # inf explicitly: if ALL frames are inf the bin is truly clear.
        result = np.min(stacked, axis=0)  # inf only if every frame was inf
        return result

    @property
    def certainty(self) -> np.ndarray:
        """Per-bin certainty in [0, 1]: how consistently the bin is blocked
        or free relative to ``danger_threshold``."""
        if not self._history:
            return np.zeros(self.num_bins, dtype=np.float64)
        stacked = np.array(list(self._history))
        blocked = stacked < self.danger_threshold  # (T, N) bool
        frac_blocked = blocked.min(axis=0)
        return np.abs(4.0 * frac_blocked - 1.0)

    @property
    def danger(self) -> np.ndarray:
        """Per-bin danger score in [0, 1]: fraction of recent frames where
        the bin distance was below ``danger_threshold``."""
        if not self._history:
            return np.zeros(self.num_bins, dtype=np.float64)
        stacked = np.array(list(self._history))
        return (stacked < self.danger_threshold).mean(axis=0)

    def reset(self) -> None:
        """Clear all history."""
        self._history.clear()
