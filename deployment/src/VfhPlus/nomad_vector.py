"""
VfhPlus/nomad_vector.py – NoMaD waypoint → direction vector + reference bin.

Converts the 2-D waypoint ``(dx, dy)`` produced by the NoMaD navigation
policy into:

1. A **nomad vector** of dimension ``num_bins``, a binary indicator where the
   bin matching the waypoint's direction is set to 1.
2. The **reference bin** index (integer) suitable for the VFH* planner.

Angle convention
----------------
    dx = forward,   dy = left  (robot frame, consistent with PD controller)
    angle = atan2(dy, dx)      0 = forward, + = left, − = right
"""

from __future__ import annotations

import math
from typing import Tuple

import numpy as np

from VfhPlus.defaults import NUM_BINS, FOV_DEG, NUM_VFH_WAYPOINTS


def waypoint_to_reference(
    dx: float,
    dy: float,
    *,
    num_bins: int = NUM_BINS,
    fov_deg: float = FOV_DEG,
) -> Tuple[np.ndarray, int]:
    """Convert a NoMaD waypoint to a nomad direction vector and reference bin.

    Parameters
    ----------
    dx, dy : float
        Forward and lateral (left-positive) components of the waypoint.
    num_bins : int
        Number of angular bins (must match the distance vector).
    fov_deg : float
        Horizontal FOV in degrees.

    Returns
    -------
    nomad_vector : ndarray of shape ``(num_bins,)``
        Binary vector with a single 1 at the reference bin.
    reference_bin : int
        Index of the bin matching the waypoint direction.
    """
    fov_rad = math.radians(fov_deg)
    half_fov = fov_rad / 2.0
    bin_width = fov_rad / num_bins
    print(f"Waypoint to reference: dx={dx}, dy={dy}, angle={math.degrees(math.atan2(dy, dx)):.1f}°")
    print(f"FOV: {fov_deg}°, Bin width: {math.degrees(bin_width):.1f}°")
    angle = math.atan2(dy, dx)

    # Map angle → bin index (bin 0 = leftmost, bin N-1 = rightmost)
    bin_idx = int((half_fov - angle) / bin_width)
    print(f"Initial bin index (before clamping): {bin_idx}")
    bin_idx = max(0, min(bin_idx, num_bins - 1))
    print(f"Clamped bin index: {bin_idx}")
    nomad_vector = np.zeros(num_bins, dtype=np.float64)
    print(f"Nomad vector before setting bin: {nomad_vector}")
    nomad_vector[bin_idx] = 1.0
    print(f"Nomad vector after setting bin {bin_idx}: {nomad_vector}")

    return nomad_vector, bin_idx


def bin_to_waypoint(
    bin_idx: int,
    magnitude: float,
    *,
    num_bins: int = NUM_BINS,
    fov_deg: float = FOV_DEG,
) -> np.ndarray:
    """Convert a bin index back to a 2-D waypoint ``(dx, dy)``.

    Parameters
    ----------
    bin_idx : int
        Angular bin index.
    magnitude : float
        Desired step length (preserves original waypoint magnitude).
    num_bins, fov_deg :
        Must match the values used in :func:`waypoint_to_reference`.

    Returns
    -------
    waypoint : ndarray of shape ``(2,)``
        ``(dx, dy)`` in robot frame.
    """
    fov_rad = math.radians(fov_deg)
    bin_width = fov_rad / num_bins
    angle = fov_rad / 2 - (bin_idx + 0.5) * bin_width
    bin_to_waypoint = np.array([magnitude * math.cos(angle), magnitude * math.sin(angle)])

    print(f"Bin to waypoint or Nomad Vector {bin_to_waypoint}")

    return bin_to_waypoint


def generate_direction_waypoints(
    angle: float,
    max_magnitude: float,
    *,
    num_waypoints: int = NUM_VFH_WAYPOINTS,
) -> np.ndarray:
    """Generate waypoints at evenly spaced distances along a direction.

    Parameters
    ----------
    angle : float
        Direction angle in radians (0 = forward, + = left, − = right).
    max_magnitude : float
        Distance of the farthest waypoint (typically the original NoMaD
        waypoint magnitude scaled by ``speed_reduction``).
    num_waypoints : int
        Number of waypoints to generate along the direction.

    Returns
    -------
    waypoints : ndarray of shape ``(num_waypoints, 2)``
        ``(dx, dy)`` waypoints ordered from nearest to farthest.
    """
    if num_waypoints < 1:
        num_waypoints = 1
    magnitudes = np.linspace(max_magnitude / num_waypoints, max_magnitude, num_waypoints)
    print(f"Generating {num_waypoints} waypoints with magnitudes: {magnitudes}")
    waypoints = np.column_stack([
        magnitudes * math.cos(angle),
        magnitudes * math.sin(angle),
    ])
    print(f"Generated {num_waypoints} waypoints along angle {math.degrees(angle):.1f}°: {waypoints}")
    # print(f"Waypoints shape: {waypoints.shape}")
    return waypoints