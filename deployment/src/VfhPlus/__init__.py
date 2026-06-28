"""
VfhPlus – Collision avoidance via VFH* on top of a vision-based navigation policy.

Provides a reactive modular layer that modifies every action outputted by a
base navigation policy (NoMaD) by re-checking the chosen direction against a
depth-derived distance vector and applying the VFH* algorithm to select a
safe alternative when the reference direction is blocked.

Modules
-------
vfh_star          Core VFH* planner (polar histogram, valley search, cost function).
depth_processing  DA2 metric depth map → per-bin minimum distance vector +
                  temporal aggregation.
nomad_vector      NoMaD waypoint → binary direction vector + reference bin index.
depth_markers     RViz MarkerArray publisher for distance-vector visualisation.
"""

from VfhPlus import defaults
from VfhPlus.vfh_star import VFHStar
from VfhPlus.depth_processing import compute_distance_vector, TemporalAggregator, pad_distance_vector
from VfhPlus.nomad_vector import waypoint_to_reference, bin_to_waypoint, generate_direction_waypoints
from VfhPlus.depth_markers import DepthMarkerPublisher

__all__ = [
    "defaults",
    "VFHStar",
    "compute_distance_vector",
    "TemporalAggregator",
    "pad_distance_vector",
    "waypoint_to_reference",
    "bin_to_waypoint",
    "generate_direction_waypoints",
    "DepthMarkerPublisher",
]
