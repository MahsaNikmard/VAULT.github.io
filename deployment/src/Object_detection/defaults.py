"""
VfhPlus/defaults.py – Single source of truth for all VFH* parameters.

Every tunable constant lives here.  Individual modules import from this
file instead of hard-coding their own defaults.  The YAML configuration
(``config/vfh.yaml``) overrides these at runtime via the ROS nodes.

Keeping one authoritative location avoids silent drift between modules.
"""

# ── Robot physical dimensions ───────────────────────────────────────────
ROBOT_RADIUS: float = 0.18          # metres – half-width including safety buffer
                                     # TurtleBot4 Create 3: ~34 cm diameter → 17 cm + margin

# ── Angular binning ──────────────────────────────────────────────────────
NUM_BINS: int = 72
FOV_DEG: float = 89.0

# ── Obstacle thresholds ─────────────────────────────────────────────────
SAFETY_THRESHOLD: float = 0.60       # metres – bins closer than this are blocked
MAX_RANGE: float = 3.0              # metres – ignore depth beyond this
VERTICAL_MARGIN: float = 0.10       # metres – ceiling exclusion (Y < -ε)
FLOOR_MARGIN: float = 0.22          # metres – floor exclusion (Y > this)
SAFETY_MARGIN: float = 0.60     # metres – subtracted from Z for clearance
FOV_PADDING_BINS: int = 10           # virtual blocked bins added to each side of FOV
                                     # prevents VFH* from selecting edge valleys

# ── Depth scale ─────────────────────────────────────────────────────────
DEPTH_SCALE: float = 0.80            # multiplicative correction on raw DA2 depth

# ── Temporal aggregation ────────────────────────────────────────────────
TEMPORAL_WINDOW: int = 3

# ── VFH* valley selection ──────────────────────────────────────────────
S_MAX: int = 10

# ── Cost function weights ──────────────────────────────────────────────
MU1: float = 5.0                    # target alignment
MU2: float = 1.0                   # heading alignment
MU3: float = 0.05                   # previous-direction smoothness

# ── Speed / waypoints ──────────────────────────────────────────────────
SPEED_REDUCTION: float = 0.8
NUM_VFH_WAYPOINTS: int = 5
VFH_WAYPOINT_INDEX: int = 0

# ── Recovery behaviour ─────────────────────────────────────────────────
RECOVERY_REVERSE_CYCLES: int = 8    # cycles to back up before turning
RECOVERY_TURN_CYCLES: int = 6       # cycles to rotate ~90° before resuming normal pipeline
