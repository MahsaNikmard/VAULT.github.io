
"""
VfhPlus/vfh_star.py – Core VFH* collision avoidance algorithm.

Implements the Vector Field Histogram Star (VFH*) algorithm for reactive
obstacle avoidance.  Given a distance vector (minimum obstacle distance per
angular bin from depth sensing) and a reference direction (from the NoMaD
navigation policy), the algorithm selects a safe travel direction that
minimises a cost function balancing target alignment, heading continuity,
and previous-direction smoothness.

Reference
---------
I. Ulrich and J. Borenstein, "VFH*: Local Obstacle Avoidance with
Look-Ahead Verification", Proc. IEEE Int. Conf. Robotics and Automation
(ICRA), San Francisco, CA, 2000.
"""

from __future__ import annotations

import math
from typing import List, Optional, Tuple

import numpy as np

from defaults import (
    NUM_BINS, FOV_DEG, SAFETY_THRESHOLD, S_MAX, MU1, MU2, MU3,
    ROBOT_RADIUS, RECOVERY_REVERSE_CYCLES, RECOVERY_TURN_CYCLES,
    FOV_PADDING_BINS,
)


class VFHStar:
    """VFH* planner operating on a 1-D polar histogram.

    Parameters
    ----------
    num_bins : int
        Number of angular sectors spanning the camera FOV.
    fov_deg : float
        Horizontal field of view (degrees).
    safety_threshold : float
        Minimum clearance (metres).  Bins whose closest obstacle is nearer
        than this value are marked *blocked*.
    s_max : int
        Wide-valley threshold.  Valleys wider than ``s_max`` bins produce
        edge + centre candidate directions; narrower valleys yield only the
        centre.
    mu1, mu2, mu3 : float
        Cost-function weights for target alignment, heading alignment, and
        previous-direction smoothness respectively.  ``mu1 > mu2 + mu3``
        is recommended to keep the robot target-oriented.
    robot_radius : float
        Effective half-width of the robot (metres), including safety buffer.
        Valleys whose physical gap is smaller than ``2 * robot_radius`` at
        the distance of the nearest obstacle on the valley edges are
        discarded so the robot never attempts to squeeze through openings
        it cannot physically fit.
    recovery_reverse_cycles : int
        Number of inference cycles the robot backs up before returning to
        normal operation.  After reversing, the normal NoMaD → VFH*
        pipeline resumes direction selection.
    """

    # Recovery phases
    _PHASE_NORMAL = 0
    _PHASE_REVERSE = 1
    _PHASE_TURN = 2

    def __init__(
        self,
        num_bins: int = NUM_BINS,
        fov_deg: float = FOV_DEG,
        safety_threshold: float = SAFETY_THRESHOLD,
        s_max: int = S_MAX,
        mu1: float = MU1,
        mu2: float = MU2,
        mu3: float = MU3,
        robot_radius: float = ROBOT_RADIUS,
        recovery_reverse_cycles: int = RECOVERY_REVERSE_CYCLES,
        recovery_turn_cycles: int = RECOVERY_TURN_CYCLES,
        fov_padding_bins: int = FOV_PADDING_BINS,
    ) -> None:
        self.num_bins = num_bins
        self.fov_rad = math.radians(fov_deg)
        self.bin_width = self.fov_rad / num_bins
        self.safety_threshold = safety_threshold
        self.s_max = s_max
        self.mu1 = mu1
        self.mu2 = mu2
        self.mu3 = mu3
        self.robot_radius = robot_radius
        self.fov_padding_bins = fov_padding_bins

        # Recovery configuration
        self.recovery_reverse_cycles = recovery_reverse_cycles
        self.recovery_turn_cycles = recovery_turn_cycles

        # Recovery state
        self._recovery_phase: int = self._PHASE_NORMAL
        self._recovery_reverse_count: int = 0
        self._recovery_turn_count: int = 0
        self._recovery_turn_angle: float = 0.0
        self.recovery_just_completed: bool = False

        self.prev_selected_bin: Optional[int] = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def compute(
        self,
        distance_vector: np.ndarray,
        reference_bins: List[int],
        robot_heading_bin: Optional[int] = None,
    ) -> Tuple[int, float, bool]:
        """Select a safe travel direction.

        Parameters
        ----------
        distance_vector : ndarray of shape ``(num_bins,)``
            Minimum obstacle distance per angular bin.  Use ``np.inf`` for
            bins with no obstacles detected.
        reference_bins : list of int
            Bin indices of all candidate directions from NoMaD (one per
            trajectory sample).  VFH* selects the safest among them, so
            trajectory selection and collision avoidance happen jointly.
        robot_heading_bin : int or None
            Bin corresponding to the current forward heading.  Defaults to
            the centre bin (straight ahead).

        Returns
        -------
        best_bin : int
            Selected bin index.
        best_angle : float
            Corresponding angle in radians (0 = forward, + = left, − = right).
        was_modified : bool
            ``True`` if the selected direction is not among the reference bins.
        """
        if robot_heading_bin is None:
            robot_heading_bin = self.num_bins // 2

        # 1) Binary polar histogram
        blocked = distance_vector < self.safety_threshold
        print(f"[VFH*] safety_threshold={self.safety_threshold}  robot_radius={self.robot_radius}")
        print(f"The binary polar distance vector is {blocked} which is directly from the thresholding, no temporal and no padding yet")

        # 1b) Mark FOV padding bins as blocked on both sides
        if self.fov_padding_bins > 0:
            blocked[:self.fov_padding_bins] = True
            blocked[-self.fov_padding_bins:] = True
            print(f"[VFH*] Padded {self.fov_padding_bins} bins on each side → blocked: {blocked}")

        # 1c) Clamp all reference bins into the valid (non-padding) range
        if self.fov_padding_bins > 0:
            lo = self.fov_padding_bins
            hi = self.num_bins - 1 - self.fov_padding_bins
            clamped = []
            for rb in reference_bins:
                if rb < lo:
                    print(f"[VFH*] reference_bin {rb} in left padding → clamped to {lo}")
                    clamped.append(lo)
                elif rb > hi:
                    print(f"[VFH*] reference_bin {rb} in right padding → clamped to {hi}")
                    clamped.append(hi)
                else:
                    clamped.append(rb)
            reference_bins = clamped

        # Primary reference: median bin (used for candidate generation direction bias)
        primary_ref = int(np.median(reference_bins))

        # 2) Fast path: collect all reference bins that are free AND have clearance
        safe_refs = [
            rb for rb in reference_bins
            if not blocked[rb] and self._bin_has_clearance(rb, blocked, distance_vector)
        ]
        print(f"[VFH*] reference_bins={reference_bins}, safe_refs={safe_refs}")

        if safe_refs:
            self._reset_recovery()
            prev_bin = self.prev_selected_bin if self.prev_selected_bin is not None else primary_ref
            best_bin = self._select_best(safe_refs, reference_bins, robot_heading_bin, prev_bin)
            self.prev_selected_bin = best_bin
            bin_to_angle = self._bin_to_angle(best_bin)
            print(f"[VFH*] Fast path: best_bin={best_bin}, angle={math.degrees(bin_to_angle):.1f}°")
            return best_bin, bin_to_angle, False

        print(f"[VFH*] No safe reference bin found — searching valleys")

        # 3) Find free valleys (contiguous unblocked sectors)
        all_valleys = self._find_valleys(blocked)
        print(f"The reference idx was blocked and this is the vallyes: {all_valleys}")

        if not all_valleys:
            print(f"No free valleys at all – triggering recovery")
            return self._recover(distance_vector)

        # 3b) Prefer valleys wide enough for the robot body;
        #     trigger recovery if none qualify.
        wide_valleys = self._filter_narrow_valleys(all_valleys, distance_vector)
        if wide_valleys:
            self._reset_recovery()
            valleys = wide_valleys
            print(f"Valleys after robot-width filtering: {valleys}")
        else:
            return self._recover(distance_vector)

        # 4) Generate candidate directions from valleys
        candidates = self._generate_candidates(valleys, primary_ref)
        print(f"The candidates of the valleys are {candidates}")
        if not candidates:
            print(f"No candidate chosen from valleys! :(")
            return primary_ref, self._bin_to_angle(primary_ref), True

        # 4b) Filter candidates that lack robot-body clearance from valley edges
        safe_candidates = [
            c for c in candidates
            if self._bin_has_clearance(c, blocked, distance_vector)
        ]
        if safe_candidates:
            print(f"Candidates after clearance filtering: {safe_candidates} (from {candidates})")
            candidates = safe_candidates
        else:
            print(f"[VFH*] All candidates fail clearance check — triggering recovery")
            return self._recover(distance_vector)

        # 5) Select best candidate via cost function
        prev_bin = (
            self.prev_selected_bin
            if self.prev_selected_bin is not None
            else primary_ref
        )
        print(f"Previous selected bin: {prev_bin} (primary_ref if None)")

        best_bin = self._select_best(
            candidates, reference_bins, robot_heading_bin, prev_bin
        )
        print(f"prev_bin={prev_bin}, best_bin={best_bin}, angle={math.degrees(self._bin_to_angle(best_bin)):.1f}°")
        self.prev_selected_bin = best_bin
        return best_bin, self._bin_to_angle(best_bin), True

    # ------------------------------------------------------------------
    # Conversion helpers
    # ------------------------------------------------------------------
    def angle_to_bin(self, angle: float) -> int:
        """Map angle (rad, 0=forward, +=left) → bin index clamped to [0, N).

        Bin 0 = leftmost (most positive angle), bin N-1 = rightmost."""
        idx = int((self.fov_rad / 2 - angle) / self.bin_width)
        return max(0, min(idx, self.num_bins - 1))

    def _bin_to_angle(self, bin_idx: int) -> float:
        """Map bin index → angle (rad, 0=forward, +=left, −=right).

        Bin 0 = leftmost (+half_fov), bin N-1 = rightmost (−half_fov)."""
        return self.fov_rad / 2 - (bin_idx + 0.5) * self.bin_width

    # ------------------------------------------------------------------
    # Internal algorithm steps
    # ------------------------------------------------------------------
    def _find_valleys(self, blocked: np.ndarray) -> List[Tuple[int, int]]:
        """Return list of ``(start, end)`` inclusive indices of free runs."""
        valleys: List[Tuple[int, int]] = []
        n = len(blocked)
        i = 0
        while i < n:
            if not blocked[i]:
                start = i
                while i < n and not blocked[i]:
                    i += 1
                valleys.append((start, i - 1))
            else:
                i += 1
        return valleys

    def _is_padding_bin(self, idx: int) -> bool:
        """Return True if *idx* falls inside the FOV padding zone."""
        if self.fov_padding_bins <= 0:
            return False
        return idx < self.fov_padding_bins or idx >= self.num_bins - self.fov_padding_bins

    def _gap_width(
        self, n_bins: int, distance_vector: np.ndarray, left_idx: int, right_idx: int
    ) -> float:
        """Compute the physical gap width (metres) of a valley.

        Uses the minimum obstacle distance at the two edges to estimate the
        narrowest cross-section the robot would have to pass through.
        If an edge borders the FOV boundary **or a padding bin** (no real
        obstacle on that side), we fall back to the minimum distance
        *inside* the valley instead.

        Physical width ≈ ``d · 2 · tan(angular_width / 2)``
        """
        angular_width = n_bins * self.bin_width
        print(f"Calculating gap width for valley ({left_idx}, {right_idx}) with angular width {math.degrees(angular_width):.1f}°")

        # Distance at the blocking edges (just outside the valley),
        # but SKIP edges that sit inside the virtual padding zone —
        # those are not real obstacles and would collapse the gap to 0.
        edge_dists: List[float] = []
        if left_idx > 0 and not self._is_padding_bin(left_idx - 1):
            edge_dists.append(float(distance_vector[left_idx - 1]))

        if right_idx < self.num_bins - 1 and not self._is_padding_bin(right_idx + 1):
            edge_dists.append(float(distance_vector[right_idx + 1]))

        # Use the mean distance *inside* the valley as the depth at which the
        # robot will pass through.  Edge obstacle distances are the walls that
        # define the opening but are often much closer (e.g. a nearby wall on
        # one side), which causes the formula to severely underestimate the
        # physical gap width at the actual passage depth.
        valley_dists = distance_vector[left_idx : right_idx + 1]
        finite_valley = valley_dists[np.isfinite(valley_dists)]
        if len(finite_valley) > 0:
            d = float(np.mean(finite_valley))
        elif edge_dists:
            d = min(edge_dists)
        else:
            d = self.safety_threshold

        return d * 2.0 * math.tan(angular_width / 2.0)

    def _filter_narrow_valleys(
        self,
        valleys: List[Tuple[int, int]],
        distance_vector: np.ndarray,
    ) -> List[Tuple[int, int]]:
        """Remove valleys whose physical gap is too narrow for the robot.

        A valley is kept only if its gap width ≥ ``2 * robot_radius``.
        """
        robot_diameter = 2.0 * self.robot_radius
        kept: List[Tuple[int, int]] = []
        for start, end in valleys:
            n_bins = end - start + 1
            gap = self._gap_width(n_bins, distance_vector, start, end)
            if gap >= robot_diameter + 0.50:
                kept.append((start, end))
            else:
                print(f"  Valley ({start},{end}) rejected: gap={gap:.3f}m < robot_diameter={robot_diameter:.3f}m")
        return kept

    def _bin_has_clearance(
        self, bin_idx: int, blocked: np.ndarray, distance_vector: np.ndarray
    ) -> bool:
        """Check the reference bin has enough room on *both* sides.

        Padding bins are virtual (they mark space outside the camera FOV, not
        real obstacles) so they do not consume margin.  Walk outwards past
        padding and stop only at a real obstacle or the array boundary, then
        require the full robot-body margin on each side that is actually
        bounded by a real obstacle.  Sides bounded only by padding / array
        edge are treated as unbounded — the robot will reveal that space as
        it rotates toward the chosen direction.
        """

        def is_real_obstacle(idx: int) -> bool:
            return bool(blocked[idx]) and not self._is_padding_bin(idx)

        # Walk left until a real obstacle (or array edge).  Padding is skipped.
        start = bin_idx
        left_real = False
        while start > 0:
            prev = start - 1
            if is_real_obstacle(prev):
                left_real = True
                break
            start -= 1

        # Walk right until a real obstacle (or array edge).
        end = bin_idx
        right_real = False
        while end < self.num_bins - 1:
            nxt = end + 1
            if is_real_obstacle(nxt):
                right_real = True
                break
            end += 1

        # Gap-width sanity check only makes sense when real obstacles bound
        # the valley — otherwise the valley is physically open.
        if left_real or right_real:
            n_bins = end - start + 1
            gap = self._gap_width(n_bins, distance_vector, start, end)
            robot_diameter = 2.0 * self.robot_radius
            if gap < robot_diameter + 0.5:
                return False

        # Angular margin the robot body subtends at local depth around bin_idx.
        neighbourhood = distance_vector[max(0, bin_idx - 1) : bin_idx + 2]
        finite_vals = neighbourhood[np.isfinite(neighbourhood)]
        d = float(np.mean(finite_vals)) if len(finite_vals) > 0 else self.safety_threshold
        margin_angle = math.atan2(self.robot_radius, d)
        margin_bins = max(1, math.ceil(margin_angle / self.bin_width))

        left_margin = bin_idx - start
        right_margin = end - bin_idx
        left_ok  = (not left_real)  or left_margin  >= margin_bins
        right_ok = (not right_real) or right_margin >= margin_bins
        ok = left_ok and right_ok
        if not ok:
            print(f"[VFH*] Bin {bin_idx} clearance fail: "
                  f"valley=({start},{end}), "
                  f"L_margin={left_margin} (real={left_real}), "
                  f"R_margin={right_margin} (real={right_real}), "
                  f"need={margin_bins} "
                  f"(d={d:.2f}m, robot_r={self.robot_radius}m)")
        return ok

    # ------------------------------------------------------------------
    # Recovery state machine
    # ------------------------------------------------------------------
    def _reset_recovery(self) -> None:
        """Reset recovery state — called when a safe valley is found."""
        if self._recovery_phase != self._PHASE_NORMAL:
            print("[VFH* RECOVERY] Recovery complete — resuming normal operation.")
            self.recovery_just_completed = True
        self._recovery_phase = self._PHASE_NORMAL
        self._recovery_reverse_count = 0
        self._recovery_turn_count = 0
        self._recovery_turn_angle = 0.0

    def _pick_turn_direction(self, distance_vector: np.ndarray) -> float:
        """Choose turn angle (+π/2 left, −π/2 right) toward the more open side."""
        mid = self.num_bins // 2
        left_clearance = float(np.sum(distance_vector[:mid]))
        right_clearance = float(np.sum(distance_vector[mid:]))
        # Bins 0..mid-1 are left (positive angles), mid..N-1 are right
        if left_clearance >= right_clearance:
            print(f"[VFH* RECOVERY] Turning LEFT (left_clearance={left_clearance:.2f} >= right={right_clearance:.2f})")
            return math.pi / 2.0
        else:
            print(f"[VFH* RECOVERY] Turning RIGHT (right_clearance={right_clearance:.2f} > left={left_clearance:.2f})")
            return -math.pi / 2.0

    def _recover(
        self,
        distance_vector: np.ndarray,
    ) -> Tuple[int, float, bool]:
        """Back up, then turn ~90° toward the open side, then resume.

        State machine:
          NORMAL → REVERSE (back up for ``recovery_reverse_cycles``)
                 → TURN   (rotate ~90° for ``recovery_turn_cycles``)
                 → NORMAL  (let the next ``compute()`` run the full
                            NoMaD → distance-vector → VFH* pipeline)

        If the pipeline still finds no safe valley after resuming, a new
        recovery cycle starts automatically.

        Returns ``(bin, angle, was_modified)`` consumed by the caller's
        waypoint generation → PD controller.
        """
        # ── Enter REVERSE on first trigger ────────────────────────────
        if self._recovery_phase == self._PHASE_NORMAL:
            self._recovery_phase = self._PHASE_REVERSE
            self._recovery_reverse_count = 0
            print(f"[VFH* RECOVERY] No safe valley — backing up for "
                  f"{self.recovery_reverse_cycles} cycles, then turning")

        # ── REVERSE phase: move backward ──────────────────────────────
        if self._recovery_phase == self._PHASE_REVERSE:
            self._recovery_reverse_count += 1
            if self._recovery_reverse_count > self.recovery_reverse_cycles:
                # Done reversing → switch to TURN
                self._recovery_phase = self._PHASE_TURN
                self._recovery_turn_count = 0
                self._recovery_turn_angle = self._pick_turn_direction(distance_vector)
                print(f"The robot should rotate {self._recovery_turn_angle}  ")
                print(f"[VFH* RECOVERY] Reverse done — starting {self.recovery_turn_cycles}-cycle turn")
            else:
                print(f"[VFH* RECOVERY] Reversing — step "
                      f"{self._recovery_reverse_count}/{self.recovery_reverse_cycles}")
                return self.num_bins // 2, math.pi, True

        # ── TURN phase: rotate in place ~90° ──────────────────────────
        if self._recovery_phase == self._PHASE_TURN:
            self._recovery_turn_count += 1
            if self._recovery_turn_count > self.recovery_turn_cycles:
                # Done turning → resume normal pipeline
                print("[VFH* RECOVERY] Turn complete — resuming normal pipeline")
                self._reset_recovery()
                return self.num_bins // 2, 0.0, False
            else:
                print(f"[VFH* RECOVERY] Turning — step "
                      f"{self._recovery_turn_count}/{self.recovery_turn_cycles}")
                return self.num_bins // 2, self._recovery_turn_angle, True

    def _generate_candidates(
        self, valleys: List[Tuple[int, int]], reference_bin: int
    ) -> List[int]:
        """Produce candidate bin indices from each valley.

        Narrow valleys (< s_max) → centre only.
        Wide valleys (≥ s_max)   → near-edge + far-edge biased by s_max/2, plus centre.
        """
        candidates: List[int] = []
        half_s = self.s_max // 2

        for start, end in valleys:
            width = end - start + 1
            if width < self.s_max:
                candidates.append((start + end) // 2)
            else:
                # Determine which edge is nearer to the reference direction
                if abs(start - reference_bin) <= abs(end - reference_bin):
                    near, far = start, end
                else:
                    near, far = end, start

                # Candidate at half_s from each edge (clamped inside valley)
                c_near = near + half_s if near == start else near - half_s
                c_far = far - half_s if far == end else far + half_s
                c_near = max(start, min(c_near, end))
                c_far = max(start, min(c_far, end))

                candidates.append(c_near)
                candidates.append(c_far)
                candidates.append((start + end) // 2)

        # Clamp and deduplicate
        clamped = list({max(0, min(c, self.num_bins - 1)) for c in candidates})
        return clamped

    def _select_best(
        self,
        candidates: List[int],
        reference_bins: List[int],
        heading_bin: int,
        prev_bin: int,
    ) -> int:
        """Pick the candidate bin minimising the VFH* cost function.

        g(c) = μ₁·min_r Δ(c, r) + μ₂·Δ(c, heading) + μ₃·Δ(c, prev)

        μ₁ uses the minimum distance to *any* NoMaD reference bin so that
        all trajectory samples contribute equally to target alignment.
        """
        best_cost = float("inf")
        best = candidates[0]
        for c in candidates:
            target_cost = min(self._bin_distance(c, r) for r in reference_bins)
            cost = (
                self.mu1 * target_cost
                + self.mu2 * self._bin_distance(c, heading_bin)
                + self.mu3 * self._bin_distance(c, prev_bin)
            )
            if cost < best_cost:
                best_cost = cost
                best = c
        return best

    def _bin_distance(self, a: int, b: int) -> int:
        """Circular distance between two bins."""
        diff = abs(a - b)
        return min(diff, self.num_bins - diff)
