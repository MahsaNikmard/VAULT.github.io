#!/usr/bin/env bash
# ============================================================================
# run_sim.sh – One-command VAULT Webots simulation (headless by default)
# ============================================================================
# Runs the full NoMaD + Depth Anything V2 + VFH* loop against the simulation arena
# world. No GUI and no manual terminal juggling: this script starts Webots,
# the Webots <-> ROS 2 bridge, and the navigation node, then shuts them down
# cleanly on Ctrl+C or when the time budget expires.
#
#   bash simulation/run_sim.sh                 # VAULT, headless, 120 s
#   bash simulation/run_sim.sh --method nomad  # NoMaD-only baseline
#   bash simulation/run_sim.sh --gui           # show the Webots window
#   bash simulation/run_sim.sh --rviz          # open RViz (needs a display)
#   bash simulation/run_sim.sh --world W.wbt   # use a different Webots world
#   bash simulation/run_sim.sh --duration 0    # run until Ctrl+C
#
# Prerequisites (see README): Webots R2023b+, ROS 2 Humble, the `vault` env.
# ============================================================================

set -euo pipefail

# ── Arguments ───────────────────────────────────────────────────────────────
METHOD="vault"        # vault | nomad
GUI=0
GPU=0                 # use the current $DISPLAY (GPU) instead of software-GL xvfb
RVIZ=0                # open RViz with the simulation visualization layout
WORLD_ARG=""          # path to a Webots world; empty = default arena
DURATION=120          # seconds; 0 = run until interrupted

while [[ $# -gt 0 ]]; do
  case "$1" in
    --method)   METHOD="$2"; shift 2 ;;
    --gui)      GUI=1; shift ;;
    --gpu)      GPU=1; shift ;;
    --rviz)     RVIZ=1; shift ;;
    --world)    WORLD_ARG="$2"; shift 2 ;;
    --duration) DURATION="$2"; shift 2 ;;
    -h|--help)  sed -n '2,18p' "$0"; exit 0 ;;
    *) echo "Unknown option: $1" >&2; exit 1 ;;
  esac
done

NAV_ARGS=()
case "$METHOD" in
  vault) NAV_NODE="deployment/src/explore_vfh.py"; CONTROL="vfh";   METRICS_DIR="metrics_output" ;;
  nomad) NAV_NODE="deployment/src/explore_vfh.py"; CONTROL="nomad"; METRICS_DIR="metrics_output_nomad"; NAV_ARGS+=(--baseline) ;;
  *) echo "Unknown method '$METHOD' (use: vault | nomad)" >&2; exit 1 ;;
esac

# ── Paths and environment ───────────────────────────────────────────────────
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
WEBOTS_HOME="${WEBOTS_HOME:-/usr/local/webots}"
if [[ -n "$WORLD_ARG" ]]; then
  [[ "$WORLD_ARG" = /* ]] && WORLD="$WORLD_ARG" || WORLD="$ROOT/$WORLD_ARG"
else
  WORLD="$SCRIPT_DIR/worlds/multi_robot.wbt"
fi
CONFIG_DIR="simulation/config"
RVIZ_CFG="$SCRIPT_DIR/config/webots_rviz.rviz"

export WEBOTS_HOME
export PYTHONPATH="$ROOT/deployment/src:$ROOT/simulation/src:$ROOT/train:$ROOT/third_party:${PYTHONPATH:-}"
export ROS_DOMAIN_ID="${ROS_DOMAIN_ID:-7}"
export PYTHONUNBUFFERED=1
export VAULT_METRICS_DIR="$ROOT/$METRICS_DIR"
# Webots derives the extern-controller IPC path from the username. In a bare
# container USER is unset, so Webots and webots-controller pick different paths
# and never connect (the robot then never moves). Pin a consistent value.
export USER="${USER:-root}"

cd "$ROOT"

echo "============================================"
echo "  VAULT – Webots simulation"
echo "  method   : $METHOD ($NAV_NODE)"
echo "  display  : $([ $GUI -eq 1 ] && echo GUI || echo headless)$([ $RVIZ -eq 1 ] && echo ' + RViz')"
echo "  duration : $([ "$DURATION" -eq 0 ] && echo 'until Ctrl+C' || echo "${DURATION}s")"
echo "============================================"

# ── Graceful shutdown ───────────────────────────────────────────────────────
PIDS=()
cleanup() {
  trap - INT TERM EXIT
  echo
  echo "[shutdown] stopping nodes..."
  # SIGINT first so the ROS nodes save metrics and close their context cleanly.
  for pid in "${PIDS[@]:-}"; do kill -INT "$pid" 2>/dev/null || true; done
  sleep 3
  for pid in "${PIDS[@]:-}"; do kill "$pid" 2>/dev/null || true; done
  pkill -f webots-controller 2>/dev/null || true
  pkill -f "webots.*$(basename "$WORLD")" 2>/dev/null || true
  echo "[shutdown] done."
}
trap cleanup INT TERM EXIT

# ── 1. Webots ───────────────────────────────────────────────────────────────
echo "[1/3] starting Webots..."
if [[ $GUI -eq 1 ]]; then
  "$WEBOTS_HOME/webots" "$WORLD" &
elif [[ $GPU -eq 1 ]]; then
  # Render on the current $DISPLAY (GPU). ~10x faster than software-GL xvfb,
  # at the cost of requiring a GPU-backed X server (or EGL).
  "$WEBOTS_HOME/webots" --batch --stdout --stderr --no-rendering --minimize \
    --mode=fast "$WORLD" &
else
  xvfb-run -a -s "-screen 0 1280x720x24" \
    "$WEBOTS_HOME/webots" --batch --stdout --stderr --no-rendering --minimize \
    --mode=fast "$WORLD" &
fi
PIDS+=($!)
sleep 20

# ── 2. Webots <-> ROS 2 bridge (integrated PD control) ──────────────────────
echo "[2/3] starting Webots controller bridge..."
"$WEBOTS_HOME/webots-controller" --robot-name=turtlebot4 \
  "$SCRIPT_DIR/src/webots_controller.py" \
  --robot turtlebot4 --control "$CONTROL" --config-dir "$CONFIG_DIR" &
PIDS+=($!)
sleep 8

# ── 3. Navigation node (NoMaD + DA2 + VFH*) ─────────────────────────────────
echo "[3/3] starting navigation node..."
python3 "$NAV_NODE" --robot turtlebot4 --model nomad --config-dir "$CONFIG_DIR" "${NAV_ARGS[@]}" &
PIDS+=($!)

# ── Optional RViz (depth fan, NoMaD/VFH* bins, camera overlay) ───────────────
if [[ $RVIZ -eq 1 ]]; then
  if [[ -z "${DISPLAY:-}" ]]; then
    echo "[rviz] no DISPLAY set; skipping RViz (use --gui or --gpu with a display)."
  else
    echo "[rviz] opening RViz..."
    rviz2 -d "$RVIZ_CFG" &
    PIDS+=($!)
  fi
fi

echo "[ready] simulation running."
if [[ "$DURATION" -eq 0 ]]; then
  wait "${PIDS[-1]}"
else
  sleep "$DURATION"
fi
