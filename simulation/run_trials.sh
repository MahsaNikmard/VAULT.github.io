#!/usr/bin/env bash
# ============================================================================
# run_trials.sh – Repeated-trial evaluation: VAULT vs NoMaD-only
# ============================================================================
# Runs each method N times in the arena and collects the per-trial
# ground-truth safety metrics (true distance travelled, minimum clearance,
# contact count). NoMaD's diffusion policy is stochastic, so multiple trials
# are needed for a mean +/- std comparison.  Results land in results/.
#
#   bash simulation/run_trials.sh                       # 5 trials, headless
#   bash simulation/run_trials.sh --trials 10 --gpu     # 10 trials, GPU (fast)
#   bash simulation/run_trials.sh --duration 120        # longer episodes
#
# Prerequisites: same as run_sim.sh (Webots, ROS 2 Humble, the `vault` env
# active).  Pass --gpu to render on the GPU (~10x faster than software xvfb).
# ============================================================================

set -euo pipefail

TRIALS=5
DURATION=90
GPU_FLAG=""
METHODS=(vault nomad)

while [[ $# -gt 0 ]]; do
  case "$1" in
    --trials)   TRIALS="$2"; shift 2 ;;
    --duration) DURATION="$2"; shift 2 ;;
    --gpu)      GPU_FLAG="--gpu"; shift ;;
    --methods)  read -r -a METHODS <<< "$2"; shift 2 ;;
    -h|--help)  sed -n '2,20p' "$0"; exit 0 ;;
    *) echo "Unknown option: $1" >&2; exit 1 ;;
  esac
done

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
RESULTS="$ROOT/results"
cd "$ROOT"

echo "============================================"
echo "  VAULT repeated-trial evaluation"
echo "  methods  : ${METHODS[*]}"
echo "  trials   : $TRIALS    duration: ${DURATION}s/trial"
echo "  results  : $RESULTS"
echo "============================================"

for method in "${METHODS[@]}"; do
  src_dir="metrics_output"; [[ "$method" == "nomad" ]] && src_dir="metrics_output_nomad"
  out_dir="$RESULTS/trials/$method"
  mkdir -p "$out_dir"
  for n in $(seq 1 "$TRIALS"); do
    echo "── $method trial $n/$TRIALS ──────────────────────────────"
    # Fully tear down any lingering processes from the previous trial before
    # starting the next, so they do not clash on the extern-controller port.
    pkill -9 -f webots 2>/dev/null || true
    pkill -9 -f explore_vfh 2>/dev/null || true
    sleep 6
    rm -rf "$src_dir"
    bash simulation/run_sim.sh --method "$method" --duration "$DURATION" $GPU_FLAG \
      > "$out_dir/trial_${n}.log" 2>&1 || true
    if [[ -f "$src_dir/ground_truth.json" ]]; then
      cp "$src_dir/ground_truth.json" "$out_dir/trial_${n}.json"
      echo "   saved $out_dir/trial_${n}.json"
    else
      echo "   WARNING: no ground_truth.json for $method trial $n" >&2
    fi
  done
done

pkill -9 -f webots 2>/dev/null || true

echo
echo "Aggregating results..."
python3 "$SCRIPT_DIR/aggregate_trials.py" --results "$RESULTS"
