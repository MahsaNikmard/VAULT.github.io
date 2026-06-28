#!/usr/bin/env python3
"""
aggregate_trials.py – Summarise repeated-trial ground-truth metrics.

Reads results/trials/<method>/trial_*.json (written by run_trials.sh) and
reports, per method, the mean +/- std of:
    - safe traversal distance  (true metres travelled in the episode)
    - minimum clearance        (closest approach to any obstacle, m)
    - contact count            (clearance below the contact margin)

Writes results/trend_results.csv and results/trend_results.md, and prints a
table to stdout. The headline comparison is traversal distance: VAULT should
travel substantially farther than the NoMaD-only baseline before stalling.
"""

import argparse
import csv
import json
import math
from pathlib import Path


def _mean_std(values):
    vals = [v for v in values if v is not None and not math.isinf(v)]
    if not vals:
        return float("nan"), float("nan")
    mean = sum(vals) / len(vals)
    var = sum((v - mean) ** 2 for v in vals) / len(vals)
    return mean, math.sqrt(var)


def _load(method_dir: Path):
    rows = []
    for f in sorted(method_dir.glob("trial_*.json")):
        with open(f) as fh:
            rows.append(json.load(fh))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", required=True, help="results/ directory")
    args = ap.parse_args()

    trials_dir = Path(args.results) / "trials"
    methods = sorted(d.name for d in trials_dir.iterdir() if d.is_dir())

    def _fmt(mean, std, prec):
        # Time and distance to first collision are only defined for trials that
        # actually collided; show n/a when no trial in the group did.
        if mean != mean:  # NaN
            return "n/a"
        return f"{mean:.{prec}f} ± {std:.{prec}f}"

    table = []
    for method in methods:
        rows = _load(trials_dir / method)
        if not rows:
            continue
        path_m, path_s = _mean_std([float(r["traversed_path_m"]) for r in rows])
        col_m, col_s = _mean_std([float(r["collisions"]) for r in rows])
        ttf_m, ttf_s = _mean_std([r["time_to_first_collision_s"] for r in rows])
        dtf_m, dtf_s = _mean_std([r["distance_to_first_collision_m"] for r in rows])
        table.append({
            "method": method,
            "n": len(rows),
            "path_mean": path_m, "path_std": path_s,
            "collisions_mean": col_m, "collisions_std": col_s,
            "ttfc_mean": ttf_m, "ttfc_std": ttf_s,
            "dtfc_mean": dtf_m, "dtfc_std": dtf_s,
        })

    # ── stdout ──────────────────────────────────────────────────────────────
    hdr = (f"{'method':8} {'n':>3}  {'path (m)':>14}  {'collisions':>13}  "
           f"{'time-to-1st (s)':>16}  {'dist-to-1st (m)':>16}")
    print("\n" + "=" * len(hdr))
    print("  REPEATED-TRIAL RESULTS  (mean +/- std)")
    print("=" * len(hdr))
    print(hdr)
    print("-" * len(hdr))
    for r in table:
        print(f"{r['method']:8} {r['n']:>3}  "
              f"{_fmt(r['path_mean'], r['path_std'], 2):>14}  "
              f"{_fmt(r['collisions_mean'], r['collisions_std'], 2):>13}  "
              f"{_fmt(r['ttfc_mean'], r['ttfc_std'], 1):>16}  "
              f"{_fmt(r['dtfc_mean'], r['dtfc_std'], 2):>16}")
    print("=" * len(hdr) + "\n")

    # ── CSV ─────────────────────────────────────────────────────────────────
    out = Path(args.results)
    with open(out / "trend_results.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(table[0].keys()))
        w.writeheader()
        w.writerows(table)

    # ── Markdown (for the README) ───────────────────────────────────────────
    with open(out / "trend_results.md", "w") as f:
        f.write("| Method | Trials | Traversed path (m) | Collisions | "
                "Time to first collision (s) | Distance to first collision (m) |\n")
        f.write("|--------|:------:|:------------------:|:----------:|"
                ":---------------------------:|:-------------------------------:|\n")
        for r in table:
            f.write(f"| {r['method']} | {r['n']} | "
                    f"{_fmt(r['path_mean'], r['path_std'], 2)} | "
                    f"{_fmt(r['collisions_mean'], r['collisions_std'], 2)} | "
                    f"{_fmt(r['ttfc_mean'], r['ttfc_std'], 1)} | "
                    f"{_fmt(r['dtfc_mean'], r['dtfc_std'], 2)} |\n")
    print(f"Saved {out/'trend_results.csv'} and {out/'trend_results.md'}")


if __name__ == "__main__":
    main()
