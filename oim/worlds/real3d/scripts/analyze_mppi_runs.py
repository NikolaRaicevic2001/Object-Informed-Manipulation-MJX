#!/usr/bin/env python3
"""Turn run JSONs into a scene-by-scene table, a failure-mode diagnosis and plots.

Reads run files written by `oim.utils.results.save_run` and derives everything
from `dynamic.object_pose` vs `static.goal` -- nothing here is read back from a
stored metric, matching that module's "a run file is evidence, metrics are
recomputed" split.

    # one sweep
    python scripts/analyze_mppi_runs.py oim/results/sweeps/A_asis/manifest.tsv \
        -o oim/results/sweeps/A_asis

    # two sweeps side by side (before/after a change)
    python scripts/analyze_mppi_runs.py \
        oim/results/sweeps/A_asis/manifest.tsv \
        oim/results/sweeps/C_parity/manifest.tsv \
        -o oim/results/sweeps/compare

    # or point it straight at run files
    python scripts/analyze_mppi_runs.py oim/results/runs/pusht3d_xarm6_mock_*_mppi_*.json

Outputs (in -o, default alongside the first input):
    summary.md   markdown table + per-run failure-mode diagnosis
    summary.csv  the same numbers, machine-readable
    curves.png   pos_err / theta_err against control step, one column per scene
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys
from typing import Any, Dict, List, Optional

import numpy as np

# Scene order: easy -> hard, the order the sweep runs them in.
SCENE_ORDER = [
    "open_table",
    "single_obstacle",
    "shelf_gap",
    "ycb_clutter",
    "icra_sign",
]

# Matches the exact-zero signature MuJoCo stiction produces (the same 1e-4 the
# flat sim loop uses to decide a step made no progress).
STALL_EPS = 1e-4
# Fraction of the total error reduction that has to be in hand before a run is
# called plateaued at that step.
PLATEAU_FRAC = 0.98


def wrap_angle(a: np.ndarray) -> np.ndarray:
    """Wrap an angle (or array of them) into [-pi, pi)."""
    return (a + np.pi) % (2.0 * np.pi) - np.pi


# ----------------------------------------------------------------------
# loading
# ----------------------------------------------------------------------


def read_manifest(path: str) -> List[Dict[str, str]]:
    """Rows of a sweep manifest written by scripts/mppi_sweep.sh."""
    rows = []
    with open(path) as f:
        header = f.readline().rstrip("\n").split("\t")
        for line in f:
            if not line.strip():
                continue
            values = line.rstrip("\n").split("\t")
            rows.append(dict(zip(header, values)))
    return rows


def collect_inputs(paths: List[str]) -> List[Dict[str, Optional[str]]]:
    """Resolve CLI arguments into `{variant, scene, json, log}` entries.

    Accepts sweep manifests, run JSONs, and shell globs of either.
    """
    entries: List[Dict[str, Optional[str]]] = []
    for raw in paths:
        for path in sorted(glob.glob(raw)) or [raw]:
            if path.endswith(".tsv"):
                for row in read_manifest(path):
                    if not row.get("result_json"):
                        # A failed or JSON-less run still deserves a row in the
                        # table -- an empty result is a finding, not a gap.
                        entries.append({
                            "variant": row.get("variant", "?"),
                            "scene": row.get("scene", "?"),
                            "json": None,
                            "log": row.get("log"),
                            "status": row.get("status", "?"),
                            "elapsed_s": row.get("elapsed_s"),
                        })
                        continue
                    entries.append({
                        "variant": row.get("variant", "?"),
                        "scene": row.get("scene", "?"),
                        "json": row["result_json"],
                        "log": row.get("log"),
                        "status": row.get("status", "ok"),
                        "elapsed_s": row.get("elapsed_s"),
                    })
            elif path.endswith(".json"):
                entries.append({"variant": None, "scene": None, "json": path,
                                "log": None, "status": "ok", "elapsed_s": None})
            else:
                print(f"skipping unrecognised input: {path}", file=sys.stderr)
    return entries


# ----------------------------------------------------------------------
# metrics
# ----------------------------------------------------------------------


def analyse(payload: Dict[str, Any]) -> Dict[str, Any]:
    """Derive every reported number for one run file."""
    dyn = payload["dynamic"]
    static = payload["static"]
    hyp = payload.get("hyperparameters", {})
    run = payload.get("run", {})

    pose = np.asarray(dyn["object_pose"], dtype=float)     # (n, 3)
    goal = np.asarray(static["goal"], dtype=float)
    pos_err = np.linalg.norm(pose[:, :2] - goal[:2], axis=1)
    theta_err = np.abs(wrap_angle(pose[:, 2] - goal[2]))

    pos_tol = float(static.get("goal_pos_tol", 0.05))
    theta_tol = float(static.get("goal_theta_tol", 0.05))

    ok = (pos_err < pos_tol) & (theta_err < theta_tol)
    reached_at = int(np.argmax(ok)) if ok.any() else None

    # Scale-free progress measure: both errors in units of their own tolerance,
    # so "how far from done" means the same thing on both axes.
    norm = np.maximum(pos_err / pos_tol, theta_err / theta_tol)
    run_min = np.minimum.accumulate(norm)
    total_gain = float(norm[0] - run_min[-1])
    if total_gain > 1e-9:
        target = norm[0] - PLATEAU_FRAC * total_gain
        plateau = int(np.argmax(run_min <= target))
    else:
        plateau = 0  # never improved at all

    n_steps = len(pos_err) - 1
    d_pos = np.abs(np.diff(pos_err))
    d_theta = np.abs(np.diff(theta_err))
    frozen = (d_pos < STALL_EPS) & (d_theta < STALL_EPS)
    # Longest consecutive frozen stretch, in control steps.
    longest = best = 0
    for f in frozen:
        best = best + 1 if f else 0
        longest = max(longest, best)

    def series(key: str) -> Optional[np.ndarray]:
        return np.asarray(dyn[key], dtype=float) if key in dyn else None

    contact = series("robot_contact_force")
    tilt = series("tip_tilt")
    fz = series("contact_normal_force_z")
    solve = series("compute_time")

    res: Dict[str, Any] = {
        "scene": run.get("task"),
        "algorithm": run.get("algorithm"),
        "backend": run.get("backend"),
        "seed": run.get("seed"),
        "samples": hyp.get("samples"),
        "horizon": hyp.get("horizon"),
        "vel_limit": hyp.get("vel_limit"),
        "exact_twist": hyp.get("exact_twist"),
        "steps_cap": hyp.get("steps"),
        "steps_run": n_steps,
        "reached": reached_at is not None,
        "reached_at": reached_at,
        "pos_err_final": float(pos_err[-1]),
        "theta_err_final": float(theta_err[-1]),
        "pos_err_min": float(pos_err.min()),
        "pos_err_min_at": int(pos_err.argmin()),
        "theta_err_min": float(theta_err.min()),
        "theta_err_min_at": int(theta_err.argmin()),
        "plateau_at": plateau,
        "plateau_frac": plateau / max(n_steps, 1),
        "frozen_frac": float(frozen.mean()) if frozen.size else 0.0,
        "longest_freeze": longest,
        "solve_ms": float(solve.mean() * 1e3) if solve is not None and solve.size else None,
        "obstacle_hit_steps": int((contact > 1e-6).sum()) if contact is not None else None,
        "obstacle_hit_max_N": float(contact.max()) if contact is not None and contact.size else None,
        "tip_tilt_mean_deg": float(np.degrees(tilt.mean())) if tilt is not None and tilt.size else None,
        "top_ride_steps": int((np.abs(fz) > 0.5).sum()) if fz is not None else None,
        "costs": hyp.get("costs", {}),
        "_pos_err": pos_err,
        "_theta_err": theta_err,
        "_pos_tol": pos_tol,
        "_theta_tol": theta_tol,
    }
    res["failure_mode"], res["diagnosis"] = classify(res)
    return res


def classify(m: Dict[str, Any]) -> tuple:
    """Assign one primary failure mode and a one-line reason.

    The four modes are the handoff document's own: stall, orientation,
    obstacle collision, wrong contact point -- plus success and a residual
    "not converged" for a run that is still moving when the step cap hits.
    """
    if m["reached"]:
        return "SUCCESS", f"step {m['reached_at']}에서 목표 도달"

    pos_ok = m["pos_err_final"] < m["_pos_tol"]
    theta_ok = m["theta_err_final"] < m["_theta_tol"]
    plateaued_early = m["plateau_frac"] < 0.6
    long_freeze = m["longest_freeze"] >= 100
    hits = m["obstacle_hit_steps"] or 0
    top_ride = m["top_ride_steps"] or 0

    notes = []
    if long_freeze:
        notes.append(f"최장 {m['longest_freeze']}스텝 완전 정지")
    if plateaued_early:
        notes.append(
            f"스텝 {m['plateau_at']}/{m['steps_run']}에서 개선 종료"
        )
    if hits > 0.05 * max(m["steps_run"], 1):
        notes.append(f"로봇-장애물 접촉 {hits}스텝(max {m['obstacle_hit_max_N']:.1f}N)")
    if top_ride > 0.2 * max(m["steps_run"], 1):
        notes.append(f"팁이 블록 윗면 타는 구간 {top_ride}스텝")
    if m["tip_tilt_mean_deg"] is not None and m["tip_tilt_mean_deg"] > 25.0:
        notes.append(f"평균 팁 기울기 {m['tip_tilt_mean_deg']:.0f}deg")
    reason = "; ".join(notes) if notes else "스텝 상한까지 계속 움직였으나 미도달"

    if pos_ok and not theta_ok:
        return "THETA", f"위치는 수렴(θ만 {m['theta_err_final']:.3f}rad) -- {reason}"
    if theta_ok and not pos_ok:
        return "POS", f"자세는 맞았으나 위치 {m['pos_err_final']:.3f}m -- {reason}"
    if hits > 0.15 * max(m["steps_run"], 1):
        return "OBSTACLE", reason
    if top_ride > 0.3 * max(m["steps_run"], 1):
        return "CONTACT", reason
    if long_freeze or plateaued_early:
        return "STALL", reason
    return "SLOW", reason


# ----------------------------------------------------------------------
# reporting
# ----------------------------------------------------------------------

COLUMNS = [
    ("variant", "variant"),
    ("scene", "scene"),
    ("reached", "reached"),
    ("pos_err_final", "pos_f[m]"),
    ("theta_err_final", "th_f[rad]"),
    ("pos_err_min", "pos_min"),
    ("theta_err_min", "th_min"),
    ("plateau_at", "plateau"),
    ("steps_run", "steps"),
    ("longest_freeze", "freeze"),
    ("obstacle_hit_steps", "obs_hit"),
    ("solve_ms", "solve[ms]"),
    ("failure_mode", "mode"),
]


def fmt(value: Any) -> str:
    if value is None:
        return "-"
    if isinstance(value, bool):
        return "Y" if value else "N"
    if isinstance(value, float):
        return f"{value:.3f}"
    return str(value)


def write_report(rows: List[Dict[str, Any]], out_dir: str) -> str:
    lines = ["# flat MPPI (mock, real3d driver) -- scene sweep", ""]

    header = "| " + " | ".join(label for _, label in COLUMNS) + " |"
    rule = "|" + "|".join("---" for _ in COLUMNS) + "|"
    lines += [header, rule]
    for r in rows:
        lines.append("| " + " | ".join(fmt(r.get(key)) for key, _ in COLUMNS) + " |")

    lines += ["", "## 설정", ""]
    seen = set()
    for r in rows:
        key = (r.get("variant"), r.get("samples"), r.get("horizon"),
               r.get("vel_limit"), r.get("backend"), r.get("seed"))
        if key in seen:
            continue
        seen.add(key)
        lines.append(
            f"- **{r.get('variant')}**: samples={r.get('samples')}, "
            f"horizon={r.get('horizon')}, vel_limit={r.get('vel_limit')}, "
            f"backend={r.get('backend')}, seed={r.get('seed')}, "
            f"steps_cap={r.get('steps_cap')}"
        )

    lines += ["", "## 실패모드 진단", ""]
    for r in rows:
        lines.append(
            f"- **{r.get('variant')} / {r.get('scene')}** -- "
            f"`{r.get('failure_mode')}`: {r.get('diagnosis')}"
        )

    lines += [
        "",
        "## 판정 기준",
        "",
        "- `reached` = pos_err < goal_pos_tol 이고 theta_err < goal_theta_tol 인 스텝이 하나라도 있음",
        f"- `plateau` = 전체 오차 감소분의 {PLATEAU_FRAC:.0%}가 이미 달성된 첫 스텝 (정체 시작점)",
        f"- `freeze` = |Δpos_err| < {STALL_EPS} 이고 |Δtheta_err| < {STALL_EPS} 인 최장 연속 구간",
        "- `obs_hit` = robot_contact_force > 0 인 스텝 수 (로봇-장애물 접촉)",
        "- 모드: SUCCESS / THETA(회전만 실패) / POS(병진만 실패) / STALL(정체) / "
        "OBSTACLE(장애물 충돌) / CONTACT(접촉지점 이상) / SLOW(느리지만 진행 중)",
        "",
    ]

    path = os.path.join(out_dir, "summary.md")
    with open(path, "w") as f:
        f.write("\n".join(lines))

    csv_path = os.path.join(out_dir, "summary.csv")
    keys = [k for k, _ in COLUMNS] + [
        "reached_at", "plateau_frac", "frozen_frac", "tip_tilt_mean_deg",
        "top_ride_steps", "samples", "horizon", "vel_limit", "seed", "backend",
    ]
    with open(csv_path, "w") as f:
        f.write(",".join(keys) + "\n")
        for r in rows:
            f.write(",".join(str(r.get(k, "")) for k in keys) + "\n")
    return path


def write_plots(rows: List[Dict[str, Any]], out_dir: str) -> Optional[str]:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not available -- skipping curves.png", file=sys.stderr)
        return None

    scenes = sorted({r["scene"] for r in rows if r.get("scene")},
                    key=lambda s: (SCENE_ORDER.index(s) if s in SCENE_ORDER else 99, s))
    if not scenes:
        return None
    variants = sorted({r.get("variant") for r in rows})

    fig, axes = plt.subplots(2, len(scenes), figsize=(3.6 * len(scenes), 6.0),
                             squeeze=False, sharex=True)
    for col, scene in enumerate(scenes):
        for variant in variants:
            match = [r for r in rows
                     if r.get("scene") == scene and r.get("variant") == variant
                     and "_pos_err" in r]
            for r in match:
                label = str(variant)
                axes[0][col].plot(r["_pos_err"], lw=1.2, label=label)
                axes[1][col].plot(r["_theta_err"], lw=1.2, label=label)
                axes[0][col].axvline(r["plateau_at"], ls=":", lw=0.8, alpha=0.5)
                axes[1][col].axvline(r["plateau_at"], ls=":", lw=0.8, alpha=0.5)
        axes[0][col].axhline(0.05, color="k", ls="--", lw=0.8)
        axes[1][col].axhline(0.05, color="k", ls="--", lw=0.8)
        axes[0][col].set_title(scene, fontsize=10)
        axes[0][col].set_yscale("log")
        axes[1][col].set_xlabel("control step")
    axes[0][0].set_ylabel("pos_err [m]")
    axes[1][0].set_ylabel("theta_err [rad]")
    handles, labels = axes[0][0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="upper right", fontsize=8)
    fig.tight_layout()
    path = os.path.join(out_dir, "curves.png")
    fig.savefig(path, dpi=140)
    plt.close(fig)
    return path


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("inputs", nargs="+",
                   help="sweep manifest(s) (.tsv) and/or run file(s) (.json)")
    p.add_argument("-o", "--out-dir", default=None,
                   help="where to write summary.md / summary.csv / curves.png")
    args = p.parse_args()

    entries = collect_inputs(args.inputs)
    if not entries:
        sys.exit("no inputs resolved")

    out_dir = args.out_dir or os.path.dirname(os.path.abspath(args.inputs[0]))
    os.makedirs(out_dir, exist_ok=True)

    rows: List[Dict[str, Any]] = []
    for entry in entries:
        if not entry["json"] or not os.path.exists(entry["json"] or ""):
            rows.append({
                "variant": entry["variant"], "scene": entry["scene"],
                "failure_mode": "NO_RUN",
                "diagnosis": f"결과 JSON 없음 (status={entry.get('status')}), "
                             f"로그 확인: {entry.get('log')}",
            })
            continue
        with open(entry["json"]) as f:
            payload = json.load(f)
        row = analyse(payload)
        # The manifest's labels win: a run file records the scene but not which
        # sweep it belonged to.
        row["variant"] = entry["variant"] or os.path.basename(
            os.path.dirname(entry["json"]))
        row["scene"] = entry["scene"] or row["scene"]
        row["path"] = entry["json"]
        rows.append(row)

    rows.sort(key=lambda r: (
        SCENE_ORDER.index(r["scene"]) if r.get("scene") in SCENE_ORDER else 99,
        str(r.get("variant")),
    ))

    md = write_report(rows, out_dir)
    png = write_plots(rows, out_dir)
    print(open(md).read())
    print(f"\nwrote {md}")
    if png:
        print(f"wrote {png}")


if __name__ == "__main__":
    main()
