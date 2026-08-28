#!/usr/bin/env python3
"""Trajectory + diagnostics + cost-breakdown figure from a saved run JSON.

`examples/pusht/pusht_real.py` has no `--plot` (the sim's figure is drawn in
`oim/experiment.py`, which the real driver deliberately does not import). But
`oim.utils.plotting.plot_run_3d` only needs a `PushT` and a log dict, and a run
file holds everything both are built from -- so the figure is recoverable after
the fact, without re-running anything.

    python oim/worlds/real3d/scripts/plot_run_from_json.py \
        oim/results/runs/pusht3d_xarm6_mock_open_table_real_mppi_*.json

    # every run of a sweep at once
    python oim/worlds/real3d/scripts/plot_run_from_json.py \
        --manifest oim/results/sweeps/real_costs/manifest.tsv

Writes `<run>.png` next to the JSON unless `-o` says otherwise. Panels:
  * left   -- the object's swept footprint, the pusher path, goal, obstacles
  * middle -- position and heading error against control step
  * right  -- every cost term's per-step value, totals in the legend

The task is rebuilt from the run file's own `run.task` and
`hyperparameters.costs`, i.e. the merged weights the run actually optimized,
not whatever the config says today.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys

import numpy as np

# Same as the entry point: keep the XLA cache warm and off the default path.
os.environ.setdefault("JAX_COMPILATION_CACHE_DIR",
                      os.path.expanduser("~/.cache/jax"))

from oim.objects import wrap_angle  # noqa: E402
from oim.tasks.pusht import PushT  # noqa: E402
from oim.utils.plotting import plot_run_3d  # noqa: E402

PLAN_DT = 0.05  # pusht_real.py's own constant


def rebuild_task(payload: dict) -> PushT:
    """The `PushT` the run was produced with, from what the run recorded.

    `hyperparameters.costs` is `task.costs` -- already merged over
    `DEFAULT_COSTS` -- so this reproduces the exact objective, including any
    `--cost KEY=VAL` the run was given.
    """
    run = payload["run"]
    return PushT(
        impl="jax",              # backend is irrelevant: nothing rolls out
        clutter=True,
        planning_dt=PLAN_DT,
        robot=run.get("robot", "xarm6"),
        consensus_source="twist",
        env=run["task"],
        costs=payload["hyperparameters"]["costs"],
    )


def rebuild_log(payload: dict) -> dict:
    """The log dict `plot_run_3d` expects, from `dynamic` + derived fields.

    `pos_err` / `theta_err` / `reached` are deliberately NOT in a run file --
    they are definitions, recomputed here the same way `oim.utils.metrics`
    does, from `object_pose` against `static.goal`.
    """
    dyn = payload["dynamic"]
    hyp = payload["hyperparameters"]
    log = {k: np.asarray(v) for k, v in dyn.items()}

    goal = np.asarray(payload["static"]["goal"], dtype=float)
    pose = np.asarray(dyn["object_pose"], dtype=float)
    pos_err = np.linalg.norm(pose[:, :2] - goal[:2], axis=1)
    theta_err = np.abs(np.asarray(wrap_angle(pose[:, 2] - goal[2])))

    log["pos_err"] = pos_err
    log["theta_err"] = theta_err
    log["reached"] = bool(
        pos_err[-1] < float(hyp.get("goal_pos_tol", 0.05))
        and theta_err[-1] < float(hyp.get("goal_theta_tol", 0.05))
    )
    return log


def manifest_jsons(path: str) -> list:
    """Every `result_json` a sweep manifest lists, skipping failed rows."""
    out = []
    with open(path) as f:
        header = f.readline().rstrip("\n").split("\t")
        for line in f:
            if not line.strip():
                continue
            row = dict(zip(
                header, line.rstrip("\n").split("\t"), strict=False
            ))
            if row.get("result_json"):
                out.append(row["result_json"])
    return out


def main() -> None:
    """Re-plot the run files named on the command line."""
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("runs", nargs="*", help="run JSON path(s); globs are fine")
    p.add_argument("--manifest",
                   help="a sweep manifest.tsv to plot every run of")
    p.add_argument("-o", "--out-dir", default=None,
                   help="where to write the PNGs (default: beside each JSON)")
    p.add_argument("--stride", type=int, default=5,
                   help="draw the object footprint every N control steps")
    args = p.parse_args()

    paths = []
    for raw in args.runs:
        paths.extend(sorted(glob.glob(raw)) or [raw])
    if args.manifest:
        paths.extend(manifest_jsons(args.manifest))
    if not paths:
        sys.exit("no run files given")

    for path in paths:
        if not os.path.exists(path):
            print(f"skip (missing): {path}", file=sys.stderr)
            continue
        with open(path) as f:
            payload = json.load(f)
        print(f"[{payload['run']['task']}] {os.path.basename(path)}")
        task = rebuild_task(payload)
        log = rebuild_log(payload)
        out_dir = args.out_dir or os.path.dirname(os.path.abspath(path))
        os.makedirs(out_dir, exist_ok=True)
        stem = os.path.splitext(os.path.basename(path))[0]
        plot_run_3d(task, log, os.path.join(out_dir, f"{stem}.png"),
                    stride=args.stride)


if __name__ == "__main__":
    main()
