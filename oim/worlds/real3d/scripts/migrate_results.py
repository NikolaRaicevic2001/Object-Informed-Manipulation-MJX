"""Move already-written REAL runs out of `results/runs` into the new layout.

`examples/pusht/pusht_real.py` now files new runs under

    oim/results/real/{algorithm}/{scene}/{date}/

while the sim path keeps writing flat into `oim/results/runs`. This moves the
real runs that are already sitting in the flat directory, and leaves every sim
run exactly where it is.

A real run is identified by its content, not its filename: only the real
driver writes a `mock` field into the run's `run` block, and only the sim
driver writes `interactive`. Filenames are not a safe discriminator -- both
paths produce `pusht3d_xarm6_..._{algorithm}_{timestamp}.json`.

Everything belonging to one run moves together: the JSON, the `--plot` PNG,
any video, and any `_states` / `_metrics` sidecar. They all share the run's
stem and its one fixed timestamp, which is exactly what `RunName` guarantees.

Dry run by default -- nothing moves until `--apply`.

    python oim/worlds/real3d/scripts/migrate_results.py
    python oim/worlds/real3d/scripts/migrate_results.py --apply
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import re
import shutil
from typing import List, Tuple

from oim import ROOT

TIMESTAMP = re.compile(r"_(\d{8})_(\d{6})$")


def classify(payload: dict) -> str:
    """`"real"`, `"sim"`, or `"other"` for one loaded JSON file.

    Only the real driver writes a `mock` field into the run block and only
    the sim driver writes `interactive`, so this reads content rather than
    filenames -- both paths produce
    `pusht3d_xarm6_..._{algorithm}_{timestamp}.json`. Anything without a run
    block at all (a `_states` or `_metrics` sidecar) is `"other"`.
    """
    run = payload.get("run")
    if not isinstance(run, dict):
        return "other"
    return "real" if "mock" in run else "sim"


def split_stem(basename: str) -> Tuple[str, str, str]:
    """`("pusht3d_xarm6_mock_scene_mppi", "20260822", "151447")`, or None."""
    stem = os.path.splitext(basename)[0]
    m = TIMESTAMP.search(stem)
    if not m:
        return None
    return stem[: m.start()], m.group(1), m.group(2)


def siblings(
    src_dir: str, stem: str, date: str, clock: str
) -> List[str]:
    """Every file of the one run: `{stem}_{ts}.*` and `{stem}_{kind}_{ts}.*`."""
    ts = f"{date}_{clock}"
    found = set(glob.glob(os.path.join(src_dir, f"{stem}_{ts}.*")))
    found |= set(glob.glob(os.path.join(src_dir, f"{stem}_*_{ts}.*")))
    return sorted(found)


def main() -> None:
    """Move one run's files into the layout `run_eval` expects."""
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--src", default=os.path.join(ROOT, "results", "runs"),
                   help="flat directory to migrate out of")
    p.add_argument("--dest-root", default=os.path.join(ROOT, "results", "real"),
                   help="root of the new layout")
    p.add_argument("--apply", action="store_true",
                   help="actually move the files; without it, only print")
    args = p.parse_args()

    paths = sorted(glob.glob(os.path.join(args.src, "*.json")))
    if not paths:
        raise SystemExit(f"no JSON files in {args.src}")

    moved = skipped_sim = skipped_bad = collided = 0
    consumed: set = set()
    for path in paths:
        if path in consumed:
            # Already moved as a sidecar of its own run.
            continue
        base = os.path.basename(path)
        try:
            with open(path) as fh:
                payload = json.load(fh)
        except (OSError, json.JSONDecodeError) as exc:
            print(f"  ?? {base}: unreadable ({exc}); left alone")
            skipped_bad += 1
            continue

        kind = classify(payload)
        if kind != "real":
            skipped_sim += kind == "sim"
            continue

        parts = split_stem(base)
        if parts is None:
            print(f"  ?? {base}: no _YYYYMMDD_HHMMSS suffix; left alone")
            skipped_bad += 1
            continue
        stem, date, clock = parts

        run = payload["run"]
        algorithm = str(run.get("algorithm") or "unknown")
        scene = str(run.get("task") or "unknown")
        dest = os.path.join(args.dest_root, algorithm, scene, date)

        group = siblings(args.src, stem, date, clock)
        clash = [f for f in group
                 if os.path.exists(os.path.join(dest, os.path.basename(f)))]
        if clash:
            print(f"  !! {base}: already present in {dest}; left alone")
            collided += 1
            continue

        rel = os.path.relpath(dest, ROOT)
        if rel.startswith(os.pardir):
            rel = dest
        print(f"  -> {rel}/  ({len(group)} file"
              f"{'s' if len(group) != 1 else ''}: {base} ...)")
        if args.apply:
            os.makedirs(dest, exist_ok=True)
            for f in group:
                shutil.move(f, os.path.join(dest, os.path.basename(f)))
        consumed.update(group)
        moved += 1

    src_rel = os.path.relpath(args.src, ROOT)
    if src_rel.startswith(os.pardir):
        src_rel = args.src
    print(f"\n{len(paths)} JSON files in {src_rel}")
    print(f"  real runs {'moved' if args.apply else 'to move'}: {moved}")
    print(f"  sim runs left in place:                {skipped_sim}")
    if collided:
        print(f"  already at the destination:            {collided}")
    if skipped_bad:
        print(f"  unreadable / unnamed, left alone:      {skipped_bad}")
    if not args.apply and moved:
        print("\nDry run -- nothing moved. Re-run with --apply.")


if __name__ == "__main__":
    main()
