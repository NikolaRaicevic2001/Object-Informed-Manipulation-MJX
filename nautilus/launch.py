#!/usr/bin/env python3
"""Launch an interactive pod, or an ablation sweep as a Job, on Nautilus.

Two commands, one template each:

    python nautilus/launch.py pod                       # a GPU and a shell
    python nautilus/launch.py job                       # the whole sweep
    python nautilus/launch.py job --shard task          # one Job per task
    python nautilus/launch.py job --only algorithm=mppi # a slice of it
    python nautilus/launch.py job --dry-run             # print, submit nothing

The Job runs `oim.run_launch` against a sweep config under
`oim/configs/sweeps/`, so what gets run is decided by `ablation.yaml` and
not duplicated here. `--shard` reads that same file to split one sweep
across several Jobs by the values of one axis -- the only parallelism
offered, because a cell already saturates its GPU and `run_launch` runs
its cells one at a time.

Results land on the PVC under `/nikola-volume/oim/<run>/`, so they outlive
the Job.
"""

from __future__ import annotations

import argparse
import copy
import os
import re
import subprocess
import sys
import tempfile
from datetime import datetime
from typing import Any, Dict, List, Optional

import yaml

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.dirname(_HERE)

#: Where the image checks out this repository. The Dockerfile must agree.
IMAGE_WORKDIR = "/workspace"
#: Mount point of the PVC, matching `pod.yaml` / `job.yaml`.
VOLUME = "/nikola-volume"
DEFAULT_IMAGE = "nikolaraicevic2001/contact-mpc:latest"
DEFAULT_CONFIG = "ablation"

#: RFC 1123 label cap. Kubernetes allows a longer `metadata.name`, but a
#: Pod's hostname is a label, so staying under this keeps names portable.
MAX_NAME = 63


def _slug(text: str) -> str:
    """Lowercase `text` down to what a DNS label may contain."""
    return re.sub(r"[^a-z0-9]+", "-", str(text).lower()).strip("-")


def resource_name(prefix: str, *parts: str) -> str:
    """A DNS-label-safe, unique-per-minute name.

    The timestamp is the last component and is never truncated: two Jobs
    of the same sweep differ only there, and a name collision is rejected
    by the API server rather than queued.

    Args:
        prefix: Leading component, e.g. "oim-job".
        parts: Further components; empty ones are dropped.

    Returns:
        The name, at most `MAX_NAME` characters.
    """
    stamp = datetime.now().strftime("%m%d-%H%M%S")
    middle = "-".join(p for p in (_slug(p) for p in parts) if p)
    head = "-".join(p for p in (prefix, middle) if p)
    return f"{head[: MAX_NAME - len(stamp) - 1]}-{stamp}".replace("--", "-")


def load_template(name: str) -> Dict[str, Any]:
    """Read `pod.yaml` or `job.yaml` from beside this script."""
    with open(os.path.join(_HERE, name)) as f:
        return yaml.safe_load(f)


#: `--only` matches a cell's FLAT keys, and a `task` entry is the dict
#: `{script: open_table}` -- so the filter that selects one task names
#: `script`, not `task`. Shard axes whose name differs from their filter
#: key are listed here; anything else filters under its own name.
_SHARD_FILTER_KEY = {"task": "script"}


def sweep_axis_values(config: str, axis: str) -> List[str]:
    """The values of one axis of a sweep config, for `--shard`.

    Read straight from the YAML rather than through `oim.run_launch`:
    this script runs on a login node, and importing `oim` initializes JAX
    and claims a GPU that is not there.

    Args:
        config: A name under `oim/configs/sweeps/`, or a path.
        axis: The axis to split on, e.g. "task" or "object".

    Returns:
        One string per value, in the order the config lists them.

    Raises:
        ValueError: If the config has no such axis, or an entry of it has
            no value this script knows how to name.
    """
    path = config if os.path.exists(config) else os.path.join(
        _REPO, "oim", "configs", "sweeps", f"{config}.yaml"
    )
    with open(path) as f:
        sweep = (yaml.safe_load(f) or {}).get("sweep", {})
    values = sweep.get(axis)
    if not values:
        raise ValueError(
            f"{path} has no `sweep.{axis}` to shard on "
            f"(axes present: {sorted(k for k, v in sweep.items() if v)})"
        )
    out = []
    for value in values:
        if isinstance(value, dict):
            # `task` entries are `{script: name}`; `algorithm` entries may
            # be `{algorithm: admm, consensus: ...}`. Both name themselves
            # under a key `--only` can then match on.
            named = value.get("script") or value.get(axis)
            if named is None:
                raise ValueError(
                    f"cannot name the {axis} entry {value!r} for --only"
                )
            out.append(str(named))
        else:
            out.append(str(value))
    return out


def sweep_command(
    config: str, only: List[str], extra: List[str], results: str
) -> str:
    """The container's bash: point the outputs at the PVC, then sweep.

    `oim` writes beside its own package (`oim/results`, `oim/recordings`),
    which in a container is a layer that disappears with the Job. The
    symlinks redirect both onto the PVC without the runner needing to know
    it is containerized.

    Args:
        config: `--config` for `oim.run_launch`.
        only: `KEY=VALUE` filters, passed through as `--only`.
        extra: `KEY=VALUE` overrides, passed through as `--set`.
        results: Directory on the PVC for this Job's outputs.

    Returns:
        A single bash command line.
    """
    flags = " ".join(
        [f"--only {f}" for f in only] + [f"--set {s}" for s in extra]
    )
    return "\n".join(
        [
            # A failing step must fail the Job, not be scrolled past.
            "set -euo pipefail",
            f"cd {IMAGE_WORKDIR}",
            f"mkdir -p {results}/runs {results}/recordings",
            "rm -rf oim/results oim/recordings",
            f"ln -s {results} oim/results",
            f"ln -s {results}/recordings oim/recordings",
            f"uv run python -m oim.run_launch --config {config} "
            f"{flags}".rstrip(),
        ]
    )


def build_pod(args: argparse.Namespace) -> List[Dict[str, Any]]:
    """One interactive pod."""
    spec = load_template("pod.yaml")
    spec["metadata"]["name"] = args.name or resource_name("oim-pod")
    container = spec["spec"]["containers"][0]
    container["image"] = args.image
    _apply_resources(container, args)
    return [spec]


def build_jobs(args: argparse.Namespace) -> List[Dict[str, Any]]:
    """One Job for the whole sweep, or one per value of `--shard`."""
    shards: List[Optional[str]] = [None]
    if args.shard:
        shards = list(sweep_axis_values(args.config, args.shard))

    jobs = []
    for shard in shards:
        spec = copy.deepcopy(load_template("job.yaml"))
        name = args.name or resource_name("oim", args.config, shard or "")
        if len(shards) > 1 and args.name:
            name = resource_name(args.name, shard or "")
        spec["metadata"]["name"] = name

        only = list(args.only)
        if shard is not None:
            key = _SHARD_FILTER_KEY.get(args.shard, args.shard)
            only.append(f"{key}={shard}")

        container = spec["spec"]["template"]["spec"]["containers"][0]
        container["image"] = args.image
        container["args"] = [
            sweep_command(
                args.config, only, args.set, f"{VOLUME}/oim/{name}"
            )
        ]
        _apply_resources(container, args)
        jobs.append(spec)
    return jobs


def _apply_resources(
    container: Dict[str, Any], args: argparse.Namespace
) -> None:
    """Override the template's requests/limits, for flags that were given."""
    overrides = {
        "nvidia.com/gpu": None if args.gpu is None else str(args.gpu),
        "cpu": None if args.cpu is None else str(args.cpu),
        "memory": args.memory,
    }
    for side in ("limits", "requests"):
        block = container["resources"].get(side)
        if block is None:
            continue
        for key, value in overrides.items():
            if value is not None:
                block[key] = value


class _Dumper(yaml.SafeDumper):
    """`SafeDumper` that writes multi-line strings as literal blocks.

    Its own subclass rather than a representer on `SafeDumper`: that is a
    global mutation, and this module has no business changing how anything
    else in the process serializes a string.
    """


def _literal_block(dumper: yaml.Dumper, data: str) -> Any:
    """Dump a multi-line string as `|`, not as a quoted blob.

    The container script is the part of a manifest worth reading before
    submitting it, and the default single-quoted style renders it as one
    run-on line with blank lines standing in for newlines.
    """
    style = "|" if "\n" in data else None
    return dumper.represent_scalar("tag:yaml.org,2002:str", data, style=style)


_Dumper.add_representer(str, _literal_block)


def submit(specs: List[Dict[str, Any]], dry_run: bool) -> int:
    """`kubectl apply` each spec, or print it.

    Args:
        specs: Rendered Pod/Job manifests.
        dry_run: Print the YAML and submit nothing.

    Returns:
        A process exit code.
    """
    for spec in specs:
        rendered = yaml.dump(spec, Dumper=_Dumper, sort_keys=False)
        if dry_run:
            print(f"--- # {spec['metadata']['name']}\n{rendered}")
            continue
        with tempfile.NamedTemporaryFile(
            "w", suffix=".yaml", delete=False
        ) as f:
            f.write(rendered)
            path = f.name
        try:
            subprocess.run(["kubectl", "apply", "-f", path], check=True)
        except subprocess.CalledProcessError as exc:
            print(f"kubectl failed on {spec['metadata']['name']}")
            return exc.returncode
        finally:
            os.remove(path)
    if dry_run:
        print(f"# {len(specs)} manifest(s), nothing submitted", file=sys.stderr)
    return 0


def _add_common(p: argparse.ArgumentParser, sub: bool = False) -> None:
    """Flags both subcommands take.

    Added to the top-level parser AND to each subparser, so they may be
    written on either side of the subcommand -- `launch.py job --dry-run`
    is the order everyone reaches for, and argparse accepts a top-level
    flag only before the subcommand.

    Args:
        p: The parser to extend.
        sub: True for a subparser copy, whose defaults are SUPPRESSed.
            Without that, an unset subparser flag writes its default over
            the value the top-level parser already put in the namespace,
            and `--dry-run job` silently submits.
    """
    default = (lambda v: argparse.SUPPRESS) if sub else (lambda v: v)
    p.add_argument(
        "--image",
        default=default(DEFAULT_IMAGE),
        help=f"Container image (default: {DEFAULT_IMAGE}).",
    )
    p.add_argument(
        "--name",
        default=default(None),
        help="Resource name; default is generated.",
    )
    # Default None, not a number: the template is the source of truth for
    # what a pod asks for, and a flag that silently replaced it would make
    # `pod.yaml` and `job.yaml` decorative.
    for flag, kind in (("--gpu", int), ("--cpu", int), ("--memory", str)):
        p.add_argument(
            flag,
            type=kind,
            default=default(None),
            help=f"{flag[2:]} per pod; default: the template's.",
        )
    p.add_argument(
        "--dry-run",
        action="store_true",
        default=default(False),
        help="Print the manifests and submit nothing.",
    )


def build_parser() -> argparse.ArgumentParser:
    """The two subcommands and the flags they share."""
    p = argparse.ArgumentParser(description=__doc__)
    _add_common(p)
    sub = p.add_subparsers(dest="command", required=True)

    _add_common(
        sub.add_parser("pod", help="An idle GPU pod to exec into."), sub=True
    )

    job = sub.add_parser("job", help="Run a sweep to completion.")
    _add_common(job, sub=True)
    job.add_argument(
        "--config",
        default=DEFAULT_CONFIG,
        help=f"Sweep config under oim/configs/sweeps/ "
        f"(default: {DEFAULT_CONFIG}).",
    )
    job.add_argument(
        "--shard",
        help="Split the sweep across one Job per value of this axis, e.g. "
        "`task`. Each Job gets the matching `--only` filter.",
    )
    job.add_argument(
        "--only",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="Passed through to run_launch --only; repeatable.",
    )
    job.add_argument(
        "--set",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="Passed through to run_launch --set; repeatable.",
    )
    return p


def main() -> int:
    """Render the manifests for one command and submit them."""
    parser = build_parser()
    args = parser.parse_args()
    try:
        specs = build_pod(args) if args.command == "pod" else build_jobs(args)
    except (OSError, ValueError) as exc:
        # A missing sweep config or an axis that is not in it: the user's
        # mistake, so report it as one rather than as a traceback.
        parser.error(str(exc))
    return submit(specs, bool(args.dry_run))


if __name__ == "__main__":
    sys.exit(main())
