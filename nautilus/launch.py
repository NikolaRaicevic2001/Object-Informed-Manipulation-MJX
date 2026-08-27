#!/usr/bin/env python3
"""Launch an interactive pod, or an ablation sweep as a Job, on Nautilus.

Two commands, one template each:

    python nautilus/launch.py pod                       # a GPU and a shell
    python nautilus/launch.py job                       # the whole sweep
    python nautilus/launch.py job --shard task          # one Job per task
    python nautilus/launch.py job --only algorithm=mppi # a slice of it
    python nautilus/launch.py job --dry-run             # print, submit nothing

    # one GPU model, either command:
    python nautilus/launch.py pod --gpu-type NVIDIA-GeForce-RTX-4090

What runs is decided by the sweep config, not duplicated here. `--shard`
splits one sweep across several Jobs by the values of one axis. Results
land on the PVC under `/nikola-volume/oim/<name>/`.
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

#: Where the pod checks this repo out; the Dockerfile's WORKDIR and
#: PYTHONPATH must agree.
IMAGE_WORKDIR = "/workspace"
#: Mount point of the PVC, matching the templates.
VOLUME = "/nikola-volume"
DEFAULT_IMAGE = "nikolaraicevic2001/contact-mpc:latest"
DEFAULT_CONFIG = "ablation"
#: Cloned at pod start; the image carries dependencies only.
DEFAULT_REPO = (
    "https://github.com/NikolaRaicevic2001/"
    "Object-Informed-Manipulation-MJX.git"
)
#: Hash of the `uv.lock` the image was built from; `_clone_repo` warns on
#: drift.
IMAGE_LOCK_HASH = "/opt/oim/uv.lock.sha256"
#: What a pod does once its clone and outputs are wired up.
IDLE = "sleep 86400"

#: RFC 1123 label cap: a Pod's hostname is a label, so names stay under it.
MAX_NAME = 63


def _slug(text: str) -> str:
    """Lowercase `text` down to what a DNS label may contain."""
    return re.sub(r"[^a-z0-9]+", "-", str(text).lower()).strip("-")


def resource_name(prefix: str, *parts: str) -> str:
    """A DNS-label-safe, unique-per-second name.

    The timestamp is last and never truncated: two Jobs of one sweep
    differ only there, and a collision is rejected rather than queued.

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


#: The Pod/Job manifests this script renders.
TEMPLATE_DIR = os.path.join(_HERE, "templates")


def load_template(name: str) -> Dict[str, Any]:
    """Read `pod.yaml` or `job.yaml` from `templates/`."""
    with open(os.path.join(TEMPLATE_DIR, name)) as f:
        return yaml.safe_load(f)


#: Shard axes whose `--only` key differs from their name: a `task` entry
#: is `{script: open_table}`, and `--only` matches flat keys.
_SHARD_FILTER_KEY = {"task": "script"}


def sweep_axis_values(config: str, axis: str) -> List[str]:
    """The values of one axis of a sweep config, for `--shard`.

    Read from the YAML, not through `oim.run_launch`: importing `oim`
    initializes JAX, which a login node cannot do.

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
            # `{script: name}` or `{algorithm: admm, ...}`; both name
            # themselves under a key `--only` matches.
            named = value.get("script") or value.get(axis)
            if named is None:
                raise ValueError(
                    f"cannot name the {axis} entry {value!r} for --only"
                )
            out.append(str(named))
        else:
            out.append(str(value))
    return out


def _clone_repo(repo: str, ref: Optional[str]) -> List[str]:
    """Bash cloning `repo` at `IMAGE_WORKDIR`, on `ref` if given.

    A plain shallow clone, so the checkout is the remote's default branch
    as a real local branch and `git pull` works in the pod. `ref` adds
    `--branch`, falling back to fetch/checkout for a commit SHA, which
    `--branch` does not accept (that one is necessarily detached).

    The token goes in via `insteadOf`, so it stays out of `.git/config`
    and `git pull` still authenticates.

    Args:
        repo: Clone URL, https.
        ref: Branch, tag or commit SHA, or None for the default branch.

    Returns:
        The commands, one per line.
    """
    clone = f"git clone -q --depth 1 {repo} {IMAGE_WORKDIR}"
    if ref:
        clone = (
            f"git clone -q --depth 1 --branch {ref} {repo} {IMAGE_WORKDIR}"
            f" 2>/dev/null || {{ git init -q {IMAGE_WORKDIR}"
            f" && cd {IMAGE_WORKDIR} && git remote add origin {repo}"
            f" && git fetch -q --depth 1 origin {ref}"
            f" && git checkout -q FETCH_HEAD; }}"
        )
    return [
        'if [ -n "${GIT_ACCESS_TOKEN:-}" ]; then',
        '  git config --global url."https://x-access-token:'
        '${GIT_ACCESS_TOKEN}@github.com/".insteadOf "https://github.com/"',
        "fi",
        clone,
        f"cd {IMAGE_WORKDIR}",
        'echo "repo $(git rev-parse --short HEAD)'
        ' ($(git rev-parse --abbrev-ref HEAD))"',
        # The clone's uv.lock may have moved past the image's. Warn, not
        # fail: newer code on older deps is usually fine.
        f"if [ -f {IMAGE_LOCK_HASH} ] && "
        f'[ "$(sha256sum uv.lock | cut -d\' \' -f1)" '
        f'!= "$(cat {IMAGE_LOCK_HASH})" ]; then',
        '  echo "WARNING: uv.lock differs from the one this image was '
        'built from; rebuild and push the image (./docker/build.sh) if a '
        'dependency changed." >&2',
        "fi",
    ]


def _redirect_results(results: str) -> List[str]:
    """Bash putting `oim`'s two output directories on the PVC.

    `oim` writes beside its own package, a layer that dies with the pod.

    Args:
        results: Directory on the PVC for this pod's outputs.

    Returns:
        The commands, one per line.
    """
    return [
        f"cd {IMAGE_WORKDIR}",
        f"mkdir -p {results}/runs {results}/recordings",
        "rm -rf oim/results oim/recordings",
        f"ln -s {results} oim/results",
        f"ln -s {results}/recordings oim/recordings",
    ]


def sweep_command(
    args: argparse.Namespace,
    only: List[str],
    results: str,
) -> str:
    """The container's bash: clone, point the outputs at the PVC, sweep.

    `python`, not `uv run`: PATH already holds the image's venv.

    Args:
        args: The parsed command line, for `--config`, `--repo`, `--ref`
            and `--set`.
        only: `KEY=VALUE` filters, passed through as `--only`.
        results: Directory on the PVC for this Job's outputs.

    Returns:
        A single bash command line.
    """
    flags = " ".join(
        [f"--only {f}" for f in only] + [f"--set {s}" for s in args.set]
    )
    return "\n".join(
        [
            # A failing step must fail the Job, not be scrolled past.
            "set -euo pipefail",
            *_clone_repo(args.repo, args.ref),
            *_redirect_results(results),
            f"python -m oim.run_launch --config {args.config} "
            f"{flags}".rstrip(),
        ]
    )


def idle_command(args: argparse.Namespace, results: str) -> str:
    """The interactive pod's bash: clone, redirect the outputs, then wait.

    Both before the `sleep`, so a run typed by hand finds code in place.

    Args:
        args: The parsed command line, for `--repo` and `--ref`.
        results: Directory on the PVC for this pod's outputs.

    Returns:
        A single bash command line.
    """
    return "\n".join(
        [
            # No `-e`: `sleep` must run even if the clone fails, or the
            # pod dies before anyone can exec in and read why.
            "set -uo pipefail",
            *_clone_repo(args.repo, args.ref),
            *_redirect_results(results),
            IDLE,
        ]
    )


def build_pod(args: argparse.Namespace) -> List[Dict[str, Any]]:
    """One interactive pod."""
    spec = load_template("pod.yaml")
    name = args.name or resource_name("oim-pod")
    spec["metadata"]["name"] = name
    container = spec["spec"]["containers"][0]
    container["image"] = args.image
    container["args"] = [idle_command(args, f"{VOLUME}/oim/{name}")]
    _apply_resources(container, args)
    _apply_gpu_type(spec["spec"], args.gpu_type)
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

        pod_spec = spec["spec"]["template"]["spec"]
        container = pod_spec["containers"][0]
        container["image"] = args.image
        container["args"] = [
            sweep_command(args, only, f"{VOLUME}/oim/{name}")
        ]
        _apply_resources(container, args)
        _apply_gpu_type(pod_spec, args.gpu_type)
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


#: The node label a GPU model is advertised under; `gpu_summary.txt`'s
#: "GPU Model" column holds its values.
GPU_PRODUCT_KEY = "nvidia.com/gpu.product"


def _apply_gpu_type(
    pod_spec: Dict[str, Any], gpu_types: Optional[List[str]]
) -> None:
    """Narrow the template's GPU-model affinity to `gpu_types`.

    Replaces the list rather than intersecting it, so a model the template
    does not name is still reachable.

    Args:
        pod_spec: `spec` of a Pod, `spec.template.spec` of a Job.
        gpu_types: Models to allow, or None to keep the template's.

    Raises:
        ValueError: If the template has no `nvidia.com/gpu.product`
            expression; doing nothing would let the pod land anywhere.
    """
    if not gpu_types:
        return
    terms = (
        pod_spec.get("affinity", {})
        .get("nodeAffinity", {})
        .get("requiredDuringSchedulingIgnoredDuringExecution", {})
        .get("nodeSelectorTerms", [])
    )
    matches = [
        expr
        for term in terms
        for expr in term.get("matchExpressions", [])
        if expr.get("key") == GPU_PRODUCT_KEY
    ]
    if not matches:
        raise ValueError(
            f"--gpu-type needs a {GPU_PRODUCT_KEY} nodeAffinity in the "
            f"template to narrow; templates/ has none"
        )
    for expr in matches:
        expr["operator"] = "In"
        expr["values"] = list(gpu_types)


class _Dumper(yaml.SafeDumper):
    """`SafeDumper` writing multi-line strings as literal blocks.

    A subclass, not a representer on `SafeDumper` itself, which would be a
    global mutation.
    """


def _literal_block(dumper: yaml.Dumper, data: str) -> Any:
    """Dump a multi-line string as `|`.

    So `--dry-run` shows the container script as script, not as one quoted
    run-on line.
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

    Added to the top-level parser and to each subparser, so they work on
    either side of the subcommand.

    Args:
        p: The parser to extend.
        sub: True for a subparser copy, whose defaults are SUPPRESSed --
            otherwise an unset subparser flag overwrites the top-level
            value and `--dry-run job` silently submits.
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
    # The code the pod runs -- flags, not image contents.
    p.add_argument(
        "--repo",
        default=default(DEFAULT_REPO),
        help=f"Clone URL (default: {DEFAULT_REPO}).",
    )
    p.add_argument(
        "--ref",
        default=default(None),
        help="Branch, tag or commit SHA to check out; default is the "
        "remote's default branch. A SHA pins a sweep to exact code, but "
        "checks out detached.",
    )
    # Default None, not a number: the template is the source of truth.
    for flag, kind in (("--gpu", int), ("--cpu", int), ("--memory", str)):
        p.add_argument(
            flag,
            type=kind,
            default=default(None),
            help=f"{flag[2:]} per pod; default: the template's.",
        )
    # Repeatable: the useful ask is usually a family, not one model.
    p.add_argument(
        "--gpu-type",
        action="append",
        default=default(None),
        metavar="MODEL",
        help="Pin the GPU model, e.g. NVIDIA-GeForce-RTX-4090; "
        "repeatable. Values are the `nvidia.com/gpu.product` label -- "
        "gpu_summary.txt's 'GPU Model' column. Default: the template's.",
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
        # A bad sweep config or axis is a usage error, not a traceback.
        parser.error(str(exc))
    return submit(specs, bool(args.dry_run))


if __name__ == "__main__":
    sys.exit(main())
