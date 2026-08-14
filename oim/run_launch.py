r"""Sweep driver: run one `examples/` script per parameter combination.

A benchmark or ablation is a cartesian product -- tasks x algorithms x
horizons x sampler budgets x seeds -- and this runs it. It holds no
planning code of its own: each cell is a subprocess invocation of the
task's own script, so a cell is exactly a command you could have typed,
and the launcher prints that command before running it.

The `task` axis names the script: `{ script: shelf_gap }` runs
`examples/shelf_gap.py`, and any other key in that entry becomes a flag for
it (`{ script: clutter, robot: xarm6 }`). Each script advertises only the
flags its world has, so the launcher reads the parser of the script it is
about to run rather than assuming one shared CLI.

    # the whole product in oim/configs/run_launch_config.yaml
    uv run python -m oim.run_launch

    # see what would run, without running it
    uv run python -m oim.run_launch --dry-run

    # narrow a sweep without editing the config
    uv run python -m oim.run_launch --only algorithm=admm --only horizon=15

Scoring is deliberately elsewhere: this writes run files, and
`oim/run_eval.py` turns them into tables. A sweep is expensive and a metric
is cheap, so they must not share a lifetime.

Subprocess isolation is the point of not importing the runners directly: a
crashed or OOM cell loses one combination instead of the sweep, and each
run starts with a clean JAX allocator. The cost is a recompile per cell,
which is unavoidable anyway whenever the horizon or sample count changes
the traced shapes.
"""

import argparse
import functools
import importlib.util
import itertools
import json
import os
import subprocess
import sys
import time
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

import jax
import yaml

from oim import ROOT
from oim.experiment import build_parser

CONFIG_DIR = os.path.join(os.path.dirname(__file__), "configs")
EXAMPLES_DIR = os.path.join(os.path.dirname(ROOT), "examples")

# Flags a flat baseline has no use for: it has no consensus to iterate, no
# penalty and no proximal term. `expand` already drops `n_admm` as an axis;
# this also keeps a `fixed:` block from putting one on a flat command line,
# where the script would now reject it rather than ignore it.
_ADMM_ONLY = (
    "robot_opt",
    "object_opt",
    "n_admm",
    "rho",
    "rho_torque",
    "gamma",
    "consensus_alpha",
    "local_goal",
)

# The mirror image: flags only a *flat* baseline's subparser defines, so an
# `iterations:` in `fixed:` reaches `mppi`/`ps` and is dropped before it can
# reach `admm`, which does not accept it. Without this the two lists are
# asymmetric and a perfectly reasonable `fixed:` block kills every ADMM cell
# of a mixed sweep with an argparse error, once per cell, minutes apart --
# the exact failure `_check_fixed` exists to prevent.
_FLAT_ONLY = ("iterations",)

# Sweep axes, outermost first -- this is the nesting order, so everything
# for task 1 finishes before task 2 starts. `start`/`goal` sit beside
# `seed` because all three vary a trial rather than the method.
#
# The ADMM penalty knobs are axes and not just `fixed:` entries because
# they are what an ablation is usually *about*: `rho` sets how hard the
# consensus penalty pulls, and it has the largest measured effect of any
# parameter here. They are all in `_ADMM_ONLY`, so a flat cell drops them
# and `expand` collapses the duplicates -- sweeping `rho` over k values
# therefore costs k ADMM cells and still only one flat cell.
_AXES = (
    "task",
    # `object_only` only: it takes one scene per run rather than one per
    # script, so the scene is an axis there where `task` is one elsewhere.
    "scene",
    "algorithm",
    # `object_only` only: which dynamics execute the block's wrench.
    "plant",
    "robot_opt",
    "object_opt",
    "horizon",
    "samples",
    "object_samples",
    "n_admm",
    "rho",
    "gamma",
    "consensus_alpha",
    "wrench_fraction",
    "w_rate",
    "noise_level",
    "temperature",
    "start",
    "goal",
    "seed",
)


def script_path(name: str) -> str:
    """`examples/{name}.py`, checked to exist.

    Args:
        name: A `script:` value from the sweep's `task` axis.

    Returns:
        The absolute path.

    Raises:
        ValueError: If no such script exists, listing the ones that do.
    """
    path = os.path.join(EXAMPLES_DIR, f"{name}.py")
    if not os.path.exists(path):
        available = sorted(
            f[:-3]
            for f in os.listdir(EXAMPLES_DIR)
            if f.endswith(".py") and not f.startswith("_")
        )
        raise ValueError(
            f"no examples/{name}.py; available: {available}"
        )
    return path


@functools.lru_cache(maxsize=None)
def _load_script(path: str) -> Any:
    """Import an `examples/` script for its `EXPERIMENT` and its parser.

    Cached: a sweep asks the same handful of scripts once per cell
    otherwise, and each import compiles an MJCF.
    """
    # Importing a script pulls in oim.utils.scenes, whose SCENES registry
    # builds module-level `jnp` arrays (goal/obstacles per scene), which
    # initializes JAX's GPU backend and claims ~75% of the device -- in
    # *this* process, which then holds it for the whole sweep and starves
    # every cell of it. The launcher only reads a parser, so pin it to CPU
    # first.
    #
    # `jax.config`, not `JAX_PLATFORMS`: the environment variable is read
    # when `jax` is first imported, and `oim/__init__.py` has already done
    # that by the time this runs. Setting it here would silently do
    # nothing. Children are unaffected either way -- they are separate
    # processes with their own JAX.
    jax.config.update("jax_platforms", "cpu")
    spec = importlib.util.spec_from_file_location("_oim_example", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def script_world(name: str) -> str:
    """Which world a script runs.

    Every `examples/` script declares an `Experiment`, and that one
    attribute is the whole contract a script needs to be sweepable.
    `object_only.py` used to be the exception -- no per-script scene, no
    algorithm subcommand, so it carried a `SWEEP_WORLD` string instead --
    until `Experiment` grew the `"object"` world it needed.
    """
    return _load_script(script_path(name)).EXPERIMENT.world


@functools.lru_cache(maxsize=None)
def _flag_spec(name: str) -> Tuple[Dict[str, bool], Dict[str, bool]]:
    """Ask one `examples/` script which flags it takes, and where.

    Its parser splits on the algorithm name -- scene and world flags before
    it, solver and run flags after -- and mixes store_true switches with
    valued options. Rather than restate that here (a copy that silently
    rots the moment a flag moves), the classification is read off the real
    parser: a wrong guess would otherwise surface as every cell of a long
    sweep failing identically. Read per script, because a 2D script and a
    3D one genuinely offer different flags.

    Args:
        name: A `script:` value from the sweep's `task` axis.

    Returns:
        `(top_level, per_algorithm)`, each mapping a dest name to whether
        the flag takes a value (False means a bare switch). A script with
        no algorithm subcommand returns an empty `per_algorithm`, which is
        what tells `build_command` not to emit the positional.
    """
    parser = build_parser(_load_script(script_path(name)).EXPERIMENT)

    top: Dict[str, bool] = {}
    sub: Dict[str, bool] = {}
    for action in parser._actions:
        if isinstance(action, argparse._SubParsersAction):
            for subparser in action.choices.values():
                for a in subparser._actions:
                    if a.dest != "help":
                        sub[a.dest] = a.nargs != 0
        elif action.dest not in ("help",):
            top[action.dest] = action.nargs != 0
    return top, sub


def load_config(path: str) -> Dict[str, Any]:
    """Load a sweep config.

    Args:
        path: Path to the YAML, or a bare name resolved in `oim/configs/`.

    Returns:
        The parsed config, with `sweep` and `fixed` keys.

    Raises:
        FileNotFoundError: If the file does not exist.
    """
    if not os.path.isabs(path) and not os.path.exists(path):
        path = os.path.join(CONFIG_DIR, path)
    if not path.endswith((".yaml", ".yml")):
        path += ".yaml"
    with open(path) as f:
        return yaml.safe_load(f)


def expand(sweep: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Expand the sweep axes into one dict per combination.

    Cells that cannot exist are dropped rather than left to fail at
    runtime: 2D has no flat baselines, and a flat baseline has no ADMM
    blocks, so sweeping `robot_opt` would otherwise re-run the identical
    baseline once per pair.

    Args:
        sweep: The config's `sweep` block, `{axis: [values]}`.

    Returns:
        One dict per valid combination, in nesting order.

    Raises:
        ValueError: If the sweep names an axis that is not in `_AXES`.
            Unknown axes used to be dropped in silence, so a sweep over a
            misspelled -- or simply unlisted -- key ran as a single cell
            and looked like it had worked.
    """
    unknown = [a for a in sweep if a not in _AXES]
    if unknown:
        raise ValueError(
            f"sweep axes {sorted(unknown)} are not sweepable "
            f"(available: {', '.join(_AXES)}). A flag that is not an axis "
            f"goes in `fixed:` instead."
        )
    axes = [a for a in _AXES if sweep.get(a)]
    combos = []
    seen = set()
    for values in itertools.product(*(sweep[a] for a in axes)):
        cell = dict(zip(axes, values, strict=True))
        task = cell.get("task", {})
        algorithm = cell.get("algorithm", "admm")
        script = task["script"]

        # Drop axes this script has no flag for *before* the dedup below,
        # so sweeping one (e.g. `scene`, which only `object_only` takes)
        # collapses to a single cell instead of running the same command k
        # times. `build_command` would drop them anyway, silently and too
        # late to prevent the duplicates.
        accepted = _accepted(script)
        cell = {
            k: v
            for k, v in cell.items()
            if k in ("task", "algorithm") or k in accepted
        }

        if not _flag_spec(script)[1]:
            # No algorithm subcommand (object_only): the axis is
            # meaningless, so strip it rather than run the script once per
            # value of something it never sees.
            cell.pop("algorithm", None)
        elif script_world(script) == "2d" and algorithm != "admm":
            continue  # PushT2D implements only ConsensusTask
        elif algorithm != "admm":
            # Flat baselines have no blocks; collapse the duplicates.
            cell = {k: v for k, v in cell.items() if k not in _ADMM_ONLY}
        key = json.dumps(cell, sort_keys=True)
        if key in seen:
            continue
        seen.add(key)
        combos.append(cell)
    return combos


def build_command(
    cell: Dict[str, Any],
    fixed: Dict[str, Any],
    spec: Optional[Tuple[Dict[str, bool], Dict[str, bool]]] = None,
) -> List[str]:
    """Turn one sweep cell into a command line for its own script.

    Args:
        cell: One combination from `expand`.
        fixed: The config's `fixed` block, applied to every cell.
        spec: Unused; kept so callers may pass a cached spec. The spec is
            per script and cached on `_flag_spec`, so it is looked up here.

    Returns:
        The argv list, algorithm name in its required position.

    Raises:
        ValueError: If a config key is not a flag that script accepts --
            caught here rather than as an argparse error repeated once per
            cell.
    """
    del spec
    task = dict(cell.get("task", {}))
    script = task.pop("script")
    algorithm = cell.get("algorithm", "admm")
    top, sub = _flag_spec(script)
    has_subcommand = bool(sub)

    settings = {
        **fixed,
        **{k: v for k, v in cell.items() if k not in ("task", "algorithm")},
        **task,
    }
    # `fixed:` is applied to every cell, so a knob belonging to the other
    # algorithm family would otherwise land on this command line, where the
    # subparser now rejects it rather than ignoring it. A script with no
    # subcommand has no families to keep apart, and the membership test
    # below already drops anything it does not accept.
    if has_subcommand:
        drop = _ADMM_ONLY if algorithm != "admm" else _FLAT_ONLY
        settings = {k: v for k, v in settings.items() if k not in drop}

    pre: List[str] = []
    post: List[str] = []
    for key, value in settings.items():
        flag = "--" + key.replace("_", "-")
        if key in top:
            target, takes_value = pre, top[key]
        elif key in sub:
            target, takes_value = post, sub[key]
        else:
            # Not a typo -- `_check_fixed` has already rejected those.
            # `fixed:` applies to every cell, but a 2D script genuinely has
            # no --record and a 3D one no --animate, so a mixed sweep needs
            # the inapplicable ones dropped. `describe_dropped` reports
            # what, once, before the first cell runs.
            continue
        if takes_value and isinstance(value, (list, tuple)):
            # A multi-value flag (`--w-rate fx fy tau`): one token each, or
            # argparse sees the Python repr of the list as a single value.
            target += [flag, *(str(v) for v in value)]
        elif takes_value:
            target += [flag, str(value)]
        elif value:
            target.append(flag)

    if not has_subcommand:
        return [sys.executable, script_path(script), *pre]
    return [sys.executable, script_path(script), *pre, algorithm, *post]


def _gpu_memory() -> Optional[Tuple[int, int]]:
    """`(free, total)` MiB on GPU 0, or None if there is no `nvidia-smi`.

    Returns:
        The reading, or None on a machine without an NVIDIA GPU -- the
        sweep must still run there, just without the memory barrier.
    """
    try:
        out = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=memory.free,memory.total",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            check=True,
            timeout=10,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    free, total = out.stdout.strip().split("\n")[0].split(",")
    return int(free), int(total)


def _await_gpu(timeout: float = 120.0, poll: float = 0.5) -> Optional[int]:
    """Wait for the previous cell's GPU memory to come back.

    A cell is its own process, so the driver reclaims everything it held on
    exit -- measured at well under a second, since the process is only
    reaped after CUDA teardown. Waiting for the reading to *settle* (two
    equal samples) is therefore enough: it catches the reclaim still being
    in flight, and equally a cell that crashed rather than exited, or a
    viewer someone left on the card.

    No headroom threshold: the previous cell releases all of its memory,
    so what the next one needs is not the launcher's business -- and a
    fixed number would be wrong the moment the scene or sample count
    changed.

    Args:
        timeout: Give up waiting after this long and run anyway -- a
            genuine shortage is reported by the cell that fails, which is
            more informative than the launcher refusing to start.
        poll: Seconds between readings.

    Returns:
        Free MiB at the moment of returning, or None without a GPU.
    """
    reading = _gpu_memory()
    if reading is None:
        return None

    deadline = time.time() + timeout
    free, _ = reading
    previous = -1
    while time.time() < deadline:
        if free == previous:
            return free
        previous = free
        time.sleep(poll)
        free = _gpu_memory()[0]

    print(
        f"  warning: GPU memory still changing after {timeout:.0f}s "
        f"({free} MiB free); running anyway"
    )
    return free


def _label(cell: Dict[str, Any]) -> str:
    """A short human-readable name for one cell, for progress output."""
    task = cell.get("task", {})
    parts = [str(task.get("script", "?"))]
    parts += [f"{k}={task[k]}" for k in sorted(task) if k != "script"]
    # Absent for a script with no algorithm subcommand -- `expand` strips
    # it there, and defaulting to "admm" would label an object-only cell
    # with a method it never ran.
    if "algorithm" in cell:
        parts.append(str(cell["algorithm"]))
    parts += [
        f"{k}={cell[k]}"
        for k in (
            "scene", "plant", "horizon", "samples", "n_admm", "rho", "gamma",
            "consensus_alpha", "wrench_fraction", "w_rate",
            "noise_level", "temperature", "start", "goal", "seed",
        )
        if k in cell
    ]
    return " ".join(parts)


def run_sweep(
    combos: Sequence[Dict[str, Any]],
    fixed: Dict[str, Any],
    dry_run: bool = False,
    keep_going: bool = True,
    gpu_timeout: float = 120.0,
) -> List[Dict[str, Any]]:
    """Run every combination, reporting progress and collecting outcomes.

    Args:
        combos: Cells from `expand`.
        fixed: The config's `fixed` block.
        dry_run: Print the commands without running them.
        keep_going: Continue after a cell fails. On by default -- a sweep
            is long and one bad combination should not discard the rest.
        gpu_timeout: How long to wait for the previous cell's GPU memory;
            see `_await_gpu`.

    Returns:
        One record per cell: its settings, command, exit status, duration,
        and the free GPU memory it started with.
    """
    records = []
    for i, cell in enumerate(combos, 1):
        cmd = build_command(cell, fixed)
        print(f"\n[{i}/{len(combos)}] {_label(cell)}")
        print("  " + " ".join(cmd))
        if dry_run:
            records.append({"cell": cell, "command": cmd, "status": "skipped"})
            continue

        # Between cells, not only after: this also catches memory held by
        # something that was not part of this sweep.
        free = _await_gpu(gpu_timeout)
        if free is not None:
            print(f"  {free} MiB free")

        t0 = time.time()
        result = subprocess.run(cmd, check=False)
        elapsed = time.time() - t0
        ok = result.returncode == 0
        print(f"  {'ok' if ok else 'FAILED'} in {elapsed:.1f}s")
        records.append(
            {
                "cell": cell,
                "command": cmd,
                "status": "ok" if ok else "failed",
                "returncode": result.returncode,
                "seconds": elapsed,
                "gpu_free_mib_at_start": free,
            }
        )
        if not ok and not keep_going:
            print("  stopping (--stop-on-error)")
            break
    return records


def _parse_only(only: Sequence[str]) -> Dict[str, str]:
    """Parse repeated `--only key=value` filters."""
    return dict(o.split("=", 1) for o in only)


def _parse_overrides(overrides: Sequence[str]) -> Dict[str, Any]:
    """Parse repeated `--set key=value` into typed values.

    Values go through the YAML loader, so `true`, `50` and `0.1` arrive as
    a bool, an int and a float rather than as strings -- `warp=false` has
    to be falsy, or switching a `fixed:` entry off from the command line
    would silently switch it on.

    Args:
        overrides: Raw `key=value` strings.

    Returns:
        The parsed overrides.

    Raises:
        ValueError: If an entry has no `=`.
    """
    parsed: Dict[str, Any] = {}
    for item in overrides:
        if "=" not in item:
            raise ValueError(f"--set expects KEY=VALUE, got '{item}'")
        key, value = item.split("=", 1)
        parsed[key.strip()] = yaml.safe_load(value)
    return parsed


def _scripts(combos: Sequence[Dict[str, Any]]) -> List[str]:
    """The distinct scripts a set of cells will invoke, in first-seen order."""
    seen: Dict[str, None] = {}
    for cell in combos:
        seen.setdefault(cell.get("task", {}).get("script"), None)
    return [s for s in seen if s]


def _accepted(script: str) -> Set[str]:
    """Every flag dest one script takes, before or after the algorithm."""
    top, sub = _flag_spec(script)
    return set(top) | set(sub)


def _check_fixed(fixed: Dict[str, Any], scripts: Sequence[str]) -> None:
    """Fail early if a `fixed:` key is no script's flag.

    Otherwise a typo surfaces as every cell of a long sweep failing
    identically, minutes apart, with an argparse error buried in each
    subprocess's output. Checked against the union over the sweep's
    scripts, not against one of them: a key only some accept is legitimate
    (see `describe_dropped`), a key none accepts is a mistake.

    Args:
        fixed: The merged `fixed:` block and CLI overrides.
        scripts: The scripts this sweep will invoke.

    Raises:
        ValueError: If any key is no script's flag.
    """
    known: Set[str] = set()
    for script in scripts:
        known |= _accepted(script)
    unknown = sorted(set(fixed) - known)
    if unknown:
        raise ValueError(
            f"not flags of any script in this sweep: {unknown}. "
            f"Known: {sorted(known)}"
        )


def describe_dropped(
    fixed: Dict[str, Any], scripts: Sequence[str]
) -> List[str]:
    """Which `fixed:` keys each script cannot take, and so will not get.

    A 2D script has no `--record` or `--warp`; a 3D one has no `--animate`.
    Dropping them per cell is what lets one `fixed:` block serve a mixed
    sweep -- but dropping anything silently is how a sweep quietly runs
    something other than what was asked for, so this is printed once,
    before the first cell.

    Args:
        fixed: The merged `fixed:` block and CLI overrides.
        scripts: The scripts this sweep will invoke.

    Returns:
        One human-readable line per script that drops something.
    """
    lines = []
    for script in scripts:
        missing = sorted(set(fixed) - _accepted(script))
        if missing:
            lines.append(f"{script}: ignores {', '.join(missing)}")
    return lines


def _apply_only(
    combos: List[Dict[str, Any]], only: Dict[str, str]
) -> List[Dict[str, Any]]:
    """Keep cells matching every filter, comparing as strings."""
    kept = []
    for cell in combos:
        flat = {**cell.get("task", {}), **cell}
        if all(str(flat.get(k)) == v for k, v in only.items()):
            kept.append(cell)
    return kept


def main() -> None:
    """Parse arguments, expand the sweep, run it, save a manifest."""
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--config",
        default="run_launch_config",
        help="Sweep config: a path, or a name under oim/configs/.",
    )
    p.add_argument(
        "--only",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="Keep only cells matching this; repeatable.",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the commands and the cell count, run nothing.",
    )
    p.add_argument(
        "--stop-on-error",
        action="store_true",
        help="Abort the sweep on the first failing cell.",
    )
    p.add_argument(
        "--manifest-dir",
        default=os.path.join(ROOT, "results", "sweeps"),
        help="Where to write the record of what ran.",
    )
    p.add_argument(
        "--gpu-timeout",
        type=float,
        default=120.0,
        help="Seconds to wait for GPU memory before running anyway.",
    )
    p.add_argument(
        "--warp",
        action="store_true",
        help="Run every cell on the MuJoCo Warp backend "
        "(shorthand for --set warp=true).",
    )
    p.add_argument(
        "--set",
        action="append",
        default=[],
        dest="overrides",
        metavar="KEY=VALUE",
        help="Override a `fixed:` entry for this sweep only; repeatable. "
        "Any flag of the sweep's scripts works, e.g. --set steps=50 "
        "--set record=false.",
    )
    args = p.parse_args()

    cfg = load_config(args.config)
    combos = expand(cfg.get("sweep", {}))
    if args.only:
        combos = _apply_only(combos, _parse_only(args.only))

    fixed = dict(cfg.get("fixed", {}))
    if args.warp:
        fixed["warp"] = True
    try:
        fixed.update(_parse_overrides(args.overrides))
        _check_fixed(fixed, _scripts(combos))
    except ValueError as e:
        p.error(str(e))

    for line in describe_dropped(fixed, _scripts(combos)):
        print(f"note: {line}")

    if not combos:
        print("no cells to run (check the sweep config and --only filters)")
        return

    print(f"{len(combos)} cells from {args.config}")
    t0 = time.time()
    records = run_sweep(
        combos,
        fixed,
        dry_run=args.dry_run,
        keep_going=not args.stop_on_error,
        gpu_timeout=args.gpu_timeout,
    )
    total = time.time() - t0

    failed = [r for r in records if r["status"] == "failed"]
    print(
        f"\n{len(records)} cells in {total / 60:.1f} min, {len(failed)} failed"
    )
    for r in failed:
        print(f"  FAILED: {_label(r['cell'])}")
    if args.dry_run:
        return

    os.makedirs(args.manifest_dir, exist_ok=True)
    stamp = time.strftime("%Y%m%d_%H%M%S")
    path = os.path.join(args.manifest_dir, f"sweep_{stamp}.json")
    with open(path, "w") as f:
        json.dump(
            {"config": args.config, "fixed": fixed, "runs": records},
            f,
            indent=2,
        )
    print(f"saved manifest to {path}")
    print("\nnext: uv run python -m oim.run_eval")


if __name__ == "__main__":
    main()
