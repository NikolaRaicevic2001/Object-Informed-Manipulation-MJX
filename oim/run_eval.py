r"""Compute metrics over saved runs -- without re-running any experiment.

An `examples/` script produces one run file per run; `oim/run_launch.py`
produces many. This reads them, derives every metric on the fly via
`oim.utils.metrics`, and emits the paper's table: one block per task, one
row per method inside it, and a final `Mean` block averaging each method
over the tasks.

    # every run under oim/results/runs
    uv run python -m oim.run_eval

    # the five tabletop tasks, ADMM against flat MPPI, as LaTeX
    uv run python -m oim.run_eval --format latex \
        --filter task=xarm6_open_table,xarm6_shelf_gap \
        --filter algorithm=admm,mppi

    # ablate a swept setting as method rows (pin the rest with --filter)
    uv run python -m oim.run_eval --ablate rho \
        --filter n_admm=4 --filter gamma=0.1 --format latex --plot

    # split a setting into row blocks instead of method rows
    uv run python -m oim.run_eval --group-by task horizon

    # re-score old runs against a stricter success tolerance
    uv run python -m oim.run_eval --pos-tol 0.02 --theta-tol 0.02

Everything not grouped on and not ablated is *averaged into* the cell.
`--filter` pins a value, `--ablate` folds a field into the method label,
and `--group-by` splits it into row blocks. Whatever still varies is
printed above the table so a mixed cell is never silently reported as a
clean one.

Nothing here touches the simulator, JAX or MuJoCo: changing a metric, a
tolerance, or how results are grouped costs a second, not a GPU-week. That
is the entire reason running and scoring are separate programs.
"""

import argparse
import glob
import json
import os
from collections import defaultdict
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

from oim import ROOT
from oim.utils.metrics import (
    aggregate_metrics,
    aggregate_step_series,
    step_series,
    trial_metrics,
)
from oim.utils.results import RunName, load_run

# Label of the block that averages each method over the row groups.
MEAN_LABEL = "Mean"

# Recorded fields that `method` is built from, so they name a row rather
# than being averaged into one.
_METHOD_FIELDS = frozenset(
    {"method", "algorithm", "robot_opt", "object_opt"}
)

# Most distinct values of a field to name in the "averaged over" line
# before summarising the rest as a count.
_MAX_LISTED_VALUES = 6

# Columns of the emitted table, as (header, metric key, format). Matches
# the paper's simulation table, plus the trial count and the orientation
# error the paper omits. Standard deviations are in the JSON, not here.
_COLUMNS: Tuple[Tuple[str, str, str], ...] = (
    ("n", "n_trials", "d"),
    ("SR", "success_rate", ".2f"),
    ("eps_d", "pos_err_mean", ".3f"),
    ("eps_d^s", "pos_err_mean_success", ".3f"),
    ("theta", "theta_err_mean", ".3f"),
    # Steps to the goal, censored at the run's step cap when it never got
    # there -- see `_steps_to_goal`. Distinct from T, which is wall-clock
    # simulated time credited the sweep-wide worst case.
    ("steps", "mean_steps_to_goal", ".0f"),
    ("f (Hz)", "mean_frequency_hz", ".2f"),
    ("T (s)", "mean_execution_time", ".2f"),
)

_LATEX_HEADERS = {
    "n": r"$n$",
    "SR": r"SR $\uparrow$",
    "eps_d": r"$\epsilon_d$ $\downarrow$",
    "eps_d^s": r"$\epsilon_d^s$ $\downarrow$",
    "theta": r"$\epsilon_\theta$ $\downarrow$",
    "steps": r"$N$ $\downarrow$",
    "f (Hz)": r"$\bar{f}$ $\uparrow$",
    "T (s)": r"$T$ $\downarrow$",
}


def _run_fields(
    run: Dict[str, Any], ablate: Sequence[str] = ()
) -> Dict[str, Any]:
    """Flatten a run's identity and settings into one lookup table.

    Adds a derived `method` field -- the algorithm plus, for ADMM, its two
    block optimizers -- since that is what a table column compares and no
    single recorded field carries it. Fields named in `ablate` are appended
    as `key=value` so a sweep over e.g. `rho` becomes distinct method rows
    rather than being averaged into one ADMM cell. Values of `None` are
    skipped so a flat baseline stays `mppi` when ablating an ADMM-only knob.
    """
    fields = dict(run.get("run", {}))
    fields.update(run.get("hyperparameters", {}))
    algorithm = fields.get("algorithm")
    robot_opt, object_opt = fields.get("robot_opt"), fields.get("object_opt")
    # Only a consensus algorithm has two blocks; a flat baseline records
    # `object_opt: None` and is named by its algorithm alone.
    method = (
        f"{algorithm}({robot_opt}/{object_opt})"
        if object_opt
        else str(algorithm)
    )
    extras = [
        f"{key}={fields[key]}"
        for key in ablate
        if key in fields and fields[key] is not None
    ]
    if extras:
        method = f"{method} {' '.join(extras)}"
    fields["method"] = method
    return fields


def parse_filters(items: Sequence[str]) -> Dict[str, Set[str]]:
    """Parse `--filter key=a,b` and repeated flags into `{key: {values}}`.

    Values accumulate rather than overwrite, so `--filter task=a --filter
    task=b` and `--filter task=a,b` mean the same thing: *any of these*.
    Overwriting would silently evaluate a different set of runs than the
    one asked for.

    Args:
        items: Raw `key=value[,value...]` strings.

    Returns:
        One set of accepted values per field.

    Raises:
        ValueError: If an entry has no `=`.
    """
    parsed: Dict[str, Set[str]] = defaultdict(set)
    for item in items:
        if "=" not in item:
            raise ValueError(f"--filter expects KEY=VALUE, got '{item}'")
        key, values = item.split("=", 1)
        parsed[key.strip()].update(v.strip() for v in values.split(","))
    return dict(parsed)


def _matches(
    run: Dict[str, Any],
    filters: Dict[str, Set[str]],
    ablate: Sequence[str] = (),
) -> bool:
    """Whether a run is any of the accepted values for every filtered field.

    Within a field the values are OR-ed, across fields AND-ed -- so
    `--filter task=a,b --filter algorithm=admm` reads as "either task, but
    only ADMM".
    """
    fields = _run_fields(run, ablate)
    return all(str(fields.get(k)) in values for k, values in filters.items())


def averaged_fields(
    runs: List[Dict[str, Any]],
    group_by: Sequence[str],
    ablate: Sequence[str] = (),
) -> Dict[str, List[str]]:
    """Settings that vary across `runs` but are not grouped on or ablated.

    Every one of these is averaged into the cells rather than splitting
    them, which is the point of the table -- but it also means a cell can
    silently mix, say, two horizons. Reporting them alongside the table is
    what keeps that honest.

    `algorithm`/`robot_opt`/`object_opt` and every ablated field are
    excluded: they vary across any real comparison, but they are not
    averaged over -- they are exactly what `method` splits the rows by.

    Args:
        runs: Payloads from `oim.utils.results.load_run`.
        group_by: The fields forming the row groups.
        ablate: Fields folded into the method label.

    Returns:
        `{field: sorted distinct values}`, for the fields that vary.
    """
    fields = [_run_fields(r, ablate) for r in runs]
    fixed = set(group_by) | _METHOD_FIELDS | set(ablate)
    out: Dict[str, List[str]] = {}
    for key in {k for f in fields for k in f}:
        if key in fixed:
            continue
        seen = sorted({str(f.get(key)) for f in fields})
        if len(seen) > 1:
            out[key] = seen
    return out


def _mean_over_groups(cells: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Average one method's per-group metrics into its `Mean` row.

    Unweighted: every row group counts once, whatever its trial count, so a
    task that happened to get more seeds does not dominate. `n_trials` is
    summed instead (it is a count, not a score), and `n_groups` records how
    many rows went in. A metric that is undefined in some groups -- e.g.
    `pos_err_mean_success` where nothing succeeded -- averages over the
    groups where it is defined, and stays `None` if that is none of them.

    Args:
        cells: `aggregate_metrics` output, one per row group.

    Returns:
        One aggregate in the same shape, plus `n_groups`.
    """
    out: Dict[str, Any] = {"n_groups": len(cells)}
    for key in {k for c in cells for k in c}:
        values = [c[key] for c in cells if c.get(key) is not None]
        if not values:
            out[key] = None
        elif key == "n_trials":
            out[key] = sum(values)
        else:
            out[key] = sum(values) / len(values)
    return out


def evaluate(
    runs: List[Dict[str, Any]],
    group_by: Sequence[str] = ("task",),
    shared_max_time: bool = True,
    ablate: Sequence[str] = (),
) -> Dict[str, Dict[str, Dict[str, Any]]]:
    """Aggregate runs into one cell per (row group, method), plus `Mean`.

    Args:
        runs: Payloads from `oim.utils.results.load_run`.
        group_by: Fields forming each row group -- the table's left column.
            Default `("task",)`. `method` is always the inner dimension and
            need not be listed. Any field not named here (and not ablated)
            is averaged into the cells; see `averaged_fields`.
        shared_max_time: Credit unsuccessful trials the slowest execution
            time *across every loaded run* rather than within their own
            cell. Keeps the T column comparable between methods: scored per
            cell, a method that fails quickly and consistently would be
            credited its own short worst case and beat one that nearly
            succeeded.
        ablate: Fields folded into each run's method label so a sweep over
            e.g. `rho` produces one method row per value instead of mixing
            them.

    Returns:
        `{row label: {method: metrics}}`, row groups sorted, with
        `MEAN_LABEL` last.

    Raises:
        ValueError: If `runs` is empty.
    """
    if not runs:
        raise ValueError("evaluate needs at least one run")

    trials: Dict[Tuple[str, str], List[Dict[str, Any]]] = defaultdict(list)
    for run in runs:
        fields = _run_fields(run, ablate)
        row = "/".join(str(fields.get(k)) for k in group_by)
        trials[(row, fields["method"])].append(trial_metrics(run))

    max_time = None
    if shared_max_time:
        max_time = max(
            t["execution_time"] for ts in trials.values() for t in ts
        )

    table: Dict[str, Dict[str, Dict[str, Any]]] = defaultdict(dict)
    for (row, method), ts in trials.items():
        table[row][method] = aggregate_metrics(ts, max_time=max_time)

    out = {row: dict(sorted(table[row].items())) for row in sorted(table)}

    # The Mean block: each method averaged over the row groups it appears
    # in. Omitted for a single row group, where it would just repeat it.
    if len(out) > 1:
        by_method: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        for methods in out.values():
            for method, metrics in methods.items():
                by_method[method].append(metrics)
        out[MEAN_LABEL] = {
            method: _mean_over_groups(cells)
            for method, cells in sorted(by_method.items())
        }
    return out


def evaluate_step_curves(
    runs: List[Dict[str, Any]],
    group_by: Sequence[str] = ("task",),
    ablate: Sequence[str] = (),
) -> Dict[str, Dict[str, Dict[str, Any]]]:
    """Aggregate per-step series into one cell per (row group, method).

    Same grouping as `evaluate`, without a `Mean` block (curves do not
    average across tasks). Used by `--plot`.

    Args:
        runs: Payloads from `oim.utils.results.load_run`.
        group_by: Fields forming each row group.
        ablate: Fields folded into the method label.

    Returns:
        `{row label: {method: aggregate_step_series output}}`.

    Raises:
        ValueError: If `runs` is empty.
    """
    if not runs:
        raise ValueError("evaluate_step_curves needs at least one run")

    trials: Dict[Tuple[str, str], List[Dict[str, Any]]] = defaultdict(list)
    dts: Dict[Tuple[str, str], List[float]] = defaultdict(list)
    for run in runs:
        fields = _run_fields(run, ablate)
        row = "/".join(str(fields.get(k)) for k in group_by)
        key = (row, fields["method"])
        trials[key].append(step_series(run))
        dts[key].append(float(run["hyperparameters"]["control_dt"]))

    table: Dict[str, Dict[str, Dict[str, Any]]] = defaultdict(dict)
    for (row, method), series in trials.items():
        control_dt = dts[(row, method)][0]
        table[row][method] = aggregate_step_series(series, control_dt)
    return {
        row: dict(sorted(methods.items()))
        for row, methods in sorted(table.items())
    }


def _known_fields(runs: Sequence[Dict[str, Any]]) -> Set[str]:
    """Union of keys available via `_run_fields` across `runs`."""
    return {k for r in runs for k in _run_fields(r)}


def validate_ablate(
    runs: Sequence[Dict[str, Any]], ablate: Sequence[str]
) -> None:
    """Reject ablate keys no loaded run records.

    Args:
        runs: Loaded run payloads (before or after filtering).
        ablate: Field names from `--ablate`.

    Raises:
        ValueError: If any ablate field is unknown.
    """
    if not ablate:
        return
    known = _known_fields(runs)
    unknown = sorted(set(ablate) - known)
    if unknown:
        raise ValueError(
            f"no run records the ablate field(s) {unknown}. "
            f"Available: {sorted(known)}"
        )


def _strip_common_prefix(labels: Sequence[str]) -> Tuple[List[str], str]:
    """Drop the `_`-delimited prefix every label shares.

    Task labels are built as `{robot}_{scene}`, so a table of five
    scenes repeats `xarm6_` in every row and the column that
    actually distinguishes them gets pushed off to the right. The stripped
    prefix is returned so it can be stated once above the table instead.

    Args:
        labels: The row labels.

    Returns:
        The shortened labels and the removed prefix (`""` if none).
    """
    parts = [label.split("_") for label in labels]
    if len(parts) < 2:
        return list(labels), ""
    common = 0
    while (
        all(len(p) > common + 1 for p in parts)
        and len({p[common] for p in parts}) == 1
    ):
        common += 1
    if not common:
        return list(labels), ""
    return (
        ["_".join(p[common:]) for p in parts],
        "_".join(parts[0][:common]) + "_",
    )


def _rows(
    summary: Dict[str, Dict[str, Dict[str, Any]]],
) -> Tuple[List[Tuple[str, str, List[str]]], str]:
    r"""Flatten the summary into (row label, method, formatted cells).

    The row label is blank on all but the first method of each block, the
    way the paper's `\multirow` renders it.
    """
    groups = [g for g in summary if g != MEAN_LABEL]
    short, prefix = _strip_common_prefix(groups)
    labels = dict(zip(groups, short, strict=True))
    labels[MEAN_LABEL] = MEAN_LABEL

    rows = []
    for group, methods in summary.items():
        for i, (method, metrics) in enumerate(methods.items()):
            cells = []
            for _, key, fmt in _COLUMNS:
                value = metrics.get(key)
                cells.append("-" if value is None else format(value, fmt))
            rows.append((labels[group] if i == 0 else "", method, cells))
    return rows, prefix


def format_table(
    summary: Dict[str, Dict[str, Dict[str, Any]]],
    group_by: Sequence[str] = ("task",),
    style: str = "text",
) -> str:
    """Render the summary as text, markdown or a LaTeX tabular.

    Args:
        summary: `evaluate` output.
        group_by: The fields its row labels were built from, for the header.
        style: `"text"`, `"markdown"` or `"latex"`.

    Returns:
        The rendered table.

    Raises:
        ValueError: If `style` is unknown.
    """
    rows, prefix = _rows(summary)
    group_header = "/".join(group_by)
    headers = [group_header, "method"] + [c[0] for c in _COLUMNS]
    note = f"({group_header} prefix '{prefix}' omitted)\n\n" if prefix else ""

    if style == "markdown":
        body = [
            "| " + " | ".join(headers) + " |",
            "| " + " | ".join("---" for _ in headers) + " |",
        ]
        body += [
            "| " + " | ".join([g, m, *cells]) + " |" for g, m, cells in rows
        ]
        return note + "\n".join(body)

    if style == "latex":
        return note + _latex_table(summary, group_header)

    if style != "text":
        raise ValueError(f"unknown table style {style!r}")

    table = [[g, m, *cells] for g, m, cells in rows]
    widths = [
        max(len(headers[i]), *(len(r[i]) for r in table))
        for i in range(len(headers))
    ]

    def _line(cells: Sequence[str]) -> str:
        pairs = zip(cells, widths, strict=True)
        return "  ".join(c.ljust(w) for c, w in pairs)

    head = _line(headers)
    out = [head, "-" * len(head)]
    for i, row in enumerate(table):
        # A rule before each new block, so the eye can find them.
        if i and row[0]:
            out.append("-" * len(head))
        out.append(_line(row))
    return note + "\n".join(out)


def _latex_table(
    summary: Dict[str, Dict[str, Dict[str, Any]]], group_header: str
) -> str:
    r"""A `tabular` body using `\multirow` for the row-group column."""
    rows, _ = _rows(summary)
    header = " & ".join(
        [group_header, "Method"] + [_LATEX_HEADERS[c[0]] for c in _COLUMNS]
    )
    # How many method rows each block spans, for \multirow.
    spans = [len(methods) for methods in summary.values()]

    out = [
        r"\begin{tabular}{@{}ll" + "c" * len(_COLUMNS) + r"@{}}",
        r"    \toprule",
        f"    {header} \\\\",
        r"    \midrule",
    ]
    block = -1
    for group, method, cells in rows:
        if group:
            block += 1
            if block:
                # \midrule\midrule sets the Mean block apart, as the paper
                # does; a single rule between the task blocks.
                out.append(
                    r"    \midrule\midrule"
                    if group == MEAN_LABEL
                    else r"    \midrule"
                )
            label = rf"\multirow{{{spans[block]}}}{{*}}{{{group}}}"
        else:
            label = ""
        escaped = [c.replace("_", r"\_") for c in [label, method]]
        # "--" rather than the plain-text "-": that is how the paper marks
        # a metric with no defined value, and a lone hyphen sets as one.
        shown = ["--" if c == "-" else c for c in cells]
        out.append("    " + " & ".join(escaped + shown) + r" \\")
    out += [r"    \bottomrule", r"\end{tabular}"]
    return "\n".join(out)


def load_runs(
    runs_dir: str,
    filters: Optional[Dict[str, Set[str]]] = None,
    pos_tol: Optional[float] = None,
    theta_tol: Optional[float] = None,
) -> List[Dict[str, Any]]:
    """Load every run file under `runs_dir`, optionally filtered and re-scored.

    Args:
        runs_dir: Directory of `save_run` JSON files, searched
            recursively.
        filters: `{field: {accepted values}}` from `parse_filters`,
            compared as strings.
        pos_tol: Override the recorded positional success tolerance.
        theta_tol: Override the recorded angular one. Overriding either
            re-scores old runs under a new definition of success, which is
            the point of storing poses rather than a `reached` flag.

    Returns:
        The matching payloads.

    Raises:
        FileNotFoundError: If no run files match.
        ValueError: If a filter names a field no run records.
    """
    # Recursive, so one function reads both layouts: the sim path writes
    # flat into results/runs, the real path nests under
    # results/real/{algorithm}/{scene}/{date}/. Point `runs_dir` at any level
    # of that tree. Do NOT point it at `results/` itself -- results/object
    # holds runs this table is deliberately not allowed to average in (see
    # `Experiment.results_dir`).
    paths = sorted(
        glob.glob(os.path.join(runs_dir, "**", "*.json"), recursive=True)
    )
    loaded = [load_run(p) for p in paths]
    if filters:
        known = _known_fields(loaded)
        unknown = sorted(set(filters) - known)
        if unknown:
            raise ValueError(
                f"no run records the field(s) {unknown}. "
                f"Available: {sorted(known)}"
            )

    runs = []
    for run in loaded:
        if filters and not _matches(run, filters):
            continue
        if pos_tol is not None:
            run["hyperparameters"]["goal_pos_tol"] = pos_tol
        if theta_tol is not None:
            run["hyperparameters"]["goal_theta_tol"] = theta_tol
        runs.append(run)
    if not runs:
        raise FileNotFoundError(
            f"no run files matched in {runs_dir} "
            f"({len(paths)} present, {filters or 'no'} filters)"
        )
    return runs


def _describe(averaged: Dict[str, List[str]]) -> str:
    """One line naming what got averaged into every cell.

    Long value lists are summarised by count -- a sweep over 20 seeds says
    so rather than printing all 20 and burying the fields that matter.

    Args:
        averaged: `averaged_fields` output.

    Returns:
        A comma-separated summary.
    """
    parts = []
    for key, values in sorted(averaged.items()):
        if len(values) > _MAX_LISTED_VALUES:
            parts.append(f"{key} ({len(values)} values)")
        else:
            parts.append(f"{key} ({', '.join(values)})")
    return ", ".join(parts)


def _build_parser() -> argparse.ArgumentParser:
    """CLI for `python -m oim.run_eval`."""
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--runs-dir",
        default=os.path.join(ROOT, "results", "runs"),
        help="Directory of run files written by the examples/ scripts.",
    )
    p.add_argument(
        "--group-by",
        nargs="+",
        default=["task"],
        help="Fields forming each row block (default: task). Methods are "
        "always the rows within a block; everything else is averaged in.",
    )
    p.add_argument(
        "--ablate",
        nargs="+",
        default=[],
        metavar="FIELD",
        help="Fold these fields into the method label (e.g. rho) so each "
        "value is its own row instead of being averaged. Pin other swept "
        "axes with --filter.",
    )
    p.add_argument(
        "--filter",
        action="append",
        default=[],
        metavar="KEY=VALUE[,VALUE...]",
        help="Keep only runs whose field is one of these values; "
        "repeatable, and comma-separated lists are accepted. Values of one "
        "field are OR-ed, different fields AND-ed.",
    )
    p.add_argument("--pos-tol", type=float, help="Re-score success with this.")
    p.add_argument("--theta-tol", type=float, help="Re-score success.")
    p.add_argument(
        "--format",
        choices=["text", "markdown", "latex"],
        default="text",
        help="Extra table style to save beside the always-written .txt "
        "(stdout always prints the human-readable text table).",
    )
    p.add_argument(
        "--plot",
        action="store_true",
        help="Write a step-curve figure (eps_d, eps_theta, primal/dual "
        "residuals) under --out-dir.",
    )
    p.add_argument(
        "--out-dir",
        default=os.path.join(ROOT, "results", "eval"),
        help="Where to write the summary JSON, table, and optional figure.",
    )
    p.add_argument("--no-save", action="store_true")
    return p


def _print_header(
    runs: List[Dict[str, Any]],
    summary: Dict[str, Dict[str, Dict[str, Any]]],
    group_by: Sequence[str],
    ablate: Sequence[str],
    filters: Dict[str, Set[str]],
    averaged: Dict[str, List[str]],
) -> None:
    """Print run counts, ablate/filter pins, and averaged-over warnings."""
    n_rows = len([g for g in summary if g != MEAN_LABEL])
    print(f"\n{len(runs)} runs -> {n_rows} {'/'.join(group_by)} groups")
    if ablate:
        print(f"ablate: {', '.join(ablate)}")
    if filters:
        print(
            "filter: "
            + ", ".join(
                f"{k}={','.join(sorted(v))}" for k, v in sorted(filters.items())
            )
        )
    if averaged:
        print(f"averaged over: {_describe(averaged)}")


def _maybe_plot(
    runs: List[Dict[str, Any]],
    group_by: Sequence[str],
    ablate: Sequence[str],
    filters: Dict[str, Set[str]],
    out_dir: str,
    stem: str,
) -> Optional[str]:
    """Write the step-curve figure when requested; return its path or None."""
    from oim.utils.eval_plots import plot_step_curves  # noqa: PLC0415

    curves = evaluate_step_curves(runs, group_by, ablate=ablate)
    os.makedirs(out_dir, exist_ok=True)
    plot_path = os.path.join(out_dir, f"{stem}.png")
    plot_step_curves(
        curves,
        plot_path,
        group_by=group_by,
        ablate=ablate,
        filters={k: sorted(v) for k, v in filters.items()},
    )
    print(f"\nsaved {plot_path}")
    return plot_path


def _save_outputs(
    *,
    out_dir: str,
    stem: str,
    runs_dir: str,
    runs: List[Dict[str, Any]],
    filters: Dict[str, Set[str]],
    group_by: Sequence[str],
    ablate: Sequence[str],
    averaged: Dict[str, List[str]],
    pos_tol: Optional[float],
    theta_tol: Optional[float],
    summary: Dict[str, Dict[str, Dict[str, Any]]],
    table: str,
    text_table: str,
    fmt: str,
    plot_path: Optional[str],
) -> None:
    """Write the eval JSON, a human-readable `.txt` table, and `--format`."""
    os.makedirs(out_dir, exist_ok=True)
    json_path = os.path.join(out_dir, f"{stem}.json")
    with open(json_path, "w") as f:
        json.dump(
            {
                "runs_dir": runs_dir,
                "n_runs": len(runs),
                "filters": {k: sorted(v) for k, v in filters.items()},
                "group_by": list(group_by),
                "ablate": list(ablate),
                "averaged_over": averaged,
                "goal_pos_tol": pos_tol,
                "goal_theta_tol": theta_tol,
                "results": summary,
                "plot": plot_path,
            },
            f,
            indent=2,
        )
    # Always write a plain-text table so results are readable without LaTeX
    # or markdown tooling; `--format` adds a second file when it differs.
    text_path = os.path.join(out_dir, f"{stem}.txt")
    with open(text_path, "w") as f:
        f.write(text_table + "\n")
    saved = [json_path, text_path]
    if fmt != "text":
        ext = {"markdown": "md", "latex": "tex"}[fmt]
        table_path = os.path.join(out_dir, f"{stem}.{ext}")
        with open(table_path, "w") as f:
            f.write(table + "\n")
        saved.append(table_path)
    print("\n" + "\n".join(f"saved {p}" for p in saved))


def main() -> None:
    """Parse arguments, aggregate the runs, print and save the table."""
    p = _build_parser()
    args = p.parse_args()

    try:
        filters = parse_filters(args.filter)
        runs = load_runs(args.runs_dir, filters, args.pos_tol, args.theta_tol)
        validate_ablate(runs, args.ablate)
    except (ValueError, FileNotFoundError) as e:
        p.error(str(e))

    summary = evaluate(runs, args.group_by, ablate=args.ablate)
    text_table = format_table(summary, args.group_by, style="text")
    table = (
        text_table
        if args.format == "text"
        else format_table(summary, args.group_by, style=args.format)
    )
    averaged = averaged_fields(runs, args.group_by, ablate=args.ablate)
    _print_header(
        runs, summary, args.group_by, args.ablate, filters, averaged
    )
    print()
    # Always show the human-readable table in the terminal; LaTeX/markdown
    # stay in the saved files so the console stays scannable.
    print(text_table)

    name = RunName("eval")
    stem = name()
    plot_path = None
    if args.plot:
        plot_path = _maybe_plot(
            runs,
            args.group_by,
            args.ablate,
            filters,
            args.out_dir,
            stem,
        )
    if args.no_save:
        return
    _save_outputs(
        out_dir=args.out_dir,
        stem=stem,
        runs_dir=args.runs_dir,
        runs=runs,
        filters=filters,
        group_by=args.group_by,
        ablate=args.ablate,
        averaged=averaged,
        pos_tol=args.pos_tol,
        theta_tol=args.theta_tol,
        summary=summary,
        table=table,
        text_table=text_table,
        fmt=args.format,
        plot_path=plot_path,
    )


if __name__ == "__main__":
    main()
