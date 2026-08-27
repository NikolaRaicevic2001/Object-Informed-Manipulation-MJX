"""Tests for the run-file evaluation in `oim/run_eval.py`.

Built on synthetic run payloads rather than real runs: the point of
separating running from scoring is that scoring is a pure function of the
recorded JSON, so it can be tested without a simulator.
"""

import math
from typing import Any, Dict, List, Optional

import numpy as np
import pytest
from PIL import Image

from oim.run_eval import (
    MEAN_LABEL,
    _build_parser,
    _describe,
    _run_fields,
    _strip_common_prefix,
    averaged_fields,
    evaluate,
    evaluate_step_curves,
    format_table,
    parse_filters,
    reference_values,
    validate_ablate,
)
from oim.utils.eval_plots import plot_step_curves
from oim.utils.metrics import step_series, trial_metrics


def make_run(
    task: str,
    algorithm: str = "admm",
    object_opt: Optional[str] = "mppi",
    seed: int = 0,
    steps: int = 10,
    final_pos_err: float = 0.5,
    final_theta_err: float = 0.0,
    compute_time: float = 0.5,
    horizon: int = 16,
    rho: Optional[float] = 10.0,
    n_admm: Optional[int] = 4,
    gamma: Optional[float] = 0.1,
    reach_at: Optional[int] = None,
    primal_residual: Optional[List[float]] = None,
) -> Dict[str, Any]:
    """A minimal run payload shaped like `oim.utils.results.save_run`.

    The object walks in a straight line along +x, ending `final_pos_err`
    from a goal at the origin, so `trial_metrics` sees a plausible series
    rather than a constant. `reach_at` forces success from that step onward
    by dropping the pose error below the tolerance.
    """
    # Flat baselines record ADMM-only knobs as None, matching experiment.py.
    if algorithm != "admm":
        rho = n_admm = gamma = None
        object_opt = None

    poses = []
    for i in range(steps):
        if reach_at is not None and i + 1 >= reach_at:
            poses.append([0.01, 0.0, 0.0])
        else:
            poses.append(
                [final_pos_err * (i + 1) / steps, 0.0, final_theta_err]
            )
    dynamic: Dict[str, Any] = {
        "object_pose": [[0.0, 0.0, final_theta_err], *poses],
        "compute_time": [compute_time] * steps,
    }
    if algorithm == "admm":
        dynamic["primal_residual"] = (
            primal_residual
            if primal_residual is not None
            else [3.0 - 0.1 * i for i in range(steps)]
        )
        dynamic["dual_residual"] = [1.5 - 0.05 * i for i in range(steps)]
    return {
        "run": {
            "world": "3d",
            "task": task,
            "robot": "xarm6",
            "algorithm": algorithm,
            "robot_opt": "mppi" if algorithm == "admm" else None,
            "object_opt": object_opt,
            "seed": seed,
        },
        "hyperparameters": {
            "steps": steps,
            "horizon": horizon,
            "control_dt": 0.05,
            "goal_pos_tol": 0.05,
            "goal_theta_tol": 0.05,
            "rho": rho,
            "n_admm": n_admm,
            "gamma": gamma,
        },
        "static": {"goal": [0.0, 0.0, 0.0]},
        "dynamic": dynamic,
    }


def _sr(summary: Dict[str, Any], group: str, method: str) -> float:
    return summary[group][method]["success_rate"]


def test_method_label_distinguishes_admm_blocks_from_flat() -> None:
    """A flat baseline is named by its algorithm, ADMM by its two blocks."""
    summary = evaluate(
        [
            make_run("t1", algorithm="admm", object_opt="mppi"),
            make_run("t1", algorithm="mppi", object_opt=None),
        ]
    )
    assert set(summary["t1"]) == {"admm(mppi/mppi)", "mppi"}


def test_settings_that_are_not_grouped_on_are_averaged_not_split() -> None:
    """Two horizons and three seeds still make one cell, not six.

    This is the whole point of the table: a sweep collapses to one number
    per (task, method), rather than a row per hyperparameter combination.
    """
    runs = [
        make_run("t1", seed=s, horizon=h)
        for s in (0, 1, 2)
        for h in (16, 32)
    ]
    summary = evaluate(runs)
    assert list(summary) == ["t1"]
    assert summary["t1"]["admm(mppi/mppi)"]["n_trials"] == 6


def test_group_by_splits_a_setting_back_out() -> None:
    """`--group-by task horizon` ablates what the default averages over."""
    runs = [make_run("t1", seed=s, horizon=h) for s in (0, 1) for h in (16, 32)]
    summary = evaluate(runs, group_by=("task", "horizon"))
    assert set(summary) == {"t1/16", "t1/32", MEAN_LABEL}
    assert summary["t1/16"]["admm(mppi/mppi)"]["n_trials"] == 2


def test_averaged_fields_reports_what_was_mixed_into_a_cell() -> None:
    """A cell mixing two horizons says so; a fixed setting is not listed."""
    runs = [make_run("t1", seed=s, horizon=h) for s in (0, 1) for h in (16, 32)]
    averaged = averaged_fields(runs, ("task",))
    assert averaged["horizon"] == ["16", "32"]
    assert averaged["seed"] == ["0", "1"]
    assert "control_dt" not in averaged
    assert "task" not in averaged


def test_mean_block_is_unweighted_over_tasks() -> None:
    """Each task counts once in `Mean`, whatever its trial count.

    Matching the paper's convention. Weighting by trials would let one
    heavily-seeded task dominate the headline number, which is exactly the
    comparison the table exists to prevent.
    """
    # t1: 4 trials, all succeed. t2: 1 trial, fails.
    runs = [make_run("t1", seed=s, final_pos_err=0.01) for s in range(4)]
    runs.append(make_run("t2", seed=0, final_pos_err=0.9))
    summary = evaluate(runs)

    assert _sr(summary, "t1", "admm(mppi/mppi)") == 1.0
    assert _sr(summary, "t2", "admm(mppi/mppi)") == 0.0
    # Unweighted: (1.0 + 0.0) / 2, not the trial-weighted 4/5.
    assert _sr(summary, MEAN_LABEL, "admm(mppi/mppi)") == pytest.approx(0.5)
    # n_trials is a count, so it sums rather than averaging.
    assert summary[MEAN_LABEL]["admm(mppi/mppi)"]["n_trials"] == 5
    assert summary[MEAN_LABEL]["admm(mppi/mppi)"]["n_groups"] == 2


def test_mean_matches_the_mean_of_the_task_rows() -> None:
    """Every displayed column of `Mean` is the mean of the column above."""
    runs = [
        make_run(f"t{i}", seed=s, final_pos_err=0.1 * i, compute_time=0.1 * i)
        for i in (1, 2, 3)
        for s in (0, 1)
    ]
    summary = evaluate(runs)
    method = "admm(mppi/mppi)"
    for key in ("success_rate", "pos_err_mean", "mean_frequency_hz"):
        per_task = [summary[f"t{i}"][method][key] for i in (1, 2, 3)]
        assert summary[MEAN_LABEL][method][key] == pytest.approx(
            float(np.mean(per_task))
        )


def test_mean_skips_groups_where_a_metric_is_undefined() -> None:
    """`eps_d^s` averages over tasks that had a success, not over zeros.

    A task where nothing succeeded has no success-conditioned error at all;
    treating that as 0 would make a method look better the more often it
    failed.
    """
    runs = [make_run("t1", seed=0, final_pos_err=0.01)]  # succeeds
    runs.append(make_run("t2", seed=0, final_pos_err=0.9))  # fails
    summary = evaluate(runs)
    method = "admm(mppi/mppi)"
    assert summary["t2"][method]["pos_err_mean_success"] is None
    mean = summary[MEAN_LABEL][method]["pos_err_mean_success"]
    assert mean == pytest.approx(summary["t1"][method]["pos_err_mean_success"])


def test_mean_is_none_when_no_group_defines_a_metric() -> None:
    """All-failing everywhere leaves the success column empty, not zero."""
    runs = [make_run(f"t{i}", final_pos_err=0.9) for i in (1, 2)]
    summary = evaluate(runs)
    mean = summary[MEAN_LABEL]["admm(mppi/mppi)"]
    assert mean["pos_err_mean_success"] is None


def test_no_mean_block_for_a_single_group() -> None:
    """One task means `Mean` would just repeat it."""
    summary = evaluate([make_run("t1")])
    assert MEAN_LABEL not in summary


def test_failed_trials_are_credited_the_sweep_wide_worst_time() -> None:
    """T stays comparable when one method fails fast and another nearly wins.

    Scored per cell, the quick failure would be credited its own short
    worst case and win the T column outright.
    """
    quick_failure = make_run("t1", algorithm="mppi", object_opt=None, steps=4,
                             final_pos_err=0.9)
    slow_success = make_run("t1", steps=40, final_pos_err=0.01)
    summary = evaluate([quick_failure, slow_success])
    assert summary["t1"]["mppi"]["mean_execution_time"] == pytest.approx(
        40 * 0.05
    )
    assert summary["t1"]["admm(mppi/mppi)"]["mean_execution_time"] == (
        pytest.approx(40 * 0.05)
    )


def test_strip_common_prefix_shortens_task_labels() -> None:
    """`pusht3d_xarm6_` drops out of a table of tabletop scenes."""
    short, prefix = _strip_common_prefix(
        ["pusht3d_xarm6_open_table", "pusht3d_xarm6_shelf_gap"]
    )
    assert short == ["open_table", "shelf_gap"]
    assert prefix == "pusht3d_xarm6_"


def test_strip_common_prefix_never_empties_a_label() -> None:
    """Nothing shared, nothing stripped; and one label is never consumed.

    `a_b` and `a_b_c` share `a_b`, but stripping all of it would leave the
    first row with a blank name, so stripping stops one component short.
    """
    assert _strip_common_prefix(["alpha", "beta"]) == (["alpha", "beta"], "")
    assert _strip_common_prefix(["a_b", "a_b_c"]) == (["b", "b_c"], "a_")
    assert _strip_common_prefix(["only_one"]) == (["only_one"], "")


@pytest.mark.parametrize("style", ["text", "markdown", "latex"])
def test_every_format_renders_all_rows(style: str) -> None:
    """Each style emits one line per (group, method), Mean included."""
    runs = [
        make_run(f"t{i}", algorithm=a, object_opt=o)
        for i in (1, 2)
        for a, o in (("admm", "mppi"), ("mppi", None))
    ]
    summary = evaluate(runs)
    table = format_table(summary, ("task",), style=style)
    # 2 tasks x 2 methods + 2 Mean rows.
    assert table.count("admm(mppi/mppi)") + table.count(
        r"admm(mppi/mppi)"
    ) >= 3
    assert MEAN_LABEL in table
    if style == "latex":
        assert r"\multirow{2}{*}{Mean}" in table
        assert r"\midrule\midrule" in table
        # Underscores in labels must be escaped or LaTeX will not compile.
        assert "t1" in table


def test_unknown_format_is_rejected() -> None:
    """A typo'd style fails loudly rather than silently emitting text."""
    summary = evaluate([make_run("t1")])
    with pytest.raises(ValueError, match="unknown table style"):
        format_table(summary, ("task",), style="csv")


def test_evaluate_rejects_no_runs() -> None:
    """Scoring nothing is a mistake, not an empty table."""
    with pytest.raises(ValueError, match="at least one run"):
        evaluate([])


def test_parse_filters_ors_within_a_field() -> None:
    """Repeated flags and comma lists both accumulate."""
    assert parse_filters(["task=a,b", "task=c"]) == {"task": {"a", "b", "c"}}
    assert parse_filters(["a=1", "b=2"]) == {"a": {"1"}, "b": {"2"}}
    with pytest.raises(ValueError, match="KEY=VALUE"):
        parse_filters(["nope"])


def test_trial_metrics_reproduce_the_synthetic_run() -> None:
    """The fixture really does exercise `trial_metrics`, not a stub."""
    run = make_run("t1", steps=10, final_pos_err=0.4, compute_time=0.25)
    m = trial_metrics(run)
    assert m["steps_run"] == 10
    assert m["pos_err_final"] == pytest.approx(0.4)
    assert m["execution_time"] == pytest.approx(0.5)
    assert m["mean_frequency_hz"] == pytest.approx(4.0)
    assert m["reached"] is False


def _converging_run(errors: List[float], cap: int = 500) -> Dict[str, Any]:
    """A run whose position error follows `errors`, closing on the goal.

    `make_run` walks *away* from a goal it starts on, which is the right
    shape for the error-magnitude tests but not for this one: a step count
    is only meaningful for a run that approaches. Orientation is held at
    zero so the position tolerance alone decides.
    """
    run = make_run("t1", steps=len(errors))
    run["hyperparameters"]["steps"] = cap
    run["dynamic"]["object_pose"] = [[errors[0], 0.0, 0.0]] + [
        [e, 0.0, 0.0] for e in errors
    ]
    run["dynamic"]["compute_time"] = [0.5] * len(errors)
    return run


def test_steps_to_goal_is_the_first_crossing_not_the_last_step() -> None:
    """A run reaching at step 3 is credited 3, not the 6 steps it logged.

    They differ whenever the tolerance in force is not the one the closed
    loop exited on -- the whole point of re-scoring old runs.
    """
    run = _converging_run([0.9, 0.5, 0.01, 0.01, 0.01, 0.01])
    m = trial_metrics(run)
    assert m["reached"] is True
    assert m["steps_to_goal"] == 3


def test_steps_to_goal_follows_a_retuned_tolerance() -> None:
    """Loosening the tolerance moves the crossing earlier, with no re-run."""
    run = _converging_run([0.9, 0.5, 0.2, 0.2, 0.2], cap=500)
    assert trial_metrics(run)["steps_to_goal"] == 500  # never met 0.05
    run["hyperparameters"]["goal_pos_tol"] = 0.25
    assert trial_metrics(run)["steps_to_goal"] == 3


def test_steps_to_goal_is_censored_at_the_step_cap_on_a_failure() -> None:
    """Never reaching costs the whole budget, not the steps it happened to run.

    Otherwise a method that diverges and is cut short would score a *better*
    step count than one that nearly succeeded -- the same trap `max_time`
    exists to close for the execution-time column.
    """
    run = make_run("t1", steps=12, final_pos_err=0.9)
    run["hyperparameters"]["steps"] = 500
    m = trial_metrics(run)
    assert m["reached"] is False
    assert m["steps_to_goal"] == 500


def test_success_is_rescored_from_the_recorded_tolerance() -> None:
    """Changing the tolerance re-labels an old run without re-running it."""
    run = make_run("t1", final_pos_err=0.2, final_theta_err=0.0)
    assert trial_metrics(run)["reached"] is False
    run["hyperparameters"]["goal_pos_tol"] = 0.5
    assert trial_metrics(run)["reached"] is True


def _flat(values: List[float]) -> float:
    return float(np.mean(values))


def test_theta_error_wraps() -> None:
    """An angle just past pi is a small error, not a nearly-2pi one."""
    run = make_run("t1", final_theta_err=math.pi + 0.1)
    assert trial_metrics(run)["theta_err_final"] == pytest.approx(
        math.pi - 0.1, abs=1e-9
    )


def test_method_components_are_not_reported_as_averaged() -> None:
    """`algorithm` and `object_opt` name a row; they are not averaged in.

    Listing them under "averaged over" would claim the opposite of what
    the table does with them.
    """
    runs = [
        make_run("t1", algorithm="admm", object_opt="mppi", horizon=16),
        make_run("t1", algorithm="mppi", object_opt=None, horizon=32),
    ]
    averaged = averaged_fields(runs, ("task",))
    assert "horizon" in averaged
    assert "algorithm" not in averaged
    assert "object_opt" not in averaged
    assert "method" not in averaged
    assert "robot_opt" not in averaged


def test_long_value_lists_are_summarised_by_count() -> None:
    """A 20-seed sweep says "20 values", not all twenty."""
    many = {"seed": [str(i) for i in range(20)], "horizon": ["16", "32"]}
    described = _describe(many)
    assert "seed (20 values)" in described
    assert "horizon (16, 32)" in described


def test_latex_marks_undefined_metrics_with_an_endash() -> None:
    """A lone "-" sets as a hyphen in LaTeX; the paper writes "--"."""
    runs = [make_run(f"t{i}", final_pos_err=0.9) for i in (1, 2)]
    table = format_table(evaluate(runs), ("task",), style="latex")
    assert " -- " in table
    assert " - " not in table


def test_ablate_folds_field_into_method_label() -> None:
    """`--ablate rho` makes each rho its own method row, not one average."""
    runs = [
        make_run("t1", rho=0.1, seed=0),
        make_run("t1", rho=10.0, seed=1),
        make_run("t1", algorithm="mppi", seed=2),
    ]
    summary = evaluate(runs, ablate=("rho",))
    assert set(summary["t1"]) == {
        "admm(mppi/mppi) rho=0.1",
        "admm(mppi/mppi) rho=10.0",
        "mppi",
    }
    assert summary["t1"]["admm(mppi/mppi) rho=0.1"]["n_trials"] == 1


def test_ablate_is_excluded_from_averaged_fields() -> None:
    """An ablated knob must not also be listed as silently averaged."""
    runs = [
        make_run("t1", rho=0.1, seed=0, horizon=16),
        make_run("t1", rho=10.0, seed=1, horizon=32),
    ]
    averaged = averaged_fields(runs, ("task",), ablate=("rho",))
    assert "rho" not in averaged
    assert averaged["horizon"] == ["16", "32"]


def test_filter_plus_ablate_pins_other_axes() -> None:
    """Pinning n_admm leaves one method row per rho value only."""
    runs = [
        make_run("t1", rho=r, n_admm=n, seed=i)
        for i, (r, n) in enumerate(
            [(0.1, 4), (10.0, 4), (0.1, 8), (10.0, 8)]
        )
    ]
    # Mimic --filter n_admm=4 by keeping only those runs.
    pinned = [r for r in runs if r["hyperparameters"]["n_admm"] == 4]
    summary = evaluate(pinned, ablate=("rho",))
    assert set(summary["t1"]) == {
        "admm(mppi/mppi) rho=0.1",
        "admm(mppi/mppi) rho=10.0",
    }


def test_ablate_repeats_accumulate() -> None:
    """Repeating --ablate must add axes, not replace the previous one."""
    args = _build_parser().parse_args(
        ["--ablate", "samples", "--ablate", "horizon", "n_admm"]
    )
    assert args.ablate == ["samples", "horizon", "n_admm"]


def test_ablate_names_only_the_axis_a_run_moves() -> None:
    """One-at-a-time sweeps: a row is labelled by its single deviation."""
    runs = [
        make_run("t1", seed=0),
        make_run("t1", seed=1),
        make_run("t1", seed=2),
        make_run("t1", rho=0.1, seed=3),
        make_run("t1", horizon=32, seed=4),
    ]
    ablate = reference_values(runs, ("rho", "horizon"))
    assert ablate == {"rho": 10.0, "horizon": 16}
    summary = evaluate(runs, ablate=ablate)
    assert set(summary["t1"]) == {
        "admm(mppi/mppi)",
        "admm(mppi/mppi) rho=0.1",
        "admm(mppi/mppi) horizon=32",
    }
    # The base cell keeps every trial that moved no ablated axis.
    assert summary["t1"]["admm(mppi/mppi)"]["n_trials"] == 3


def test_consensus_variants_are_separate_methods_by_default() -> None:
    """Six ADMM variants must not average into one row without --ablate."""
    wrench = make_run("t1")
    wrench["hyperparameters"].update(consensus="wrench", local_goal=False)
    carrot = make_run("t1", seed=1)
    carrot["hyperparameters"].update(
        consensus="wrench", local_goal=True, local_goal_lookahead=0.25
    )
    assert _run_fields(wrench)["method"] == (
        "admm(mppi/mppi) consensus=wrench"
    )
    assert _run_fields(carrot)["method"] == (
        "admm(mppi/mppi) consensus=wrench local_goal_lookahead=0.25"
    )


def test_validate_ablate_rejects_unknown_fields() -> None:
    """A typo'd ablate key fails before scoring."""
    with pytest.raises(ValueError, match="ablate field"):
        validate_ablate([make_run("t1")], ("not_a_field",))


def test_cum_success_stays_one_after_early_reach() -> None:
    """Once the goal is met, later steps remain successful."""
    run = make_run("t1", steps=10, final_pos_err=0.9, reach_at=4)
    series = step_series(run)
    assert series["cum_success"][:3].tolist() == [0.0, 0.0, 0.0]
    assert series["cum_success"][3:].tolist() == [1.0] * 7


def test_aggregate_step_series_pads_early_success_with_last() -> None:
    """A short successful trial still contributes 1.0 at later steps."""
    short = make_run("t1", steps=4, reach_at=2, final_pos_err=0.9)
    long = make_run("t1", steps=8, final_pos_err=0.9, seed=1)
    curves = evaluate_step_curves([short, long], ablate=())
    cell = curves["t1"]["admm(mppi/mppi)"]
    # Short run reached at step 2; padded cum_success must stay 1.
    assert cell["cum_success_mean"][-1] == pytest.approx(0.5)
    assert len(cell["steps"]) == 8


def test_plot_step_curves_smoke(tmp_path) -> None:  # noqa: ANN001
    """`--plot` writes a PNG from tiny synthetic curves."""
    runs = [
        make_run("pusht3d_xarm6_open_table", rho=0.1, seed=0),
        make_run("pusht3d_xarm6_open_table", rho=10.0, seed=1),
        make_run("pusht3d_xarm6_shelf_gap", rho=0.1, seed=0),
        make_run(
            "pusht3d_xarm6_open_table", algorithm="mppi", seed=2
        ),
    ]
    curves = evaluate_step_curves(runs, ablate=("rho",))
    path = tmp_path / "ablation.png"
    plot_step_curves(
        curves,
        str(path),
        group_by=("task",),
        ablate=("rho",),
        filters={"n_admm": ["4"]},
    )
    assert path.is_file()
    assert path.stat().st_size > 0


def test_plot_drops_the_residual_column_when_no_run_has_residuals(
    tmp_path,  # noqa: ANN001
) -> None:
    """An object-only sweep gets two columns, not two plus an empty one.

    Object runs have no consensus and so no primal/dual residual. Drawing
    the panel anyway spends a third of the figure on axes that read as a
    plot that failed rather than a quantity that does not exist.
    """
    steps = list(range(6))
    errors = {
        "steps": steps,
        "pos_err_mean": [0.5] * 6,
        "pos_err_std": [0.0] * 6,
        "theta_err_mean": [0.5] * 6,
        "theta_err_std": [0.0] * 6,
    }
    residuals = {
        "primal_residual_mean": [1.0] * 6,
        "primal_residual_std": [0.0] * 6,
        "dual_residual_mean": [0.5] * 6,
        "dual_residual_std": [0.0] * 6,
    }

    def width(curves: Dict[str, Any], name: str) -> int:
        path = tmp_path / name
        plot_step_curves(curves, str(path))
        return Image.open(path).size[0]

    narrow = width({"s": {"object_only": dict(errors)}}, "obj.png")
    wide = width({"s": {"admm": {**errors, **residuals}}}, "admm.png")
    assert narrow < wide


def test_run_fields_skips_none_ablate_values_on_flat() -> None:
    """Ablating rho must not rename a flat baseline to `mppi rho=None`."""
    run = make_run("t1", algorithm="mppi")
    assert _run_fields(run, ablate=("rho",))["method"] == "mppi"
