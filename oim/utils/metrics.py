"""Metrics derived from saved run files -- computed, never recorded.

Everything here is a function of what `oim.utils.results.save_run` already
wrote, so a new metric costs a re-read of the JSON rather than a re-run of
the experiment. That is the whole reason the two are separate: goal errors,
success, execution time and control frequency are *definitions*, and
definitions change.

Follows the OI-MPPI paper (`Documentation/IROS2026.pdf`, Sec. VI): success
rate (SR), position error (eps_d, eps_d^s), control frequency (f_bar) and
total execution time (T) -- plus orientation error, which the paper omits,
and a standard deviation beside every mean.

Deliberately numpy-only: `oim/run_eval.py` reads files and should not pay
for importing JAX or MuJoCo.
"""

from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np


def _wrap(angle: np.ndarray) -> np.ndarray:
    """Wrap an angle to (-pi, pi]."""
    return (np.asarray(angle) + np.pi) % (2 * np.pi) - np.pi


def goal_errors(run: Dict[str, Any]) -> Dict[str, np.ndarray]:
    """Per-step distance from the goal, over the executed steps.

    Uses `object_pose[1:]` -- the pose *after* each control step -- so the
    series aligns with the input and residual series rather than with the
    initial condition.

    Args:
        run: A payload from `oim.utils.results.load_run`.

    Returns:
        `pos_err` and `theta_err`, each of length `steps_run`.
    """
    goal = np.asarray(run["static"]["goal"], dtype=float)
    poses = np.asarray(run["dynamic"]["object_pose"], dtype=float)[1:]
    return {
        "pos_err": np.linalg.norm(poses[:, :2] - goal[:2], axis=1),
        "theta_err": np.abs(_wrap(poses[:, 2] - goal[2])),
    }


def step_series(run: Dict[str, Any]) -> Dict[str, np.ndarray]:
    """Per-step curves for ablation plots, length `steps_run`.

    `cum_success[t]` is 1 once both tolerances have been met at any step
    `<= t` (and stays 1 after an early exit). Goal errors come from
    `goal_errors`. ADMM runs also carry `primal_residual` and
    `dual_residual` when logged.

    Args:
        run: A payload from `oim.utils.results.load_run`.

    Returns:
        `cum_success`, `pos_err`, `theta_err`, and optionally the ADMM
        residual series.
    """
    hp = run["hyperparameters"]
    err = goal_errors(run)
    pos_err = err["pos_err"]
    theta_err = err["theta_err"]
    steps = int(len(pos_err))
    pos_tol = float(hp["goal_pos_tol"])
    theta_tol = float(hp["goal_theta_tol"])
    at_goal = (pos_err < pos_tol) & (theta_err < theta_tol)
    cum_success = np.maximum.accumulate(at_goal.astype(np.float64))

    out: Dict[str, np.ndarray] = {
        "cum_success": cum_success,
        "pos_err": np.asarray(pos_err, dtype=np.float64),
        "theta_err": np.asarray(theta_err, dtype=np.float64),
    }
    for key in ("primal_residual", "dual_residual"):
        residual = run["dynamic"].get(key)
        if residual is not None and len(residual) > 0:
            out[key] = np.asarray(residual, dtype=np.float64)[:steps]
    return out


def _pad_series(
    series: np.ndarray, length: int, *, fill: str
) -> np.ndarray:
    """Extend a shorter series to `length` for cross-trial alignment.

    Args:
        series: Values of shape `(T,)`.
        length: Target length `T_max`.
        fill: `"last"` repeats the final value (for cumulative success after
            early exit); `"nan"` pads with NaN (errors / residuals so short
            runs do not invent later values).
    """
    series = np.asarray(series, dtype=np.float64)
    if len(series) >= length:
        return series[:length]
    pad = length - len(series)
    if fill == "last":
        value = series[-1] if len(series) else 0.0
        return np.concatenate([series, np.full(pad, value)])
    return np.concatenate([series, np.full(pad, np.nan)])


def aggregate_step_series(
    trials: Sequence[Dict[str, np.ndarray]],
    control_dt: float,
) -> Dict[str, Any]:
    """Mean ± std of aligned per-step series across trials.

    Cumulative success pads with the last value; errors and residuals
    pad with NaN and aggregate with `nanmean` / `nanstd`.

    Args:
        trials: `step_series` outputs for one (group, method) cell.
        control_dt: Seconds per control step, for the shared x-axis.

    Returns:
        `steps`, `time`, and for each metric present in any trial
        `{metric}_mean` / `{metric}_std` arrays of length `T_max`. Empty
        `trials` yields empty arrays.
    """
    if not trials:
        empty = np.zeros(0, dtype=np.float64)
        return {"steps": empty, "time": empty}

    t_max = max(len(t["pos_err"]) for t in trials)
    steps = np.arange(t_max, dtype=np.float64)
    out: Dict[str, Any] = {
        "steps": steps,
        "time": steps * float(control_dt),
        "n_trials": len(trials),
    }

    keys_fill: Tuple[Tuple[str, str], ...] = (
        ("cum_success", "last"),
        ("pos_err", "nan"),
        ("theta_err", "nan"),
        ("primal_residual", "nan"),
        ("dual_residual", "nan"),
    )
    for key, fill in keys_fill:
        present = [t[key] for t in trials if key in t]
        if not present:
            continue
        stacked = np.stack(
            [_pad_series(s, t_max, fill=fill) for s in present], axis=0
        )
        if fill == "nan":
            out[f"{key}_mean"] = np.nanmean(stacked, axis=0)
            out[f"{key}_std"] = np.nanstd(stacked, axis=0)
        else:
            out[f"{key}_mean"] = np.mean(stacked, axis=0)
            out[f"{key}_std"] = np.std(stacked, axis=0)
    return out


def _steps_to_goal(
    pos_err: np.ndarray,
    theta_err: np.ndarray,
    pos_tol: float,
    theta_tol: float,
    hp: Dict[str, Any],
    steps_run: int,
) -> int:
    """Control steps until both tolerances are first met; the cap if never.

    The *first* crossing, not `steps_run`, because the two come apart under
    `--pos-tol`/`--theta-tol` re-scoring: a looser tolerance was met earlier
    than the run's own early exit, and a tighter one may never have been met
    at all despite the run having stopped.

    A trial that never reaches is censored at the run's configured `steps`
    rather than dropped, so a method that fails is not rewarded for failing
    quickly -- the same reasoning as `aggregate_metrics`' `max_time`. It
    falls back to `steps_run` for a run file with no `steps` recorded, which
    is the same number for any run that used its whole budget.

    Args:
        pos_err: Per-step position error.
        theta_err: Per-step orientation error.
        pos_tol: Position tolerance.
        theta_tol: Orientation tolerance.
        hp: The run's `hyperparameters`, for the configured step cap.
        steps_run: Steps the run actually executed.

    Returns:
        The step index of the first success, or the censoring cap.
    """
    at_goal = (pos_err < pos_tol) & (theta_err < theta_tol)
    hit = np.flatnonzero(at_goal)
    if len(hit):
        # +1: `at_goal[i]` is the state *after* control step i, so meeting
        # the tolerance there took i + 1 steps.
        return int(hit[0]) + 1
    return int(hp.get("steps") or steps_run)


def trial_metrics(run: Dict[str, Any]) -> Dict[str, Any]:
    """Reduce one run file to the scalars an aggregate is built from.

    `reached` is re-derived from the final pose against the tolerances the
    run recorded, rather than trusted from a stored flag. That is exactly
    equivalent -- the closed loop exits the moment both tolerances are met,
    so a run that used all its steps never met them -- and it means a
    changed tolerance can be applied to old runs without re-running them.

    Args:
        run: A payload from `oim.utils.results.load_run`.

    Returns:
        `reached`, `pos_err_final`, `theta_err_final`, `pos_err_mean`,
        `theta_err_mean`, `steps_run`, `steps_to_goal`, `execution_time`,
        and `mean_frequency_hz` if the run recorded `compute_time`.
    """
    hp = run["hyperparameters"]
    err = goal_errors(run)
    pos_err, theta_err = err["pos_err"], err["theta_err"]
    steps_run = int(len(pos_err))

    pos_tol = float(hp["goal_pos_tol"])
    theta_tol = float(hp["goal_theta_tol"])
    reached = bool(
        steps_run > 0
        and pos_err[-1] < pos_tol
        and theta_err[-1] < theta_tol
    )

    out: Dict[str, Any] = {
        "reached": reached,
        "steps_to_goal": _steps_to_goal(
            pos_err, theta_err, pos_tol, theta_tol, hp, steps_run
        ),
        "pos_err_final": float(pos_err[-1]) if steps_run else float("nan"),
        "theta_err_final": float(theta_err[-1]) if steps_run else float("nan"),
        "pos_err_mean": float(np.mean(pos_err)) if steps_run else float("nan"),
        "theta_err_mean": (
            float(np.mean(theta_err)) if steps_run else float("nan")
        ),
        "steps_run": steps_run,
        "execution_time": steps_run * float(hp["control_dt"]),
    }
    compute_time = run["dynamic"].get("compute_time")
    if compute_time:
        out["mean_frequency_hz"] = float(1.0 / np.mean(compute_time))
    return out


def numerical_failure_step(run: Dict[str, Any]) -> Optional[int]:
    """First step at which the planner's own state went non-finite, or None.

    A run that hits this is CENSORED, not failed. `ADMM._finite_or` and the
    warm-start sanitizer contain the damage now, but a run recorded before
    them froze outright: `z` and both duals persist across control steps, so
    one non-finite rollout made every sample's penalty NaN, `MPPI.
    update_params` then found no finite cost and held its mean, and the arm
    replayed that mean for the rest of the episode. Its final pose is
    whatever the object happened to be at when the arithmetic broke, which
    is not a measurement of the method.

    Deliberately NOT folded into `trial_metrics`: that would silently change
    every number a results table already reports. This is a separate
    question a caller asks on purpose.

    `nonfinite_rounds` is the direct signal where a run recorded it (the
    guard counts what it repaired); the residual series is the fallback for
    runs predating it, which are exactly the runs that froze.

    Args:
        run: A payload from `oim.utils.results.load_run`.

    Returns:
        The first affected step, or None if the run is clean.
    """
    dyn = run.get("dynamic", {})
    repaired = dyn.get("nonfinite_rounds")
    if repaired:
        hits = np.flatnonzero(np.asarray(repaired, dtype=float) > 0)
        if hits.size:
            return int(hits[0])
    first: Optional[int] = None
    for key in ("primal_residual", "dual_residual"):
        series = dyn.get(key)
        if not series:
            continue
        bad = np.flatnonzero(~np.isfinite(np.asarray(series, dtype=float)))
        if bad.size:
            first = int(bad[0]) if first is None else min(first, int(bad[0]))
    return first


def _mean_std(values: List[float]) -> Dict[str, Optional[float]]:
    """Mean and population std of `values`, or `(None, None)` if empty."""
    if not values:
        return {"mean": None, "std": None}
    return {"mean": float(np.mean(values)), "std": float(np.std(values))}


def aggregate_metrics(
    trials: List[Dict[str, Any]], max_time: Optional[float] = None
) -> Dict[str, Any]:
    """Aggregate several trials of one method into the paper's columns.

    Args:
        trials: `trial_metrics` output, one per trial -- same task, method
            and hyperparameters, varying only the seed.
        max_time: Simulated execution time credited to a trial that never
            reached the goal. Defaults to the slowest trial in this group;
            pass the sweep-wide maximum to compare groups whose worst cases
            differ, since otherwise a method that fails fast scores better
            on T than one that nearly succeeds.

    Returns:
        `n_trials`, `success_rate`; `pos_err_mean`/`_std` over all trials
        and `pos_err_mean_success`/`_std_success` over successful ones; the
        same four for `theta_err`; `mean_`/`std_execution_time`;
        `mean_`/`std_steps_to_goal` over all trials and
        `mean_steps_to_goal_success` over successful ones; and
        `mean_`/`std_frequency_hz` when the trials recorded timings.

    Raises:
        ValueError: If `trials` is empty.
    """
    if not trials:
        raise ValueError("aggregate_metrics needs at least one trial")

    if max_time is None:
        max_time = max(t["execution_time"] for t in trials)

    successes = [t for t in trials if t["reached"]]
    exec_times = [
        t["execution_time"] if t["reached"] else max_time for t in trials
    ]
    freqs = [t["mean_frequency_hz"] for t in trials if "mean_frequency_hz" in t]

    pos_err = _mean_std([t["pos_err_mean"] for t in trials])
    pos_err_s = _mean_std([t["pos_err_mean"] for t in successes])
    theta_err = _mean_std([t["theta_err_mean"] for t in trials])
    theta_err_s = _mean_std([t["theta_err_mean"] for t in successes])
    exec_time = _mean_std(exec_times)
    # Already censored at the step cap per trial by `_steps_to_goal`, so
    # unlike `execution_time` this needs no group-wide worst case: the cap
    # is a property of the experiment, not of the other runs beside it.
    steps_goal = _mean_std([t["steps_to_goal"] for t in trials])
    steps_goal_s = _mean_std([t["steps_to_goal"] for t in successes])

    result: Dict[str, Any] = {
        "n_trials": len(trials),
        "success_rate": len(successes) / len(trials),
        "pos_err_mean": pos_err["mean"],
        "pos_err_std": pos_err["std"],
        "pos_err_mean_success": pos_err_s["mean"],
        "pos_err_std_success": pos_err_s["std"],
        "theta_err_mean": theta_err["mean"],
        "theta_err_std": theta_err["std"],
        "theta_err_mean_success": theta_err_s["mean"],
        "theta_err_std_success": theta_err_s["std"],
        "mean_execution_time": exec_time["mean"],
        "std_execution_time": exec_time["std"],
        "mean_steps_to_goal": steps_goal["mean"],
        "std_steps_to_goal": steps_goal["std"],
        "mean_steps_to_goal_success": steps_goal_s["mean"],
    }
    if freqs:
        freq = _mean_std(freqs)
        result["mean_frequency_hz"] = freq["mean"]
        result["std_frequency_hz"] = freq["std"]
    return result
