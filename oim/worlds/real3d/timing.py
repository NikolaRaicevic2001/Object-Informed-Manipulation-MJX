"""When each thing happened: loop clocks, transport stamps, the window.

Two independent questions. `_PhaseTimer` answers "where did this run's
control period go", from our own `perf_counter`. The `_STAMP_*` series
answer "how stale was the state we solved from, and when did the command
reach the arm", by comparing the driver's ROS header stamps against ours --
the only way to see the transport delay the planner has to predict across.
"""

from __future__ import annotations

import time
from typing import Any, Dict, Optional, Tuple

import numpy as np

# Clock readings per control step, for the loop-timing comparison. Read
# side from `WorldState.stamps` (the /joint_states header stamp, its arrival
# in our callback, and the `read_state` call, on the ROS clock and on
# `perf_counter`); publish side from the publisher thread, the first command
# it sent out of that step's plan. NaN wherever the interface has no stamps
# (mock) or the plan never reached the publisher (last step).
_STAMP_READ_KEYS = ("ros_js_stamp", "ros_js_recv", "ros_read",
                    "perf_js_recv", "perf_read")

_STAMP_PUB_KEYS = ("ros_cmd_pub", "perf_cmd_pub", "cmd_pub_index")

_STAMP_KEYS = (*_STAMP_READ_KEYS, *_STAMP_PUB_KEYS)

def _log_read_stamps(log: Dict[str, Any], world: Any) -> None:
    st = getattr(world, "stamps", None) or {}
    for key in _STAMP_READ_KEYS:
        log.setdefault(key, []).append(float(st.get(key, float("nan"))))

def _log_publish_stamps(log: Dict[str, Any],
                        pub_first: Dict[int, Tuple[float, float, int]]) -> None:
    """Align the publisher's first-command stamps (keyed by step) with the
    read-side lists, so every stamp key has one entry per logged step."""
    n = len(log.get("ros_read", []))
    for key in _STAMP_PUB_KEYS:
        log[key] = []
    for k in range(n):
        ros, perf, idx = pub_first.get(k, (float("nan"), float("nan"), -1))
        log["ros_cmd_pub"].append(ros)
        log["perf_cmd_pub"].append(perf)
        log["cmd_pub_index"].append(int(idx))

def _stamp_summary(log: Dict[str, Any], step: Optional[int] = None) -> str:
    """One line of the timing comparison: for one step, or run medians.

    js_age: how old the /joint_states sample was when `read_state` ran
    (ROS clock). transport: driver stamp to our callback. e2e_ros: encoder
    read to first command out, on the ROS clock. e2e_perf: `read_state` to
    first command out, on our clock -- the same interval minus js_age, if
    the two clocks agree.
    """
    if not log.get("ros_read") or not log.get("ros_cmd_pub"):
        return ""
    a = {k: np.asarray(log[k], dtype=float) for k in _STAMP_KEYS}
    n = min(len(a["ros_read"]), len(a["ros_cmd_pub"]))
    if n == 0 or (step is not None and step >= n):
        return ""
    sl = slice(step, step + 1) if step is not None else slice(0, n)
    js_age = a["ros_read"][sl] - a["ros_js_stamp"][sl]
    transport = a["ros_js_recv"][sl] - a["ros_js_stamp"][sl]
    e2e_ros = a["ros_cmd_pub"][sl] - a["ros_js_stamp"][sl]
    e2e_perf = a["perf_cmd_pub"][sl] - a["perf_read"][sl]
    idx = a["cmd_pub_index"][sl]
    if step is not None:
        if not np.isfinite(e2e_ros).all():
            return ""
        return (f"timing[{step}]: js_age={js_age[0] * 1e3:.0f}ms "
                f"transport={transport[0] * 1e3:.0f}ms  "
                f"e2e_ros={e2e_ros[0] * 1e3:.0f}ms  "
                f"e2e_perf={e2e_perf[0] * 1e3:.0f}ms  "
                f"first_idx={int(idx[0])}")

    def _med(x: np.ndarray) -> str:
        x = x[np.isfinite(x)]
        if x.size == 0:
            return "n/a"
        return f"{np.median(x) * 1e3:.0f}/{np.percentile(x, 95) * 1e3:.0f}ms"

    good = idx[idx >= 0]
    return ("timing (median/p95): "
            f"js_age={_med(js_age)}  transport={_med(transport)}  "
            f"e2e_ros={_med(e2e_ros)}  e2e_perf={_med(e2e_perf)}  "
            f"first_idx median={np.median(good):.0f}" if good.size else
            "timing: no publish stamps recorded")

class _PhaseTimer:
    """Wall-clock of each loop phase, one entry per control step.

    Always on: two variants of the loop diverge onto different trajectories
    within a few steps and then do different amounts of work (different
    contact counts, different sample populations), so per-step MEANS from
    two separate runs are not comparable at this granularity -- one
    `lagged_consensus` pair differed by 130 ms/step of overhead on nothing
    but trajectory. A single run's median/p95 per phase is the comparison
    that holds. Cost is one `perf_counter` per phase.

    `compute_time` is the solve phase, so the series the rest of the code
    already reads stays the only copy of it. It ends when `params` is ready:
    the rollouts the same jit produced are still in flight, so their tail
    lands in whichever phase first touches them (the reducer).
    """

    KEYS = ("t_read", "t_assemble", "compute_time", "t_reduce", "t_send",
            "t_plan", "t_log")

    def __init__(self, log: Dict[str, Any]) -> None:
        self._log = log
        for key in self.KEYS:
            log.setdefault(key, [])
        self._t = time.perf_counter()

    def step_start(self) -> None:
        self._t = time.perf_counter()

    def mark(self, key: str) -> None:
        now = time.perf_counter()
        self._log[key].append(now - self._t)
        self._t = now

    def finish(self) -> None:
        """Pad to a rectangle -- a step can break out mid-phase."""
        n = max(len(self._log[key]) for key in self.KEYS)
        for key in self.KEYS:
            self._log[key] += [float("nan")] * (n - len(self._log[key]))

def _phase_summary(log: Dict[str, Any]) -> str:
    """One line: median/p95 [ms] per phase, over every logged step."""
    keys = [k for k in _PhaseTimer.KEYS if log.get(k)]
    if not keys:
        return ""
    rows = []
    total = np.zeros(len(log[keys[0]]))
    for key in keys:
        x = np.asarray(log[key], dtype=float)
        total = total + np.nan_to_num(x)
        good = x[np.isfinite(x)]
        if good.size:
            rows.append(f"{key.removeprefix('t_')} "
                        f"{np.median(good) * 1e3:.0f}/"
                        f"{np.percentile(good, 95) * 1e3:.0f}")
    return (f"phase median/p95 [ms] over {total.size} steps: "
            + "  ".join(rows)
            + f"  | step {np.median(total) * 1e3:.0f}/"
            f"{np.percentile(total, 95) * 1e3:.0f}")

def execution_window(
    handoff: str, t_c: float, replan_rate: float
) -> float:
    """Plan time [s] one solve executes before the next plan replaces it.

    One definition for both loops, so the mock runs the window hardware
    runs. Under `deterministic` that is `t_c` -- the same key
    `_run_overlapped` pins its anchor to -- and the two agree by
    construction. `replan_rate` is the `responsive` fallback only: it has
    no YAML key, and its 2 Hz default used to give the mock a 0.5 s window
    against hardware's 0.4 s. Under `responsive` hardware has no fixed
    window either (the period is the solve), so a stand-in is all there is.
    """
    return float(t_c) if handoff == "deterministic" else 1.0 / replan_rate

def publish_index(
    elapsed: float, n: int, control_dt: float
) -> Optional[int]:
    """Which command of an `n`-sample plan to send `elapsed` s past its anchor.

    Module-level and pure so the handoff policies can be tested without a
    robot: `--mock` runs `_run_serial`, so nothing in `_run_overlapped` --
    including this -- is exercised by a mock run. See
    `tests/test_plan_handoff.py`.

    Two guards, in order:

    * `elapsed` is clamped at 0. A plan published BEFORE its own anchor
      (the solve beat `lat`, which under an EMA anchor is roughly half of
      them) leaves `elapsed` negative, and `int()` truncates toward zero --
      `int(-0.07 / 0.02)` is -3, and `s[-3]` is the third-from-LAST command
      in the plan. Entering at index 0 instead means the arm is slightly
      behind the state the plan was solved for, bounded by the EMA's error
      and corrected by the next solve.
    * Past the plan's end, `None`: the caller sends zeros rather than the
      tail. A stalled solver must not leave the arm executing an old plan,
      and the interface watchdog cannot catch that because the publisher is
      still sending.

    Args:
        elapsed: Seconds since the plan's anchor. May be negative.
        n: Number of samples in the plan.
        control_dt: Publisher tick [s]; the plan's own sample spacing.

    Returns:
        The index to publish, or None for "exhausted, send zeros".
    """
    elapsed = max(elapsed, 0.0)
    if elapsed > n * control_dt:
        return None
    return min(int(elapsed / control_dt), n - 1)
