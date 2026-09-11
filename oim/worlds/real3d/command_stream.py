"""Time integration of a piecewise-constant command stream.

The hardware loop predicts the arm's joint state forward by integrating the
velocity commands the arm is executing. Two streams feed that prediction:
what the publisher thread has already sent (or, when the interface can see
it, what the CBF actually forwarded to the controller), and the plan
samples it is about to send. Both are step functions of time -- a command
holds until the next one -- so the displacement over a window is an exact
sum of value x duration. This module is pure numpy so the arithmetic can be
tested without JAX, MuJoCo or ROS.
"""

from __future__ import annotations

from typing import Tuple

import numpy as np


def integrate_stream(
    times: np.ndarray, values: np.ndarray, t0: float, t1: float
) -> np.ndarray:
    """Integral over [t0, t1] of a zero-order-hold stream.

    `values[k]` is in force from `times[k]` until `times[k + 1]`; the last
    value holds indefinitely, and nothing is in force before `times[0]`
    (the arm was not being commanded). Times must be sorted.

    Args:
        times: (N,) sample times [s], sorted ascending.
        values: (N, D) command values.
        t0, t1: window [s]. Empty or inverted windows integrate to zero.

    Returns:
        (D,) displacement.
    """
    times = np.asarray(times, dtype=float)
    values = np.asarray(values, dtype=float)
    if values.ndim == 1:
        values = values[:, None]
    out = np.zeros(values.shape[1], dtype=float)
    if times.size == 0 or t1 <= t0:
        return out
    lo = max(float(t0), float(times[0]))
    hi = float(t1)
    if hi <= lo:
        return out
    # Segment k spans [times[k], times[k + 1]) with times[N] = +inf.
    edges = np.append(times, np.inf)
    k0 = int(np.searchsorted(edges, lo, side="right") - 1)
    k1 = int(np.searchsorted(edges, hi, side="right") - 1)
    k0 = max(k0, 0)
    k1 = min(k1, times.size - 1)
    if k0 == k1:
        return values[k0] * (hi - lo)
    out = values[k0] * (edges[k0 + 1] - lo)
    if k1 > k0 + 1:
        seg = edges[k0 + 2:k1 + 1] - edges[k0 + 1:k1]
        out = out + (values[k0 + 1:k1] * seg[:, None]).sum(axis=0)
    out = out + values[k1] * (hi - edges[k1])
    return out


def window_split(
    t_state: float, t_now: float, t_publish: float, delay: float
) -> Tuple[Tuple[float, float], Tuple[float, float]]:
    """The two integration windows behind one predicted start state.

    The arm's joint velocity at time t is the command in force at t - delay
    (CBF output -> measured motion is ~35 ms on the xArm6; see the 09-11
    step test). The state the next solve should start from is the arm's
    state when the new plan's first command takes effect, i.e. at
    `t_publish + delay`. Integrating the measured state (valid at `t_state`)
    forward to there, in command time:

        [t_state - delay, t_publish]

    Only the part up to `t_now` has been sent yet, so it is split into a
    PAST window served by the record of commands already sent (or executed)
    and a FUTURE window served by the plan samples the publisher will send
    next. Returned as ((past_lo, past_hi), (future_lo, future_hi)) in
    absolute time; the future window is empty when `t_publish <= t_now`.
    """
    past = (t_state - delay, t_now)
    future = (t_now, max(t_publish, t_now))
    return past, future
