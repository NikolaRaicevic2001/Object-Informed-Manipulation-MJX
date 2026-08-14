"""Derived quantities every runner appends to its log after the loop.

Kept out of the world packages because all four produced their own copy of
`finite_difference` and two of them had quietly dropped the angle wrap --
a heading crossing pi then read as a ~2*pi/dt velocity spike in the
diagnostics panel of one world but not another.
"""

from typing import Optional

import numpy as np

from oim.objects import wrap_angle


def finite_difference(
    series: np.ndarray, dt: float, angle_col: Optional[int] = None
) -> np.ndarray:
    """Per-step velocity of a logged series, same length as the series.

    Args:
        series: Positions over time, shape (steps, dim).
        dt: The control timestep.
        angle_col: Index of a column holding an angle, if any. Its
            differences are wrapped to (-pi, pi] so a crossing does not
            register as a ~2*pi/dt spike.

    Returns:
        Velocities, with a leading zero row since nothing precedes step 0.
    """
    if len(series) < 2:
        return np.zeros_like(series)
    deltas = np.diff(series, axis=0)
    if angle_col is not None:
        deltas[:, angle_col] = np.asarray(wrap_angle(deltas[:, angle_col]))
    return np.vstack([np.zeros_like(series[:1]), deltas / dt])
