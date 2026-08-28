"""Validate the planar-pushing LCS against the analytic object model.

INCREMENT 1 gate. Before building the C3+ solver on top of the LCS, confirm the
LCS encodes the same physics as the trusted `PlanarPushingObject.step`:

  * Axis-aligned wrenches: the box limit surface and the ellipsoidal one
    coincide on each axis, so the one-step motion must match to numerical
    tolerance. This is the hard assertion -- a failure here means a sign, unit,
    axis, or dt bug in the LCS.
  * Diagonal wrenches: box vs ellipse genuinely differ, so we only report the
    gap (and check the motion direction is sane) rather than assert
    equality.

Run standalone:  python tests/test_c3_lcs.py
Or under pytest: pytest tests/test_c3_lcs.py
"""

import jax
import jax.numpy as jnp

from oim.algs.c3 import build_planar_pushing_lcs, lcs_step
from oim.objects.planar_pushing import PlanarPushingObject, t_shape_footprint


def _make_object() -> PlanarPushingObject:
    """A T-shaped object with the module's default pushing physics."""
    return PlanarPushingObject(
        dt=0.05,
        goal=jnp.zeros(3),
        footprint=t_shape_footprint(),
    )


def _lcs_one_step(
    obj: PlanarPushingObject, pose: jnp.ndarray, wrench: jnp.ndarray
) -> jnp.ndarray:
    """One LCS step, reading physics straight off the object model."""
    lcs = build_planar_pushing_lcs(obj.wrench_limit, obj.dt)
    x_next, lam = lcs_step(lcs, pose, wrench)
    return x_next, lam


def test_axis_aligned_wrenches_match() -> None:
    """Along a single axis, LCS motion must equal the analytic model's."""
    obj = _make_object()
    fl = obj.wrench_limit  # [fl_x, fl_y, fl_theta]
    pose = jnp.array([0.1, -0.2, 0.3])

    # For each axis, sweep a range of magnitudes straddling the friction limit
    # (sub-threshold -> no motion; super-threshold -> the excess moves it).
    max_err = 0.0
    for axis in range(3):
        for scale in [0.0, 0.5, 1.0, 1.5, 3.0, 8.0, -0.5, -1.5, -8.0]:
            w = jnp.zeros(3).at[axis].set(scale * fl[axis])
            analytic = obj.step(pose, w)
            lcs_next, _ = _lcs_one_step(obj, pose, w)
            err = float(jnp.max(jnp.abs(analytic - lcs_next)))
            max_err = max(max_err, err)
            assert err < 1e-5, (
                f"axis {axis}, scale {scale}: analytic {analytic} "
                f"vs lcs {lcs_next} (err {err:.2e})"
            )
    print(f"[axis-aligned] max abs error over all cases: {max_err:.2e}")


def test_diagonal_wrenches_are_sane() -> None:
    """Off-axis: box vs ellipse differ; just report the gap and check sanity."""
    obj = _make_object()
    fl = obj.wrench_limit
    pose = jnp.zeros(3)

    print("[diagonal] analytic vs LCS one-step displacement:")
    for fx, fy, ft in [(2.0, 2.0, 0.0), (3.0, 1.0, 2.0), (-2.0, 2.0, -3.0)]:
        w = jnp.array([fx * fl[0], fy * fl[1], ft * fl[2]])
        analytic = obj.step(pose, w)
        lcs_next, _ = _lcs_one_step(obj, pose, w)
        gap = float(jnp.linalg.norm(analytic - lcs_next))
        # Sanity: both should move the object the same general direction.
        dot = float(jnp.dot(analytic[:2], lcs_next[:2]))
        print(
            f"  w=({fx},{fy},{ft})*limit  analytic={analytic}  "
            f"lcs={lcs_next}  gap={gap:.4f}  xy_dot={dot:+.4e}"
        )
        assert dot >= 0.0, "LCS and analytic disagree on translation direction"


if __name__ == "__main__":
    jax.config.update("jax_enable_x64", True)  # tight tolerances need float64
    test_axis_aligned_wrenches_match()
    test_diagonal_wrenches_are_sane()
    print("\nOK: increment-1 LCS matches the analytic model on axis-aligned "
          "wrenches; diagonal gaps are the expected box-vs-ellipse difference.")
