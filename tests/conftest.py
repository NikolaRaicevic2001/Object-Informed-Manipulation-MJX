"""Shared test helpers, existing mostly to keep the suite fast.

The suite's runtime is not spread across its tests -- it is concentrated in
a handful of JAX compilations, and most of those were accidental. An
un-jitted `mjx.forward` compiles every primitive of the physics pipeline
separately: measured on the clutter scene, 32.66 s eager against 8.51 s for
one jitted call and 0.0033 s for every call after it. Four test files were
paying the eager price, and one test that never touches ADMM at all was the
second-slowest in the suite because of it.

So the rule here is: a test that needs MuJoCo state calls `mjx_forward`,
never `mjx.forward`. The compiled function is cached for the whole session,
so the first caller pays once and the rest are free.
"""

import jax
from mujoco import mjx

# One compiled `mjx.forward` per model shape, cached for the session. JAX
# keys its own compilation cache on the traced signature, so distinct tasks
# still compile separately -- what this shares is the *jit wrapper*, which
# is what turns the second and later calls into cache hits instead of full
# eager re-dispatch.
_JIT_FORWARD = jax.jit(mjx.forward)


def mjx_forward(model: mjx.Model, data: mjx.Data) -> mjx.Data:
    """Run MJX forward kinematics, compiled.

    Use in place of a bare `mjx.forward` in tests. Eager MJX is not merely
    slower -- it is ~4x slower on the *first* call and ~400x slower on
    every subsequent one, because each primitive is dispatched and compiled
    on its own.

    Args:
        model: The MJX model.
        data: The state to populate.

    Returns:
        The state with kinematics (site_xpos, xpos, sensordata) filled in.
    """
    return _JIT_FORWARD(model, data)
