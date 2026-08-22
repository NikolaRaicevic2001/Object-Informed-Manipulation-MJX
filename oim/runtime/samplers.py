"""Choosing and sizing a sampling optimizer, identically for every world.

`build_sub_optimizer` is what makes "same algorithm, different dynamics" a
fact rather than a claim for the comparison that rests on it: the 3D world
(`oim.worlds.sim3d.build`) and the object-only world
(`oim.worlds.object_only.build`) construct their blocks through it, from
the same `sampler:` config block, so an object-only run and the ADMM run it
is the upper bound for cannot differ in optimizer configuration.

`oim.worlds.sim2d.run.build_admm_2d` is deliberately *not* one of them: it
constructs MPPI directly with 2D-tuned noise levels that are not in any
config file. That world exists to tell an algorithm bug from an MJX bug,
not to be scored against the other two, so it is the one place the shared
path is not the honest one. Routing it through here would silently retune
it.

`consensus_space` has no such exception -- all three builders had their own
copy of the same construction, and only the 3D one had ever been taught
that a pose consensus needs a per-dimension dual bound.
"""

from typing import Any, Dict, Optional, Union

import numpy as np

from oim.algs import (
    CBO,
    CEM,
    MPPI,
    ContactPointConsensus,
    PredictiveSampling,
    WrenchConsensus,
)

SUB_OPTIMIZERS = ["mppi", "cem", "ps", "cbo"]


def object_sample_count(
    sampler_cfg: Dict[str, Any],
    samples: int,
    override: Optional[int] = None,
) -> int:
    """How many rollouts the ADMM object block gets.

    A flag beats `sampler.object.num_samples`, which beats the shared
    `samples`. Shared with the run-file writer so what a run *records* is
    resolved by the same rule that built it, rather than recording the
    flag and leaving a `None` to mean "whichever config, whenever read".

    Args:
        sampler_cfg: The config's `sampler` block.
        samples: The shared per-block budget.
        override: A command-line `--object-samples`, or None.

    Returns:
        The object block's sample count.
    """
    if override is not None:
        return override
    return sampler_cfg.get("object", {}).get("num_samples", samples)


def build_sub_optimizer(
    name: str,
    task: object,
    *,
    plan_horizon: float,
    num_knots: int,
    spline: str,
    seed: int,
    num_samples: int,
    sampler_cfg: Dict[str, Any],
    iterations: int = 1,
    overrides: Optional[Dict[str, Any]] = None,
) -> object:
    """Build one sub-optimizer by name, from the config's block for it.

    Any `SamplingBasedController` works for either ADMM block -- the ADMM
    layer only ever calls `sample_knots`/`update_params` -- so the object-
    and robot-level optimizers are chosen independently. Each one's own
    parameters come from `sampler_cfg[name]`, so the same numbers are used
    whether a method runs as an ADMM block or as a flat baseline.

    Args:
        name: `mppi`, `cem`, `ps` or `cbo`.
        task: The task to build against.
        plan_horizon: Planning horizon in seconds.
        num_knots: Spline knots.
        spline: Spline type.
        seed: RNG seed.
        num_samples: Rollouts per iteration.
        sampler_cfg: The config's `sampler` block.
        iterations: Optimizer passes per `optimize()` call -- the
            "vanilla, more inner iterations" side of the
            iterations-vs-n_admm ablation. Raising it on ADMM's own blocks
            was measured to hurt: each converges harder to its individual
            optimum before the next consensus round.
        overrides: Per-call replacements for entries of
            `sampler_cfg[name]`. The object block needs them because it
            samples in a *different space* from the robot block --
            normalized wrench against joint velocity in rad/s -- so a
            single `noise_level` cannot be right for both. Sharing one was
            measured to leave the object block's torque channel saturated
            while its force channel explored ~6% of its range.

    Returns:
        The controller.

    Raises:
        ValueError: If `name` is not a known sub-optimizer, or `overrides`
            names a parameter that sub-optimizer does not take.
    """
    if name not in SUB_OPTIMIZERS:
        raise ValueError(f"unknown sub-optimizer '{name}'")
    common = dict(
        plan_horizon=plan_horizon,
        spline_type=spline,
        num_knots=num_knots,
        seed=seed,
        num_samples=num_samples,
        iterations=iterations,
    )
    own = dict(sampler_cfg[name])
    unknown = sorted(set(overrides or {}) - set(own))
    if unknown:
        raise ValueError(
            f"sampler override(s) {unknown} are not parameters of "
            f"'{name}' (known: {sorted(own)})"
        )
    own.update(overrides or {})
    if name == "mppi":
        return MPPI(task, **own, **common)
    if name == "cem":
        return CEM(task, **own, **common)
    if name == "ps":
        return PredictiveSampling(task, **own, **common)
    return CBO(task, **own, **common)


def consensus_space(
    task: Any, variable: str = "wrench"
) -> Union[WrenchConsensus, ContactPointConsensus]:
    """The consensus space for a task, scaled by the task's own scale.

    `max_dual` is twice the scale: the dual accumulates the running sum of
    primal residuals, and left unbounded it winds up during the many steps
    where the two blocks genuinely disagree (the object wants a push the
    robot cannot reach yet) and then dominates both objectives.

    Args:
        task: Anything implementing `ConsensusTask.consensus_scale`.
        variable: `"wrench"` (the paper's own, eq. 24) or
            `"contact_point"`, which makes the blocks agree on where on
            the object's boundary to push and how hard.

    Returns:
        The consensus space, built the same way for every world -- so a
        difference between two runs is not a difference in dual clipping.
    """
    scale = task.consensus_scale()
    if variable == "contact_point":
        # Per-dimension: metres in the two point channels and newtons in
        # lambda, so a single scalar bound taken from the first would
        # leave the force dual effectively unclipped.
        return ContactPointConsensus(
            max_dual=2.0 * np.asarray(scale), scale=scale
        )
    return WrenchConsensus(max_dual=2.0 * float(scale[0]), scale=scale)
