"""Choosing and sizing a sampling optimizer, identically for every world.

`build_sub_optimizer` is what makes "same algorithm, different dynamics" a
fact rather than a claim for the comparison that rests on it: the 3D world
(`oim.worlds.sim3d.build`) and the object-only world
(`oim.worlds.object_only.build`) construct their blocks through it, from
the same `sampler:` config block, so an object-only run and the ADMM run it
is the upper bound for cannot differ in optimizer configuration.

`consensus_space` is shared for the same reason: both builders had their
own copy of the same construction, and only the 3D one had ever been
taught that a per-dimension dual bound is needed when the consensus
variable's channels carry different units.
"""

from typing import Any, Dict, Optional, Union

import numpy as np

from oim.algs import (
    CBO,
    CEM,
    MPPI,
    ContactPointConsensus,
    ObjectPoseConsensus,
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
    noise_scale: Optional[Any] = None,
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
        noise_scale: Per-channel multiplier applied to the resolved
            `noise_level`, or None to use it as given. Restores the
            unit-box reading of `noise_level` -- "a fraction of the largest
            admissible decision" -- for an action space that carries real
            units instead. Under `consensus="contact_point"` the action is
            [p_x, p_y, lambda] in metres and newtons, where one scalar
            cannot mean the same thing in both: pass the task's
            `consensus_scale()` = [r_body, r_body, f_max]. Ignored by
            optimizers with no `noise_level` (CEM, CBO name theirs
            differently and want setting in action units by hand).

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
    if noise_scale is not None and "noise_level" in own:
        own["noise_level"] = np.asarray(own["noise_level"]) * np.asarray(
            noise_scale
        )
    if name == "mppi":
        return MPPI(task, **own, **common)
    if name == "cem":
        return CEM(task, **own, **common)
    if name == "ps":
        return PredictiveSampling(task, **own, **common)
    return CBO(task, **own, **common)


def object_noise_scale(task: Any, consensus: str) -> Optional[Any]:
    """`build_sub_optimizer`'s `noise_scale` for an object block, or None.

    The object block's action is a unit box under `consensus="wrench"`
    (`object_action_scale` carries the physics), so `noise_level` is
    already a fraction of the largest admissible decision and needs no
    scaling. Under `"contact_point"` the action is [p_x, p_y, lambda] in
    metres and newtons and the optimizer samples it directly -- there is
    no task-side proposal distribution -- so the same scalar has to be
    converted into those units. `consensus_scale()` supplies them:
    `r_body` for the point channels and `f_max` for lambda.

    Which makes `noise_level` the reach knob under `contact_point`.
    `project_object_action` sends every sample to the *nearest* boundary
    point, so a draw that wanders far enough from the nominal lands on a
    different face on its own -- that, not a special sampler, is how the
    block re-chooses where to push. Below roughly 0.2 * r_body the
    population cannot leave the face it started on.

    In lambda the same scalar has to clear the friction cone or nothing
    moves at all: `initial_object_action` seeds lambda at 0.25 * f_max
    while a pure normal force needs 4.90-7.85 N to break away (measured
    over the T footprint's 32 boundary points, median 6.75; the C is
    6.13-7.85, median 7.43). At `f_max = 11.77 N` the shipped
    `noise_level: 0.2` gives sigma_lambda = 2.35 N, which puts the
    easiest boundary point inside 1 sigma. Over-sizing is cheap --
    `project_object_action` clips lambda into [0, f_max].

    Args:
        task: Anything implementing `ConsensusTask.consensus_scale`.
        consensus: The consensus variable the block was built for.

    Returns:
        The per-channel multiplier, or None when none is needed.
    """
    if consensus != "contact_point":
        return None
    return np.asarray(task.consensus_scale())


def consensus_space(
    task: Any, variable: str = "wrench"
) -> Union[WrenchConsensus, ContactPointConsensus, ObjectPoseConsensus]:
    """The consensus space for a task, scaled by the task's own scale.

    `max_dual` is twice the scale: the dual accumulates the running sum of
    primal residuals, and left unbounded it winds up during the many steps
    where the two blocks genuinely disagree (the object wants a push the
    robot cannot reach yet) and then dominates both objectives.

    Args:
        task: Anything implementing `ConsensusTask.consensus_scale`.
        variable: `"wrench"` (the paper's own, eq. 24); `"contact_point"`,
            which makes the blocks agree on where on the object's boundary
            to push and how hard; or `"object_pose"`, which makes them
            agree on where the object ends up.

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
    if variable == "object_pose":
        # Per-dimension for the same reason -- metres and radians -- and
        # the one space whose yaw channel is a circle, so `difference` and
        # `increment` wrap. See `ObjectPoseConsensus`.
        return ObjectPoseConsensus(
            max_dual=2.0 * np.asarray(scale), scale=scale
        )
    # 2.0 -> 0.5 (2026-09-01): on hardware the wrench duals sat pinned at
    # the 2.0*scale clip for 70-90% of every ADMM run -- the object block
    # asks for a wrench on every horizon step while the robot is out of
    # contact on most of them, so y integrates A^r - z = -z until the clip.
    # A railed dual turns the penalty into a stale demand of up to
    # 3x the friction limit, which no contact can pay down; it dragged the
    # tip to the arm's 0.75 m reach boundary in the 13:03 run's 480-step
    # stall. 0.5 bounds the demand at 1.5x limit -- one real push's worth
    # -- so the bias stays advice, not a debt spiral.
    return WrenchConsensus(max_dual=0.5 * float(scale[0]), scale=scale)
