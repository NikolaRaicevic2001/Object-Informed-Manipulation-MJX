"""Build the object-level subproblem with no robot attached.

The object block is reused *verbatim* -- this module builds the same
`oim.tasks.pusht.PushT` that `oim.worlds.sim3d.build.build_admm_3d` builds,
from the same scene registry and the same `costs:` block, and drives it
with the same `oim.algs.admm.ObjectSubproblem`. That matters more than it
might look: `PushT`'s object side carries scene-gated behaviour (the
`open_table` action-bounds branch, `wrench_sample_fraction`), and a
hand-rolled object model would quietly study a different planner than the
one ADMM runs.

ADMM is removed by making its coupling vanish rather than by bypassing it:
`rho = 0` kills the consensus penalty, `z` and the duals are zeros, and the
proximal weight is zero. `ObjectSubproblem.optimize` is then exactly the
paper's eq. 24 with the penalty and proximal terms dropped -- an ordinary
sampling-based MPC on the limit-surface model.

`oim.worlds.object_only.run` closes the loop; `oim.worlds.object_only.plant`
owns the `--plant` mode table, which decides both what the block predicts
with and what executes.
"""


from typing import Any, Dict, Optional, Sequence, Tuple, Union

import jax
import jax.numpy as jnp
import numpy as np

from oim.algs import ObjectSubproblem, make_object_shim
from oim.objects import wrench_weights
from oim.objects.contact import contact_point_to_wrench
from oim.runtime.object_mjx import PREDICT_SUBSTEPS, build_object_rollout
from oim.runtime.samplers import (
    build_sub_optimizer,
    consensus_space,
    object_noise_scale,
)
from oim.tasks.pusht import PushT
from oim.worlds.object_only.plant import resolve_plant


def _contact_point_budget(
    task: PushT, limit: np.ndarray, verbose: bool
) -> float:
    """`check_action_budget`'s contact-point branch: measure, don't derive.

    Every boundary point at full normal force, through the same
    `contact_point_to_wrench` the rollout uses, so this reports the wrench
    set the block can actually reach rather than a formula that could
    drift from it. Evaluated at the identity pose: `D` is isotropic in
    translation and the torque is a scalar, so rotating the object rotates
    the force and leaves both norms unchanged.

    The per-channel figure is `max_i |w_i| / limit_i` over the same set --
    what MuJoCo's per-DoF `frictionloss` gates on -- and is generally
    reached at a *different* boundary point than the norm's maximum, which
    is why both are taken over the whole set rather than off one wrench.
    """
    footprint = task.object_model.footprint
    points = np.asarray(footprint.sample_boundary(4))
    lam = np.full((points.shape[0], 1), task._contact_f_max)
    cps = jnp.asarray(np.concatenate([points, lam], axis=1))
    wrenches = np.asarray(
        jax.vmap(contact_point_to_wrench, in_axes=(None, None, 0))(
            footprint, jnp.zeros(3), cps
        )
    )
    normalized = np.abs(wrenches) / limit
    ceiling = float(np.max(np.linalg.norm(normalized, axis=-1)))
    per_channel = float(np.max(normalized))
    if verbose:
        print(
            f"  budget    lambda <= {task._contact_f_max:.2f} N on the "
            f"boundary  ->  max |w|/limit {ceiling:.2f}"
        )
        if ceiling < 1.0:
            print(
                "            WARNING: no boundary point at full normal force "
                "escapes the friction cone, so nothing this block "
                "proposes moves the object at all. Raise "
                "--wrench-fraction (it sets f_max here)."
            )
        elif per_channel <= 1.0:
            print(
                f"            WARNING: the most any single channel "
                f"reaches is "
                f"{per_channel:.2f} x its own limit, so under a per-DoF "
                f"friction plant the object will barely move. Raise "
                f"--wrench-fraction."
            )
    return ceiling


def check_action_budget(
    task: PushT, verbose: bool = True, plant: str = "analytic"
) -> float:
    """Largest `||w / w_limit||` the object block can express, vs. the deadzone.

    Two deadzones, because the two plants gate differently and a budget
    that clears one can be structurally unable to clear the other:

    * `analytic` gates on the coupled norm `||w / w_limit|| >= 1`, so what
      matters is the ceiling `fraction * sqrt(3)`.
    * `mujoco` gates per DoF, on `|w_i| >= limit_i`, so what matters is the
      per-channel ceiling `fraction` alone. At the analytic default of 1.0
      that is *exactly* the friction threshold on every channel -- the net
      generalized force is ~0 and the object does not move at all, however
      healthy the norm looks. Measured on open_table: 200 steps leave
      `pos_err` at 0.732 with fraction 1.0, and reach the goal at step 91
      with fraction 2.0.

    `PlanarPushingObject.step` zeroes any wrench under 1.0, so if this
    ceiling is below 1.0 the block is *structurally* unable to move the
    object -- no cost weight, sample count or horizon can help, and the
    run will sit at its start pose forever while the optimizer quietly
    converges to `w = 0` (with every rollout frozen, effort is the only
    term that still varies across samples, and it is minimized at zero).

    Reported at build time rather than left to be inferred from a flat
    `pos_err` column, because the two failures look identical from the log
    and have nothing in common: this one is a reachability bug, the other
    is a planner that cannot find the route.

    The ceiling is `fraction * sqrt(3)`: the action box is the unit cube
    (`ConsensusTask.object_action_bounds`) and `object_action_to_consensus`
    scales it by `action_scale = fraction * w_limit`. A pure-force push,
    with no torque, only reaches `fraction * sqrt(2)`.

    Under `consensus="contact_point"` none of that reasoning
    applies: the action is [p_x, p_y, lambda] in real units, not a scaled
    wrench, so what bounds the wrench is `f_max` and the footprint's own
    lever arms. The ceiling is measured directly instead -- the largest
    `||w / w_limit||` any boundary point reaches at full normal force --
    which is the same number in the same units, so the two
    parameterizations stay comparable and both deadzone warnings below
    keep their meaning.

    Args:
        task: The `PushT` under study.
        verbose: Print the finding.
        plant: Which plant will execute, selecting which deadzone to check.

    Returns:
        The ceiling, in units of the friction-cone limit.
    """
    limit = np.asarray(task.object_model.wrench_limit)
    if task.consensus == "contact_point":
        return _contact_point_budget(task, limit, verbose)
    scale = np.asarray(task.object_action_scale())
    ceiling = float(np.linalg.norm(scale / limit))
    per_channel = float(np.max(scale / limit))
    if verbose:
        fraction = float(np.mean(scale / limit))
        print(
            f"  budget    unit action = {fraction:.2f} x friction cone  ->  "
            f"max |w|/limit {ceiling:.2f}"
        )
        if ceiling < 1.0:
            print(
                "            WARNING: the largest wrench this block can "
                "express is "
                "inside the friction cone, so nothing it proposes moves "
                "the object at all and the run cannot progress. Raise "
                "--wrench-fraction."
            )
        elif per_channel <= 1.0:
            # Both plants, for different reasons, and both measured.
            # Analytic: `step` subtracts friction, so a wrench barely over
            # the cone yields a correspondingly tiny step -- only the cube
            # diagonal clears it at all, and 0/5 scenes reached the goal.
            # MuJoCo: friction is per DoF, so a saturated single channel
            # nets ~zero force. 2.0 fixes both.
            print(
                f"            WARNING: the most this block can put on any "
                f"one "
                f"channel is {per_channel:.2f} x that channel's own "
                f"limit, so friction cancels nearly all of it and the "
                f"object will barely move -- however healthy the "
                f"||w||/limit ceiling above looks. Measured: 0/5 scenes "
                f"reach the goal at 1.0, 15/15 at 1.5-3.0. Raise "
                f"--wrench-fraction (2.0 is the shipped default)."
            )
    return ceiling


def report_softmax_ess(
    task: PushT,
    block: ObjectSubproblem,
    params: Any,
    obj_state0: jnp.ndarray,
) -> float:
    """Effective sample size of one MPPI reweighting, out of `num_samples`.

    `temperature` has no meaningful default: it must be read against the
    *spread* of rollout costs, which depends on the cost weights, the
    horizon and the scene. Far below that spread the softmax collapses
    onto the single best sample, so the "weighted average" averages one
    white-noise draw and the mean re-randomizes every control step -- the
    jitter looks like a physics or noise problem and is neither.

    Measured on shelf_gap+xarm6 at the shipped `temperature: 0.5`: cost
    std 31.9 across 128 samples, ESS 1.0, top weight 1.000. At lambda near
    the cost std, ESS was 50/128.

    Only meaningful for samplers with a temperature; others return -1.

    Args:
        task: The task, for the object cost.
        block: Its `ObjectSubproblem`.
        params: The object optimizer's policy parameters.
        obj_state0: The pose to sample from.

    Returns:
        The effective sample size, or -1 when the sampler has no softmax.
    """
    opt = block.optimizer
    temperature = getattr(opt, "temperature", None)
    if temperature is None:
        return -1.0

    knots, _ = opt.sample_knots(params)
    knots = jnp.clip(knots, opt.task.u_min, opt.task.u_max)
    knots = task.project_object_action(knots, obj_state0)
    states, ws, _ = jax.vmap(block._rollout, in_axes=(None, 0))(
        obj_state0, knots
    )
    running = task.dt * jax.vmap(jax.vmap(task.object_running_cost))(
        states[:, :-1], ws[:, :-1]
    )
    terminal = jax.vmap(task.object_terminal_cost)(states[:, -1])
    costs = jnp.concatenate([running, terminal[:, None]], axis=1).sum(axis=1)

    weights = np.asarray(jax.nn.softmax(-costs / temperature))
    ess = float(1.0 / np.sum(weights**2))
    spread = float(np.std(np.asarray(costs)))
    print(
        f"  softmax   cost std {spread:.1f}, ESS {ess:.1f}/{len(weights)}, "
        f"top weight {weights.max():.3f}"
    )
    if ess < 0.05 * len(weights):
        print(
            f"            WARNING: the average is one sample -- an argmax, "
            f"not MPPI, so the mean\n            re-randomizes every step. "
            f"Raise temperature toward the cost std ({spread:.0f})."
        )
    return ess


def build_object_only(
    scene: str,
    robot: str,
    cfg: Dict[str, Any],
    *,
    horizon: int,
    samples: int,
    seed: int,
    object_opt: str = "mppi",
    iterations: int = 1,
    consensus: str = "wrench",
    plant: str = "analytic",
    object_substeps: int = PREDICT_SUBSTEPS,
    wrench_fraction: Optional[float] = None,
    w_rate: Optional[Union[float, Sequence[float]]] = None,
    w_contact_rate: Optional[Union[float, Sequence[float]]] = None,
    noise_level: Optional[float] = None,
    temperature: Optional[float] = None,
    goal: Optional[Sequence[float]] = None,
    start: Optional[Sequence[float]] = None,
    verbose: bool = True,
) -> Tuple[PushT, ObjectSubproblem, Any, jnp.ndarray]:
    """Task, object subproblem, initial params and start pose.

    Args:
        scene: A key of `oim.utils.scenes.SCENES`.
        robot: `"point"` or `"xarm6"`. There is no robot in this world, but
            it still selects the scene variant, the config file, and the
            `object_action_bounds` branch -- so an object-only study of an
            xArm6 scene must pass `"xarm6"` to be studying the same object
            block that scene's ADMM runs use.
        cfg: A parsed `oim/configs/robots/*.yaml`.
        horizon: Planning horizon H, in steps of `world3d.planning_dt`.
        samples: Rollouts per iteration.
        seed: RNG seed.
        object_opt: Sub-optimizer -- `mppi`, `cem`, `ps` or `cbo`.
        iterations: Optimizer passes per control step. ADMM runs the object
            block `n_admm` times per step (once per consensus round), so
            this is the knob that makes the comparison like-for-like: set
            it to `n_admm` to give the block the same number of updates it
            would get inside ADMM.
        consensus: `"wrench"` or `"contact_point"`. The
            consensus penalty is switched off here, but this still picks
            the block's *sampling* space, so it is the knob for isolating
            the contact parameterization from ADMM entirely.
        plant: Which dynamics this run uses, as a key of
            `oim.worlds.object_only.plant.PLANT_MODES` -- `"analytic"` or
            `"mujoco"`. One mode rather than a predict/execute pair, so
            neither mixed pair, where the planner and the plant disagree
            about the physics, can be built. See that module.

            The sampler, the costs and the block itself are identical in
            both, which is the point. Only two things read the mode
            here: which `ObjectRollout` the block predicts with, and which
            deadzone `check_action_budget` checks against, since the two
            dynamics gate it differently.
        object_substeps: MJX physics steps per planning step, when the mode
            predicts with MuJoCo. Defaults to `PREDICT_SUBSTEPS`; 1 gives it
            the same coarse integration eq. 5 gets, which is the
            like-for-like setting when the question is how good eq. 5 is
            rather than how well the block does.
        wrench_fraction: Override `PlanarPushingObject`'s
            `wrench_sample_fraction`, the fraction of the friction-cone
            limit a unit action maps to. `None` keeps whatever the scene
            ships, so the block matches its ADMM runs exactly.

            Worth a knob because it decides whether the block can move the
            object *at all*. The action box is the unit cube, so the
            largest expressible wrench is `fraction * w_limit` and the
            largest normalized magnitude is `fraction * sqrt(3)` -- while
            `PlanarPushingObject.step` zeroes anything under 1.0. At the
            shipped 0.5 that ceiling is 0.87, so every action the block can
            propose is inside the deadzone and the object never moves; the
            only cost that still varies across samples is effort, which is
            minimized at zero, so the optimizer converges *to* stillness.
            `xarm6` + `open_table` ships 1.0 and is the one configuration
            not affected. See `check_action_budget`.
        w_contact_rate: Override the contact-rate penalty
            (`PushT.object_rate_cost`), read only under
            `consensus="contact_point"`. Its own knob rather than a reuse
            of `w_rate`: those channels are metres and newtons, not
            newtons and newton-metres. `None` keeps the config's.
        w_rate: Override the wrench-rate penalty (`PlanarPushingObject.
            rate_cost`), as one number or `[f_x, f_y, tau]`. `None` keeps
            the config's.
        noise_level: Override the object sampler's exploration noise, in
            units where 1.0 is the whole friction-cone limit. `None` keeps
            the config's.
        temperature: Override the object sampler's MPPI temperature.
            `None` keeps the config's. Worth checking against the *spread*
            of rollout costs: the softmax collapses onto a single sample
            when lambda is far below it, which turns MPPI into an argmax
            over white noise and is what makes the executed wrench
            re-randomize every step. `report_softmax_ess` measures it.
        goal: Object goal pose, or `None` for the scene's own.
        start: Object start pose, or `None` for the scene's own MJCF
            keyframe value.
        verbose: Print the action-budget line and its warnings here.
            `False` where the caller prints its own header block and wants
            that line inside it rather than ahead of it; the caller then
            calls `check_action_budget` itself.

    Returns:
        `(task, object_subproblem, initial params, start pose)`.
    """
    w3, smp = cfg["world3d"], cfg["sampler"]
    plan_dt = w3["planning_dt"]
    predicts_with, executes_with = resolve_plant(plant)

    # Built exactly as `build_admm_3d` builds it, so the object block under
    # study is the one ADMM would use. `impl="jax"` because the task's own
    # MJX model is never stepped -- it is loaded only because `PushT` owns
    # the scene, and the Warp backend would cost a graph capture for a
    # rollout that never happens. A mode that predicts with MuJoCo does
    # step MJX, but through its own stripped model, not this one.
    task = PushT(
        impl="jax",
        clutter=True,
        planning_dt=plan_dt,
        robot=robot,
        consensus_source="contact" if robot == "point" else "twist",
        consensus=consensus,
        env=scene,
        goal=goal,
        costs=cfg.get("costs"),
    )

    if w_rate is not None:
        task.object_model.w_rate = wrench_weights(w_rate)
    if w_contact_rate is not None:
        # `PushT.object_rate_cost` reads this attribute directly, so
        # setting it here is equivalent to having built the task with a
        # different `costs.w_contact_rate` -- and keeps the override a
        # property of the study, never of the shipped task.
        task._w_contact_rate = wrench_weights(w_contact_rate)
    if wrench_fraction is not None:
        # `action_scale` is the only thing `wrench_sample_fraction` sets,
        # so overriding it here is equivalent to having constructed
        # `PlanarPushingObject` with a different fraction -- and keeps this
        # a property of the *study*, never of the shipped task.
        task.object_model.action_scale = wrench_fraction * (
            task.object_model.wrench_limit
        )
        # `PushT.__init__` caches `action_scale[0]` as the contact
        # parameterization's f_max (as a Python float, since every reader
        # runs under trace). Mutating `action_scale` after construction
        # leaves that cache stale, so under
        # `consensus="contact_point"` -- where f_max is what
        # bounds lambda in `object_action_bounds` and
        # `project_object_action` -- the override would otherwise be
        # silently ignored and the run would report a fraction it never
        # used.
        task._contact_f_max = float(task.object_model.action_scale[0])

    optimizer = build_sub_optimizer(
        object_opt,
        make_object_shim(task, dt=plan_dt),
        plan_horizon=horizon * plan_dt,
        num_knots=horizon,
        spline=smp["object_spline"],
        seed=seed,
        num_samples=samples,
        iterations=iterations,
        sampler_cfg=smp,
        overrides={
            **(smp.get("object", {}).get(object_opt) or {}),
            **({} if noise_level is None else {"noise_level": noise_level}),
            **({} if temperature is None else {"temperature": temperature}),
        },
        # Under `contact_point` the block samples metres and newtons, not a
        # unit box, so `noise_level` is read as a fraction of the object's
        # own size and force ceiling. See `object_noise_scale`.
        noise_scale=object_noise_scale(task, consensus),
    )

    # A consensus space is required by `ObjectSubproblem`'s constructor but
    # is inert at rho = 0: `penalty_cost` returns 0.5 * sum(0 * diff**2).
    # Constructed with the task's real scale anyway, so that switching the
    # penalty back on for a debugging session needs no other change.
    space = consensus_space(task, consensus)
    # proximal_weight = 0: gamma anchors one ADMM *iteration* to the last,
    # and there are no iterations here. Left at the config's value it would
    # instead anchor each control step to the previous step's shifted plan,
    # which is a different mechanism -- a damping term on replanning -- and
    # would quietly slow exactly the routing behaviour being measured.
    block = ObjectSubproblem(
        task,
        optimizer,
        space,
        proximal_weight=0.0,
        # `None` for the analytic backend, so the default stays decided in
        # one place -- `ObjectSubproblem` itself -- rather than here as well.
        rollout=build_object_rollout(
            predicts_with,
            task,
            robot,
            w3,
            substeps=object_substeps,
            # Stays on the JAX backend, unlike the 3D world's, which
            # follows `--warp`. This world exists to measure model error,
            # and every table it has recorded was taken here; Warp is a
            # different rollout, so adopting it would silently reprice
            # those rather than speed them up. There is no `--warp` flag
            # on this world to make the choice with, either.
            impl="jax",
        ),
    )

    params = optimizer.init_params(seed=seed)
    seed_action = task.initial_object_action()
    if seed_action is not None:
        params = params.replace(
            mean=jnp.broadcast_to(
                jnp.asarray(seed_action), (horizon, task.object_action_dim)
            )
        )

    # `verbose=False` where the caller prints its own header and wants
    # this line inside it rather than ahead of it -- see `_run_object`.
    check_action_budget(task, verbose=verbose, plant=executes_with)

    obj_state0 = (
        jnp.asarray(task.start, dtype=float)
        if start is None
        else jnp.asarray(start, dtype=float)
    )
    return task, block, params, obj_state0
