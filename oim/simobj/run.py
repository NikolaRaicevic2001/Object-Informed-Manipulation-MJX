"""Build and run the object-level subproblem with no robot attached.

The object block is reused *verbatim* -- this module builds the same
`oim.tasks.pusht.PushT` that `oim.sim3d.build.build_admm_3d` builds, from
the same scene registry and the same `costs:` block, and drives it with the
same `oim.algs.admm.ObjectSubproblem`. That matters more than it might
look: `PushT`'s object side carries scene-gated behaviour (the `open_table`
action-bounds branch, `wrench_sample_fraction`, the goal-proximity gate in
`project_object_action`), and a hand-rolled object model would quietly
study a different planner than the one ADMM runs.

ADMM is removed by making its coupling vanish rather than by bypassing it:
`rho = 0` kills the consensus penalty, `z` and the duals are zeros, and the
proximal weight is zero. `ObjectSubproblem.optimize` is then exactly the
paper's eq. 24 with the penalty and proximal terms dropped -- an ordinary
sampling-based MPC on the limit-surface model.

THE PLANT IS THE MODEL. `object_dynamics` both predicts and executes here,
so the loop is a receding-horizon MPC with no model error whatsoever. That
is the point: it upper-bounds what the object block can achieve, and the
gap between this and an ADMM run is what the robot and the consensus cost.
It also means the breakaway deadzone in `PlanarPushingObject.step` applies
to execution as well as to prediction, which is why the diagnostics plot
draws the wrench against that threshold.
"""

import time
from typing import Any, Dict, Optional, Sequence, Tuple, Union

import jax
import jax.numpy as jnp
import numpy as np

from oim.algs import ObjectSubproblem, WrenchConsensus, make_object_shim
from oim.algs.admm import shift_object_actions
from oim.objects import wrap_angle, wrench_weights
from oim.sim3d.build import build_sub_optimizer
from oim.tasks.pusht import PushT


def check_action_budget(task: PushT, verbose: bool = True) -> float:
    """Largest `||w / w_limit||` the object block can express, vs. the deadzone.

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

    Args:
        task: The `PushT` under study.
        verbose: Print the finding.

    Returns:
        The ceiling, in units of the friction-cone limit.
    """
    scale = np.asarray(task.object_action_scale())
    limit = np.asarray(task.object_model.wrench_limit)
    ceiling = float(np.linalg.norm(scale / limit))
    if verbose:
        fraction = float(np.mean(scale / limit))
        print(
            f"action budget: unit action -> {fraction:.3f} x friction-cone "
            f"limit, so max ||w||/limit = {ceiling:.3f}"
        )
        if ceiling < 1.0:
            print(
                "  WARNING: below the breakaway threshold of 1.0, so no "
                "action this block can propose moves the object at all. "
                "The run cannot progress. Raise --wrench-fraction "
                "(1.0 gives a ceiling of 1.73)."
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
        f"MPPI softmax: temperature={temperature:g}, cost std {spread:.1f}, "
        f"ESS {ess:.1f}/{len(weights)}, top weight {weights.max():.3f}"
    )
    if ess < 0.05 * len(weights):
        print(
            f"  WARNING: the average is one sample -- this is an argmax, not "
            f"MPPI, so the mean re-randomizes every step. Raise the object "
            f"temperature toward the cost std ({spread:.0f})."
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
    consensus_variable: str = "wrench",
    wrench_fraction: Optional[float] = None,
    w_rate: Optional[Union[float, Sequence[float]]] = None,
    project_gate: Optional[float] = None,
    noise_level: Optional[float] = None,
    temperature: Optional[float] = None,
    goal: Optional[Sequence[float]] = None,
    start: Optional[Sequence[float]] = None,
) -> Tuple[PushT, ObjectSubproblem, Any, jnp.ndarray]:
    """Task, object subproblem, initial params and start pose.

    Args:
        scene: A key of `oim.utils.scenes.SCENES`.
        robot: `"point"` or `"xarm6"`. There is no robot in this world, but
            it still selects the scene variant, the config file, and the
            `object_action_bounds` branch -- so an object-only study of an
            xArm6 scene must pass `"xarm6"` to be studying the same object
            block that scene's ADMM runs use.
        cfg: A parsed `oim/configs/*.yaml`.
        horizon: Planning horizon H, in steps of `world3d.planning_dt`.
        samples: Rollouts per iteration.
        seed: RNG seed.
        object_opt: Sub-optimizer -- `mppi`, `cem`, `ps` or `cbo`.
        iterations: Optimizer passes per control step. ADMM runs the object
            block `n_admm` times per step (once per consensus round), so
            this is the knob that makes the comparison like-for-like: set
            it to `n_admm` to give the block the same number of updates it
            would get inside ADMM.
        consensus_variable: Kept only so `PushT` is constructed identically
            to the ADMM path; it does not affect anything here, since the
            consensus penalty is switched off.
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
        w_rate: Override the wrench-rate penalty (`PlanarPushingObject.
            rate_cost`), as one number or `[f_x, f_y, tau]`. `None` keeps
            the config's.
        project_gate: Override `project_gate_pos`, the position error
            below which `project_object_action` snaps a sub-threshold
            action up to breakaway. `None` keeps the config's. Decides
            whether the optimizer may average sample *directions* without
            the averaged magnitude falling into the deadzone.
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

    Returns:
        `(task, object_subproblem, initial params, start pose)`.
    """
    w3, smp = cfg["world3d"], cfg["sampler"]
    plan_dt = w3["planning_dt"]

    # Built exactly as `build_admm_3d` builds it, so the object block under
    # study is the one ADMM would use. `impl="jax"` because nothing here
    # ever calls `mjx.step` -- the MJX model is loaded only because `PushT`
    # owns the scene, and the Warp backend would cost a graph capture for a
    # rollout that never happens.
    task = PushT(
        impl="jax",
        clutter=True,
        planning_dt=plan_dt,
        robot=robot,
        consensus_source="contact" if robot == "point" else "twist",
        consensus_variable=consensus_variable,
        env=scene,
        goal=goal,
        costs=cfg.get("costs"),
    )

    if w_rate is not None:
        task.object_model.w_rate = wrench_weights(w_rate)
    if project_gate is not None:
        task.project_gate_pos = project_gate
    if wrench_fraction is not None:
        # `action_scale` is the only thing `wrench_sample_fraction` sets,
        # so overriding it here is equivalent to having constructed
        # `PlanarPushingObject` with a different fraction -- and keeps this
        # a property of the *study*, never of the shipped task.
        task.object_model.action_scale = wrench_fraction * (
            task.object_model.wrench_limit
        )

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
    )

    # A consensus space is required by `ObjectSubproblem`'s constructor but
    # is inert at rho = 0: `penalty_cost` returns 0.5 * sum(0 * diff**2).
    # Constructed with the task's real scale anyway, so that switching the
    # penalty back on for a debugging session needs no other change.
    consensus = WrenchConsensus(
        max_dual=2.0 * float(task.consensus_scale()[0]),
        scale=task.consensus_scale(),
    )
    # proximal_weight = 0: gamma anchors one ADMM *iteration* to the last,
    # and there are no iterations here. Left at the config's value it would
    # instead anchor each control step to the previous step's shifted plan,
    # which is a different mechanism -- a damping term on replanning -- and
    # would quietly slow exactly the routing behaviour being measured.
    block = ObjectSubproblem(task, optimizer, consensus, proximal_weight=0.0)

    params = optimizer.init_params(seed=seed)
    seed_action = task.initial_object_action()
    if seed_action is not None:
        params = params.replace(
            mean=jnp.broadcast_to(
                jnp.asarray(seed_action), (horizon, task.object_action_dim)
            )
        )

    check_action_budget(task)

    obj_state0 = (
        jnp.asarray(task.start, dtype=float)
        if start is None
        else jnp.asarray(start, dtype=float)
    )
    return task, block, params, obj_state0


def run_object(
    task: PushT,
    block: ObjectSubproblem,
    params: Any,
    obj_state0: jnp.ndarray,
    max_steps: int = 200,
    goal_pos_tol: float = 0.05,
    goal_theta_tol: float = 0.05,
    verbose: bool = True,
    jit: bool = True,
    log_samples: bool = True,
) -> Dict[str, Any]:
    """Receding-horizon loop over the object block alone.

    Each step solves for a wrench sequence, applies its first entry through
    `object_dynamics`, and warm-starts by shifting -- the object half of an
    ADMM control step, with the consensus and the robot removed.

    Args:
        task: The `PushT` from `build_object_only`.
        block: Its `ObjectSubproblem`.
        params: The object optimizer's initial policy parameters.
        obj_state0: Object start pose, world-frame SE(2).
        max_steps: Maximum control steps.
        goal_pos_tol: Positional tolerance for declaring success.
        goal_theta_tol: Angular tolerance for declaring success.
        verbose: Print per-step progress.
        jit: Whether to jit the solve. False steps through the object
            dynamics and the sampler in a debugger, which is most of the
            reason to run this world at all.
        log_samples: Keep each step's sampled candidate trajectories for
            the plot. `(num_samples, H, 3)` per step is by far the largest
            thing in the log, so it can be switched off for a long run.

    Returns:
        A log dict in the same shape the other runners produce: `time`,
        `object_pose`, `object_velocity`, `wrench`, `object_plan`,
        `pos_err`, `theta_err`, `compute_time`, `reached` -- plus
        `object_samples` when asked for.
    """
    horizon = params.mean.shape[0]
    dim = task.consensus_dim
    # The ADMM inputs, switched off. Shapes must still be right: they are
    # traced into the penalty even though it evaluates to zero.
    z = jnp.zeros((horizon, dim))
    dual = jnp.zeros((horizon, dim))
    rho = jnp.zeros(dim)
    noise_scale = jnp.zeros(())

    def _solve(
        obj_state: jnp.ndarray, prm: Any, rng: jax.Array
    ) -> Tuple[Any, jnp.ndarray, jnp.ndarray]:
        """One object-block update; returns params, its plan, its samples."""
        new_params, _a_obj, ref_states, samples = block.optimize(
            obj_state, prm, z, dual, rho, prm.mean, noise_scale, rng
        )
        return new_params, ref_states, samples

    solve = jax.jit(_solve) if jit else _solve
    dynamics = jax.jit(task.object_dynamics) if jit else task.object_dynamics

    obj_state = jnp.asarray(obj_state0, dtype=float)
    goal = np.asarray(task.goal)
    rng = jax.random.key(0)
    if verbose:
        report_softmax_ess(task, block, params, obj_state)

    log: Dict[str, Any] = {
        "time": [0.0],
        "object_pose": [np.asarray(obj_state)],
        "wrench": [],
        "object_plan": [],
        "pos_err": [],
        "theta_err": [],
        "compute_time": [],
    }
    if log_samples:
        log["object_samples"] = []
    reached = False

    for step in range(max_steps):
        rng, step_rng = jax.random.split(rng)
        t0 = time.perf_counter()
        params, plan, samples = solve(obj_state, params, step_rng)
        jax.block_until_ready(params)
        log["compute_time"].append(time.perf_counter() - t0)
        if step == 0 and verbose:
            _report_plan_span(np.asarray(plan), np.asarray(samples))

        # The decision actually executed: the first entry of the projected
        # nominal, mapped through the same action -> wrench map the rollout
        # used, so the applied wrench is the one the plan was scored on.
        action = task.project_object_action(params.mean, obj_state)[0]
        wrench = task.object_action_to_consensus(obj_state, action)
        obj_state = dynamics(obj_state, wrench)

        pos_err, theta_err = _log_step(
            log,
            task,
            step,
            obj_state,
            wrench,
            plan,
            samples if log_samples else None,
            goal,
            verbose,
        )
        if pos_err < goal_pos_tol and theta_err < goal_theta_tol:
            reached = True
            if verbose:
                print(f"goal reached at step {step}")
            break

        # Warm start, identically to ADMM's own object warm start.
        params = params.replace(
            mean=shift_object_actions(task, params.mean)
        )

    log["reached"] = reached
    for key in ("time", "object_pose", "wrench", "object_plan"):
        log[key] = np.array(log[key])
    if log_samples:
        log["object_samples"] = np.array(log["object_samples"])
    # Matches the other worlds' logs, where the object twist is recorded:
    # forward Euler on the pose, so the difference quotient is exactly the
    # velocity the model applied.
    log["object_velocity"] = _finite_difference(
        log["object_pose"], float(task.dt)
    )
    return log


# A plan shorter than this spans well under a pixel at any sensible axis
# scale, so it is drawn but cannot be seen -- and an overlay that renders
# nothing looks exactly like one that was never switched on. Same constant
# and the same reasoning as `oim.sim3d.run._VISIBLE_SPAN_M`.
_VISIBLE_SPAN_M = 5e-3


def _report_plan_span(plan: np.ndarray, samples: np.ndarray) -> None:
    """Print how far the first plan and its candidates actually reach.

    Both spans are worth having, because they fail apart: a spread of
    candidates around a motionless nominal means the optimizer is still
    exploring and merely preferring stillness, while candidates that are
    *all* motionless mean no sampled action can move the object at all --
    the deadzone-reachability failure `check_action_budget` predicts.

    Args:
        plan: The object block's nominal, (H, 3).
        samples: Its candidate trajectories, (num_samples, H, 3).
    """
    span = float(np.linalg.norm(plan[-1, :2] - plan[0, :2]))
    spans = np.linalg.norm(samples[:, -1, :2] - samples[:, 0, :2], axis=-1)
    print(
        f"first plan: nominal spans {span:.4f} m; candidates span "
        f"{float(spans.min()):.4f}-{float(spans.max()):.4f} m"
    )
    if float(spans.max()) < _VISIBLE_SPAN_M:
        print(
            "  every candidate is motionless -- the object block cannot "
            "move the object, not merely choosing not to. See the action "
            "budget warning above."
        )
    elif span < _VISIBLE_SPAN_M:
        print(
            "  the nominal is motionless while candidates are not: the "
            "optimizer is choosing stillness, so this is the cost, not "
            "reachability."
        )


def _log_step(
    log: Dict[str, Any],
    task: PushT,
    step: int,
    obj_state: jnp.ndarray,
    wrench: jnp.ndarray,
    plan: jnp.ndarray,
    samples: Optional[jnp.ndarray],
    goal: np.ndarray,
    verbose: bool,
) -> Tuple[float, float]:
    """Append one control step to the log; return its two goal errors.

    The progress line reports the wrench as a fraction of the friction-cone
    limit rather than in newtons, because the number that matters is
    whether it cleared 1.0: below that `PlanarPushingObject.step` holds the
    object still, so a run can look busy in the cost panel and not be
    moving at all.
    """
    log["time"].append((step + 1) * float(task.dt))
    log["object_pose"].append(np.asarray(obj_state))
    log["wrench"].append(np.asarray(wrench))
    log["object_plan"].append(np.asarray(plan))
    if samples is not None:
        log["object_samples"].append(np.asarray(samples))

    pos_err = float(np.linalg.norm(np.asarray(obj_state)[:2] - goal[:2]))
    theta_err = float(
        abs(float(wrap_angle(float(obj_state[2]) - float(goal[2]))))
    )
    log["pos_err"].append(pos_err)
    log["theta_err"].append(theta_err)

    if verbose and step % 10 == 0:
        limit = np.asarray(task.object_model.wrench_limit)
        normalized = float(np.linalg.norm(np.asarray(wrench) / limit))
        held = "  (below breakaway: held)" if normalized < 1.0 else ""
        print(
            f"step {step:4d}  pos_err={pos_err:.4f}  "
            f"theta_err={theta_err:.4f}  "
            f"|w|/limit={normalized:.3f}{held}"
        )
    return pos_err, theta_err


def _finite_difference(series: np.ndarray, dt: float) -> np.ndarray:
    """Per-step velocity of a pose series, heading differences wrapped."""
    if len(series) < 2:
        return np.zeros_like(series)
    deltas = np.diff(series, axis=0)
    deltas[:, 2] = np.asarray(wrap_angle(deltas[:, 2]))
    return np.vstack([np.zeros_like(series[:1]), deltas / dt])
