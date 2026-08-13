"""Construct the 3D world: task, controller, and execution model.

Splitting this out from the runners means a flat baseline and an ADMM run
are built from the *same* scene, horizon, sampler budget and execution
model -- the only honest way to compare them. The 2D counterpart is
`oim.sim2d.run.build_admm_2d`.

Everything here reads its numbers from a config dict (`oim/configs/*.yaml`)
rather than holding constants, so retuning a method is a config edit.
"""

from copy import deepcopy
from typing import Any, Dict, Optional, Sequence, Tuple

import mujoco
import numpy as np

from oim.algs import (
    ADMM,
    CBO,
    CEM,
    MPPI,
    PoseConsensus,
    PredictiveSampling,
    WrenchConsensus,
    make_object_shim,
)
from oim.tasks.pusht import PushT

SUB_OPTIMIZERS = ["mppi", "cem", "ps", "cbo"]

# Joint config (degrees) putting the xArm6's stick tip near the clutter
# scene's block start; found via
# oim/models/xarm6_pusht_clutter/verify_reach.py. Fallback only, and used
# by `clutter` alone -- a scene with its own workspace (the tabletop
# family) defines a "start" keyframe instead, not another entry here.
XARM6_START_QPOS_DEG = [-15.43, 100.0, -185.36, 0.0, 60.0]

# The point-mass pusher's own start, for the original `clutter` scene only
# (see `point_start_qpos`): block at the origin, pusher just below it. Two
# slides and a hinge for the block, then the pusher's x/y.
POINT_START_QPOS = [0.0, 0.0, 0.0, -0.05, -0.06]


def point_start_qpos(mj_model: mujoco.MjModel) -> np.ndarray:
    """Initial qpos for a point-robot scene.

    Reads the model's own "start" keyframe if it defines one -- the five
    tabletop scenes' point variants each do, matching their own workspace
    and obstacle layout -- falling back to `POINT_START_QPOS` otherwise
    (`clutter`, which has none). Mirrors `xarm6_start_qpos` exactly;
    before this, every point-robot scene silently reused
    `POINT_START_QPOS` regardless of what its own MJCF declared, so five
    different scenes all started from the same clutter-specific offset.

    Args:
        mj_model: The compiled scene.

    Returns:
        A full qpos vector.
    """
    key_id = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_KEY, "start")
    if key_id >= 0:
        return np.array(mj_model.key_qpos[key_id])
    return np.array(POINT_START_QPOS)


def xarm6_start_qpos(mj_model: mujoco.MjModel) -> np.ndarray:
    """Initial qpos for an xArm6 scene.

    Reads the model's own "start" keyframe if it defines one -- every scene
    besides the original clutter layout does, since each has a different
    workspace -- falling back to `XARM6_START_QPOS_DEG` otherwise. Generic
    over `mj_model.nq`, so it also covers the block's own qpos entries.

    Args:
        mj_model: The compiled scene.

    Returns:
        A full qpos vector.
    """
    key_id = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_KEY, "start")
    if key_id >= 0:
        return np.array(mj_model.key_qpos[key_id])
    qpos = np.zeros(mj_model.nq)
    qpos[:5] = np.radians(XARM6_START_QPOS_DEG)
    return qpos


def named_camera(
    mj_model: mujoco.MjModel, name: str = "front"
) -> Optional[str]:
    """`name` if the model defines a camera by that name, else `None`.

    `None` means the default free camera framing the whole scene. A scene
    the free camera frames badly (the tabletop family, viewed from the side
    by default) defines its own fixed camera instead of a special case in
    the caller.

    Args:
        mj_model: The compiled scene.
        name: Camera to look for.

    Returns:
        The camera name, or None.
    """
    if mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_CAMERA, name) >= 0:
        return name
    return None


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


def mocap_id(mj_model: mujoco.MjModel, name: str) -> int:
    """The mocap index of body `name`, or -1 if the scene has no such body.

    Every marker write goes through this rather than a literal index.
    `goal` was mocap 0 only because it was the one mocap body in every
    scene; adding `local_goal` makes it a two-entry table ordered by
    declaration. The two happen to land at 0 and 1 in all seven scenes
    today, which is exactly the kind of coincidence the old literal relied
    on -- a scene that declares its marker before its goal would silently
    swap them.

    Args:
        mj_model: The compiled scene.
        name: Body name.

    Returns:
        The index into `mocap_pos`/`mocap_quat`, or -1 when the body is
        absent (scenes predating the marker) or is not a mocap body.
    """
    body_id = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_BODY, name)
    if body_id < 0:
        return -1
    return int(mj_model.body_mocapid[body_id])


def hide_body_geoms(mj_model: mujoco.MjModel, name: str) -> None:
    """Make every geom of body `name` fully transparent, in place.

    For a marker the run will not be driving. Editing alpha rather than
    moving the body out of frame keeps the fix local to rendering: the body
    stays where it is, so nothing that reads a position changes meaning.

    Args:
        mj_model: The model to edit -- pass the *execution* model, which is
            a deepcopy, not the task's own.
        name: Body name. Absent is a no-op.
    """
    body_id = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_BODY, name)
    if body_id < 0:
        return
    start = int(mj_model.body_geomadr[body_id])
    for geom in range(start, start + int(mj_model.body_geomnum[body_id])):
        mj_model.geom_rgba[geom][3] = 0.0


def set_mocap_se2(
    mj_data: mujoco.MjData, index: int, pose: Sequence[float]
) -> None:
    """Place mocap body `index` at an SE(2) pose, keeping its height.

    The markers are flat objects lying on the table, so only (x, y, yaw)
    ever move -- z stays at whatever the MJCF put the body at, which is the
    block's own resting height and differs per scene.

    Args:
        mj_data: The state to write into.
        index: A `mocap_id` result; negative is a no-op, so a caller need
            not branch on whether the scene has the marker.
        pose: World-frame `[x, y, theta]`.
    """
    if index < 0:
        return
    pose = np.asarray(pose, dtype=float)
    mj_data.mocap_pos[index][:2] = pose[:2]
    half = 0.5 * float(pose[2])
    mj_data.mocap_quat[index] = [np.cos(half), 0.0, 0.0, np.sin(half)]


def execution_model(
    task: PushT,
    robot: str,
    cfg: Dict[str, Any],
    start: Optional[Sequence[float]] = None,
    goal: Optional[Sequence[float]] = None,
) -> Tuple[mujoco.MjModel, mujoco.MjData]:
    """A separate, finer MuJoCo model for *executing* the plan.

    The planner rolls out at `planning_dt`; execution steps at
    `exec_timestep` with more solver iterations, so the closed loop is not
    graded against the same coarse integration it planned with.

    Args:
        task: The task whose `mj_model` to copy.
        robot: Embodiment, selecting the start pose.
        cfg: The config's `world3d` block.
        start: Object start pose, world-frame SE(2) `[x, y, theta]`,
            written straight into the block's slide/hinge `qpos`. `None`
            keeps the MJCF keyframe's own.
        goal: Goal pose, moving the `goal` mocap marker to match what the
            task's costs aim at. `None` keeps the MJCF's own. The costs
            themselves come from `PushT(goal=...)`; this is the marker and
            the goal-relative sensors that read off it.

    Returns:
        The execution model and its data, at the start pose.
    """
    mj_model = deepcopy(task.mj_model)
    mj_model.opt.timestep = cfg["exec_timestep"]
    mj_model.opt.iterations = cfg["exec_iterations"]
    mj_model.opt.ls_iterations = cfg["exec_ls_iterations"]
    mj_data = mujoco.MjData(mj_model)
    if robot == "xarm6":
        mj_data.qpos[:] = xarm6_start_qpos(mj_model)
    else:
        mj_data.qpos[:] = point_start_qpos(mj_model)
    if start is not None:
        # The block's joints are two slides and a hinge anchored at the
        # origin, so qpos *is* the world pose -- the same invariant
        # `PushT._block_pose` relies on when it reads qpos back.
        mj_data.qpos[np.asarray(task.block_qpos_indices)] = np.asarray(
            start, dtype=float
        )
    if goal is not None:
        set_mocap_se2(mj_data, mocap_id(mj_model, "goal"), goal)
    # Park the local-goal marker on the block's start pose, so it is not
    # sitting at the world origin for the one frame before the first plan
    # exists. Absent in scenes without the marker, where this is a no-op.
    set_mocap_se2(
        mj_data,
        mocap_id(mj_model, "local_goal"),
        mj_data.qpos[np.asarray(task.block_qpos_indices)],
    )
    # Populate xpos/site_xpos/sensordata for whatever reads them before the
    # first step -- the initial log entry, and the goal-relative sensors.
    mujoco.mj_forward(mj_model, mj_data)
    return mj_model, mj_data


def build_admm_3d(
    scene: str,
    robot: str,
    cfg: Dict[str, Any],
    *,
    warp: bool,
    horizon: int,
    samples: int,
    seed: int,
    object_samples: Optional[int] = None,
    robot_opt: str,
    object_opt: str,
    n_admm: int,
    rho: float,
    gamma: float,
    consensus_alpha: float = 1.0,
    rho_torque: Optional[float] = 10.0,
    consensus_variable: str = "wrench",
    local_goal: bool = False,
    start: Optional[Sequence[float]] = None,
    goal: Optional[Sequence[float]] = None,
) -> Tuple[PushT, ADMM, mujoco.MjModel, mujoco.MjData]:
    """Task, ADMM controller, and execution model/data for the 3D world.

    Args:
        scene: A key of `oim.utils.scenes.SCENES`.
        robot: `"point"` or `"xarm6"`.
        cfg: A parsed `oim/configs/*.yaml`.
        warp: Use the MuJoCo Warp rollout backend.
        horizon: Consensus horizon H, in planning steps. Shared by both
            blocks: H is the *consensus* horizon, and z, the duals and
            both A sequences are all (H, dim), so the two blocks cannot
            currently disagree about it. See README's implementation notes.
        samples: Rollouts per block, unless `object_samples` overrides the
            object block's.
        seed: RNG seed.
        object_samples: Rollouts for the object block alone. `None` reads
            `sampler.object.num_samples`, and falls back to `samples`.

            Worth splitting because the two blocks cost wildly different
            amounts per rollout: the object block integrates a 3-vector in
            closed form, the robot block steps MJX over the whole scene.
            Sample counts are also genuinely independent -- each block
            reweights its own population, and only the (H, dim) consensus
            values pass between them -- so this is a budget knob, not a
            formulation change.
        robot_opt: Sub-optimizer for the robot block.
        object_opt: Sub-optimizer for the object block.
        n_admm: Max ADMM iterations per control step.
        rho: Initial penalty, on the wrench's two force components (and
            the torque component too, if `rho_torque` is unset).
        gamma: Proximal weight.
        consensus_alpha: EMA weight on A^o/A^r across ADMM rounds (1.0 =
            raw). See `ADMM`.
        rho_torque: Penalty on the wrench's torque component alone, or
            `None` for the paper's single scalar. Lets the penalty pull
            harder on orientation agreement than on position independently
            of the cost. Default 10.0: an ablation found it the one
            formulation-level change that moved both position and
            orientation error together in most scenes. Under
            `consensus_variable="pose"` it weights the heading instead.
        consensus_variable: `"wrench"` (the paper's own, eq. 24) or
            `"pose"`, which makes the blocks agree on the object's SE(2)
            trajectory. Selects `WrenchConsensus` or `PoseConsensus` and
            the matching `PushT.consensus_scale()`.
        local_goal: Point the robot block's goal tracking at the object
            block's horizon endpoint instead of the global goal. See
            `PushT`'s own argument of the same name.
        start: Object start pose, or `None` for the scene's own.
        goal: Object goal pose, or `None` for the scene's own. See
            `examples/poses/`.

    Returns:
        `(task, controller, exec model, exec data)`.
    """
    w3, smp, adm = cfg["world3d"], cfg["sampler"], cfg["admm"]
    plan_dt = w3["planning_dt"]

    # "contact" (point-mass only) reads the real constraint force; "twist"
    # infers the wrench from motion and converges worse, but is the only
    # option for an articulated arm.
    consensus_source = "contact" if robot == "point" else "twist"
    # Clip on the robot block's *estimated* wrench, scene-gated because
    # ablations disagreed about it. 16 is data-driven: over a 1500-step
    # shelf_gap run |z| had median 5.81, p95 12.61, p99 16.10, max 21.79,
    # so 16 clips the true outliers only (30 clipped nothing and broke
    # convergence outright). But on open_table reverting the clip took
    # final pos_err 0.369 -> 0.046, while icra_sign went the other way
    # (0.159 with, 0.318 without). single_obstacle is excluded by
    # association with open_table, not separately ablated.
    _WRENCH_CLIP_SCENES = {"shelf_gap", "icra_sign", "clutter"}
    realized_wrench_clip = (
        [16.0, 16.0, 0.471 * 16.0 / 7.848]
        if consensus_source == "contact" and scene in _WRENCH_CLIP_SCENES
        else None
    )
    task = PushT(
        impl="warp" if warp else "jax",
        clutter=True,
        planning_dt=plan_dt,
        robot=robot,
        consensus_source=consensus_source,
        consensus_variable=consensus_variable,
        env=scene,
        goal=goal,
        costs=cfg.get("costs"),
        realized_wrench_clip=realized_wrench_clip,
        local_goal=local_goal,
    )
    # Normalizing by the characteristic magnitude (the friction-cone limit
    # for a wrench, the object's own size for a pose) keeps the ADMM
    # penalty O(1) and comparable to the task costs, so rho is a
    # meaningful knob in either space.
    scale = task.consensus_scale()
    consensus_cls = (
        PoseConsensus if consensus_variable == "pose" else WrenchConsensus
    )
    # Per-dimension for a pose: its two components have genuinely
    # different units (metres, radians) and a single scalar bound taken
    # from the first would leave the heading dual effectively unclipped.
    max_dual = (
        2.0 * np.asarray(scale)
        if consensus_variable == "pose"
        else 2.0 * float(scale[0])
    )
    consensus = consensus_cls(max_dual=max_dual, scale=scale)
    robot_optimizer = build_sub_optimizer(
        robot_opt,
        task,
        plan_horizon=horizon * plan_dt,
        num_knots=smp["robot_num_knots"],
        spline=smp["robot_spline"],
        seed=seed,
        num_samples=samples,
        sampler_cfg=smp,
    )
    object_optimizer = build_sub_optimizer(
        object_opt,
        make_object_shim(task, dt=plan_dt),
        plan_horizon=horizon * plan_dt,
        num_knots=horizon,
        spline=smp["object_spline"],
        seed=seed,
        num_samples=object_sample_count(smp, samples, object_samples),
        sampler_cfg=smp,
        # The object block samples wrenches, the robot block joint
        # velocities; `sampler.object:` is where the former's own
        # noise/temperature live. Absent, both blocks share one setting,
        # which is what shipped before this key existed. `num_samples`
        # sits beside the per-optimizer sub-blocks rather than inside
        # them, since it is not one of the sampler's own parameters --
        # `build_sub_optimizer` rejects overrides that are.
        overrides=smp.get("object", {}).get(object_opt),
    )
    # A vector rho weights the wrench's torque component separately from
    # its two forces (WrenchConsensus.penalty_cost sums rho * diff**2, so
    # this is a per-dimension penalty, not a single scalar); unset keeps
    # the paper's single scalar.
    rho_init = rho if rho_torque is None else np.array([rho, rho, rho_torque])
    ctrl = ADMM(
        task,
        robot_optimizer,
        object_optimizer,
        consensus,
        n_admm=n_admm,
        eps_r=adm["eps_r"],
        eps_s=adm["eps_s"],
        proximal_weight=gamma,
        rho_init=rho_init,
        rho_adapt=adm["rho_adapt"],
        rho_bound_factor=adm["rho_bound_factor"],
        noise_min=adm["noise_min"],
        noise_kappa=adm["noise_kappa"],
        noise_max=adm["noise_max"],
        consensus_alpha=consensus_alpha,
    )
    mj_model, mj_data = execution_model(task, robot, w3, start, goal)
    return task, ctrl, mj_model, mj_data


def build_flat_3d(
    method: str,
    scene: str,
    robot: str,
    cfg: Dict[str, Any],
    *,
    warp: bool,
    horizon: int,
    samples: int,
    seed: int,
    control_dt: float,
    iterations: int = 1,
    start: Optional[Sequence[float]] = None,
    goal: Optional[Sequence[float]] = None,
) -> Tuple[PushT, object, mujoco.MjModel, mujoco.MjData]:
    """A flat baseline on ADMM's own scene, horizon and sampler budget.

    Deliberately the same task and execution model `build_admm_3d`
    produces, so a comparison isolates the object-level hierarchy rather
    than a difference in setup.

    Args:
        method: `mppi`, `cem`, `ps` or `cbo`.
        scene: A key of `oim.utils.scenes.SCENES`.
        robot: `"point"` or `"xarm6"`.
        cfg: A parsed `oim/configs/*.yaml`.
        warp: Use the MuJoCo Warp rollout backend.
        horizon: Planning horizon, in control steps.
        samples: Rollouts per iteration.
        seed: RNG seed.
        control_dt: Replanning period, also the planner's rollout timestep.
        iterations: Internal optimizer passes per real control step (paper
            default 1). The "vanilla, more inner iterations" side of the
            iterations-vs-n_admm ablation -- does replanning harder each
            step buy what ADMM's outer consensus loop buys, or not.
        start: Object start pose, or `None` for the scene's own.
        goal: Object goal pose, or `None` for the scene's own. See
            `examples/poses/`.

    Returns:
        `(task, controller, exec model, exec data)`.
    """
    w3, smp = cfg["world3d"], cfg["sampler"]
    task = PushT(
        impl="warp" if warp else "jax",
        clutter=True,
        planning_dt=control_dt,
        robot=robot,
        env=scene,
        goal=goal,
        costs=cfg.get("costs"),
    )
    ctrl = build_sub_optimizer(
        method,
        task,
        plan_horizon=horizon * control_dt,
        num_knots=smp["robot_num_knots"],
        spline=smp["robot_spline"],
        seed=seed,
        num_samples=samples,
        iterations=iterations,
        sampler_cfg=smp,
    )
    mj_model, mj_data = execution_model(task, robot, w3, start, goal)
    return task, ctrl, mj_model, mj_data
