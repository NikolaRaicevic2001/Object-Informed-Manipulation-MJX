"""Construct the 3D world: task, controller, and execution model.

Splitting this out from the runners means a flat baseline and an ADMM run
are built from the *same* scene, horizon, sampler budget and execution
model -- the only honest way to compare them. The 2D counterpart is
`oim.sim2d.run.build_admm_2d`.

Everything here reads its numbers from a config dict (`oim/configs/*.yaml`)
rather than holding constants, so retuning a method is a config edit.
"""

from typing import Any, Dict, Optional, Sequence, Tuple

import mujoco
import numpy as np

from oim.algs import ADMM, make_object_shim
from oim.runtime.mjcf import execution_model
from oim.runtime.samplers import (
    build_sub_optimizer,
    consensus_space,
    object_sample_count,
)
from oim.tasks.pusht import PushT


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
    consensus = consensus_space(task, consensus_variable)
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
