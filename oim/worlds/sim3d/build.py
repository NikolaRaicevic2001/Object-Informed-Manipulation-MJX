"""Construct the 3D world: task, controller, and execution model.

Splitting this out from the runners means a flat baseline and an ADMM run
are built from the *same* scene, horizon, sampler budget and execution
model -- the only honest way to compare them. The 2D counterpart is
`oim.worlds.object_only.build`.

Everything here reads its numbers from a config dict
(`oim/configs/robots/*.yaml`) rather than holding constants, so retuning a
method is a config edit.
"""

from typing import Any, Dict, Optional, Sequence, Tuple

import mujoco
import numpy as np

from oim.algs import ADMM, MJXRollout, make_object_shim
from oim.objects.library import SCENE_DEFAULT
from oim.runtime.mjcf import execution_model
from oim.runtime.object_mjx import PREDICT_SUBSTEPS, build_object_rollout
from oim.runtime.samplers import (
    build_sub_optimizer,
    consensus_space,
    object_noise_scale,
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
    consensus_object_weight: float = 0.5,
    rho_torque: Optional[float] = 10.0,
    consensus: str = "wrench",
    consensus_source: Optional[str] = None,
    plant: str = "analytic",
    object_substeps: int = PREDICT_SUBSTEPS,
    robot_substeps: Optional[int] = None,
    local_goal: bool = False,
    local_goal_lookahead: float = 0.0,
    push_object: str = SCENE_DEFAULT,
    start: Optional[Sequence[float]] = None,
    goal: Optional[Sequence[float]] = None,
) -> Tuple[PushT, ADMM, mujoco.MjModel, mujoco.MjData]:
    """Task, ADMM controller, and execution model/data for the 3D world.

    Args:
        scene: A key of `oim.utils.scenes.SCENES`.
        robot: `"point"` or `"xarm6"`.
        cfg: A parsed `oim/configs/robots/*.yaml`.
        warp: Use the MuJoCo Warp rollout backend, for BOTH blocks -- the
            robot block's own rollouts and the object block's, when
            `plant="mujoco"`.
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
            closed form, while the robot block steps MJX over the whole
            scene. `plant="mujoco"` narrows that gap but does not close it
            the way it looks -- the MJX object block is latency-bound, so
            its cost is flat in this number (measured flat from 64 to 512)
            and linear in `horizon` and `n_admm` instead. The shipped 512
            against the robot block's 16 stays a reasonable ratio there.
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
        consensus_object_weight: The object block's share of the z-update
            (0.5 = the paper's average). See `ADMM`.
        rho_torque: Penalty on the wrench's torque component alone, or
            `None` for the paper's single scalar. Lets the penalty pull
            harder on orientation agreement than on position independently
            of the cost. Default 10.0: an ablation found it the one
            formulation-level change that moved both position and
            orientation error together in most scenes. Under
            Under `consensus="contact_point"` it weights the
            lambda channel instead, where "torque" has no meaning.
        consensus: `"wrench"` (the paper's own, eq. 24);
            `"contact_point"` = [p_x, p_y, lambda], which makes the blocks
            agree on where on the boundary to push and how hard; or
            `"object_pose"` = [x, y, yaw], which makes them agree on where
            the object ends up. The first two drive the object block's
            *sampling* space too, so the decision it samples is the agreed
            quantity; `"object_pose"` leaves it sampling wrenches and
            derives A^o from the rollout, which is why A^r there needs no
            force estimator. Selects `WrenchConsensus`,
            `ContactPointConsensus` or `ObjectPoseConsensus` and the
            matching `PushT.consensus_scale()`.
        consensus_source: How the robot block estimates A^r. Unset reads
            `admm.consensus_source`, then falls back to `"contact"` for
            the point robot and `"twist"` for the arm. `"twist_exact"`
            inverts the plant's own slip term -- see
            `PushT._consensus_from_twist_exact`.
        plant: Which dynamics the *object block* plans against. This
            world always executes in MuJoCo -- the robot block steps MJX
            and the run is graded by the execution model -- so unlike the
            object-only world there is no execution side to choose, and
            the mode names only the prediction.

            `"analytic"` is our formulation, the quasi-static limit surface
            of eq. 5, and the default: it is what the paper's results are,
            and changing it would silently reprice every existing 3D run.
            `"mujoco"` instead runs the object block through MJX on a
            stripped copy of this scene, in parallel with the robot block,
            so both blocks predict with the engine the run is executed in
            and the object plan is no longer quasi-static.

            Not free, but not in the way it looks. The MJX object block is
            latency-bound: its cost is ~0.89 ms per horizon step per ADMM
            round and is *flat* in `object_samples` (measured flat from 64
            to 512), because `mjx.step` must be issued once per horizon
            step sequentially while the batch stays far from saturating
            the GPU. `horizon` and `n_admm` are the knobs, not the sample
            count. See `oim.runtime.object_mjx`.
        robot_substeps: MJX physics steps per planning step in the ROBOT
            block's rollout. `None` reads `world3d.robot_substeps`, or 1
            where a config does not set it.
        object_substeps: MJX physics steps per planning step, under
            `plant="mujoco"`. Defaults to `PREDICT_SUBSTEPS`, where the
            object block's integration error against the executed model
            stops dominating; 1 gives it the same coarse integration the
            analytic model has.
        local_goal_lookahead: Distance [m] ahead along that plan the
            target sits; 0 keeps the plan's endpoint. See
            `PushT.local_goal_from_plan`.
        local_goal: Point the robot block's goal tracking at the object
            block's horizon endpoint instead of the global goal. See
            `PushT`'s own argument of the same name.
        push_object: Which object to push -- `SCENE_DEFAULT` for the
            scene's own, or a key of `oim.objects.library.PUSH_OBJECTS`.
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
    # option for an articulated arm. `admm.consensus_source` overrides,
    # which is how "twist_exact" is A/B'd against "twist" -- see
    # `PushT._consensus_from_twist_exact`.
    consensus_source = consensus_source or adm.get(
        "consensus_source", "contact" if robot == "point" else "twist"
    )
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
        twist_stick_speed=adm.get("twist_stick_speed", 0.005),
        consensus=consensus,
        env=scene,
        push_object=push_object,
        goal=goal,
        costs=cfg.get("costs"),
        # `admm:`, not `costs:` -- neither sizes a cost, they size the
        # object block's action. `PushT` falls back to the legacy `costs:`
        # key when absent, so an older config still works.
        wrench_fraction=adm.get("wrench_fraction"),
        contact_fraction=adm.get("contact_fraction"),
        realized_wrench_clip=realized_wrench_clip,
        local_goal=local_goal,
        local_goal_lookahead=local_goal_lookahead,
    )
    # Normalizing by the characteristic magnitude (the friction-cone limit
    # for a wrench, the object's own size for a pose) keeps the ADMM
    # penalty O(1) and comparable to the task costs, so rho is a
    # meaningful knob in either space.
    space = consensus_space(task, consensus)
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
        # Under `contact_point` the block samples metres and newtons, not
        # a unit box, so `noise_level` is read as a fraction of the
        # object's own size and force ceiling.
        noise_scale=object_noise_scale(task, consensus),
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
        space,
        n_admm=n_admm,
        eps_r=adm["eps_r"],
        eps_s=adm["eps_s"],
        proximal_weight=gamma,
        rho_init=rho_init,
        rho_adapt=adm["rho_adapt"],
        rho_bound_factor=adm["rho_bound_factor"],
        consensus_object_weight=consensus_object_weight,
        # The robot block integrates contact at `planning_dt /
        # robot_substeps`. Absent from a config, 1 -- the pre-existing
        # single coarse `mjx.step`, so no config is changed by this
        # existing. See `MJXRollout` and `point.yaml` for the measured
        # planner-vs-execution gap it closes.
        rollout=MJXRollout(
            substeps=(
                int(w3.get("robot_substeps", 1))
                if robot_substeps is None
                else robot_substeps
            )
        ),
        # `None` for the analytic backend: the default lives in
        # `ObjectSubproblem`, not in each builder.
        object_rollout=build_object_rollout(
            plant,
            task,
            robot,
            w3,
            substeps=object_substeps,
            # The same backend the robot block got above. `--warp` used to
            # reach only that one, so a "warp run" was half warp; the two
            # blocks now share a pipeline as well as a scene.
            impl="warp" if warp else "jax",
            # Warp's contact arenas are shared across the batch, so they
            # have to be sized from the real sample count -- the same
            # number given to the optimizer above, not a default.
            num_samples=object_sample_count(smp, samples, object_samples),
        ),
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
    push_object: str = SCENE_DEFAULT,
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
        cfg: A parsed `oim/configs/robots/*.yaml`.
        warp: Use the MuJoCo Warp rollout backend. One block here, so
            unlike `build_admm_3d` this reaches only the flat controller's
            own rollouts.
        horizon: Planning horizon, in control steps.
        samples: Rollouts per iteration.
        seed: RNG seed.
        control_dt: Replanning period, also the planner's rollout timestep.
        iterations: Internal optimizer passes per real control step (paper
            default 1). The "vanilla, more inner iterations" side of the
            iterations-vs-n_admm ablation -- does replanning harder each
            step buy what ADMM's outer consensus loop buys, or not.
        push_object: Which object to push -- `SCENE_DEFAULT` for the
            scene's own, or a key of `oim.objects.library.PUSH_OBJECTS`.
        start: Object start pose, or `None` for the scene's own.
        goal: Object goal pose, or `None` for the scene's own. See
            `examples/poses/`.

    Returns:
        `(task, controller, exec model, exec data)`.
    """
    w3, smp, adm = cfg["world3d"], cfg["sampler"], cfg["admm"]
    task = PushT(
        impl="warp" if warp else "jax",
        clutter=True,
        planning_dt=control_dt,
        # Flat baseline only -- build_admm_3d builds its own PushT
        # separately and never reads these, so ADMM/the point robot's
        # planning fidelity is untouched regardless of what these say.
        planning_iterations=w3.get("planning_iterations"),
        planning_ls_iterations=w3.get("planning_ls_iterations"),
        robot=robot,
        env=scene,
        push_object=push_object,
        goal=goal,
        costs=cfg.get("costs"),
        # A flat baseline has no object block, so neither is read; passed
        # anyway so `task.object_model.action_scale` is the same array it
        # would be under ADMM and the two are comparable.
        wrench_fraction=adm.get("wrench_fraction"),
        contact_fraction=adm.get("contact_fraction"),
    )
    if method == "c3":
        # C3+ (Push Anything): a SamplingBasedController subclass, so it runs
        # through the same run_3d_plain path -- but it is constructed here, not
        # via build_sub_optimizer, which is a sampling-optimizer-only factory
        # (reads sampler_cfg[name] noise/temperature, takes num_samples). C3 has
        # no sample population or those params. N=10 from
        # sampling_c3plus_options.yaml.
        from oim.algs.c3_dynamic import C3MJXSampling  # noqa: PLC0415
        ctrl = C3MJXSampling(task, plan_horizon=10 * control_dt, num_knots=10,
                        seed=seed, num_random=8, q_theta=40.0)
    else:
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
