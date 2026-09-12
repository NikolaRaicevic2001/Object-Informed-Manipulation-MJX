"""Build the hardware/mock stack: the task + controller, and the interface.

The sim worlds each have a builder (`oim.worlds.sim3d.build`,
`oim.worlds.object_only.build`); this is the real world's, and it lived in
`examples/pusht/pusht_real.py` until that script had grown past a thousand
lines. Everything algorithmic still goes through `build_admm_3d` /
`build_flat_3d`, for the reason `build_controller` spells out -- what is
here is the hardware wrapping around them: the velocity clamp, the
execution-fidelity mock, the ROS bridge.

The config arrives as an argument rather than being read from a module
global, so a caller that loaded a different `--config` gets what it loaded.
"""

import math
import os
import time
from copy import copy, deepcopy
from typing import Any, Callable, Optional

import jax
import jax.numpy as jnp
import mujoco
import numpy as np
import yaml

from oim import ROOT
from oim.control_projection import projection_settings
from oim.utils.scenes import SCENES
from oim.worlds.real3d.interface import MujocoMockInterface
from oim.worlds.sim3d.build import build_admm_3d, build_flat_3d

# Planner timestep [s]. A module constant, not `world3d.planning_dt`, so a
# config retune of the rollout step cannot silently move the flat
# baseline's control period out from under a comparison.
PLAN_DT = 0.05

def load_robot_config(name: str) -> dict:
    """Parse `oim/configs/robots/{name}.yaml` -- the file `load_config` reads.

    Reading the SAME file the sim reads is what keeps dt, sampler budget and
    cost weights one source of truth across the two worlds.
    """
    with open(os.path.join(ROOT, "configs", "robots", f"{name}.yaml")) as f:
        return yaml.safe_load(f)


def build_controller(args: Any, cfg: dict) -> tuple:
    """Build the xArm6 PushT task + controller.

    ADMM, or a flat sampler under `--algorithm mppi` -- the real-side twin
    of the sim's `build_flat_3d` / `run_3d_plain`.

    Args:
        args: The parsed command line.
        cfg: The robot config, already rebound to `--config` by the caller.

    Returns:
        `(task, ctrl)`, with the task's velocity bounds clamped to
        `--vel-limit`.
    """
    w3, smp, adm_cfg = cfg["world3d"], cfg["sampler"], cfg["admm"]
    t = time.perf_counter()
    print(f"[setup] loading task/scene '{args.scene}' "
          "(MJCF compile + MJX build)...")

    # Same costs: block for every algorithm -- 2026-09-07, per Shahid: a
    # separate costs_admm: overlay meant MPPI and ADMM could silently be
    # optimizing different tasks (different weights on the same named
    # terms), which makes any comparison between them a comparison of
    # tasks, not of planners. `costs_admm:` no longer exists in the
    # config at all -- see Tasks.md if the old per-algorithm values are
    # ever needed for reference.
    costs = dict(cfg.get("costs") or {})
    for kv in args.cost:
        k, v = kv.split("=", 1)
        # Enumerated keys (`tip_z_form=...`,
        # `align_ref=...`) stay strings; everything else is a float.
        try:
            costs[k] = float(v)
        except ValueError:
            costs[k] = v

    # ADMM goes through the SAME builder the sim uses. This driver used to
    # construct its own PushT + ADMM, and that duplication silently diverged
    # four times, each caught only after hardware runs were produced under
    # it: the robot rollout ran at 1 substep while sim read the config's,
    # the object sampler's noise/temperature were substituted, `rho_torque`
    # arrived as a bare scalar so torque was penalised 10x weaker, and
    # `planning_iterations`/`planning_ls_iterations` were never passed at
    # all, so every hardware run solved contacts at the MJCF's 20/20 against
    # sim's 40/30. `tests/test_sim_real_parity.py` pins the two together.
    #
    # Hardware specifics stay OUT of the builder and are applied around it:
    # the velocity clamp below, and everything in `run_real` (latency
    # compensation, watchdogs, pose gating), which are compensations for
    # running on a real arm, not changes to the algorithm.
    # CLI overrides are folded into the config the builders read, so there
    # is exactly one place each knob is resolved. `cfg` is the one the caller
    # loaded, `--config` included.
    cfg = dict(cfg)
    cfg["costs"] = costs
    cfg["control_projection"] = projection_settings(
        cfg.get("control_projection"), args
    )
    adm = dict(adm_cfg)

    if args.algorithm == "admm":
        cfg["admm"] = adm
        task, ctrl, _, _ = build_admm_3d(
            args.scene, "xarm6", cfg,
            warp=args.warp,
            horizon=args.horizon,
            samples=args.num_samples,
            seed=args.seed,
            robot_opt=args.robot_opt,
            object_opt=args.object_opt,
            n_admm=args.n_admm,
            rho=args.rho,
            gamma=args.gamma,
            consensus_object_weight=float(
                adm.get("consensus_object_weight", 0.5)
            ),
            rho_torque=args.rho_torque,
            consensus=args.consensus,
            lagged_consensus=adm.get("lagged_consensus"),
            plant=args.plant,
            object_substeps=args.object_substeps,
            robot_substeps=int(w3.get("robot_substeps", 1)),
            # `--goal` / `--goal-yaw-deg`; None keeps the scene's own goal.
            goal=resolve_goal(args),
        )
        # The published command is capped at --vel-limit, so cap the
        # planner's sample bounds at the same value. Otherwise it samples up
        # to the model's ctrlrange (+-1.0) and predicts ~5x the object motion
        # the arm can produce; harmless while approaching, but at contact the
        # two blocks argue over an unrealisable wrench and the primal
        # residual runs away.
        task.u_min = jnp.full_like(task.u_min, -args.vel_limit)
        task.u_max = jnp.full_like(task.u_max, args.vel_limit)
        print(f"[setup] task ready in {time.perf_counter() - t:.1f}s; "
              f"ADMM via build_admm_3d (n_admm={args.n_admm}, "
              f"consensus={args.consensus}, plant={args.plant})")
        return task, ctrl

    # The flat baseline goes through the SAME builder the sim uses, for the
    # same reason ADMM does above. This driver used to construct its own
    # PushT + optimizer here; `build_flat_3d` sets `robot_samples` (Warp
    # contact-arena sizing), `robot_substeps` (so the baseline integrates
    # contact at the fidelity ADMM's robot block does -- the gap that once
    # handed ADMM 5x the contact resolution in a head-to-head) and the
    # shared planner-model solver depth, all from the same config keys.
    task, ctrl, _, _ = build_flat_3d(
        args.robot_opt, args.scene, "xarm6", cfg,
        warp=args.warp,
        horizon=args.horizon,
        samples=args.num_samples,
        seed=args.seed,
        control_dt=PLAN_DT,
        iterations=int(smp.get("iterations", 1)),
        robot_substeps=int(w3.get("robot_substeps", 1)),
        goal=resolve_goal(args),
    )
    task.u_min = jnp.full_like(task.u_min, -args.vel_limit)
    task.u_max = jnp.full_like(task.u_max, args.vel_limit)
    print(f"[setup] task ready in {time.perf_counter() - t:.1f}s; "
          f"flat {args.robot_opt} via build_flat_3d, no ADMM "
          f"(knots={smp['robot_num_knots']}, "
          f"noise={smp[args.robot_opt]['noise_level']}, "
          f"substeps={task.robot_substeps})")
    return task, ctrl


def resolve_goal(args: Any) -> Optional[list]:
    """The goal pose this run scores against, or None for the scene's."""
    if args.goal is not None and args.goal_yaw_deg is not None:
        raise SystemExit("--goal and --goal-yaw-deg are mutually exclusive")
    if args.goal is not None:
        return [args.goal[0], args.goal[1], math.radians(args.goal[2])]
    if args.goal_yaw_deg is not None:
        g = SCENES[args.scene].goal
        return [float(g[0]), float(g[1]), math.radians(args.goal_yaw_deg)]
    return None


def _mock_control_filter(projector: Any, mj_data: Any) -> Optional[Callable]:
    """The external CBF node, emulated per control tick, or None.

    Placed on the CPU: one 6-variable QP is pure kernel-launch latency on
    the GPU -- 3.00 ms/tick measured against a 0.12 ms dispatch floor,
    versus 0.67 ms on the CPU, agreeing to 2.5e-9. At 20 ticks a control
    step that is ~47 ms/step of MOCK-ONLY cost, which hardware never pays
    (there the filter is another process). Falls back to the default device
    if the projector's arrays cannot be moved.
    """
    if projector is None:
        return None

    def build(device):
        proj = copy(projector)
        for attr in ("model", "data", "dofs", "jnt_qadr", "jnt_lo", "jnt_hi",
                     "jnt_limited"):
            setattr(proj, attr,
                    jax.device_put(getattr(projector, attr), device))

        @jax.jit
        def filter_command(qpos, qvel, mocap_pos, mocap_quat, command):
            state = proj.data.replace(
                qpos=qpos, qvel=qvel,
                mocap_pos=mocap_pos, mocap_quat=mocap_quat,
            )
            return proj.project(command, proj.prepare(state))[0]

        def control_filter(data, command):
            # Committed to `device`, which is what puts the trace there.
            def put(x):
                return jax.device_put(np.asarray(x, np.float32), device)

            return np.asarray(filter_command(
                put(data.qpos), put(data.qvel), put(data.mocap_pos),
                put(data.mocap_quat), put(command),
            ))

        return control_filter

    candidate = build(jax.devices("cpu")[0])
    try:
        candidate(mj_data, np.zeros(projector.task.model.nu))
    except Exception as exc:  # noqa: BLE001 -- placement only, never fatal
        print(f"[setup] mock filter stays on the default device: {exc!r}")
        return build(jax.devices()[0])
    return candidate


def build_mock_interface(task: Any, control_rate: float, cfg: dict,
                         exact_twist: bool = False,
                         block_start: Any = None) -> MujocoMockInterface:
    """A MuJoCo sim behind the hardware interface, for laptop testing.

    Each `send_velocity` applies the commanded velocity and advances the sim by
    one control tick (1/control_rate). `run_real` calls it `num_ticks` times per
    replanning period, so the sim advances exactly one period per plan.

    exact_twist=True reads the sim's true block qvel (like the sim driver
    run_3d_admm); False (default) finite-differences the pose, as real hardware
    must from FoundationPose. This affects what the MOCK reports, not how A^r
    is formed -- A^r is summed from the planning rollout's contact forces and
    never from an observed twist.
    """
    w3 = cfg["world3d"]
    mj_model = deepcopy(task.mj_model)
    mj_model.opt.timestep = w3["exec_timestep"]
    mj_model.opt.iterations = w3["exec_iterations"]
    mj_model.opt.ls_iterations = w3["exec_ls_iterations"]
    mj_data = mujoco.MjData(mj_model)
    # Start pose: the scene's arm home config (from SCENES[...]["arm_start_deg"],
    # reachable + collision-free for that scene's base) and block start SE(2).
    # Sim scenes leave it None -- fall back to the model's own default qpos0
    # rather than raising TypeError, so --mock runs for them too. A scene that
    # wants a specific mock start pose sets its own xarm6_arm_start_deg.
    if task.arm_start_deg is not None:
        mj_data.qpos[:5] = [math.radians(q) for q in task.arm_start_deg]
    # block_start overrides the scene's nominal block SE(2) -- e.g. rehearse
    # tomorrow's run in the mock from the real block pose FoundationPose reports.
    mj_data.qpos[5:8] = list(block_start if block_start is not None else task.start)
    sim_steps_per_send = max(1, round((1.0 / control_rate) / w3["exec_timestep"]))
    # The real bridge publishes nominal commands to an external filter.
    # Reproduce that boundary in the mock, refreshing constraints at every
    # control tick rather than reusing the planner's frozen state.
    control_filter = _mock_control_filter(
        getattr(task, "control_projector", None), mj_data
    )

    return MujocoMockInterface(mj_model, mj_data, sim_steps_per_send,
                               emulate_pose_only=not exact_twist,
                               control_filter=control_filter)


def build_real_interface(task: Any, velocity_topic: str,
                         enable_commands: bool,
                         object_origin_offset: tuple = (0.0, 0.0)) -> Any:
    """The real ROS2 <-> xArm6 bridge. Import is lazy so --mock needs no ROS.

    Frames, joint naming and watchdog default from the OI-MPPI reference in
    Ros2Interface.__init__. `enable_commands=False` is the dry run (reads
    state/TF, publishes nothing). `task.world_frame` selects the planner's TF
    frame: "xarm_device" for base-at-origin scenes (reads FoundationPose's TF
    directly, no world->base transform), or "world" otherwise.
    """
    from oim.worlds.real3d.interface import Ros2Interface  # noqa: PLC0415

    return Ros2Interface(
        world_frame=task.world_frame,
        object_origin_offset=object_origin_offset,
        base_pos=task.base_pos,
        base_yaw_deg=task.base_yaw_deg,
        base_z=task.base_z,
        velocity_command_topic=velocity_topic,
        enable_commands=enable_commands,
    )

