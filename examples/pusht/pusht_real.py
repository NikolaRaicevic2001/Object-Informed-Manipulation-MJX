"""Entry point: run the push-T ADMM controller on the real xArm6 (or a mock).

The hardware sibling of `examples/clutter.py`'s `--robot xarm6 admm --headless`
path. The task and the ADMM controller are built exactly as there -- same
scene, same weights, same hyperparameters -- so any difference in behaviour
is the sim-to-real gap, not a different planner. Only the execution driver
changes: `oim.worlds.real3d.run_real` instead of
`oim.worlds.sim3d.run.run_3d_admm`.

    # Laptop / dev: drive a MuJoCo sim through the hardware interface.
    #   Validates the whole loop with no robot and no ROS.
    python examples/pusht_real.py --mock --steps 200

    # Robot (at the lab, with the arm + FoundationPose + velocity controller):
    python examples/pusht_real.py --steps 400

The states/metrics JSON is written with the same schema as the simulation
run, so `oim.utils.metrics` / a diff of the two files gives the sim-to-real
comparison directly.
"""

import argparse
import math
import os
import time
import warnings
from copy import deepcopy

# Persist XLA compilations across runs so the minutes-long JIT warm-up only
# happens once per config (later runs load from disk). Set before JAX is
# imported (via the oim modules below); override with the env var if needed.
os.environ.setdefault("JAX_COMPILATION_CACHE_DIR",
                      os.path.expanduser("~/.cache/jax"))

# Cosmetic warnings during CPU JAX tracing / MuJoCo model compile. The results
# are unaffected (the saved states contain no NaNs); filtered here at the entry
# point only so the closed-loop log stays readable.
warnings.filterwarnings("ignore", message="overflow encountered in cast")
warnings.filterwarnings("ignore", message=".*coplanar face.*")

import jax.numpy as jnp
import mujoco
import numpy as np
import yaml

from oim import ROOT
from oim.algs import (
    ADMM,
    CBO,
    CEM,
    MPPI,
    MJXRollout,
    PredictiveSampling,
    make_object_shim,
)
from oim.runtime.object_mjx import build_object_rollout
from oim.runtime.samplers import build_sub_optimizer as build_cfg_optimizer
from oim.runtime.samplers import consensus_space, object_noise_scale
from oim.tasks.pusht import PushT
from oim.utils.results import RunName, save_run
from oim.utils.scenes import SCENES
from oim.worlds.real3d.interface import MujocoMockInterface
from oim.worlds.real3d.run_real import run_real


def _load_cfg(name):
    """Parse `oim/configs/robots/{name}.yaml` -- the file `load_config` reads.

    Reading the SAME file the sim reads is what keeps dt, sampler budget and
    cost weights one source of truth across the two worlds.
    """
    with open(os.path.join(ROOT, "configs", "robots", f"{name}.yaml")) as f:
        return yaml.safe_load(f)


_CFG = _load_cfg("xarm6")

PLAN_DT = 0.05      # planner timestep (matches examples/clutter.py)
# Mock execution model = the sim's, from the same yaml (build.py reads world3d
# exec_* into opt too), so mock and sim advance identical physics.
_W3 = _CFG["world3d"]
_SMP = _CFG["sampler"]
_RUN = _CFG["run"]
_ADM = _CFG["admm"]
# (arm start config is per-scene: SCENES[...]["arm_start_deg"] in oim/tasks/pusht.py)


def build_sub_optimizer(name, task, *, plan_horizon, num_knots, spline, seed,
                        num_samples):
    """Like examples/clutter.py::build_sub_optimizer, but with a tunable sample
    count -- xarm6 needs a smaller budget than the point mass (64 samples can
    exhaust an 11 GB GPU for the arm; see oim/configs/robots/xarm6.yaml).
    """
    common = dict(
        plan_horizon=plan_horizon,
        spline_type=spline,
        num_knots=num_knots,
        seed=seed,
    )
    if name == "mppi":
        return MPPI(task,
                    num_samples=num_samples,
                    noise_level=0.5,
                    temperature=0.5,
                    **common)
    if name == "cem":
        return CEM(task,
                   num_samples=num_samples,
                   num_elites=8,
                   sigma_start=0.5,
                   sigma_min=0.1,
                   **common)
    if name == "ps":
        return PredictiveSampling(task,
                                  num_samples=num_samples,
                                  noise_level=0.5,
                                  **common)
    if name == "cbo":
        return CBO(task,
                   num_samples=num_samples,
                   initial_noise_level=0.5,
                   temperature=0.5,
                   consensus_weight=1.0,
                   noise_weight=1.0,
                   step_size=0.1,
                   **common)
    raise ValueError(f"unknown sub-optimizer '{name}'")


def build_controller(args):
    """Build the xArm6 PushT task + controller: ADMM, or a flat sampler when
    --algorithm mppi (the real-side twin of sim build_flat_3d / run_3d_plain).
    """
    t = time.perf_counter()
    print(
        f"[setup] loading task/scene '{args.scene}' (MJCF compile + MJX build)..."
    )

    costs = dict(_CFG.get("costs") or {})
    for kv in args.cost:
        k, v = kv.split("=", 1)
        costs[k] = float(v)

    task = PushT(
        impl="warp"
        if args.warp else "jax",  # --warp: MuJoCo Warp rollout backend
        clutter=True,
        planning_dt=PLAN_DT,
        robot="xarm6",
        # `"contact"` is invalid for an arm (J^T f, not a single DOF pair),
        # so the real choice is which twist inversion to use. Read from the
        # config, not hardcoded, so the two can be A/B'd by editing one
        # line -- see `PushT._consensus_from_twist_exact`.
        consensus_source=str(
            args.consensus_source or _ADM.get("consensus_source", "twist")
        ),
        twist_stick_speed=float(_ADM.get("twist_stick_speed", 0.005)),
        # Both were hardcoded here while `build_admm_3d` read them from the
        # config, so a sim run and a real run of "the same" ADMM could differ
        # in the consensus space itself. Unused on the flat path.
        consensus=args.consensus,
        local_goal=args.local_goal,
        env=args.scene,
        # Same cost weights the sim reads; without this the real driver silently
        # falls back to DEFAULT_COSTS (w_ee 40 vs yaml 10, w_tilt 30 vs yaml 100),
        # so sim and real would optimize different objectives.
        costs=costs,
    )

    # The published command is capped at --vel-limit, so cap the planner's own
    # sample bounds at the same value. Otherwise it samples up to the model's
    # ctrlrange (+/-1.0) and predicts ~5x the object motion the arm can produce;
    # harmless while approaching, but at contact the object and robot blocks
    # argue over an unrealisable wrench and the primal residual runs away.
    task.u_min = jnp.full_like(task.u_min, -args.vel_limit)
    task.u_max = jnp.full_like(task.u_max, args.vel_limit)

    if args.algorithm == "mppi":
        # Flat baseline: the robot sampler optimises the task directly -- no
        # object subproblem, no consensus, no duals (Nikola's baseline; sim
        # equivalent is build_flat_3d + run_3d_plain). rho / gamma / n_admm /
        # object_opt are all unused on this path.
        #
        # Built through `oim.runtime.samplers.build_sub_optimizer` against the
        # yaml's `sampler:` block -- the same call, with the same arguments,
        # that `oim.worlds.sim3d.build.build_flat_3d` makes. Before this, the
        # driver's own builder below silently substituted a different
        # optimizer for the same `--algorithm mppi`: scalar noise_level 0.5
        # instead of the per-joint [0.45, 0.3, 0.3, 0.2, 0.2], 4 spline knots
        # instead of `sampler.robot_num_knots`, and none of the flat-only
        # mechanisms (stuck_kick_*) the sim's tuning rounds validated.
        # "Same planner in sim and real" held for ADMM but not for the
        # flat baseline.
        robot_optimizer = build_cfg_optimizer(
            args.robot_opt, task,
            plan_horizon=args.horizon * PLAN_DT,
            num_knots=_SMP["robot_num_knots"],
            spline=_SMP["robot_spline"],
            seed=args.seed,
            num_samples=args.num_samples,
            sampler_cfg=_SMP,
            iterations=_SMP.get("iterations", 1),
        )
        # Physics steps per planning step in the sampler's own rollout, read
        # by `oim.alg_base.SamplingBasedController.eval_rollouts`. Set on the
        # TASK because that is the only object both the sampler and this
        # driver hold; 1 (absent) is the old single coarse step.
        #
        # Set on the flat path only. ADMM's robot block never goes through
        # `eval_rollouts` -- `RobotSubproblem` has its own `MJXRollout`, which
        # is given the same number below -- so setting it globally would
        # substep that path twice.
        task.robot_substeps = int(_W3.get("robot_substeps", 1))
        print(f"[setup] task ready in {time.perf_counter() - t:.1f}s; "
              f"flat {args.robot_opt}, no ADMM (knots="
              f"{_SMP['robot_num_knots']}, noise={_SMP['mppi']['noise_level']}, "
              f"stuck_kick={_SMP['mppi'].get('stuck_kick_steps')}, "
              f"substeps={task.robot_substeps})")
        return task, robot_optimizer

    # ADMM's robot block, through the SAME builder and the SAME config block
    # `oim/worlds/sim3d/build.py:206` uses. The comment that used to sit here
    # claimed the driver's own builder was "already matched to the sim". It was
    # not: that builder hardcodes `noise_level=0.5` (a scalar, against the
    # per-joint [0.45, 0.15, 0.15, 0.2, 0.2] the config carries),
    # `temperature=0.5` (against 10) and `num_knots=4` (against
    # `robot_num_knots: 8`). The flat baseline was rerouted here for exactly
    # this reason; ADMM was left behind, so every ADMM run on hardware ignored
    # the whole `sampler.mppi:` block.
    #
    # The scalar 0.5 is the damaging one. `task.u_min/u_max` are clamped to
    # +-vel_limit just above, so at --vel-limit 0.25 the sampling std was TWICE
    # the entire admissible range: nearly every sample landed on a corner of
    # the box, and with `temperature=0.5` the softmax then picked one of them
    # outright. That is bang-bang random search, and it is what a saturated
    # `|u|max = 0.250` in a different direction every step looks like.
    robot_optimizer = build_cfg_optimizer(
        args.robot_opt, task,
        plan_horizon=args.horizon * PLAN_DT,
        num_knots=_SMP["robot_num_knots"],
        spline=_SMP["robot_spline"],
        seed=args.seed,
        num_samples=args.num_samples,
        sampler_cfg=_SMP,
        iterations=_SMP.get("iterations", 1),
    )

    print(f"[setup] task ready in {time.perf_counter() - t:.1f}s; building ADMM...")
    # Same construction `build_admm_3d` uses: a pose consensus needs a
    # per-dimension dual bound, which the hardcoded scalar version here got
    # wrong by construction.
    consensus = consensus_space(task, args.consensus)
    obj_samples = (_CFG["sampler"].get("object") or {}).get(
        "num_samples", args.num_samples)
    # Same rerouting for the object block: `sampler.object:` is where its own
    # noise/temperature live (0.25 / 0.5 here), and the driver's builder was
    # substituting 0.5 / 0.5 for them.
    object_optimizer = build_cfg_optimizer(
        args.object_opt, make_object_shim(task, dt=PLAN_DT),
        plan_horizon=args.horizon * PLAN_DT,
        num_knots=args.horizon,
        spline=_SMP["object_spline"],
        seed=args.seed,
        num_samples=obj_samples,
        sampler_cfg=_SMP,
        overrides=(_SMP.get("object") or {}).get(args.object_opt),
        noise_scale=object_noise_scale(task, args.consensus),
    )
    # A vector rho penalises the wrench's torque component separately from its
    # two forces. The sim has defaulted this to 10.0 since the ablation that
    # found it the one formulation change moving position and orientation error
    # together; this driver passed a bare scalar, so the torque channel was
    # penalised 10x more weakly on hardware than in simulation.
    rho_init = (
        args.rho if args.rho_torque is None
        else np.array([args.rho, args.rho, args.rho_torque])
    )
    ctrl = ADMM(
        task, robot_optimizer, object_optimizer, consensus,
        n_admm=args.n_admm,
        eps_r=float(_ADM["eps_r"]), eps_s=float(_ADM["eps_s"]),
        proximal_weight=args.gamma, rho_init=rho_init,
        rho_adapt=bool(_ADM["rho_adapt"]),
        rho_bound_factor=float(_ADM["rho_bound_factor"]),
        # The robot block integrates contact at `planning_dt /
        # robot_substeps`, the same wiring `oim/worlds/sim3d/build.py` gives
        # the sim path. This driver passed no `rollout` at all, so it has been
        # running at 1 while the sim ran at its config's value.
        rollout=MJXRollout(substeps=int(_W3.get("robot_substeps", 1))),
        # `noise_min`/`noise_kappa`/`noise_max` and `consensus_alpha` used to be
        # passed here. Dropped in the merge, not by choice: main's [ADMM]
        # cleanup removed all four from `ADMM.__init__`, so passing them is a
        # TypeError now. The `--consensus-alpha` flag went with them, which is
        # a real loss on this path -- the real driver ran 0.3 (hardware contact
        # noise) against sim's 1.0, deliberately. Re-add it as an ADMM kwarg if
        # that difference still matters.
        #
        # Which dynamics the object block plans against: the paper's
        # quasi-static limit surface, or MJX on a stripped copy of the scene.
        # `None` for "analytic" -- ObjectSubproblem owns that default.
        object_rollout=build_object_rollout(
            args.plant, task, "xarm6", _W3, substeps=args.object_substeps,
            # Both were missing, so under `--plant mujoco --warp` the object
            # block silently ran the JAX backend (3.6-4.6x slower, measured in
            # `build_object_rollout`'s own docstring) with its Warp contact
            # arenas sized for 1 sample while 256 were being batched. The sim
            # path in `oim/worlds/sim3d/build.py` passes both; this one did
            # not, so a "warp run" was half warp here and only here.
            impl="warp" if args.warp else "jax",
            num_samples=obj_samples,
        ),
        # OFF: its jax.debug.print forces a GPU->host sync every ADMM iteration
        # (~200 s/optimize on a 2080 Ti). The real-time killer.
        debug_print=False,
    )
    return task, ctrl


def build_mock_interface(task, control_rate, exact_twist=False, block_start=None):
    """A MuJoCo sim behind the hardware interface, for laptop testing.

    Each `send_velocity` applies the commanded velocity and advances the sim by
    one control tick (1/control_rate). `run_real` calls it `num_ticks` times per
    replanning period, so the sim advances exactly one period per plan.

    exact_twist=True reads the sim's true block qvel (like the sim driver
    run_3d_admm); False (default) finite-differences the pose, as real hardware
    must from FoundationPose. With consensus_source="twist" this choice matters:
    the pose-derived twist is the sim-to-real gap.
    """
    mj_model = deepcopy(task.mj_model)
    mj_model.opt.timestep = _W3["exec_timestep"]
    mj_model.opt.iterations = _W3["exec_iterations"]
    mj_model.opt.ls_iterations = _W3["exec_ls_iterations"]
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
    sim_steps_per_send = max(1, round((1.0 / control_rate) / _W3["exec_timestep"]))
    return MujocoMockInterface(mj_model, mj_data, sim_steps_per_send,
                               emulate_pose_only=not exact_twist)


def build_real_interface(task, velocity_topic, enable_commands, object_origin_offset=(0.0, 0.0)):
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


def _dump_setup(args, task):
    """Print every number this run actually resolved to.

    Not a convenience. Three separate bugs on hardware were invisible because
    the value in the yaml was not the value in the loop: argparse defaults read
    from the wrong config, an ADMM block that ignored `sampler.mppi:`, and a
    stick attached at the wrong place. Each cost a session. Whatever is on this
    screen is what ran, so a log is enough to reconstruct a run without also
    needing the yaml that produced it.
    """
    spec = SCENES[args.scene]
    smp, w3, cost = _SMP, _W3, task.costs
    mppi = smp.get("mppi", {})
    obj = smp.get("object", {}) or {}
    span = args.horizon * PLAN_DT

    def row(label, body):
        print(f"[setup] {label:<9s} {body}")

    row("run", f"scene={args.scene} algorithm={args.algorithm} "
               f"config={args.config}.yaml backend={'warp' if args.warp else 'jax'} "
               f"seed={args.seed} steps={args.steps} "
               f"{'DRY-RUN (no commands)' if args.dry_run else 'LIVE'}")
    row("exec", f"vel_limit={args.vel_limit} rad/s  control={args.control_rate:g} Hz  "
                f"topic={args.velocity_topic}  "
                f"object_origin_offset={tuple(args.object_origin_offset)}")
    row("sampler", f"num_samples={args.num_samples} horizon={args.horizon} "
                   f"({span:.2f}s @ dt={w3['planning_dt']}) "
                   f"knots={smp['robot_num_knots']}/{smp['robot_spline']} "
                   f"substeps={w3.get('robot_substeps', 1)} "
                   f"iterations={smp.get('iterations', 1)}")
    row("robot", f"noise={mppi.get('noise_level')} temperature={mppi.get('temperature')} "
                 f"stuck_kick={mppi.get('stuck_kick_steps')}x{mppi.get('stuck_kick_scale')}")
    if args.algorithm == "admm":
        om = (obj.get(args.object_opt) or {})
        row("object", f"num_samples={obj.get('num_samples', args.num_samples)} "
                      f"noise={om.get('noise_level')} temperature={om.get('temperature')} "
                      f"spline={smp['object_spline']}")
        row("admm", f"plant={args.plant} n_admm={args.n_admm} rho={args.rho} "
                    f"rho_torque={args.rho_torque} gamma={args.gamma} "
                    f"consensus={args.consensus} local_goal={args.local_goal} "
                    f"wrench_fraction={_CFG['admm'].get('wrench_fraction')} "
                    f"eps=({_CFG['admm']['eps_r']}, {_CFG['admm']['eps_s']})")
        _src = args.consensus_source or _ADM.get("consensus_source", "twist")
        row("consensus", f"source={_src} "
                         f"stick_speed="
                         f"{_ADM.get('twist_stick_speed', 0.005)} m/s")
    # Only the weights that have moved a real run. The rest are in the yaml.
    row("costs", f"q_pos={cost.get('q_pos')} q_theta={cost.get('q_theta')} "
                 f"ramp={cost.get('q_ramp_per_step')}->{cost.get('q_ramp_max')} "
                 f"w_approach={cost.get('w_approach')} r0={cost.get('r0')} "
                 f"w_align={cost.get('w_align')}@{cost.get('gamma0_deg')}deg "
                 f"w_tilt={cost.get('w_tilt')} fade={cost.get('shaping_fade_dist')}")
    row("tip", f"w_z_tip={cost.get('w_z_tip')} w_z_tip_exp={cost.get('w_z_tip_exp')} "
               f"tip_floor_z={cost.get('tip_floor_z')} "
               f"w_contact_z_exp={cost.get('w_contact_z_exp')} "
               f"slab={cost.get('contact_z_slab')} margin={cost.get('contact_z_margin')}")
    goal = np.asarray(task.goal)
    row("scene", f"start={tuple(round(v, 4) for v in spec.object_start)} "
                 f"goal=({goal[0]:.4f}, {goal[1]:.4f}, {math.degrees(goal[2]):.1f}deg) "
                 f"base_z={spec.xarm6_base_z} arm_home={spec.xarm6_arm_start_deg}")
    row("object", f"mass={spec.mass} mu={spec.mu} "
                  f"limit_surface_radius={spec.limit_surface_radius} "
                  f"wrench_limit={np.round(np.asarray(task.object_model.wrench_limit), 5)}")
    row("tol", f"goal_pos_tol={_RUN['goal_pos_tol']} "
               f"goal_theta_tol={_RUN['goal_theta_tol']} "
               f"(plan span {span:.2f}s -- keep the solve under {span / 3:.2f}s)")


def _tee_console(log_dir: str) -> "str | None":
    """Mirror everything this process writes to stdout/stderr into a
    timestamped file under `<repo>/<log_dir>/real/<date>/`, while still
    showing it on the terminal.

    Done at the FILE-DESCRIPTOR level with a `tee` child rather than by
    swapping `sys.stdout`: the ROS logger (`pose rejected`, `re-baselined`,
    `stuck -- kicked` neighbours) is written by rcutils in C straight to
    fd 1/2 and never passes through Python's streams, and those lines are
    exactly the ones a post-mortem needs. Returns the log path, or None
    when disabled.
    """
    if not log_dir:
        return None
    import atexit  # noqa: PLC0415
    import shutil  # noqa: PLC0415
    import subprocess  # noqa: PLC0415
    import sys  # noqa: PLC0415
    from datetime import datetime  # noqa: PLC0415

    if shutil.which("tee") is None:
        print("[log] 'tee' not found; console is not being saved")
        return None
    now = datetime.now()
    out_dir = os.path.join(os.path.dirname(ROOT), log_dir, "real",
                           now.strftime("%Y%m%d"))
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir,
                        f"pusht_real_{now.strftime('%Y%m%d_%H%M%S')}.log")

    # rcutils defaults: severity-split streams and full buffering once the
    # fd is a pipe, which would land the ROS lines late and out of order
    # relative to the step lines. Must be set before rclpy initialises.
    os.environ.setdefault("RCUTILS_LOGGING_USE_STDOUT", "1")
    os.environ.setdefault("RCUTILS_LOGGING_BUFFERED_STREAM", "0")

    sys.stdout.flush()
    sys.stderr.flush()
    saved_out, saved_err = os.dup(1), os.dup(2)
    # -i: survive the Ctrl-C that stops the run so the tail is flushed too.
    tee = subprocess.Popen(["tee", "-a", "-i", path], stdin=subprocess.PIPE)
    os.dup2(tee.stdin.fileno(), 1)
    os.dup2(tee.stdin.fileno(), 2)
    # Python's own streams now sit on a pipe: keep them line-buffered so the
    # file keeps pace with the terminal.
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(line_buffering=True)
        except (AttributeError, ValueError):
            pass

    def _close() -> None:
        try:
            sys.stdout.flush()
            sys.stderr.flush()
        except Exception:  # noqa: BLE001
            pass
        os.dup2(saved_out, 1)
        os.dup2(saved_err, 2)
        try:
            tee.stdin.close()
            tee.wait(timeout=5.0)
        except Exception:  # noqa: BLE001
            pass
        print(f"[log] console saved to {path}")

    atexit.register(_close)
    print(f"[log] console -> {path}")
    return path


def main():
    # Declared here, not at the reassignment below: `main` reads _CFG for the
    # ADMM argparse defaults before that point, and Python requires the
    # global declaration to precede every use of the name in the function.
    global _CFG, _W3, _SMP, _RUN

    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--mock", action="store_true",
                   help="drive a MuJoCo sim instead of the real robot")
    p.add_argument("--scene", default="box_clutter_real",
                   help="scene from oim.tasks.pusht.SCENES (e.g. open_table_real, "
                        "single_obstacle_real, box_clutter_real)")
    p.add_argument("--steps", type=int, default=200, help="max control steps")
    p.add_argument("--replan-rate", type=float, default=2,
                   help="replanning frequency (Hz); must be <= 1/optimize time")
    p.add_argument("--control-rate", type=float, default=50,
                   help="velocity command streaming rate (Hz)")
    p.add_argument("--warp", action="store_true",
                   help="use the MuJoCo Warp rollout backend (speed A/B)")
    p.add_argument("--velocity-topic",
                   default="velocity_controller/commands_nominal",
                   help="topic to publish to. Default feeds the CBF safety "
                        "filter (commands_nominal -> CBF -> commands); the arm "
                        "moves (filtered) when the CBF node is up")
    p.add_argument("--dry-run", action="store_true",
                   help="publish no command at all (no motion), like OI-MPPI's "
                        "enable_velocity_commands:=false; state/TF are still "
                        "read so you can watch the plan in RViz")
    p.add_argument("--block-start", type=float, nargs=3, default=None,
                   metavar=("X", "Y", "YAW"),
                   help="mock only: override the block start SE(2) [x y yaw], "
                        "e.g. the real block pose from FoundationPose, to "
                        "rehearse a specific run in the mock before enabling motors")
    p.add_argument("--object-origin-offset", type=float, nargs=2,
                default=(0.0, 0.030), metavar=("DX", "DY"),
                help="real only: (dx, dy) in the OBJECT's own frame from the "
                    "perception mesh origin to the MJCF block origin [m]. "
                    "FoundationPose publishes the mesh origin, which for "
                    "meshes/T_block/T_block.ply is the bounding-box centre, "
                    "while tee_real.xml's origin is the crossbar/stem "
                    "junction -- 0.030 m along the object's +y. Confirm with "
                    "oim/worlds/real3d/scripts/check_object_tf.py before "
                    "trusting it; 0 0 (the default) keeps the old behaviour")
    p.add_argument("--exact-twist", action="store_true",
                   help="mock only: feed the sim's true block qvel to the "
                        "planner (like run_3d_admm) instead of a pose finite "
                        "difference. Isolates the FoundationPose twist gap")
    p.add_argument("--consensus-source", default=None,
                   choices=["twist", "twist_exact", "contact"],
                   help="how the robot block estimates A^r; overrides "
                        "admm.consensus_source in the config. NOT the same "
                        "knob as --exact-twist above -- that picks how the "
                        "MOCK reports the block's velocity, this picks which "
                        "equation is inverted to turn a velocity into a "
                        "wrench. `twist` inverts xdot = D w (the paper's "
                        "relation), `twist_exact` inverts xdot = D w (1-1/s), "
                        "which is what the object plant has actually "
                        "integrated since the near-goal-stall fix")
    p.add_argument("--algorithm", default="admm", choices=["admm", "mppi"],
                   help="admm = object-informed ADMM (default); mppi = flat "
                        "MPPI baseline, the real twin of the sim's "
                        "build_flat_3d / run_3d_plain")
    p.add_argument("--cost", action="append", default=[], metavar="KEY=VAL",
                   help="override a cost weight, real only, repeatable: "
                        "--cost w_tip_z=30 --cost w_ee=60")
    p.add_argument("--num-samples", type=int, default=None,
                   help="rollouts per sub-optimizer. Default: the config's "
                        "sampler.num_samples, shared by every algorithm")
    p.add_argument("--horizon", type=int, default=None,
                   help="planning horizon H, in PLAN_DT steps. Default: the "
                        "config's sampler.horizon, shared by every algorithm")
    p.add_argument("--vel-limit", type=float, default=0.2,
                   help="joint velocity cap [rad/s], applied to BOTH the "
                        "planner's sample bounds and the published command")
    p.add_argument("--preflight", type=float, default=5.0,
                   help="seconds to watch the raw FoundationPose stream "
                        "(block still) before the first command; a FAILing "
                        "stream (upside-down/mirror fit, yaw hopping, "
                        "floated bbox) aborts the run before the arm moves. "
                        "0 disables. LIVE only; ignored with --mock")
    p.add_argument("--log-dir", default="logs",
                   help="mirror the whole console (setup banner, per-step "
                        "lines, ROS gate warnings) into "
                        "<repo>/<log-dir>/real/<date>/pusht_real_<stamp>.log "
                        "while still printing it; '' disables. LIVE only")
    p.add_argument("--n-admm", type=int, default=None)
    p.add_argument("--rho-torque", type=float,
                   default=None,
                   help="ADMM only: initial penalty on the wrench's torque "
                        "component alone, split from --rho (the force "
                        "penalty). Same default and same rule the sim uses. "
                        "A negative value selects the paper's single scalar")
    p.add_argument("--consensus", choices=["wrench", "pose"],
                   default=None,
                   help="ADMM only: what the two blocks agree on -- the "
                        "contact wrench (paper eq. 24) or the object's SE(2) "
                        "pose trajectory")
    p.add_argument("--local-goal", action="store_true",
                   default=None,
                   help="ADMM only: robot block tracks the object block's "
                        "horizon endpoint instead of the global goal")
    p.add_argument("--plant", choices=["analytic", "mujoco"],
                   default=None,
                   help="ADMM only: which dynamics the object block plans "
                        "against")
    p.add_argument("--object-substeps", type=int,
                   default=None,
                   help="ADMM only: MJX physics steps per planning step, "
                        "under --plant mujoco")
    p.add_argument("--rho", type=float, default=None)
    p.add_argument("--gamma", type=float, default=None)
    p.add_argument("--robot-opt", default="mppi", choices=["mppi", "cem", "ps", "cbo"])
    p.add_argument("--object-opt", default="mppi", choices=["mppi", "cem", "ps", "cbo"])
    p.add_argument("--seed", type=int, default=5)
    p.add_argument("--config", default="xarm6_real", metavar="NAME",
                   help="robot config under oim/configs/robots/NAME.yaml. "
                        "xarm6_real is the lab T-block: same sampler and "
                        "execution model, with the cost terms carrying a "
                        "length or force scale re-derived for the 89x99 mm / "
                        "0.1 kg object. NOTE: --n-admm/--rho/--gamma take "
                        "their defaults from xarm6.yaml at parse time, so "
                        "pass them explicitly on the ADMM path")
    p.add_argument("--plot", action="store_true",
                   help="write the trajectory/diagnostics/cost-breakdown "
                        "figure next to the run JSON. Same `plot_run_3d` the "
                        "sim draws from oim/experiment.py -- this driver does "
                        "not import that module, hence the flag rather than "
                        "it being automatic. Equivalent to running "
                        "oim/worlds/real3d/scripts/plot_run_from_json.py on "
                        "the saved run afterwards")
    args = p.parse_args()

    # Console -> file from here on, so a run's terminal output is never lost
    # to scrollback again (the JSON keeps the states; the gate warnings,
    # kicks and per-step cost lines only ever existed on the terminal).
    if not args.mock:
        _tee_console(args.log_dir)

    # Rebind the config globals before anything reads them. Safe here because
    # every yaml-derived value is resolved after this point: --num-samples and
    # --horizon default to None (resolved just below), and the cost dict,
    # execution model and tolerances are read inside build_controller /
    # build_mock_interface / the run_real call.
    if args.config != "xarm6":
        _CFG = _load_cfg(args.config)
        _W3, _SMP, _RUN = _CFG["world3d"], _CFG["sampler"], _CFG["run"]
        print(f"[setup] config: {args.config}.yaml")

    # ADMM's knobs are resolved HERE, not in `add_argument`. argparse
    # evaluates its defaults at PARSE time, before `--config` has been read,
    # so every one of them used to come from xarm6.yaml no matter which config
    # was asked for. `--config xarm6_real` therefore ran the SIM's ADMM
    # settings -- plant=mujoco instead of analytic, rho=2.0 instead of 1.0,
    # rho_torque=2.0 instead of 10.0 -- which is how a "real" ADMM run spent
    # 0.81 s per solve rolling 256 object samples through MJX. An explicit
    # flag still wins: it leaves the value non-None and nothing below fires.
    admm_cfg = _CFG["admm"]
    if args.n_admm is None:
        args.n_admm = int(admm_cfg["n_admm"])
    if args.rho is None:
        args.rho = float(admm_cfg["rho"])
    if args.gamma is None:
        args.gamma = float(admm_cfg["gamma"])
    if args.rho_torque is None:
        args.rho_torque = float(admm_cfg.get("rho_torque", 10.0))
    if args.consensus is None:
        args.consensus = admm_cfg.get("consensus", "wrench")
    if args.local_goal is None:
        args.local_goal = bool(admm_cfg.get("local_goal", False))
    if args.plant is None:
        args.plant = admm_cfg.get("plant", "analytic")
    if args.object_substeps is None:
        args.object_substeps = int(admm_cfg.get("object_substeps", 1))
    print(f"[setup] admm: plant={args.plant} n_admm={args.n_admm} "
          f"rho={args.rho} rho_torque={args.rho_torque} "
          f"consensus={args.consensus}")

    # One sampler budget for every algorithm -- the same rule
    # oim/experiment.py::_run_3d applies. A baseline is only worth
    # something if it faces the budget ADMM faces; pass --num-samples /
    # --horizon to give a particular run its own.
    if args.num_samples is None:
        args.num_samples = _SMP["num_samples"]
    if args.horizon is None:
        args.horizon = _SMP["horizon"]

    # A negative --rho-torque selects the paper's single scalar rho, which is
    # what `rho_torque=None` means to build_admm_3d. argparse has no
    # "None or a float" type, so the sign carries the sentinel.
    if args.rho_torque is not None and args.rho_torque < 0:
        args.rho_torque = None

    task, ctrl = build_controller(args)
    _dump_setup(args, task)
    print(f"[setup] cache dir: {os.environ['JAX_COMPILATION_CACHE_DIR']}")

    t = time.perf_counter()
    if args.mock:
        interface = build_mock_interface(task, args.control_rate,
                                         exact_twist=args.exact_twist,
                                         block_start=args.block_start)
        real_time = False
    else:
        # Normal path publishes to the CBF filter's input (commands_nominal),
        # and the CBF node drives the motors. --dry-run publishes nothing.
        interface = build_real_interface(
            task, args.velocity_topic, enable_commands=not args.dry_run,
            object_origin_offset=tuple(args.object_origin_offset),
        )
        real_time = True
    print(f"[setup] interface ready in {time.perf_counter() - t:.1f}s")

    try:
        log = run_real(
            task, ctrl, ctrl.init_params(seed=args.seed), interface,
            replan_rate=args.replan_rate,
            control_rate=args.control_rate,
            max_steps=args.steps,
            real_time=real_time,
            vel_limit=args.vel_limit,
            preflight=args.preflight,
            admm=(args.algorithm == "admm"),
            # From the config's `run:` block rather than run_real's own
            # defaults, so sim and real grade against one source of truth.
            goal_pos_tol=float(_RUN["goal_pos_tol"]),
            goal_theta_tol=float(_RUN["goal_theta_tol"]),
        )
    finally:
        interface.close()

    # Same file, naming and schema as a sim run, so the two compare directly
    # and `oim/run_eval.py` groups them side by side. The scene goes in the
    # name so clutter and box_clutter_real runs are never told apart by timestamp
    # alone (e.g. pusht3d_xarm6_mock_box_clutter_real_admm_...).
    variant = f"xarm6_{'mock' if args.mock else 'real'}_{args.scene}"
    is_admm = args.algorithm == "admm"
    name = RunName("pusht3d", variant, args.algorithm)
    # Real runs are filed under results/real/{algorithm}/{scene}/{date}/
    # rather than flat in results/runs, which is where the sim path still
    # writes. Same filenames, so nothing downstream has to change: the name
    # already carries algorithm, scene and timestamp, and the directories
    # only make a session findable without grepping 400 filenames. The date
    # comes off `name`, so this run's JSON, plot and video always land in
    # one folder even across midnight. `oim/run_eval.py` globs recursively,
    # so `--runs-dir oim/results/real` scores every real run and
    # `--runs-dir oim/results/real/mppi/single_obstacle_real` scores one
    # scene's.
    results_dir = os.path.join(
        ROOT, "results", "real", args.algorithm, args.scene, name.date
    )
    path = save_run(
        results_dir,
        name,
        run=dict(
            world="3d",
            task=args.scene,
            robot="xarm6",
            algorithm=args.algorithm,
            robot_opt=args.robot_opt if is_admm else args.algorithm,
            object_opt=args.object_opt if is_admm else None,
            seed=args.seed,
            backend="warp" if args.warp else "jax",
            # The one field a sim run has no equivalent of: whether this was
            # the real arm or the MuJoCo stand-in behind the same interface.
            mock=args.mock,
        ),
        hyperparameters=dict(
            steps=args.steps,
            samples=args.num_samples,
            horizon=args.horizon,
            # Execution/observation conditions a sim run has no equivalent of,
            # but which decide what a number means here: the joint-velocity cap
            # the arm actually has, whether the planner saw the true block
            # twist, and which config it ran under.
            vel_limit=args.vel_limit,
            exact_twist=bool(args.exact_twist),
            object_origin_offset=list(args.object_origin_offset),
            config=args.config,
            # `oim.utils.metrics.trial_metrics` reads these two out of
            # `hyperparameters` and KeyErrors without them -- which is why
            # `python -m oim.run_eval` could not score a single run this entry
            # point wrote.
            goal_pos_tol=float(_RUN["goal_pos_tol"]),
            goal_theta_tol=float(_RUN["goal_theta_tol"]),
            n_admm=args.n_admm,
            rho=args.rho,
            rho_torque=args.rho_torque,
            consensus=args.consensus,
            local_goal=bool(args.local_goal),
            plant=args.plant,
            gamma=args.gamma,
            control_dt=1.0 / args.control_rate,
            replan_rate=args.replan_rate,
            costs=task.costs,
        ),
        task=task,
        log=log,
        # Same fields `oim/experiment.py::_mjx_static` writes, so
        # `replay_states.py` and the contact analysis read a real run and a
        # sim run through the same code path. `control_dt` here is the gap
        # between logged frames -- the log is appended once per replan, not
        # once per command -- which is what the replay plays back at.
        extra_static=dict(
            robot="xarm6",
            mock=args.mock,
            sim_timestep=float(task.mj_model.opt.timestep),
            control_dt=1.0 / args.replan_rate,
            qpos_size=int(task.mj_model.nq),
            qvel_size=int(task.mj_model.nv),
            block_qpos_adr=task.block_qpos_adr,
            block_dof_adr=task.block_dofs,
        ),
    )
    print(f"saved run to {path}")

    if args.plot:
        # Imported here, not at module scope: matplotlib is a plotting-only
        # dependency and the closed loop must not pay for it on a run that
        # does not ask for a figure. `plot_run_3d` needs `pos_err`/
        # `theta_err`/`reached`, which `run_real` already put in `log`, so
        # unlike `plot_run_from_json.py` there is nothing to recompute.
        from oim.utils.plotting import plot_run_3d  # noqa: PLC0415

        figure = os.path.splitext(path)[0] + ".png"
        plot_run_3d(task, log, figure)
        print(f"saved figure to {figure}")


if __name__ == "__main__":
    main()
