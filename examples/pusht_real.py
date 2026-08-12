"""Entry point: run the push-T ADMM controller on the real xArm6 (or a mock).

The hardware sibling of `examples/clutter.py`'s `--robot xarm6 admm --headless`
path. The task and the ADMM controller are built exactly as there -- same
scene, same weights, same hyperparameters -- so any difference in behaviour
is the sim-to-real gap, not a different planner. Only the execution driver
changes: `oim.real3d.run_real` instead of `oim.sim3d.run.run_3d_admm`.

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

import mujoco
import jax.numpy as jnp
import yaml

from oim import ROOT
from oim.algs import (
    ADMM,
    CBO,
    CEM,
    MPPI,
    PredictiveSampling,
    WrenchConsensus,
    make_object_shim,
)
from oim.real3d.interface import MujocoMockInterface
from oim.real3d.run_real import run_real
from oim.tasks.pusht import PushT
from oim.utils.results import RunName, save_run

# Read the SAME file the sim reads (oim.experiment.load_config("xarm6")), so
# sim and real share one source of truth for dt / sampler budget / ADMM knobs.
# Loaded directly rather than importing oim.experiment, which would drag the
# viewer/plot stack into a headless run.
with open(os.path.join(ROOT, "configs", "xarm6.yaml")) as _f:
    _CFG = yaml.safe_load(_f)

PLAN_DT = 0.05      # planner timestep (matches examples/clutter.py)
EXEC_TIMESTEP = 0.002  # fine execution timestep for the mock sim
# (arm start config is per-scene: SCENES[...]["arm_start_deg"] in oim/tasks/pusht.py)


def build_sub_optimizer(name, task, *, plan_horizon, num_knots, spline, seed,
                        num_samples):
    """Like examples/clutter.py::build_sub_optimizer, but with a tunable sample
    count -- xarm6 needs a smaller budget than the point mass (64 samples can
    exhaust an 11 GB GPU for the arm; see oim/configs/xarm6.yaml)."""
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
    --algorithm mppi (the real-side twin of sim build_flat_3d / run_3d_plain)."""
    t = time.perf_counter()
    print(
        f"[setup] loading task/scene '{args.scene}' (MJCF compile + MJX build)..."
    )
    task = PushT(
        impl="warp"
        if args.warp else "jax",  # --warp: MuJoCo Warp rollout backend
        clutter=True,
        planning_dt=PLAN_DT,
        robot="xarm6",
        consensus_source="twist",  # only valid estimator for an articulated arm
        env=args.scene,
    )

    # The published command is capped at --vel-limit, so cap the planner's own
    # sample bounds at the same value. Otherwise it samples up to the model's
    # ctrlrange (+/-1.0) and predicts ~5x the object motion the arm can produce;
    # harmless while approaching, but at contact the object and robot blocks
    # argue over an unrealisable wrench and the primal residual runs away.
    task.u_min = jnp.full_like(task.u_min, -args.vel_limit)
    task.u_max = jnp.full_like(task.u_max, args.vel_limit)

    # Robot-level sampler, identical for both algorithms (same num_knots / spline
    # the sim's robot ADMM block and its flat baseline both use).
    robot_optimizer = build_sub_optimizer(
        args.robot_opt, task, plan_horizon=args.horizon * PLAN_DT,
        num_knots=4, spline="linear", seed=args.seed,
        num_samples=args.num_samples,
    )
    if args.algorithm == "mppi":
        # Flat baseline: the robot sampler optimises the task directly -- no
        # object subproblem, no consensus, no duals (Nikola's baseline; sim
        # equivalent is build_flat_3d + run_3d_plain). rho / gamma / n_admm /
        # consensus_alpha / object_opt are all unused on this path.
        print(f"[setup] task ready in {time.perf_counter() - t:.1f}s; "
              f"flat {args.robot_opt}, no ADMM")
        return task, robot_optimizer

    print(f"[setup] task ready in {time.perf_counter() - t:.1f}s; building ADMM...")
    consensus = WrenchConsensus(
        max_dual=2.0 * float(task.consensus_scale()[0]),
        scale=task.consensus_scale(),
    )
    object_optimizer = build_sub_optimizer(
        args.object_opt, make_object_shim(task, dt=PLAN_DT),
        plan_horizon=args.horizon * PLAN_DT, num_knots=args.horizon, spline="zero",
        seed=args.seed, num_samples=args.num_samples,
    )
    ctrl = ADMM(
        task, robot_optimizer, object_optimizer, consensus,
        n_admm=args.n_admm, eps_r=0.5, eps_s=0.5,
        proximal_weight=args.gamma, rho_init=args.rho,
        # Noise annealing off (primal residual doesn't converge here, it just
        # pins at noise_max rather than annealing).
        noise_min=0.0, noise_kappa=0.0, noise_max=0.0,
        # EMA on A^o/A^r across ADMM rounds. Real default 0.3 (hardware contact
        # noise); NOT from the yaml (sim uses 1.0 -- a legitimate difference).
        consensus_alpha=args.consensus_alpha,
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
    mj_model.opt.timestep = EXEC_TIMESTEP
    mj_model.opt.iterations = 100
    mj_model.opt.ls_iterations = 50
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
    sim_steps_per_send = max(1, round((1.0 / control_rate) / EXEC_TIMESTEP))
    return MujocoMockInterface(mj_model, mj_data, sim_steps_per_send,
                               emulate_pose_only=not exact_twist)


def build_real_interface(task, velocity_topic, enable_commands):
    """The real ROS2 <-> xArm6 bridge. Import is lazy so --mock needs no ROS.

    Frames, joint naming and watchdog default from the OI-MPPI reference in
    Ros2Interface.__init__. `enable_commands=False` is the dry run (reads
    state/TF, publishes nothing). `task.world_frame` selects the planner's TF
    frame: "xarm_device" for base-at-origin scenes (reads FoundationPose's TF
    directly, no world->base transform), or "world" otherwise.
    """
    from oim.real3d.interface import Ros2Interface  # noqa: PLC0415

    return Ros2Interface(
        world_frame=task.world_frame,
        velocity_command_topic=velocity_topic,
        enable_commands=enable_commands,
    )


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--mock", action="store_true",
                   help="drive a MuJoCo sim instead of the real robot")
    p.add_argument("--scene", default="clutter2",
                   help="scene from oim.tasks.pusht.SCENES (e.g. clutter, clutter2)")
    p.add_argument("--steps", type=int, default=200, help="max control steps")
    p.add_argument("--replan-rate", type=float, default=2.5,
                   help="replanning frequency (Hz); must be <= 1/optimize time")
    p.add_argument("--control-rate", type=float, default=50.0,
                   help="velocity command streaming rate (Hz)")
    p.add_argument("--command-mode", default="hold", choices=["hold", "stream"],
                   help="hold a constant velocity per period, or stream the plan")
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
    p.add_argument("--exact-twist", action="store_true",
                   help="mock only: feed the sim's true block qvel to the "
                        "planner (like run_3d_admm) instead of a pose finite "
                        "difference. Isolates the FoundationPose twist gap")
    p.add_argument("--algorithm", default="admm", choices=["admm", "mppi"],
                   help="admm = object-informed ADMM (default); mppi = flat "
                        "MPPI baseline, the real twin of the sim's "
                        "build_flat_3d / run_3d_plain")
    p.add_argument("--num-samples", type=int, default=_CFG["sampler"]["num_samples"],
                   help="rollouts per sub-optimizer (default from xarm6.yaml)")
    p.add_argument("--horizon", type=int, default=_CFG["sampler"]["horizon"],
                   help="planning horizon H, in PLAN_DT steps (default from xarm6.yaml)")
    p.add_argument("--vel-limit", type=float, default=0.2,
                   help="joint velocity cap [rad/s], applied to BOTH the "
                        "planner's sample bounds and the published command")
    p.add_argument("--n-admm", type=int, default=_CFG["admm"]["n_admm"])
    p.add_argument("--rho", type=float, default=_CFG["admm"]["rho"])
    p.add_argument("--gamma", type=float, default=_CFG["admm"]["gamma"])
    p.add_argument("--consensus-alpha", type=float, default=0.3,
                   help="ADMM consensus EMA. Real default 0.3 (hardware contact "
                        "noise); sim's yaml uses 1.0 -- kept off the yaml on "
                        "purpose as a legitimate sim/real difference")
    p.add_argument("--robot-opt", default="mppi", choices=["mppi", "cem", "ps", "cbo"])
    p.add_argument("--object-opt", default="mppi", choices=["mppi", "cem", "ps", "cbo"])
    p.add_argument("--seed", type=int, default=5)
    args = p.parse_args()

    task, ctrl = build_controller(args)
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
            task, args.velocity_topic, enable_commands=not args.dry_run
        )
        real_time = True
    print(f"[setup] interface ready in {time.perf_counter() - t:.1f}s")

    try:
        log = run_real(
            task, ctrl, ctrl.init_params(seed=args.seed), interface,
            replan_rate=args.replan_rate,
            control_rate=args.control_rate,
            command_mode=args.command_mode,
            max_steps=args.steps,
            real_time=real_time,
            vel_limit=args.vel_limit,
            admm=(args.algorithm == "admm"),
        )
    finally:
        interface.close()

    # Same file, naming and schema as a sim run, so the two compare directly
    # and `oim/run_eval.py` groups them side by side. The scene goes in the
    # name so clutter and clutter2 runs are never told apart by timestamp
    # alone (e.g. pusht3d_xarm6_mock_clutter2_admm_...).
    results_dir = os.path.join(ROOT, "results", "runs")
    variant = f"xarm6_{'mock' if args.mock else 'real'}_{args.scene}"
    is_admm = args.algorithm == "admm"
    name = RunName("pusht3d", variant, args.algorithm)
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
            n_admm=args.n_admm,
            rho=args.rho,
            gamma=args.gamma,
            consensus_alpha=args.consensus_alpha,
            control_dt=1.0 / args.control_rate,
            replan_rate=args.replan_rate,
            command_mode=args.command_mode,
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


if __name__ == "__main__":
    main()
