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
import sys
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
from oim.utils.results import RunName, save_run
from oim.utils.scenes import SCENES
from oim.worlds.real3d.interface import MujocoMockInterface
from oim.worlds.sim3d.build import build_admm_3d, build_flat_3d
from oim.worlds.real3d.run_real import run_real

# Same folder every sim world's --record writes an mp4 to
# (RECORDINGS_DIR in oim/experiment.py) -- "exactly like a sim run's
# --record" means the same place, not a real-only one.
RECORDINGS_DIR = os.path.join(ROOT, "recordings")


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




def build_controller(args):
    """Build the xArm6 PushT task + controller: ADMM, or a flat sampler when
    --algorithm mppi (the real-side twin of sim build_flat_3d / run_3d_plain).
    """
    t = time.perf_counter()
    print(
        f"[setup] loading task/scene '{args.scene}' (MJCF compile + MJX build)..."
    )

    # Same costs: block for every algorithm -- 2026-09-07, per Shahid: a
    # separate costs_admm: overlay meant MPPI and ADMM could silently be
    # optimizing different tasks (different weights on the same named
    # terms), which makes any comparison between them a comparison of
    # tasks, not of planners. `costs_admm:` no longer exists in the
    # config at all -- see Tasks.md if the old per-algorithm values are
    # ever needed for reference.
    costs = dict(_CFG.get("costs") or {})
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
    # is exactly one place each knob is resolved. `_CFG`/`_ADM` are already
    # rebound to `--config` by the time this runs.
    cfg = dict(_CFG)
    cfg["costs"] = costs
    adm = dict(_ADM)

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
            robot_substeps=int(_W3.get("robot_substeps", 1)),
            # `--goal` / `--goal-yaw-deg`; None keeps the scene's own goal.
            goal=_resolve_goal(args),
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
        iterations=int(_SMP.get("iterations", 1)),
        robot_substeps=int(_W3.get("robot_substeps", 1)),
        goal=_resolve_goal(args),
    )
    task.u_min = jnp.full_like(task.u_min, -args.vel_limit)
    task.u_max = jnp.full_like(task.u_max, args.vel_limit)
    print(f"[setup] task ready in {time.perf_counter() - t:.1f}s; "
          f"flat {args.robot_opt} via build_flat_3d, no ADMM "
          f"(knots={_SMP['robot_num_knots']}, "
          f"noise={_SMP[args.robot_opt]['noise_level']}, "
          f"substeps={task.robot_substeps})")
    return task, ctrl



def _resolve_goal(args):
    """The goal pose this run scores against, or None for the scene's."""
    if args.goal is not None and args.goal_yaw_deg is not None:
        raise SystemExit("--goal and --goal-yaw-deg are mutually exclusive")
    if args.goal is not None:
        return [args.goal[0], args.goal[1], math.radians(args.goal[2])]
    if args.goal_yaw_deg is not None:
        g = SCENES[args.scene].goal
        return [float(g[0]), float(g[1]), math.radians(args.goal_yaw_deg)]
    return None


def build_mock_interface(task, control_rate, exact_twist=False, block_start=None):
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
    if args.algorithm == "admm":
        mppi = dict(mppi)
        mppi.update((smp.get("admm_robot") or {}).get(args.robot_opt) or {})
    row("robot", f"noise={mppi.get('noise_level')} temperature={mppi.get('temperature')} "
                 f"stuck_kick={mppi.get('stuck_kick_steps')}x{mppi.get('stuck_kick_scale')}")
    if args.algorithm == "admm":
        om = (obj.get(args.object_opt) or {})
        row("object", f"num_samples={obj.get('num_samples', args.num_samples)} "
                      f"noise={om.get('noise_level')} temperature={om.get('temperature')} "
                      f"spline={smp['object_spline']}")
        row("admm", f"plant={args.plant} n_admm={args.n_admm} rho={args.rho} "
                    f"rho_torque={args.rho_torque} "
                    f"gamma={args.gamma} "
                    f"consensus={args.consensus} "
                    f"wrench_fraction={float(task.object_model.action_scale[0] / task.object_model.wrench_limit[0]):.2f} (effective) "
                    f"eps=({_CFG['admm']['eps_r']}, {_CFG['admm']['eps_s']})")
        row("consensus", "A^r = contact forces of the planning rollout "
                         "(no estimator; hardware forces never read)")
    # Only the weights that have moved a real run. The rest are in the yaml.
    row("costs", f"q_pos={cost.get('q_pos')} q_theta={cost.get('q_theta')} "
                 f"ramp={cost.get('q_ramp_per_step')}->{cost.get('q_ramp_max')} "
                 f"w_approach={cost.get('w_approach')} r0={cost.get('r0')} "
                 f"w_align={cost.get('w_align')}@{cost.get('gamma0_deg')}deg "
                 f"w_tilt={cost.get('w_tilt')} fade={cost.get('shaping_fade_dist')}")
    # Resolved on the task, not `cost.get`: the banner must show what
    # actually runs, not what the yaml wrote down. There is one approach
    # form now -- the SDF wall distance -- so `r0` is a stand-off from the
    # wall, not a deadband radius from the block origin.
    row("approach", f"sdf-wall xy only, r0={getattr(task, 'r0', '?')}")
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


def _tee_console(log_dir: str, stamp: str) -> "str | None":
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

    if shutil.which("tee") is None:
        print("[log] 'tee' not found; console is not being saved")
        return None
    # `stamp` is the run-wide timestamp shared with `RunName`, so the log
    # and the results JSON carry the same one and pair up by filename.
    out_dir = os.path.join(os.path.dirname(ROOT), log_dir, "real", stamp[:8])
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f"pusht_real_{stamp}.log")

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
    global _CFG, _W3, _SMP, _RUN, _ADM

    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--mock", action="store_true",
                   help="drive a MuJoCo sim instead of the real robot")
    p.add_argument("--scene", default="box_clutter_real",
                   help="scene from oim.tasks.pusht.SCENES (e.g. open_table_real, "
                        "single_obstacle_real, box_clutter_real)")
    p.add_argument("--steps", type=int, default=None,
                   help="max control steps. Default: the config's run.steps")
    p.add_argument("--replan-rate", type=float, default=2,
                   help="replanning frequency (Hz); must be <= 1/optimize time")
    p.add_argument("--control-rate", type=float, default=50,
                   help="velocity command streaming rate (Hz)")
    p.add_argument("--warp", action="store_true", default=None,
                   help="use the MuJoCo Warp rollout backend (speed A/B). "
                        "Default: the config's run.warp")
    p.add_argument("--no-warp", dest="warp", action="store_false")
    p.add_argument("--velocity-topic",
                   default="velocity_controller/commands_nominal",
                   help="topic to publish to. Default feeds the CBF safety "
                        "filter (commands_nominal -> CBF -> commands); the arm "
                        "moves (filtered) when the CBF node is up")
    p.add_argument("--dry-run", action="store_true",
                   help="publish no command at all (no motion), like OI-MPPI's "
                        "enable_velocity_commands:=false; state/TF are still "
                        "read so you can watch the plan in RViz")
    p.add_argument("--goal", type=float, nargs=3, default=None,
                   metavar=("X", "Y", "YAW_DEG"),
                   help="override the scene's goal pose [x y yaw_deg] for this "
                        "run; default: the scene's goal")
    p.add_argument("--goal-yaw-deg", type=float, default=None,
                   help="override only the goal yaw [deg], keeping the "
                        "scene's goal position, e.g. 90 or -90")
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
    p.add_argument("--vel-limit", type=float, default=None,
                   help="joint velocity cap [rad/s], applied to BOTH the "
                        "planner's sample bounds and the published command. "
                        "Default: admm.vel_limit (ADMM) / run.vel_limit (flat)")
    p.add_argument("--latency-comp", type=float, default=None,
                   help="hardware loop: initial solve-latency guess [s] to "
                        "predict the arm state forward by before each solve "
                        "and anchor the plan's clock there (tracked per "
                        "solve afterwards). 0 disables (today's behaviour)")
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
    p.add_argument("--consensus",
                   choices=["wrench", "object_pose", "contact_point"],
                   default=None,
                   help="ADMM only: what the two blocks agree on -- the "
                        "contact wrench (paper eq. 24) or the object's SE(2) "
                        "pose trajectory")
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
    opt_choices = ["mppi", "cem", "ps", "cbo"]
    p.add_argument("--robot-opt", default="mppi", choices=opt_choices)
    p.add_argument("--object-opt", default="mppi", choices=opt_choices)
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
    # Mirrors the sim worlds' --record/--show-samples/--show-optimal
    # (oim/experiment.py): the same OffscreenRecorder, the same overlay,
    # wired into run_real's loop instead of sim3d's. --live has no sim
    # CLI equivalent to mirror -- the sim scripts open their live viewer
    # by the ABSENCE of --headless rather than a flag of its own, which
    # doesn't fit here since real has no headless/interactive split to
    # begin with.
    p.add_argument("--record", action="store_true",
                   help="Film the run (robot, object, samples, chosen "
                        "trajectory) and write an mp4 to oim/recordings/, "
                        "exactly like a sim run's --record.")
    p.add_argument("--live", action="store_true",
                   help="Open a MuJoCo window and show the run as it "
                        "happens. Independent of --record -- either, "
                        "both, or neither.")
    p.add_argument("--show-samples", action="store_true", default=True,
                   help="Overlay the sampled candidate rollouts, in "
                        "whichever of --record/--live are active.")
    p.add_argument("--no-show-samples", dest="show_samples",
                   action="store_false",
                   help="Do not overlay the candidates (smaller mp4).")
    p.add_argument("--show-optimal", action="store_true", default=True,
                   help="Overlay each block's chosen trajectory.")
    p.add_argument("--no-show-optimal", dest="show_optimal",
                   action="store_false",
                   help="Do not overlay the chosen trajectory.")
    p.add_argument("--show-object-plan", action="store_true",
                   help="ADMM only: draw the object block's plan-endpoint "
                        "ghost marker in --record/--live (hidden by "
                        "default; it mostly duplicates the goal marker).")
    p.add_argument("--camera", default=None,
                   help="Model camera name to render/view from, e.g. "
                        "'front' for the scene's fixed lab-mount camera "
                        "(oim.runtime.mjcf.named_camera). Unset uses the "
                        "default free camera, auto-framed to the scene "
                        "(mujoco.mjv_defaultFreeCamera) -- same on both "
                        "--record and --live.")
    p.add_argument("--view-azimuth", type=float, default=180.0,
                   help="Where --live's camera stands, in degrees around "
                        "the table. 180 (default) looks back along -x from "
                        "over the table's +x end, which puts its long axis "
                        "across the screen. 90 views from -y, 270 from +y.")
    p.add_argument("--view-elevation", type=float, default=-27.0,
                   help="How far --live's camera is tipped down, in "
                        "degrees. Negative looks down; -27 is the default.")
    p.add_argument("--view-distance", type=float, default=None,
                   help="How far back --live's camera stands, in metres. "
                        "Unset solves it from the table's width and the "
                        "window's own aspect ratio so the table just fills "
                        "the frame -- set this only to override that.")
    p.add_argument("--video-fps", type=float, default=None,
                   help="mp4 playback rate, and the assumed real seconds "
                        "between recorded steps (see run_real's video_fps "
                        "for why real needs this spelled out where sim "
                        "does not). Unset uses --replan-rate, which "
                        "makes a --mock recording play back true to real "
                        "time; on hardware the true interval is the "
                        "solve time itself and this is only ever an "
                        "approximation.")
    p.add_argument("--obstacle-calibration", default=None,
                   help="'live' to sample obs_1/2/3's current pose "
                        "directly off TF before the run (no --mock; "
                        "requires aruco_obstacle_node.py + "
                        "aruco_tf_broadcaster.py running on the perception "
                        "laptop on the same ROS 2 domain). Otherwise a path "
                        "to a JSON file from "
                        "Fork_FoundationPose/calibrate_obstacles.py (one "
                        "xarm_device -> obs_N_center TF lookup per "
                        "obstacle; works with --mock, so a calibrated "
                        "layout can be rehearsed). Unset keeps each "
                        "scene's plain MJCF/config obstacle poses as-is. On "
                        "--scene box_clutter_real this only repositions "
                        "the 3 obstacles that are always there; on "
                        "--scene live_real it instead determines which "
                        "obstacles exist AT ALL -- the scene is built "
                        "from scratch with one geom per obstacle this "
                        "run's calibration actually resolved, nothing "
                        "else (see oim.worlds.real3d.live_scene).")
    args = p.parse_args()

    # One timestamp for the whole run: the console log takes it here and
    # `RunName` below inherits it, so <stamp>.log and <stamp>.json match.
    from datetime import datetime  # noqa: PLC0415
    run_stamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Console -> file from here on, so a run's terminal output is never lost
    # to scrollback again (the JSON keeps the states; the gate warnings,
    # kicks and per-step cost lines only ever existed on the terminal).
    if not args.mock:
        _tee_console(args.log_dir, run_stamp)

    # The exact launch command, first thing in the tee -- CLI --cost
    # overrides are where the effective config actually lives, and two
    # config post-mortems (09-01 yaml-vs-banner, 09-02 slab_above) went
    # wrong because the log never recorded what was typed.
    print(f"[setup] cmd: {' '.join(sys.argv)}")

    # Rebind the config globals before anything reads them. Safe here because
    # every yaml-derived value is resolved after this point: --num-samples and
    # --horizon default to None (resolved just below), and the cost dict,
    # execution model and tolerances are read inside build_controller /
    # build_mock_interface / the run_real call.
    if args.config != "xarm6":
        _CFG = _load_cfg(args.config)
        _W3, _SMP, _RUN = _CFG["world3d"], _CFG["sampler"], _CFG["run"]
        # `_ADM` was left out of this rebind, so every `_ADM` read --
        # eps_r/eps_s and (once it was wired through)
        # wrench_fraction -- silently came from
        # xarm6.yaml no matter which --config was named. The 2026-08-31
        # 23:29-23:43 real runs therefore ran wrench_fraction 1.5 (xarm6)
        # while xarm6_real.yaml said 1.0; the banner, which reads _CFG,
        # printed the intended value the whole time.
        _ADM = _CFG["admm"]
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
        args.consensus = admm_cfg.get("consensus", "object_pose")
    if args.plant is None:
        args.plant = admm_cfg.get("plant", "analytic")
    if args.object_substeps is None:
        # `world3d:`, the same block `oim/experiment.py` reads it from for
        # sim. It lived under `admm:` here with its own value (5 against
        # sim's 2), so one parameter had two homes and two values.
        args.object_substeps = int(_W3.get("object_substeps", 1))
    print(f"[setup] admm: plant={args.plant} n_admm={args.n_admm} "
          f"rho={args.rho} rho_torque={args.rho_torque} "
          f"consensus={args.consensus}")

    # Sampler budget: the shared `sampler.*` values unless the yaml gives
    # ADMM its own under `sampler.admm_robot` (horizon / num_samples), the
    # same place its robot-block sampler already lives. Added 2026-09-07
    # after a flat-MPPI retune moved the shared horizon 28 -> 42 and
    # silently re-budgeted every ADMM run with it (the ADMM successes on
    # record were all at 28). --num-samples / --horizon still override.
    _own = (_SMP.get("admm_robot") or {}) if args.algorithm == "admm" else {}
    if args.num_samples is None:
        args.num_samples = int(_own.get("num_samples", _SMP["num_samples"]))
    if args.horizon is None:
        args.horizon = int(_own.get("horizon", _SMP["horizon"]))
    # Run-level defaults from the yaml, so the canonical launch line is the
    # config and a bare `--algorithm admm` / `--algorithm mppi` reproduces it.
    if args.steps is None:
        args.steps = int(_RUN.get("steps", 200))
    if args.warp is None:
        args.warp = bool(_RUN.get("warp", False))
    if args.latency_comp is None:
        args.latency_comp = float(_RUN.get("latency_comp", 0.0))
    if args.vel_limit is None:
        args.vel_limit = float(
            _ADM.get("vel_limit", _RUN.get("vel_limit", 0.2))
            if args.algorithm == "admm" else _RUN.get("vel_limit", 0.2)
        )

    # A negative --rho-torque selects the paper's single scalar rho, which is
    # what `rho_torque=None` means to build_admm_3d. argparse has no
    # "None or a float" type, so the sign carries the sentinel.
    if args.rho_torque is not None and args.rho_torque < 0:
        args.rho_torque = None

    # live_real has no fixed obstacle layout -- unlike box_clutter_real
    # (always 3 obstacles, calibration only repositions them), its model
    # is regenerated from scratch here, BEFORE build_controller compiles
    # it, with a geom for exactly whichever obstacles this run's
    # calibration resolved. A model's body count is fixed at compile
    # time, so this has to happen before PushT, not after (contrast with
    # box_clutter_real's calibration, applied inside run_real once the
    # already-compiled mocap bodies just need repositioning).
    live_calibration = None
    if args.scene == "live_real" and args.obstacle_calibration is not None:
        from oim.worlds.real3d.live_scene import (  # noqa: PLC0415
            load_live_obstacle_calibration,
            write_live_real_xml,
        )
        live_calibration = load_live_obstacle_calibration(
            args.obstacle_calibration
        )
        write_live_real_xml(live_calibration)
        # Already fully consumed above -- run_real must not also apply
        # box_clutter_real's mocap-repositioning logic against a scene
        # with no mocap obstacles at all.
        args.obstacle_calibration = None

    task, ctrl = build_controller(args)
    if live_calibration:
        from oim.worlds.real3d.live_scene import (  # noqa: PLC0415
            apply_live_obstacle_calibration_to_planner,
        )
        apply_live_obstacle_calibration_to_planner(task, live_calibration)
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

    # Named before the run rather than after (moved up from where this used
    # to sit, below `finally`): --record needs the same stem/timestamp the
    # JSON and --plot figure get, so all three of one run's artifacts share
    # one name instead of the mp4 stamping its own later.
    variant = f"xarm6_{'mock' if args.mock else 'real'}_{args.scene}"
    is_admm = args.algorithm == "admm"
    name = RunName("pusht3d", variant, args.algorithm)
    # Stamp the artifacts with the run's START time -- the same stamp the
    # console log took -- not the save time, so log, JSON and the --record
    # mp4 all pair up. Must sit here, above the run_real call that passes
    # `record_name=name()`, not below it where this used to live.
    name.timestamp = run_stamp

    # No default-to-"front": that's the scene's fixed lab-mount camera, a
    # documentation angle rather than a good live-viewing one. Unset stays
    # None, which both OffscreenRecorder and (after run_real's own fix)
    # the live viewer resolve the same way -- mujoco.mjv_defaultFreeCamera,
    # auto-framed to the scene.
    camera = args.camera
    # See run_real's video_fps docstring: unset falls back to --replan-rate,
    # which makes a --mock recording play back true to real time; hardware
    # has no equivalent notion of a fixed rate, so this is only ever an
    # approximation there -- pass --video-fps explicitly on that path if
    # the default's playback speed looks wrong.
    video_fps = args.video_fps if args.video_fps is not None else args.replan_rate

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
            record_dir=RECORDINGS_DIR if args.record else None,
            record_name=name(),
            video_fps=video_fps,
            camera=camera,
            live=args.live,
            show_samples=args.show_samples,
            show_optimal=args.show_optimal,
            show_object_plan=args.show_object_plan,
            view_azimuth=args.view_azimuth,
            view_elevation=args.view_elevation,
            view_distance=args.view_distance,
            obstacle_calibration=args.obstacle_calibration,
            latency_comp=args.latency_comp,
        )
    finally:
        interface.close()

    # Same file, naming and schema as a sim run, so the two compare directly
    # and `oim/run_eval.py` groups them side by side. The scene goes in the
    # name so clutter and box_clutter_real runs are never told apart by timestamp
    # alone (e.g. pusht3d_xarm6_mock_box_clutter_real_admm_...).
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
            plant=args.plant,
            gamma=args.gamma,
            control_dt=1.0 / args.control_rate,
            replan_rate=args.replan_rate,
            costs=task.costs,
            # The formulation switches, so a run file says which forms it
            # ran (the sim/real split lives in these and in `costs`).
            # `plant_form` and `consensus_source` used to be recorded
            # here. Both selectors are gone: the object block's dynamics
            # are `plant` (recorded above) and A^r is always the planning
            # rollout's own contact forces.
            latency_comp=float(args.latency_comp),
            goal=None if task.goal is None else [float(g) for g in task.goal],
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
