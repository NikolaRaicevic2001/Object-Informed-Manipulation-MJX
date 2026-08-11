"""One experiment, end to end -- everything an `examples/` script is not.

Each script under `examples/` declares a single `Experiment` (which world,
which scene) and calls `main`. Everything else lives here: the command
line, config loading, construction, the closed loop, recording, the run
file and the plot. So a task is a declaration, every task gets the same
flags, and adding one never means copying a runner.

    # examples/shelf_gap.py
    EXPERIMENT = Experiment(world="3d", scene="shelf_gap")
    if __name__ == "__main__":
        main(EXPERIMENT)

The CLI a script offers is derived from its `Experiment`, so it advertises
only what applies: a 3D script has `--warp`/`--record` and the `ps`, `mppi`
and `admm` algorithms, a 2D script has `--animate`/`--no-jit` and only
`admm` (`PushT2D` implements `ConsensusTask` alone). `--robot` offers
exactly the embodiments the scene's MJCF exists for.

Sweeps are `oim/run_launch.py`'s job and metrics are `oim/run_eval.py`'s.
Neither happens here, so a new metric never costs a re-run.
"""

import argparse
import os
import sys
from contextlib import nullcontext
from dataclasses import dataclass
from typing import Any, Dict, Literal, Optional, Sequence, Tuple

# XLA's GPU command buffers leak across long closed loops and hit
# RESOURCE_EXHAUSTED; disabling them is XLA's fix. Not in oim/__init__.py:
# doing it globally breaks the Warp backend.
os.environ["XLA_FLAGS"] = (
    os.environ.get("XLA_FLAGS", "") + " --xla_gpu_enable_command_buffer="
)

# Warp allocates outside JAX's pool, and JAX preallocates ~75% of the device
# the first time it touches it. On a 16 GB card that leaves Warp ~4 GB --
# not enough to build MuJoCo Warp's CUDA graphs for the xArm6 scene, which
# fails in `wp_cuda_graph_create_exec` before a single planning step. The
# two only coexist if JAX grows on demand instead.
#
# Read from argv rather than from parsed arguments: XLA fixes its allocator
# when the GPU backend first initializes, which happens long before main().
# `setdefault` so an explicit setting in the environment still wins.
if "--warp" in sys.argv:
    os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import jax  # noqa: E402
import mujoco  # noqa: E402
import numpy as np  # noqa: E402
import yaml  # noqa: E402

from oim import ROOT  # noqa: E402
from oim.objects import wrap_angle  # noqa: E402
from oim.sim2d import (  # noqa: E402
    PushT2D,
    build_admm_2d,
    build_scenario,
    run_2d,
)
from oim.sim3d.build import (  # noqa: E402
    SUB_OPTIMIZERS,
    build_admm_3d,
    build_flat_3d,
    named_camera,
)
from oim.sim3d.deterministic import run_interactive  # noqa: E402
from oim.sim3d.run import run_3d_admm, run_3d_plain  # noqa: E402
from oim.utils.plotting import (  # noqa: E402
    plot_run_2d,
    plot_run_3d,
    save_animation_2d,
)
from oim.utils.poses import load_poses  # noqa: E402
from oim.utils.results import RunName, save_run  # noqa: E402
from oim.utils.scenes import SCENES  # noqa: E402

CONFIG_DIR = os.path.join(ROOT, "configs")
RECORDINGS_DIR = os.path.join(ROOT, "recordings")
RUNS_DIR = os.path.join(ROOT, "results", "runs")

# Replanning period, shared by every 3D run so a flat baseline and an ADMM
# run are graded on the same control rate.
CONTROL_DT = 0.05


@dataclass(frozen=True)
class Experiment:
    """What one `examples/` script declares -- and nothing more.

    Everything else (embodiments, goal, obstacles, defaults) is looked up
    from the registries this names, so a task cannot drift out of sync with
    the scene it claims to run.

    Args:
        world: `"3d"` (MJX contact) or `"2d"` (analytic single contact).
        scene: 3D only -- a key of `oim.utils.scenes.SCENES`.
        env: 2D only -- a scenario name for `oim.sim2d.build_scenario`.

    Raises:
        ValueError: If the world and the named registry disagree.
    """

    world: Literal["2d", "3d"]
    scene: Optional[str] = None
    env: Optional[str] = None

    def __post_init__(self) -> None:
        """Fail at import if a script names a scene that does not exist."""
        if self.world == "3d":
            if self.scene is None or self.env is not None:
                raise ValueError("a 3D Experiment sets `scene`, not `env`")
            if self.scene not in SCENES:
                raise ValueError(
                    f"scene={self.scene!r} is not in oim.utils.scenes.SCENES "
                    f"(available: {sorted(SCENES)})"
                )
        elif self.world == "2d":
            if self.env is None or self.scene is not None:
                raise ValueError("a 2D Experiment sets `env`, not `scene`")
        else:
            raise ValueError(f"world must be '2d' or '3d', got {self.world!r}")

    @property
    def robots(self) -> Tuple[str, ...]:
        """Embodiments this experiment can run, first one the default.

        Read from the scene's own MJCF table rather than declared, so
        `--robot` offers exactly what has a model to load and a scene
        cannot advertise an embodiment it lacks.
        """
        if self.world == "2d":
            return ("disc",)
        return tuple(sorted(SCENES[self.scene].mjcf_by_robot))

    def task_id(self, robot: str) -> str:
        """The run identity `oim/run_eval.py` groups rows on."""
        if self.world == "2d":
            return f"pusht2d_{self.env}"
        return f"pusht3d_{robot}_{self.scene}"

    def run_name(self, robot: str, *method: Optional[str]) -> RunName:
        """Name every artifact of a run after the task that produced it.

        The stem starts with `task_id`, the same string the run file
        records, so a filename says which of the five tasks it came from.
        Naming it after the world instead left every tabletop scene
        writing `pusht3d_xarm6_admm_...`, told apart only by timestamp.

        Args:
            robot: Embodiment, part of the identity in 3D.
            method: Algorithm and its sub-optimizers; `None` entries are
                dropped, so a flat baseline contributes only its name.

        Returns:
            A `RunName` whose files share one timestamp.
        """
        return RunName(self.task_id(robot), *(p for p in method if p))


def config_name(robot: str) -> str:
    """Which config file an embodiment reads.

    The 2D world's disc has no file of its own and reads the point mass's,
    which is where its sampler budget and 2D physics live.
    """
    return "point" if robot == "disc" else robot


def load_config(robot: str) -> Dict[str, Any]:
    """Load the default parameters for an embodiment.

    There is no flag for this: the config is a property of the robot, so
    `--robot xarm6` reads `xarm6.yaml` and nothing else can be selected.
    One file per embodiment is the point -- the arm's sampler budget and
    step count do not transfer to the point mass, so letting a run mix them
    would quietly invalidate a comparison.

    Args:
        robot: Selects `oim/configs/{robot}.yaml`.

    Returns:
        The parsed config.

    Raises:
        FileNotFoundError: If the file does not exist.
    """
    path = os.path.join(CONFIG_DIR, f"{config_name(robot)}.yaml")
    with open(path) as f:
        return yaml.safe_load(f)


def build_parser(
    experiment: Experiment, cfg: Optional[Dict[str, Any]] = None
) -> argparse.ArgumentParser:
    """Build one script's CLI, taking every default from `cfg`.

    Only the flags that apply to this experiment's world are added, so
    `--help` never lists something that would be silently ignored -- which
    is how `--no-plot` came to be accepted, and disregarded, by 3D runs.

    Args:
        experiment: The script's declaration.
        cfg: A config from `load_config`. Omitted (the launcher's
            introspection path) falls back to the point robot's file, since
            only the *shape* of the parser matters there.

    Returns:
        The parser.
    """
    if cfg is None:
        cfg = load_config("point")
    smp, adm, run = cfg["sampler"], cfg["admm"], cfg["run"]
    three_d = experiment.world == "3d"

    parser = argparse.ArgumentParser(
        description=f"{experiment.world.upper()} experiment: "
        f"{experiment.scene or experiment.env}."
    )
    if three_d:
        parser.add_argument(
            "--robot",
            choices=list(experiment.robots),
            default=experiment.robots[0],
            help="Embodiment; only those this scene has an MJCF for.",
        )
        parser.add_argument(
            "--warp",
            action="store_true",
            help="MuJoCo Warp rollouts instead of JAX.",
        )
        parser.add_argument(
            "--record",
            action="store_true",
            help="Write an mp4 to oim/recordings/ (needs ffmpeg).",
        )
        # Not algorithm-specific: every sampling-based controller has a
        # candidate population and a chosen trajectory, so these sit beside
        # --record rather than under one subcommand. ADMM draws two blocks
        # (object and robot), a flat baseline draws its one.
        parser.add_argument(
            "--show-samples",
            action="store_true",
            default=run["show_samples"],
            help="Overlay sampled candidate rollouts (thin lines). "
            "Independent of --show-optimal.",
        )
        parser.add_argument(
            "--show-optimal",
            action="store_true",
            default=run["show_optimal"],
            help="Overlay the chosen trajectory (thick line). "
            "Independent of --show-samples.",
        )
    else:
        parser.add_argument(
            "--contact-action",
            action="store_true",
            help="Object block decides [p, f_n, f_t], not the wrench.",
        )
        parser.add_argument(
            "--no-relocate",
            action="store_true",
            help="Disable the global contact-point search.",
        )
        parser.add_argument(
            "--no-obstacles", action="store_true", help="Strip the obstacles."
        )
        parser.add_argument(
            "--no-jit",
            action="store_true",
            help="Run eagerly, steppable in a debugger.",
        )
        parser.add_argument(
            "--animate", action="store_true", help="Also write a gif."
        )
    if three_d:
        # Initial conditions, not sampler settings: varying the seed
        # redraws the planner's noise but leaves the problem identical,
        # while these change where the object starts and where it must go.
        # Unset means "draw one", so a sweep varies them without an axis.
        for kind in ("start", "goal"):
            parser.add_argument(
                f"--{kind}",
                type=str,
                default=None,
                help=f"{kind.capitalize()} pose key from "
                f"examples/poses/<task>.yaml, or 'random' "
                f"(default: random).",
            )
    parser.add_argument(
        "--no-plot", action="store_true", help="Skip the summary figure."
    )
    parser.add_argument(
        "--samples",
        type=int,
        default=smp["num_samples"],
        help="Rollouts per sub-optimizer.",
    )
    parser.add_argument(
        "--horizon",
        type=int,
        default=smp["horizon"],
        help="Consensus horizon H.",
    )

    subparsers = parser.add_subparsers(dest="algorithm")
    if three_d:
        # Flat baselines have no consensus, so they do not take --n-admm,
        # --rho or --gamma. They used to accept and ignore them, which put
        # a value in the command line that never reached the algorithm.
        for name, helptext in (
            ("ps", "Predictive sampling"),
            ("mppi", "MPPI"),
        ):
            sp = subparsers.add_parser(name, help=helptext)
            sp.add_argument("--seed", type=int, default=run["seed"])
            sp.add_argument("--steps", type=int, default=run["steps"])
            sp.add_argument(
                "--iterations",
                type=int,
                default=smp["iterations"],
                help="Internal optimizer passes per real control step.",
            )
            sp.add_argument(
                "--headless",
                action="store_true",
                help="No viewer: run --steps steps and save a run file.",
            )

    admm = subparsers.add_parser(
        "admm", help="ADMM-coordinated object-informed MPPI"
    )
    if three_d:
        # 2D's ADMM is built by `build_admm_2d`, which always uses MPPI
        # with 2D-tuned noise levels; offering a choice there would accept
        # a value it then ignores.
        admm.add_argument(
            "--robot-opt",
            choices=SUB_OPTIMIZERS,
            default=adm["robot_opt"],
            help="Sampling optimizer for the robot-level ADMM block.",
        )
        admm.add_argument(
            "--object-opt",
            choices=SUB_OPTIMIZERS,
            default=adm["object_opt"],
            help="Sampling optimizer for the object-level ADMM block.",
        )
        admm.add_argument(
            "--consensus-alpha",
            type=float,
            default=adm["consensus_alpha"],
            help="EMA weight on A^o/A^r across ADMM rounds (1.0 = raw).",
        )
        admm.add_argument(
            "--rho-torque",
            type=float,
            default=10.0,
            help="Separate initial penalty on the wrench's torque "
            "component, split from --rho (the force penalty). Found to "
            "improve both position and orientation error in most scenes, "
            "so it is now the default rather than an opt-in flag.",
        )
    admm.add_argument("--n-admm", type=int, default=adm["n_admm"])
    admm.add_argument("--rho", type=float, default=adm["rho"])
    admm.add_argument("--gamma", type=float, default=adm["gamma"])
    admm.add_argument("--seed", type=int, default=run["seed"])
    admm.add_argument("--steps", type=int, default=run["steps"])
    if three_d:
        admm.add_argument(
            "--headless",
            action="store_true",
            help="No viewer: run --steps steps and save a run file.",
        )
    return parser


def _save(
    experiment: Experiment,
    args: argparse.Namespace,
    name: RunName,
    task: Any,
    log: Dict[str, Any],
    *,
    algorithm: str,
    robot: str,
    robot_opt: str,
    object_opt: Optional[str],
    control_dt: float,
    extra_static: Dict[str, Any],
    extra_hyper: Optional[Dict[str, Any]] = None,
) -> None:
    """Write the run file, identically for every world and algorithm.

    One writer rather than one per runner: the identity and hyperparameter
    blocks are what `oim/run_eval.py` groups and filters on, so a field
    that only some runners record is a field no table can use.
    """
    run_cfg = args.cfg["run"]
    save_run(
        RUNS_DIR,
        name,
        run=dict(
            world=experiment.world,
            task=experiment.task_id(robot),
            robot=robot,
            algorithm=algorithm,
            robot_opt=robot_opt,
            object_opt=object_opt,
            seed=args.seed,
            start_index=getattr(args, "start_index", None),
            goal_index=getattr(args, "goal_index", None),
            # Rollout backend and viewer mode. Neither changes what the
            # planner is asked to do, but both change what it actually
            # does (Warp and MJX-JAX physics differ in contact handling;
            # the viewer seeds `init_params` differently than --headless),
            # so two otherwise-identical runs are not comparable without
            # them. Learned the hard way: an interactive 0.025 m run and a
            # headless 0.688 m run of the "same" configuration.
            backend="warp" if getattr(args, "warp", False) else "jax",
            interactive=not getattr(args, "headless", False),
        ),
        hyperparameters=dict(
            config=args.config_name,
            steps=args.steps,
            samples=args.samples,
            horizon=args.horizon,
            n_admm=getattr(args, "n_admm", None),
            rho=getattr(args, "rho", None),
            rho_torque=getattr(args, "rho_torque", None),
            gamma=getattr(args, "gamma", None),
            consensus_alpha=getattr(args, "consensus_alpha", None),
            iterations=getattr(args, "iterations", None),
            control_dt=control_dt,
            goal_pos_tol=run_cfg["goal_pos_tol"],
            goal_theta_tol=run_cfg["goal_theta_tol"],
            # The weights this run was scored under, not just the file
            # they came from: `costs:` is now the thing being tuned, so a
            # run file that only recorded `config: xarm6` would not say
            # which tuning it was, and two runs a retune apart would be
            # indistinguishable in `oim/run_eval.py`.
            costs=getattr(args, "cfg", {}).get("costs"),
            **(extra_hyper or {}),
        ),
        task=task,
        log=log,
        extra_static=extra_static,
    )


def _goal_reached(task: Any, run_cfg: Dict[str, Any]) -> Any:
    """A predicate for `run_interactive`: has the object reached the goal?

    The same test the headless runners apply, against the same two
    tolerances, so the viewer stops exactly where a `--headless` run of the
    identical command would have recorded success -- rather than the two
    disagreeing about what "done" means.

    Args:
        task: The `PushT` whose goal and block pose to read.
        run_cfg: The config's `run` block, holding the tolerances.

    Returns:
        A callable taking `mujoco.MjData` and returning whether both
        tolerances are met.
    """
    goal = np.asarray(task.goal)
    pos_tol = float(run_cfg["goal_pos_tol"])
    theta_tol = float(run_cfg["goal_theta_tol"])

    def reached(mj_data: mujoco.MjData) -> bool:
        pose = np.asarray(task._block_pose(mj_data))
        pos_err = float(np.linalg.norm(pose[:2] - goal[:2]))
        theta_err = abs(float(wrap_angle(pose[2] - goal[2])))
        return pos_err < pos_tol and theta_err < theta_tol

    return reached


def _mjx_static(
    task: Any, robot: str, mj_model: mujoco.MjModel
) -> Dict[str, Any]:
    """Scene facts a 3D run file carries so it can be replayed."""
    return dict(
        robot=robot,
        sim_timestep=float(mj_model.opt.timestep),
        qpos_size=int(mj_model.nq),
        qvel_size=int(mj_model.nv),
        block_qpos_adr=(task.block_qpos_adr if robot == "xarm6" else [0, 1, 2]),
        block_dof_adr=task.block_dofs,
    )


def _resolve_poses(
    experiment: Experiment, args: argparse.Namespace
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """Pick this run's start and goal, and record which were picked.

    An unset `--start`/`--goal` draws one, which is the point: re-running a
    task then varies the initial condition rather than only the sampler's
    noise. The drawn keys go onto `args` so `_save` writes them into the
    run file -- a random draw nobody recorded is a result nobody can repeat.

    The draw is seeded from `--seed`, so a given seed always yields the
    same pair, and a sweep over seeds is a sweep over initial conditions.

    Args:
        experiment: The script's declaration, naming the pose file.
        args: Parsed arguments; `start_index`/`goal_index` are set here.

    Returns:
        `(start, goal)`, either entry `None` if the task has no pose file.
    """
    poses = load_poses(experiment.scene)
    if poses is None:
        args.start_index = args.goal_index = None
        return None, None
    rng = np.random.default_rng(args.seed)
    args.start_index, start = poses.select("start", args.start, rng)
    args.goal_index, goal = poses.select("goal", args.goal, rng)
    print(
        f"poses: start {args.start_index} {np.round(start, 3).tolist()}  "
        f"goal {args.goal_index} {np.round(goal, 3).tolist()}"
    )
    return start, goal


def _run_3d(experiment: Experiment, args: argparse.Namespace) -> None:
    """One 3D run: interactive viewer, or `--headless` run file + plot."""
    is_admm = args.algorithm == "admm"
    run_cfg = args.cfg["run"]
    start, goal = _resolve_poses(experiment, args)

    if is_admm:
        print(
            f"ADMM object-informed MPPI on {experiment.scene}: "
            f"robot={args.robot_opt}, object={args.object_opt}"
        )
        task, ctrl, mj_model, mj_data = build_admm_3d(
            experiment.scene,
            args.robot,
            args.cfg,
            warp=args.warp,
            horizon=args.horizon,
            samples=args.samples,
            seed=args.seed,
            robot_opt=args.robot_opt,
            object_opt=args.object_opt,
            n_admm=args.n_admm,
            rho=args.rho,
            gamma=args.gamma,
            consensus_alpha=args.consensus_alpha,
            rho_torque=args.rho_torque,
            start=start,
            goal=goal,
        )
        name = experiment.run_name(
            args.robot, "admm", args.robot_opt, args.object_opt
        )
    else:
        print(f"Flat {args.algorithm} on {experiment.scene}")
        # Built by `build_flat_3d`, not a path of its own: a baseline is
        # only worth anything if it faces the same task, horizon, sampler
        # budget and execution model ADMM does.
        task, ctrl, mj_model, mj_data = build_flat_3d(
            args.algorithm,
            experiment.scene,
            args.robot,
            args.cfg,
            warp=args.warp,
            horizon=args.horizon,
            samples=args.samples,
            seed=args.seed,
            control_dt=CONTROL_DT,
            iterations=args.iterations,
            start=start,
            goal=goal,
        )
        name = experiment.run_name(args.robot, args.algorithm)

    camera = named_camera(mj_model)
    os.makedirs(RECORDINGS_DIR, exist_ok=True)
    if not args.headless:
        # Pin the scene camera only while recording -- a fixed cam makes
        # mp4s comparable, but locks out mouse look in the live viewer.
        # Without --record, leave the free camera so you can orbit.
        fixed_cam = None
        if args.record and camera:
            fixed_cam = mujoco.mj_name2id(
                mj_model, mujoco.mjtObj.mjOBJ_CAMERA, camera
            )
        log = run_interactive(
            ctrl,
            mj_model,
            mj_data,
            frequency=1.0 / CONTROL_DT,
            fixed_camera_id=fixed_cam,
            show_traces=False,
            record_video=args.record,
            recording_name=name(),
            show_samples=args.show_samples,
            show_optimal=args.show_optimal,
            terminate_fn=_goal_reached(task, run_cfg),
        )
    else:
        runner = run_3d_admm if is_admm else run_3d_plain
        log = runner(
            task,
            ctrl,
            ctrl.init_params(seed=args.seed),
            mj_model,
            mj_data,
            frequency=1.0 / CONTROL_DT,
            max_steps=args.steps,
            goal_pos_tol=run_cfg["goal_pos_tol"],
            goal_theta_tol=run_cfg["goal_theta_tol"],
            # The same recording the ADMM path gets: a baseline you cannot
            # watch is a baseline you cannot debug.
            record_dir=RECORDINGS_DIR if args.record else None,
            record_name=name(),
            camera=camera,
            # Both runners take these now: a flat baseline has a candidate
            # population and a chosen trajectory just as ADMM's blocks do.
            show_samples=args.show_samples,
            show_optimal=args.show_optimal,
        )

    # Live and headless share one save path: the interactive runner now
    # returns the same log dict when the task is PushT-like.
    if log is None:
        return
    _save(
        experiment,
        args,
        name,
        task,
        log,
        algorithm=args.algorithm,
        robot=args.robot,
        robot_opt=args.robot_opt if is_admm else args.algorithm,
        object_opt=args.object_opt if is_admm else None,
        control_dt=CONTROL_DT,
        extra_static=_mjx_static(task, args.robot, mj_model),
    )
    if not args.no_plot:
        plot_run_3d(task, log, os.path.join(RECORDINGS_DIR, f"{name()}.png"))


def _run_2d(experiment: Experiment, args: argparse.Namespace) -> None:
    """One 2D run: closed loop, run file, plot, optional gif."""
    w2 = args.cfg["world2d"]
    scenario = build_scenario(experiment.env)
    task = PushT2D(
        footprint=scenario.footprint,
        goal=scenario.goal,
        obstacles=None if args.no_obstacles else scenario.obstacles,
        contact_actions=args.contact_action,
        relocate_contact=not args.no_relocate,
        mass=w2["mass"],
        mu=w2["mu"],
        mu_c=w2["mu_c"],
        f_max=w2["f_max"],
    )
    print(f"scenario: {scenario.name} -- {scenario.description}")
    ctrl, params = build_admm_2d(
        task,
        horizon=args.horizon,
        num_samples=args.samples,
        n_admm=args.n_admm,
        rho=args.rho,
        gamma=args.gamma,
        seed=args.seed,
    )
    block_kind = (
        "contact action [p, f_n, f_t]"
        if args.contact_action
        else "direct wrench"
    )
    print(
        f"object block: {block_kind}  (action dim {task.object_action_dim}, "
        f"consensus dim {task.consensus_dim})"
    )

    ctx = jax.disable_jit() if args.no_jit else nullcontext()
    with ctx:
        log = run_2d(
            task,
            ctrl,
            params,
            object_pose0=scenario.object_pose0,
            robot_pos0=scenario.robot_pos0,
            max_steps=args.steps,
            jit=not args.no_jit,
        )

    op = log["object_pose"]
    goal_xy = np.asarray(scenario.goal[:2])
    d0 = float(np.linalg.norm(op[0, :2] - goal_xy))
    d1 = float(np.linalg.norm(op[-1, :2] - goal_xy))
    pct = 100 * (1 - d1 / d0) if d0 > 0 else 0.0
    print(f"position error {d0:.4f} -> {d1:.4f}  ({pct:.1f}% closer)")

    name = experiment.run_name("disc", "admm")
    _save(
        experiment,
        args,
        name,
        task,
        log,
        algorithm="admm",
        robot="disc",
        # `build_admm_2d` has no named sub-optimizer choice: it is always
        # MPPI, with 2D-tuned noise levels.
        robot_opt="mppi",
        object_opt="mppi",
        control_dt=float(task.dt),
        extra_static=dict(
            scenario=scenario.name,
            robot="disc",
            robot_radius=float(task.model.robot_radius),
            robot_max_speed=float(task.u_max[0]),
        ),
        extra_hyper=dict(
            contact_action=args.contact_action,
            relocate_contact=not args.no_relocate,
        ),
    )

    if not args.no_plot:
        os.makedirs(RECORDINGS_DIR, exist_ok=True)
        path = os.path.join(RECORDINGS_DIR, f"{name()}.png")
        plot_run_2d(task, scenario, log, path)
        if args.animate:
            gif = os.path.join(RECORDINGS_DIR, f"{name()}.gif")
            save_animation_2d(task, scenario, log, gif)


def main(experiment: Experiment, argv: Optional[Sequence[str]] = None) -> None:
    """Parse this script's CLI, then run, record and save one experiment.

    Args:
        experiment: The declaration at the top of an `examples/` script.
        argv: Command line to parse; `None` reads `sys.argv`.
    """
    # Two stages, because which config supplies the defaults depends on a
    # CLI choice: peek at --robot, load that robot's file, then build the
    # real parser on top of it. Anything passed explicitly still wins --
    # argparse defaults are exactly "the value when the flag is absent".
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--robot", default=experiment.robots[0])
    pre_args, _ = pre.parse_known_args(argv)

    cfg = load_config(pre_args.robot)
    parser = build_parser(experiment, cfg)
    args = parser.parse_args(argv)
    args.cfg = cfg
    # Provenance: which defaults produced this run, recorded alongside the
    # values themselves so a run file explains itself.
    args.config_name = config_name(pre_args.robot)
    if args.algorithm is None:
        args.algorithm = "admm"

    if experiment.world == "2d":
        _run_2d(experiment, args)
    else:
        _run_3d(experiment, args)
