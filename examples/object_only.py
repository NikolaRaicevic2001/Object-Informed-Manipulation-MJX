"""Run the object-level subproblem on its own -- no robot, no ADMM.

One script for every scene, rather than one per scene as under
`oim/experiment.py`: this world has no MJCF of its own and no embodiment to
choose between, so a scene here is just a key of `oim.utils.scenes.SCENES`
and a `--scene` flag says everything a separate file would.

    # can the object planner route the T through the shelf gap at all?
    uv run python examples/object_only.py --scene shelf_gap --robot xarm6

    # give it the update budget it would get inside ADMM (n_admm = 4)
    uv run python examples/object_only.py --scene open_table --iterations 4

    # watch the plans evolve: one gif frame per control step
    uv run python examples/object_only.py --scene shelf_gap --record

    # eager, to breakpoint inside the limit-surface dynamics
    uv run python examples/object_only.py --scene clutter --no-jit --steps 5

Writes its plot (and, with `--record`, its gif) to `oim/recordings/`, and
its run file to
`oim/results/object/` -- deliberately *not* `oim/results/runs/`, which
`oim/run_eval.py` globs: these runs have no robot and no control frequency,
so averaging them into a results table beside real ones would be
meaningless.
"""

import argparse
import os
from contextlib import nullcontext

import jax

from oim import ROOT
from oim.experiment import config_name, load_config
from oim.sim3d.build import SUB_OPTIMIZERS
from oim.simobj import build_object_only, run_object
from oim.utils.plotting import plot_run_object, save_animation_object
from oim.utils.poses import load_poses
from oim.utils.results import RunName, save_run
from oim.utils.scenes import SCENES

RECORDINGS_DIR = os.path.join(ROOT, "recordings")
RUNS_DIR = os.path.join(ROOT, "results", "object")


def build_parser(cfg: dict) -> argparse.ArgumentParser:
    """The CLI, defaulted from the embodiment's config file."""
    smp, adm, run = cfg["sampler"], cfg["admm"], cfg["run"]
    parser = argparse.ArgumentParser(
        description="Run the ADMM object-level subproblem in isolation."
    )
    parser.add_argument(
        "--scene",
        choices=sorted(SCENES),
        default="clutter",
        help="Which scene's goal, obstacles and object physics to use.",
    )
    parser.add_argument(
        "--robot",
        choices=["point", "xarm6"],
        default="point",
        help="No robot is simulated; this picks the config file and the "
        "scene variant, so the object block matches that embodiment's "
        "ADMM runs (it gates object_action_bounds).",
    )
    parser.add_argument(
        "--object-opt",
        choices=SUB_OPTIMIZERS,
        default=adm["object_opt"],
        help="Sampling optimizer for the object block.",
    )
    parser.add_argument("--samples", type=int, default=smp["num_samples"])
    parser.add_argument("--horizon", type=int, default=smp["horizon"])
    parser.add_argument(
        "--iterations",
        type=int,
        default=1,
        help="Optimizer passes per control step. ADMM gives the object "
        "block n_admm of these per step, so set it to n_admm for a "
        "like-for-like comparison.",
    )
    parser.add_argument(
        "--wrench-fraction",
        type=float,
        default=None,
        help="Fraction of the friction-cone limit a unit action maps to. "
        "Unset keeps the scene's own. Decides whether the block can move "
        "the object at all: the ceiling on ||w||/limit is fraction*sqrt(3) "
        "and the breakaway threshold is 1.0, so the shipped 0.5 (every "
        "scene but xarm6+open_table) cannot reach it. Try 1.0.",
    )
    parser.add_argument("--steps", type=int, default=run["steps"])
    parser.add_argument("--seed", type=int, default=run["seed"])
    parser.add_argument(
        "--stride",
        type=int,
        default=None,
        help="Draw the object's footprint every this many steps in the "
        "summary figure. Unset scales with --steps to keep ~40.",
    )
    # Mirrors the MuJoCo runs' --record/--show-samples/--show-optimal. The
    # trajectories live here rather than in the summary figure: one static
    # frame carrying every step's horizon is unreadable.
    parser.add_argument(
        "--record",
        action="store_true",
        help="Write an animated gif to oim/recordings/, one frame per "
        "control step, showing that step's candidate rollouts and chosen "
        "plan against where the object actually was.",
    )
    parser.add_argument(
        "--show-samples",
        action="store_true",
        default=True,
        help="Overlay the sampled candidate rollouts in the recording.",
    )
    parser.add_argument(
        "--no-show-samples",
        dest="show_samples",
        action="store_false",
        help="Do not overlay the candidates (smaller gif).",
    )
    parser.add_argument(
        "--show-optimal",
        action="store_true",
        default=True,
        help="Overlay the chosen plan, and mark its endpoint.",
    )
    parser.add_argument(
        "--no-show-optimal",
        dest="show_optimal",
        action="store_false",
        help="Do not overlay the chosen plan.",
    )
    parser.add_argument(
        "--fps", type=int, default=15, help="Recording playback rate."
    )
    for kind in ("start", "goal"):
        parser.add_argument(
            f"--{kind}",
            default=None,
            help=f"{kind.capitalize()} pose key from "
            f"examples/poses/<task>.yaml, or 'random'. Unset uses the "
            f"scene's own.",
        )
    parser.add_argument(
        "--no-jit",
        action="store_true",
        help="Run eagerly, steppable in a debugger.",
    )
    parser.add_argument(
        "--no-plot", action="store_true", help="Skip the summary figure."
    )
    return parser


def main() -> None:
    """Parse, build, run, save and plot one object-only experiment."""
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--robot", default="point")
    pre_args, _ = pre.parse_known_args()
    cfg = load_config(pre_args.robot)
    args = build_parser(cfg).parse_args()

    # Same pose files the 3D runs draw from, so an object-only run and an
    # ADMM run can be pointed at the identical problem instance.
    start = goal = None
    poses = load_poses(args.scene)
    if poses is not None and (args.start or args.goal):
        import numpy as np  # noqa: PLC0415

        rng = np.random.default_rng(args.seed)
        _, start = poses.select("start", args.start, rng)
        _, goal = poses.select("goal", args.goal, rng)
        print(f"poses: start {start} goal {goal}")

    task, block, params, obj_state0 = build_object_only(
        args.scene,
        args.robot,
        cfg,
        horizon=args.horizon,
        samples=args.samples,
        seed=args.seed,
        object_opt=args.object_opt,
        iterations=args.iterations,
        wrench_fraction=args.wrench_fraction,
        goal=goal,
        start=start,
    )
    print(
        f"object block alone on {args.scene} ({args.robot} config): "
        f"{args.object_opt}, H={args.horizon}, {args.samples} samples, "
        f"{args.iterations} pass(es)/step"
    )

    run_cfg = cfg["run"]
    ctx = jax.disable_jit() if args.no_jit else nullcontext()
    with ctx:
        log = run_object(
            task,
            block,
            params,
            obj_state0,
            max_steps=args.steps,
            goal_pos_tol=run_cfg["goal_pos_tol"],
            goal_theta_tol=run_cfg["goal_theta_tol"],
            jit=not args.no_jit,
            # Only kept when something will draw them: (steps, samples, H,
            # 3) is ~100 MB at 1000 steps / 128 samples / H=32, and it is
            # dropped from the run file either way.
            log_samples=args.record and args.show_samples,
        )

    name = RunName(f"object_{args.scene}", args.object_opt)
    save_run(
        RUNS_DIR,
        name,
        run=dict(
            world="object",
            task=f"object_{args.scene}",
            robot=args.robot,
            algorithm="object_only",
            robot_opt=None,
            object_opt=args.object_opt,
            seed=args.seed,
        ),
        hyperparameters=dict(
            config=config_name(args.robot),
            steps=args.steps,
            samples=args.samples,
            horizon=args.horizon,
            iterations=args.iterations,
            wrench_fraction=args.wrench_fraction,
            # The object block plans and executes at the same rate, and
            # there is no replanning budget distinct from it.
            control_dt=float(task.dt),
            goal_pos_tol=run_cfg["goal_pos_tol"],
            goal_theta_tol=run_cfg["goal_theta_tol"],
            costs=cfg.get("costs"),
        ),
        task=task,
        log=log,
        extra_static=dict(scene=args.scene, robot=args.robot),
    )

    if args.no_plot and not args.record:
        return
    os.makedirs(RECORDINGS_DIR, exist_ok=True)
    if not args.no_plot:
        # ~40 footprints regardless of run length: at --steps 1000 a fixed
        # stride of 5 draws 200 and they merge into one blob.
        steps_run = len(log["object_pose"]) - 1
        stride = args.stride or max(1, steps_run // 40)
        plot_run_object(
            task,
            log,
            os.path.join(RECORDINGS_DIR, f"{name()}.png"),
            stride=stride,
        )
    if args.record:
        save_animation_object(
            task,
            log,
            os.path.join(RECORDINGS_DIR, f"{name()}.gif"),
            fps=args.fps,
            show_samples=args.show_samples,
            show_optimal=args.show_optimal,
        )


if __name__ == "__main__":
    main()
