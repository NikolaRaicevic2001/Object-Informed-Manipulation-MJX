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

    # same planner, MuJoCo executing the wrench: how good is the model?
    uv run python examples/object_only.py --scene shelf_gap --plant mujoco

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
from oim.runtime.mjcf import named_camera
from oim.runtime.overlay import PlanOverlay, traces_for
from oim.runtime.samplers import SUB_OPTIMIZERS, object_sample_count
from oim.runtime.video import OffscreenRecorder
from oim.utils.plotting import plot_run_object, save_animation_object
from oim.utils.poses import load_poses
from oim.utils.results import RunName, save_run
from oim.utils.scenes import SCENES
from oim.worlds.object_only import build_object_only, build_plant, run_object

RECORDINGS_DIR = os.path.join(ROOT, "recordings")
RUNS_DIR = os.path.join(ROOT, "results", "object")

# What `oim.run_launch` reads to sweep this script. Every other example
# declares an `Experiment` and the launcher takes the world and the parser
# off that; this one has no per-script scene and no algorithm subcommand,
# so it supplies both directly instead. See `run_launch.script_world`.
SWEEP_WORLD = "object"

# One constant for both parsers below. `main` reads --robot with a throwaway
# pre-parser to pick the config file that then *defaults the real parser*, so
# a separate default in each would let the two disagree: the banner would
# name one file while every number came from the other.
DEFAULT_ROBOT = "xarm6"


def sweep_parser() -> argparse.ArgumentParser:
    """This script's parser, for `oim.run_launch` to introspect.

    The config only supplies defaults, and the launcher looks at the
    parser's *shape* -- which flags exist and which take a value -- so
    which robot's file it is built from does not matter here.
    """
    return build_parser(load_config("point"))


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
        default=DEFAULT_ROBOT,
        help="No robot is simulated; this picks the config file and the "
        "scene variant, so the object block matches that embodiment's "
        "ADMM runs (it gates object_action_bounds). Defaults to xarm6 "
        "because that is where the object block's own tuning lives -- "
        "sampler.object.num_samples, its noise_level, and costs.w_rate "
        "are all absent from point.yaml, so --robot point silently runs a "
        "differently-configured block.",
    )
    parser.add_argument(
        "--plant",
        choices=["analytic", "mujoco"],
        default="analytic",
        help="What executes the chosen wrench. 'analytic' is the limit "
        "surface executing itself -- no model error, an upper bound on the "
        "formulation. 'mujoco' applies the same wrench to the block's "
        "slide/hinge DoFs in the simulator, so the run plans with the limit "
        "surface and is graded by MuJoCo; the log then carries the "
        "per-step gap between the two.",
    )
    parser.add_argument(
        "--object-opt",
        choices=SUB_OPTIMIZERS,
        default=adm["object_opt"],
        help="Sampling optimizer for the object block.",
    )
    # Resolved through `object_sample_count`, the same rule `build_admm_3d`
    # uses, so `sampler.object.num_samples` means the same thing here as it
    # does inside ADMM. The block under study is only comparable to its
    # ADMM runs if it is given the budget those runs give it -- and there
    # is no robot here, so `--samples` *is* the object block's count.
    parser.add_argument(
        "--samples",
        type=int,
        default=object_sample_count(smp, smp["num_samples"]),
        help="Rollouts for the object block. Defaults to "
        "sampler.object.num_samples, else sampler.num_samples.",
    )
    parser.add_argument("--horizon", type=int, default=smp["horizon"])
    parser.add_argument(
        "--iterations",
        type=int,
        default=adm["n_admm"],
        help="Optimizer passes per control step. Defaults to the config's "
        "n_admm, which is the budget ADMM actually gives the object block "
        "(once per consensus round) -- so a bare run is the same "
        "experiment as the sweep rather than a quieter one.",
    )
    parser.add_argument(
        "--wrench-fraction",
        type=float,
        default=None,
        help="Fraction of the friction-cone limit a unit action maps to. "
        "Unset takes costs.wrench_fraction from the config. Decides "
        "whether the block can move the object at all, and the two plants "
        "need different values: --plant analytic gates on the coupled "
        "norm, whose ceiling is fraction*sqrt(3), so 1.0 works; --plant "
        "mujoco gates per DoF, whose ceiling is fraction alone, so 1.0 "
        "nets ~zero force and 2.0 is the measured best.",
    )
    parser.add_argument(
        "--w-rate",
        type=float,
        nargs="+",
        metavar="W",
        default=None,
        help="Penalty on the step-to-step change in wrench, normalized by "
        "the friction-cone limit. One value for all channels, or three as "
        "fx fy tau. Unset keeps the config's. Nothing else couples w_t to "
        "w_t+1 -- the block samples one independent knot per step under a "
        "zero-order hold, so this is what makes a change take several "
        "steps instead of one jump.",
    )
    parser.add_argument(
        "--project-gate",
        type=float,
        default=None,
        help="Position error below which a sub-threshold action is snapped "
        "up to breakaway (project_object_action). Unset keeps the "
        "config's. Large values snap always, which is what lets the "
        "optimizer average directions without the mean falling into the "
        "deadzone.",
    )
    parser.add_argument(
        "--noise-level",
        type=float,
        default=None,
        help="Object sampler exploration noise, where 1.0 is the whole "
        "friction-cone limit. Unset keeps the config's.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=None,
        help="Object MPPI temperature. Read it against the rollout cost "
        "spread the run prints: far below it the softmax is an argmax "
        "over white noise. Unset keeps the config's.",
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
    parser.add_argument(
        "--video-width",
        type=int,
        default=720,
        help="mp4 width. --plant mujoco only: that plant owns a real "
        "scene, so --record also films it from the scene's own camera.",
    )
    parser.add_argument(
        "--video-height", type=int, default=480, help="mp4 height."
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


def _mujoco_recording(
    args: argparse.Namespace, plant: object, base_name: str
) -> tuple:
    """A `(recorder, on_plan)` filming the MuJoCo plant, else `(None, None)`.

    Only the MuJoCo plant owns a real scene to film; the analytic one is
    three numbers and gets the matplotlib gif alone. Frames are captured
    from the plant's own `MjData` at the physics rate, so playback is real
    time, and each control step's plans are handed to the overlay just
    before that step executes -- which is the window those frames fall in.

    Args:
        args: Parsed command line.
        plant: The plant, which must be a `MujocoPlant` to be filmed.
        base_name: Filename stem, shared with the run's plot and results.

    Returns:
        The recorder (to close afterwards) and the per-step plan callback.
    """
    if not (args.record and args.plant == "mujoco"):
        return None, None

    os.makedirs(RECORDINGS_DIR, exist_ok=True)
    # The same overlay the 3D runners composite into their mp4s, with one
    # block instead of three, so candidates and the chosen plan read
    # identically across the two worlds.
    overlay = (
        PlanOverlay(horizon=args.horizon, max_blocks=1)
        if (args.show_samples or args.show_optimal)
        else None
    )
    recorder = OffscreenRecorder(
        plant.mj_model,
        output_dir=RECORDINGS_DIR,
        base_name=base_name,
        target_fps=args.fps,
        size=(args.video_width, args.video_height),
        camera=named_camera(plant.mj_model),
        overlay=overlay,
    )
    plant.attach_recorder(recorder)
    if overlay is None:
        return recorder, None

    def on_plan(plan, samples) -> None:  # noqa: ANN001
        """Hand this step's plans to the frames captured during it."""
        recorder.set_plans(
            traces_for(
                object_chosen=plan if args.show_optimal else None,
                object_samples=samples if args.show_samples else None,
            )
        )

    return recorder, on_plan


def main() -> None:
    """Parse, build, run, save and plot one object-only experiment."""
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--robot", default=DEFAULT_ROBOT)
    pre_args, _ = pre.parse_known_args()
    cfg = load_config(pre_args.robot)
    parser = build_parser(cfg)
    args = parser.parse_args()
    if args.w_rate is not None and len(args.w_rate) not in (1, 3):
        parser.error("--w-rate takes 1 value or 3 (fx fy tau)")

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
        plant=args.plant,
        wrench_fraction=args.wrench_fraction,
        w_rate=args.w_rate,
        project_gate=args.project_gate,
        noise_level=args.noise_level,
        temperature=args.temperature,
        goal=goal,
        start=start,
    )
    # Naming the config *file* rather than just the embodiment: every
    # number on the next line is defaulted from it, and `--robot point`
    # quietly selects a file with none of the object block's own tuning in
    # it. "(point config)" was too easy to read past.
    print(
        f"object block alone on {args.scene}, from "
        f"oim/configs/{config_name(args.robot)}.yaml"
    )
    print(
        f"  {args.object_opt}, H={args.horizon}, {args.samples} samples, "
        f"{args.iterations} pass(es)/step, {args.steps} steps max, "
        f"{args.plant} plant"
    )
    print(
        f"  w_rate={[float(v) for v in task.object_model.w_rate]}, "
        f"noise_level={block.optimizer.noise_level}, "
        f"temperature={getattr(block.optimizer, 'temperature', None)}, "
        f"project_gate={task.project_gate_pos}"
    )

    run_cfg = cfg["run"]
    # Built from the same start/goal the task was, so the simulator's block
    # begins where the analytic one does and the two are comparable.
    plant = build_plant(
        args.plant,
        task,
        args.robot,
        cfg["world3d"],
        control_dt=float(task.dt),
        start=obj_state0,
        goal=goal,
        jit=not args.no_jit,
    )
    name = RunName(f"object_{args.scene}", args.object_opt)
    recorder, on_plan = _mujoco_recording(args, plant, name())

    ctx = jax.disable_jit() if args.no_jit else nullcontext()
    try:
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
                plant=plant,
                # Only kept when something will draw them: (steps, samples,
                # H, 3) is ~100 MB at 1000 steps / 128 samples / H=32, and
                # it is dropped from the run file either way.
                log_samples=args.record and args.show_samples,
                on_plan=on_plan,
            )
    finally:
        # In `finally` so an interrupted run still yields a playable mp4
        # rather than a truncated pipe to ffmpeg.
        if recorder is not None:
            recorder.close()
            print(f"saved mujoco video to {recorder.recorder.video_path}")

    save_run(
        RUNS_DIR,
        name,
        run=dict(
            world="object",
            task=f"object_{args.scene}",
            robot=args.robot,
            algorithm=f"object_only_{args.plant}",
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
            plant=args.plant,
            wrench_fraction=args.wrench_fraction,
            w_rate=[float(v) for v in task.object_model.w_rate],
            project_gate=task.project_gate_pos,
            noise_level=block.optimizer.noise_level,
            temperature=getattr(block.optimizer, "temperature", None),
            # The object block plans and executes at the same rate, and
            # there is no replanning budget distinct from it.
            control_dt=float(task.dt),
            goal_pos_tol=run_cfg["goal_pos_tol"],
            goal_theta_tol=run_cfg["goal_theta_tol"],
            costs=cfg.get("costs"),
        ),
        task=task,
        log=log,
        extra_static=dict(scene=args.scene, robot=args.robot, plant=args.plant),
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
