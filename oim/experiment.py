"""One experiment, end to end -- everything an `examples/` script is not.

Each script under `examples/pusht/` declares a single `Experiment` (which
world, which scene) and calls `main`. Everything else lives here: the
command line, config loading, construction, the closed loop, recording, the
run file and the plot. So a task is a declaration, every task gets the same
flags, and adding one never means copying a runner.

    # examples/pusht/shelf_gap.py
    EXPERIMENT = Experiment(world="3d", scene="shelf_gap")
    if __name__ == "__main__":
        main(EXPERIMENT)

The CLI a script offers is derived from its `Experiment`, so it advertises
only what applies:

    3d      --warp/--record, and the ps, mppi and admm algorithms
    2d      --animate/--no-jit, and admm alone (`PushT2D` implements
            `ConsensusTask` and nothing else)
    object  --scene/--plant and the object block's own tuning knobs, and
            no algorithm subcommand at all -- there is one block, so
            there is no consensus to choose an algorithm for

`--robot` offers exactly the embodiments the scene's MJCF exists for, or,
in the object world where none is simulated, the two whose config files
the block can be built from.

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
from oim.runtime.mjcf import named_camera  # noqa: E402
from oim.runtime.overlay import PlanOverlay, traces_for  # noqa: E402
from oim.runtime.samplers import (  # noqa: E402
    SUB_OPTIMIZERS,
    object_sample_count,
)
from oim.runtime.video import OffscreenRecorder  # noqa: E402
from oim.runtime.viewer import run_interactive  # noqa: E402
from oim.utils.plotting import (  # noqa: E402
    plot_run_2d,
    plot_run_3d,
    plot_run_object,
    save_animation_2d,
    save_animation_object,
)
from oim.utils.poses import load_poses  # noqa: E402
from oim.utils.results import RunName, save_run  # noqa: E402
from oim.utils.scenes import SCENES  # noqa: E402
from oim.worlds.object_only import (  # noqa: E402
    build_object_only,
    build_plant,
    run_object,
)
from oim.worlds.object_only.plant import (  # noqa: E402
    PLANT_MODES,
    resolve_plant,
)
from oim.worlds.sim2d import (  # noqa: E402
    PushT2D,
    build_admm_2d,
    build_scenario,
    run_2d,
)
from oim.worlds.sim3d.build import build_admm_3d, build_flat_3d  # noqa: E402
from oim.worlds.sim3d.run import run_3d_admm, run_3d_plain  # noqa: E402

CONFIG_DIR = os.path.join(ROOT, "configs", "robots")
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
        world: `"3d"` (MJX contact), `"2d"` (analytic single contact), or
            `"object"` (the object block alone, no robot).
        scene: 3D and object worlds -- a key of `oim.utils.scenes.SCENES`.
            The object world may leave it `None`, which means the CLI's
            `--scene` supplies it: that world has no MJCF and no
            embodiment of its own, so a scene there is only a choice of
            goal, obstacles and object physics, and one script with a flag
            says everything five near-identical files would.
        env: 2D only -- a scenario name for `oim.worlds.sim2d.build_scenario`.

    Raises:
        ValueError: If the world and the named registry disagree.
    """

    world: Literal["2d", "3d", "object"]
    scene: Optional[str] = None
    env: Optional[str] = None

    def __post_init__(self) -> None:
        """Fail at import if a script names a scene that does not exist."""
        if self.world in ("3d", "object"):
            if self.env is not None:
                raise ValueError(
                    f"a {self.world} Experiment sets `scene`, not `env`"
                )
            if self.scene is None and self.world == "3d":
                raise ValueError("a 3D Experiment must name a `scene`")
            if self.scene is not None and self.scene not in SCENES:
                raise ValueError(
                    f"scene={self.scene!r} is not in oim.utils.scenes.SCENES "
                    f"(available: {sorted(SCENES)})"
                )
        elif self.world == "2d":
            if self.env is None or self.scene is not None:
                raise ValueError("a 2D Experiment sets `env`, not `scene`")
        else:
            raise ValueError(
                f"world must be '2d', '3d' or 'object', got {self.world!r}"
            )

    @property
    def robots(self) -> Tuple[str, ...]:
        """Embodiments this experiment can run, first one the default.

        Read from the scene's own MJCF table rather than declared, so
        `--robot` offers exactly what has a model to load and a scene
        cannot advertise an embodiment it lacks.

        The object world simulates no robot, but still takes one: it
        selects the config file, the scene variant, and the
        `object_action_bounds` branch, so an object-only study of an xArm6
        scene must say `xarm6` to be studying the block that scene's ADMM
        runs use. xArm6 leads because that is where the object block's own
        tuning lives -- `sampler.object.num_samples`, its `noise_level`
        and `costs.w_rate` are all absent from `point.yaml`.
        """
        if self.world == "2d":
            return ("disc",)
        if self.world == "object":
            return ("xarm6", "point")
        return tuple(sorted(SCENES[self.scene].mjcf_by_robot))

    def results_dir(self) -> str:
        """Where this world's run files go.

        The object world writes somewhere `oim/run_eval.py` does not glob,
        deliberately: these runs have no robot and no control frequency, so
        averaging them into a results table beside real ones would be
        meaningless.
        """
        if self.world == "object":
            return os.path.join(ROOT, "results", "object")
        return RUNS_DIR

    def task_id(self, robot: str, scene: Optional[str] = None) -> str:
        """The run identity `oim/run_eval.py` groups rows on.

        Args:
            robot: Embodiment, part of the identity in 3D.
            scene: Overrides the declared scene, for the object world where
                it arrives on the command line instead.

        Returns:
            The identity string.
        """
        scene = scene or self.scene
        if self.world == "2d":
            return f"pusht2d_{self.env}"
        if self.world == "object":
            return f"object_{scene}"
        return f"pusht3d_{robot}_{scene}"

    def run_name(
        self, robot: str, *method: Optional[str], scene: Optional[str] = None
    ) -> RunName:
        """Name every artifact of a run after the task that produced it.

        The stem starts with `task_id`, the same string the run file
        records, so a filename says which of the five tasks it came from.
        Naming it after the world instead left every tabletop scene
        writing `pusht3d_xarm6_admm_...`, told apart only by timestamp.

        Args:
            robot: Embodiment, part of the identity in 3D.
            method: Algorithm and its sub-optimizers; `None` entries are
                dropped, so a flat baseline contributes only its name.
            scene: Overrides the declared scene; see `task_id`.

        Returns:
            A `RunName` whose files share one timestamp.
        """
        return RunName(self.task_id(robot, scene), *(p for p in method if p))


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
        robot: Selects `oim/configs/robots/{robot}.yaml`.

    Returns:
        The parsed config.

    Raises:
        FileNotFoundError: If the file does not exist.
    """
    path = os.path.join(CONFIG_DIR, f"{config_name(robot)}.yaml")
    with open(path) as f:
        return yaml.safe_load(f)


def _add_object_arguments(
    parser: argparse.ArgumentParser,
    experiment: Experiment,
    cfg: Dict[str, Any],
) -> None:
    """The object world's own flags: which scene, which plant, which tuning.

    All of them override something the config supplies, so `None` means
    "keep the config's" and a bare run is the same experiment as a sweep
    cell rather than a quieter one.

    Args:
        parser: The parser to extend.
        experiment: The script's declaration; supplies the scene when it
            names one and the embodiment list either way.
        cfg: A parsed `oim/configs/*.yaml`.
    """
    adm = cfg["admm"]
    if experiment.scene is None:
        parser.add_argument(
            "--scene",
            choices=sorted(SCENES),
            default="clutter",
            help="Which scene's goal, obstacles and object physics to use.",
        )
    parser.add_argument(
        "--robot",
        choices=list(experiment.robots),
        default=experiment.robots[0],
        help="No robot is simulated; this picks the config file and the "
        "scene variant, so the object block matches that embodiment's "
        "ADMM runs (it gates object_action_bounds). See "
        "`Experiment.robots` for why xarm6 leads.",
    )
    parser.add_argument(
        "--plant",
        choices=sorted(PLANT_MODES),
        default="analytic",
        help="Which dynamics this run uses, predicting AND executing. "
        "'analytic' is the limit surface (eq. 5) on both sides -- no model "
        "error, an upper bound on the formulation. 'mujoco' is the "
        "simulator on both sides: the block plans through MJX and executes "
        "in MuJoCo, self-consistent the way a deployment is. 'model-error' "
        "plans with eq. 5 and executes in MuJoCo, which is the measurement "
        "this world exists for -- pred_pos_err is then how good eq. 5 is. "
        "One flag rather than two, so the fourth combination (a planner "
        "with a better model of the world than the world has) cannot be "
        "asked for.",
    )
    parser.add_argument(
        "--friction",
        choices=["box", "cone", "wrench"],
        default="box",
        help="--plant mujoco only: the shape of the simulated support "
        "friction. 'box' is MuJoCo's own per-DoF frictionloss and is the "
        "default because it is measurably the closest to eq. 5 in closed "
        "loop. 'cone' is the coupled ellipsoid eq. 5 assumes -- it fixes "
        "the breakaway threshold exactly and still agrees worse overall; "
        "'wrench' is eq. 5's own force balance and diverges outright. See "
        "oim/worlds/object_only/plant.py.",
    )
    _add_object_substeps_argument(parser)
    parser.add_argument(
        "--object-opt",
        choices=SUB_OPTIMIZERS,
        default=adm["object_opt"],
        help="Sampling optimizer for the object block.",
    )
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
        "config's, which is 0.0 -- off. Superseded by `step` subtracting "
        "friction rather than gating on it, and kept only to reproduce "
        "runs that predate that change.",
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
    parser.add_argument(
        "--stride",
        type=int,
        default=None,
        help="Draw the object's footprint every this many steps in the "
        "summary figure. Unset scales with --steps to keep ~40.",
    )
    # Mirrors the MuJoCo runs' --record/--show-samples/--show-optimal. The
    # trajectories live in the gif rather than in the summary figure: one
    # static frame carrying every step's horizon is unreadable.
    parser.add_argument(
        "--record",
        action="store_true",
        help="Write an animated gif to oim/recordings/, one frame per "
        "control step, showing that step's candidate rollouts and chosen "
        "plan against where the object actually was. With --plant mujoco, "
        "also films the simulator from the scene's own camera.",
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
    parser.add_argument(
        "--no-jit",
        action="store_true",
        help="Run eagerly, steppable in a debugger.",
    )


def _add_object_substeps_argument(parser: argparse.ArgumentParser) -> None:
    """Resolution of the MJX object rollout, identically for both worlds.

    A parameter of the MuJoCo prediction, not a mode of it, so it stays its
    own flag: no setting of it can produce an incoherent run the way an
    independent predict/execute pair could.

    Args:
        parser: The parser to extend.
    """
    parser.add_argument(
        "--object-substeps",
        type=int,
        default=1,
        help="Where --plant predicts with MuJoCo: MJX physics steps per "
        "planning step. 1 gives it the same coarse integration eq. 5 gets, "
        "which is the like-for-like setting; raise it to tell a modelling "
        "disagreement from an integration one.",
    )


def _add_3d_arguments(
    parser: argparse.ArgumentParser,
    experiment: Experiment,
    run: Dict[str, Any],
) -> None:
    """The 3D world's own flags: embodiment, backend, and what to draw.

    Args:
        parser: The parser to extend.
        experiment: The script's declaration, supplying the embodiments.
        run: The config's `run` block, holding the overlay defaults.
    """
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


def _add_2d_arguments(parser: argparse.ArgumentParser) -> None:
    """The 2D world's own flags: what the object block decides, and how.

    Args:
        parser: The parser to extend.
    """
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
    object_only = experiment.world == "object"

    parser = argparse.ArgumentParser(
        description=f"{experiment.world.upper()} experiment: "
        f"{experiment.scene or experiment.env or 'scene from --scene'}."
    )
    # Exactly one of these -- the three worlds' flag sets overlap by name
    # (`--record`, `--no-jit`) but not by meaning, so adding two would be
    # an argparse conflict rather than a merge.
    if object_only:
        _add_object_arguments(parser, experiment, cfg)
    elif three_d:
        _add_3d_arguments(parser, experiment, run)
    else:
        _add_2d_arguments(parser)
    if three_d or object_only:
        # Initial conditions, not sampler settings: varying the seed
        # redraws the planner's noise but leaves the problem identical,
        # while these change where the object starts and where it must go.
        # Unset means "draw one", so a sweep varies them without an axis.
        # The object world reads the same files, so an object-only run and
        # an ADMM run can be pointed at the identical problem instance.
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
        # There is no robot block in the object world, so `--samples` *is*
        # the object block's count and must resolve by the rule ADMM uses
        # for it -- otherwise a bare object-only run studies a
        # differently-sized block than the sweeps it is compared against.
        default=(
            object_sample_count(smp, smp["num_samples"])
            if object_only
            else smp["num_samples"]
        ),
        help="Rollouts for the object block. Defaults to "
        "sampler.object.num_samples, else sampler.num_samples."
        if object_only
        else "Rollouts per sub-optimizer.",
    )
    if not object_only:
        parser.add_argument(
            "--object-samples",
            type=int,
            default=None,
            help="Rollouts for the ADMM object block alone; --samples then "
            "applies to the robot block only. Unset reads "
            "sampler.object.num_samples, then falls back to --samples. An "
            "object rollout integrates a 3-vector in closed form, so it is "
            "orders cheaper than a robot rollout through MJX. Still nearly "
            "free under --plant mujoco, where the object block is "
            "latency-bound rather than throughput-bound: measured flat from "
            "64 to 512 samples, so raise it there rather than lowering it, "
            "and spend --horizon/--n-admm instead.",
        )
    parser.add_argument(
        "--horizon",
        type=int,
        default=smp["horizon"],
        help="Planning horizon H."
        if object_only
        else "Consensus horizon H. Shared by both blocks -- z and the "
        "duals are (H, dim), so they cannot disagree about it.",
    )
    if object_only:
        # No consensus and so no algorithm subcommand to hang these off;
        # every other world declares them once per subparser.
        parser.add_argument("--steps", type=int, default=run["steps"])
        parser.add_argument("--seed", type=int, default=run["seed"])
        return parser

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
        # On the admm subparser, not the shared 3D group: a flat baseline
        # has no object block to choose dynamics for, and the 2D world has
        # no MJX scene to offer as the alternative.
        admm.add_argument(
            "--plant",
            choices=["analytic", "mujoco"],
            default="analytic",
            help="Which dynamics the object block plans against. This world "
            "always executes in MuJoCo, so unlike the object-only world "
            "there is no execution side to pick and no 'model-error' mode: "
            "'analytic' (the default) already is one, our formulation "
            "planning and MuJoCo grading. 'mujoco' runs the object block "
            "through MJX in parallel with the robot block, so both predict "
            "with the engine the run is executed in. Not free: the object "
            "block costs ~0.89 ms per horizon step per pass, linear in "
            "--horizon and --n-admm and flat in --object-samples. See "
            "oim/runtime/object_mjx.py.",
        )
        _add_object_substeps_argument(admm)
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
            "--consensus",
            choices=["wrench", "pose"],
            default=adm.get("consensus_variable", "wrench"),
            help="What the two blocks agree on: the contact wrench "
            "(paper eq. 24) or the object's SE(2) pose trajectory.",
        )
        admm.add_argument(
            "--rho-torque",
            type=float,
            default=adm.get("rho_torque", 10.0),
            help="Separate initial penalty on the wrench's torque "
            "component, split from --rho (the force penalty). Reads "
            "admm.rho_torque from the robot yaml when unset; falls back "
            "to 10.0. Found to improve both position and orientation "
            "error in most scenes, so it is the default rather than an "
            "opt-in flag.",
        )
    admm.add_argument("--n-admm", type=int, default=adm["n_admm"])
    admm.add_argument("--rho", type=float, default=adm["rho"])
    admm.add_argument("--gamma", type=float, default=adm["gamma"])
    # Both worlds: `PushT` and `PushT2D` implement it identically, and the
    # analytic world is where the formulation is meant to be checked (see
    # README_ADMM.md). 3D additionally drives the `local_goal` ghost marker.
    admm.add_argument(
        "--local-goal",
        action="store_true",
        default=adm.get("local_goal", False),
        help="Robot block tracks the object block's horizon endpoint "
        "x^{o*}_H instead of the global goal (ell_o and the terminal term "
        "only; the shaping fade stays on the global goal).",
    )
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
    # Fields that only mean something where there are two blocks to
    # coordinate. The object world has one, so recording them there would
    # put a column of `null` in every run file and invite a table to group
    # on it.
    consensus = (
        {}
        if experiment.world == "object"
        else dict(
            # Resolved, not the raw flag: a `None` here would mean "read
            # whichever config happened to be current", which is exactly
            # what a run file exists to pin down. Only meaningful for
            # ADMM -- a flat baseline has no object block.
            object_samples=(
                object_sample_count(
                    args.cfg["sampler"],
                    args.samples,
                    getattr(args, "object_samples", None),
                )
                if object_opt is not None
                else None
            ),
            plant=(
                getattr(args, "plant", None)
                if object_opt is not None
                else None
            ),
            n_admm=getattr(args, "n_admm", None),
            rho=getattr(args, "rho", None),
            rho_torque=getattr(args, "rho_torque", None),
            gamma=getattr(args, "gamma", None),
            consensus_alpha=getattr(args, "consensus_alpha", None),
            consensus_variable=getattr(args, "consensus", None),
            local_goal=getattr(args, "local_goal", None),
        )
    )
    # Likewise the rollout backend and the viewer mode: neither changes
    # what the planner is asked to do, but both change what it actually
    # does (Warp and MJX-JAX physics differ in contact handling; the viewer
    # seeds `init_params` differently than --headless), so two otherwise-
    # identical runs are not comparable without them. Learned the hard way:
    # an interactive 0.025 m run and a headless 0.688 m run of the "same"
    # configuration. The object world steps neither, so it records neither.
    execution = (
        {}
        if experiment.world == "object"
        else dict(
            backend="warp" if getattr(args, "warp", False) else "jax",
            interactive=not getattr(args, "headless", False),
        )
    )
    save_run(
        experiment.results_dir(),
        name,
        run=dict(
            world=experiment.world,
            task=experiment.task_id(robot, getattr(args, "scene", None)),
            robot=robot,
            algorithm=algorithm,
            robot_opt=robot_opt,
            object_opt=object_opt,
            seed=args.seed,
            start_index=getattr(args, "start_index", None),
            goal_index=getattr(args, "goal_index", None),
            **execution,
        ),
        hyperparameters=dict(
            config=args.config_name,
            steps=args.steps,
            samples=args.samples,
            horizon=args.horizon,
            **consensus,
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
            object_samples=args.object_samples,
            seed=args.seed,
            robot_opt=args.robot_opt,
            object_opt=args.object_opt,
            n_admm=args.n_admm,
            rho=args.rho,
            gamma=args.gamma,
            consensus_alpha=args.consensus_alpha,
            rho_torque=args.rho_torque,
            consensus_variable=args.consensus,
            plant=args.plant,
            object_substeps=args.object_substeps,
            local_goal=args.local_goal,
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
        local_goal=args.local_goal,
    )
    print(f"scenario: {scenario.name} -- {scenario.description}")
    ctrl, params = build_admm_2d(
        task,
        horizon=args.horizon,
        num_samples=args.samples,
        object_samples=args.object_samples,
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


def _mujoco_recording(
    args: argparse.Namespace, plant: Any, base_name: str
) -> Tuple[Any, Any]:
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
    # Keyed on what *executes*, not on the mode: `model-error` executes in
    # MuJoCo and so has a real `MjModel` to film, even though it predicts
    # with eq. 5.
    if not (args.record and resolve_plant(args.plant)[1] == "mujoco"):
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


def _run_object(experiment: Experiment, args: argparse.Namespace) -> None:
    """One object-only run: the object block alone, on the chosen plant."""
    run_cfg = args.cfg["run"]
    scene = args.scene
    # The same pose files the 3D runs draw from, so an object-only run and
    # an ADMM run can be pointed at the identical problem instance. Unlike
    # 3D, unset means "the scene's own" rather than "draw one": there is no
    # robot to be in the way, so a fixed instance is the useful default.
    start = goal = None
    poses = load_poses(scene)
    args.start_index = args.goal_index = None
    if poses is not None and (args.start or args.goal):
        rng = np.random.default_rng(args.seed)
        args.start_index, start = poses.select("start", args.start, rng)
        args.goal_index, goal = poses.select("goal", args.goal, rng)
        print(f"poses: start {start} goal {goal}")

    task, block, params, obj_state0 = build_object_only(
        scene,
        args.robot,
        args.cfg,
        horizon=args.horizon,
        samples=args.samples,
        seed=args.seed,
        object_opt=args.object_opt,
        iterations=args.iterations,
        plant=args.plant,
        object_substeps=args.object_substeps,
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
        f"object block alone on {scene}, from "
        f"oim/configs/robots/{args.config_name}.yaml"
    )
    print(
        f"  {args.object_opt}, H={args.horizon}, {args.samples} samples, "
        f"{args.iterations} pass(es)/step, {args.steps} steps max, "
        f"{args.plant} dynamics"
    )
    print(
        f"  w_rate={[float(v) for v in task.object_model.w_rate]}, "
        f"noise_level={block.optimizer.noise_level}, "
        f"temperature={getattr(block.optimizer, 'temperature', None)}, "
        f"project_gate={task.project_gate_pos}"
    )

    # Built from the same start/goal the task was, so the simulator's block
    # begins where the analytic one does and the two are comparable. The
    # *execution* half of the mode -- `build_object_only` already took the
    # prediction half, from the same `resolve_plant` table.
    plant = build_plant(
        resolve_plant(args.plant)[1],
        task,
        args.robot,
        args.cfg["world3d"],
        control_dt=float(task.dt),
        start=obj_state0,
        goal=goal,
        jit=not args.no_jit,
        friction=args.friction,
    )
    name = experiment.run_name(args.robot, args.object_opt, scene=scene)
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

    _save(
        experiment,
        args,
        name,
        task,
        log,
        algorithm=f"object_only_{args.plant}",
        robot=args.robot,
        robot_opt=None,
        object_opt=args.object_opt,
        # The object block plans and executes at the same rate, and there
        # is no replanning budget distinct from it.
        control_dt=float(task.dt),
        extra_static=dict(scene=scene, robot=args.robot, plant=args.plant),
        extra_hyper=dict(
            plant=args.plant,
            friction=args.friction,
            wrench_fraction=args.wrench_fraction,
            w_rate=[float(v) for v in task.object_model.w_rate],
            project_gate=task.project_gate_pos,
            noise_level=block.optimizer.noise_level,
            temperature=getattr(block.optimizer, "temperature", None),
        ),
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

    if experiment.world == "object":
        # No consensus, so no algorithm subcommand and no `args.algorithm`.
        if args.w_rate is not None and len(args.w_rate) not in (1, 3):
            parser.error("--w-rate takes 1 value or 3 (fx fy tau)")
        args.scene = experiment.scene or args.scene
        _run_object(experiment, args)
        return

    if args.algorithm is None:
        args.algorithm = "admm"
    if experiment.world == "2d":
        _run_2d(experiment, args)
    else:
        _run_3d(experiment, args)
