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
# Read here rather than from parsed arguments: XLA fixes its allocator when
# the GPU backend first initializes, which happens long before main().
# `setdefault` so an explicit setting in the environment still wins.
#
# `run.warp` in a robot config has to be honoured the same way, which means
# reading yaml before `import jax` below -- hence the hand-rolled peek
# rather than `load_config`, which lives past that import. The robot may be
# implicit, so an unqualified run checks every config: preallocation is a
# throughput optimization, and disabling it for a run that turned out not
# to need Warp costs speed, while leaving it on for one that does costs the
# run.
def _warp_preallocation_hook() -> None:
    """Turn JAX preallocation off if this run will use Warp."""
    if "--no-warp" in sys.argv:
        return
    wants = "--warp" in sys.argv
    if not wants:
        robot = None
        for i, a in enumerate(sys.argv):
            if a == "--robot" and i + 1 < len(sys.argv):
                robot = sys.argv[i + 1]
            elif a.startswith("--robot="):
                robot = a.split("=", 1)[1]
        import yaml as _yaml  # noqa: PLC0415

        # `os.path.dirname(__file__)`, not `oim.ROOT`: importing `oim` here
        # would pull the package in before the allocator setting lands.
        here = os.path.dirname(os.path.abspath(__file__))
        names = [robot] if robot else ["xarm6", "point"]
        for name in names:
            path = os.path.join(here, "configs", "robots", f"{name}.yaml")
            try:
                with open(path) as f:
                    cfg = _yaml.safe_load(f) or {}
            except OSError:
                continue
            if (cfg.get("run") or {}).get("warp"):
                wants = True
                break
    if wants:
        os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")


_warp_preallocation_hook()

import jax  # noqa: E402
import mujoco  # noqa: E402
import numpy as np  # noqa: E402
import yaml  # noqa: E402

from oim import ROOT  # noqa: E402
from oim.objects import wrap_angle  # noqa: E402
from oim.runtime.mjcf import named_camera  # noqa: E402
from oim.runtime.object_mjx import PREDICT_SUBSTEPS  # noqa: E402
from oim.runtime.overlay import (  # noqa: E402
    CONTACT_POINT_HEIGHT,
    PlanOverlay,
    contact_points_world,
    traces_for,
)
from oim.runtime.samplers import (  # noqa: E402
    SUB_OPTIMIZERS,
    object_sample_count,
)
from oim.runtime.video import OffscreenRecorder  # noqa: E402
from oim.runtime.viewer import run_interactive  # noqa: E402
from oim.utils.plotting import plot_run_3d, plot_run_object  # noqa: E402
from oim.utils.poses import load_poses  # noqa: E402
from oim.utils.results import RunName, save_run  # noqa: E402
from oim.utils.scenes import SCENES  # noqa: E402
from oim.worlds.object_only import (  # noqa: E402
    build_object_only,
    build_plant,
    check_action_budget,
    run_object,
)
from oim.worlds.object_only.plant import (  # noqa: E402
    PLANT_MODES,
    resolve_plant,
)
from oim.worlds.sim3d.build import build_admm_3d, build_flat_3d  # noqa: E402
from oim.worlds.sim3d.run import run_3d_admm, run_3d_plain  # noqa: E402

CONFIG_DIR = os.path.join(ROOT, "configs", "robots")
RECORDINGS_DIR = os.path.join(ROOT, "recordings")
RUNS_DIR = os.path.join(ROOT, "results", "runs")

# Replanning period, shared by every 3D run so a flat baseline and an ADMM
# run are graded on the same control rate.
CONTROL_DT = 0.05

# Object-world recording and figure geometry. Constants rather than config
# keys or flags: nothing about a result depends on them, and they were four
# more knobs to read past in every file. `oim.worlds.sim3d.run` keeps its
# own defaults for the same reason.
OBJECT_VIDEO_FPS = 15
OBJECT_VIDEO_SIZE = (720, 480)
# Footprints drawn in the summary PNG, whatever the run length: at 1000
# steps a fixed stride of 5 would draw 200 and they merge into one blob.
OBJECT_PLOT_FOOTPRINTS = 40


@dataclass(frozen=True)
class Experiment:
    """What one `examples/` script declares -- and nothing more.

    Everything else (embodiments, goal, obstacles, defaults) is looked up
    from the registries this names, so a task cannot drift out of sync with
    the scene it claims to run.

    Args:
        world: `"3d"` (MJX contact, robot and object) or `"object"`
            (the object block alone, no robot).
        scene: 3D and object worlds -- a key of `oim.utils.scenes.SCENES`.
            The object world may leave it `None`, which means the CLI's
            `--scene` supplies it: that world has no MJCF and no
            embodiment of its own, so a scene there is only a choice of
            goal, obstacles and object physics, and one script with a flag
            says everything five near-identical files would.

    Raises:
        ValueError: If the world and the named registry disagree.
    """

    world: Literal["3d", "object"]
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
        else:
            raise ValueError(
                f"world must be '3d' or 'object', got {self.world!r}"
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
    """Which config file an embodiment reads."""
    return robot


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
    adm, run, smp = cfg["admm"], cfg["run"], cfg["sampler"]
    if experiment.scene is None:
        parser.add_argument(
            "--scene",
            choices=sorted(SCENES),
            default=run.get("scene", "clutter"),
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
        default=adm.get("plant", "analytic"),
        help="Which dynamics this run uses, predicting AND executing. "
        "'analytic' is the limit surface (eq. 5) on both sides -- an upper "
        "bound on the formulation with a perfect model. 'mujoco' is the "
        "simulator on both sides: the block plans through MJX and executes "
        "in MuJoCo. Both are self-consistent -- the planner is never given "
        "a model that differs from the plant. One flag rather than two, so "
        "neither mixed pair can be asked for.",
    )
    parser.add_argument(
        "--friction",
        choices=["box", "cone", "wrench"],
        default=adm.get("friction", "box"),
        help="--plant mujoco only: the shape of the simulated support "
        "friction. 'box' is MuJoCo's own per-DoF frictionloss and is the "
        "default because it is measurably the closest to eq. 5 in closed "
        "loop. 'cone' is the coupled ellipsoid eq. 5 assumes -- it fixes "
        "the breakaway threshold exactly and still agrees worse overall; "
        "'wrench' is eq. 5's own force balance and diverges outright. See "
        "oim/worlds/object_only/plant.py.",
    )
    _add_object_substeps_argument(
        parser, cfg["world3d"].get("object_substeps")
    )
    parser.add_argument(
        "--object-opt",
        choices=SUB_OPTIMIZERS,
        default=adm["object_opt"],
        help="Sampling optimizer for the object block.",
    )
    parser.add_argument(
        "--consensus",
        choices=["wrench", "contact_point", "object_pose"],
        default=adm.get("consensus", "wrench"),
        help="What the two blocks would agree on: the contact wrench "
        "[f_x, f_y, tau] (paper eq. 24); the contact point "
        "[p_x, p_y, lambda] -- where on the object's boundary to push, "
        "in its body frame, and how hard along the inward normal, with "
        "w = J_c^T f derived at the current pose each step; or the "
        "object pose [x, y, yaw], where the block still samples wrenches "
        "and the pose is what eq. 5 produces from them. There is no "
        "consensus to reach in this world (rho = 0, one block), so the "
        "first two select the block's action space -- which is the "
        "point: it isolates the parameterization from ADMM entirely, "
        "before it has to also agree with a robot. 'object_pose' changes "
        "no action space at all, so here it should match 'wrench'.",
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=smp.get("iterations", adm["n_admm"]),
        help="Optimizer passes per control step, from `sampler.iterations` "
        "(shared with the flat 3D baselines). ADMM gives its object block "
        "`n_admm` passes instead -- once per consensus round -- so this "
        "matching n_admm is what makes a bare object-only run the same "
        "experiment as the ADMM block rather than a quieter one.",
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
        "nets ~zero force and 2.0 is the measured best. Read under "
        "--consensus wrench and object_pose; see --contact-fraction for "
        "contact_point.",
    )
    parser.add_argument(
        "--contact-fraction",
        type=float,
        default=None,
        help="Ceiling on lambda under --consensus contact_point, as a "
        "fraction of the friction-cone limit mu*m*g. Unset takes "
        "costs.contact_fraction from the config, which itself falls back "
        "to wrench_fraction. Its own knob because the two bound different "
        "things: the wrench box's ceiling is fraction*sqrt(3) on a "
        "coupled 3-channel norm, while lambda is a single normal force "
        "that must clear breakaway alone.",
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
        "--w-contact-rate",
        type=float,
        nargs="+",
        metavar="W",
        default=None,
        help="The same penalty in the contact parameterization's units, "
        "read only under --consensus contact_point. One value for all "
        "channels, or three as px py lambda, normalized by "
        "(r_body, r_body, f_max). A separate flag from --w-rate because "
        "the channels are metres and newtons there, not newtons and "
        "newton-metres, so one number cannot mean the same thing in both. "
        "This is the only term that knows relocating a contact is a real "
        "maneuver -- the block can teleport it between steps for free, "
        "while an arm has to retract, travel and re-approach.",
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
    # Mirrors the MuJoCo runs' --record/--show-samples/--show-optimal. The
    # trajectories live in the recording rather than in the summary figure:
    # one static frame carrying every step's horizon is unreadable.
    parser.add_argument(
        "--record",
        action="store_true",
        default=run.get("record", False),
        help="Film the simulator from the scene's own camera and write an "
        "mp4 to oim/recordings/, with each step's candidate rollouts and "
        "chosen plan composited into the frames captured during it. Needs "
        "a mode that EXECUTES in MuJoCo (--plant mujoco); "
        "--plant analytic has no scene to film and records nothing.",
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
        help="Do not overlay the candidates (smaller mp4).",
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
        "--show-contact-point",
        action="store_true",
        default=run.get("show_contact_point", False),
        help="Mark the contact point the block decided on as a red dot on "
        "the object in the recording, at each pose of its plan. Requires "
        "--consensus "
        "contact_point; ignored (with a warning) under a wrench "
        "consensus, where the decision is a wrench and has no place on "
        "the object to draw.",
    )
    parser.add_argument(
        "--no-show-contact-point",
        dest="show_contact_point",
        action="store_false",
        help="Do not mark the contact point.",
    )
    parser.add_argument(
        "--no-jit",
        action="store_true",
        default=run.get("no_jit", False),
        help="Run eagerly, steppable in a debugger.",
    )
    # `--no-jit` and `--no-plot` can now default to True from the config,
    # so each needs the negative form or a `true` there would be
    # unturnoffable from the command line. `--no-plot` itself is added by
    # `build_parser`, shared with the other worlds; its inverse is only
    # meaningful where the config supplies a default, which is here.
    # `default=SUPPRESS` on both, and it is load-bearing: a `store_false`
    # action carries an implicit default of True, and argparse lets the
    # FIRST action registered for a dest own that dest's default. `--plot`
    # is added here, before `build_parser` adds `--no-plot`, so without
    # SUPPRESS it would win and force `no_plot` True however the config
    # read. (`--jit` happens to be registered after `--no-jit` and so was
    # already harmless -- suppressed anyway, since that ordering is not a
    # property anything enforces.)
    parser.add_argument(
        "--jit",
        dest="no_jit",
        action="store_false",
        default=argparse.SUPPRESS,
        help="Compile, whatever `run.no_jit` says.",
    )
    parser.add_argument(
        "--plot",
        dest="no_plot",
        action="store_false",
        default=argparse.SUPPRESS,
        help="Write the summary figure, whatever `run.no_plot` says.",
    )


def _add_object_substeps_argument(
    parser: argparse.ArgumentParser, default: Optional[int] = None
) -> None:
    """Resolution of the MJX object rollout, identically for both worlds.

    A parameter of the MuJoCo prediction, not a mode of it, so it stays its
    own flag: no setting of it can produce an incoherent run the way an
    independent predict/execute pair could.

    Args:
        parser: The parser to extend.
        default: Value when the flag is absent -- the object world passes
            `world3d.object_substeps`. `None` keeps `PREDICT_SUBSTEPS`, so
            the 3D caller and any config without the key are unchanged.
    """
    parser.add_argument(
        "--object-substeps",
        type=int,
        default=PREDICT_SUBSTEPS if default is None else default,
        help="Where --plant predicts with MuJoCo: MJX physics steps per "
        f"planning step (default {PREDICT_SUBSTEPS}, where the object "
        "block's integration error against the executed model stops "
        "dominating). 1 gives it the same coarse integration eq. 5 gets, "
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
        default=run.get("warp", False),
        help="MuJoCo Warp rollouts instead of JAX. Defaults to "
        "run.warp in the robot config.",
    )
    parser.add_argument(
        "--no-warp",
        dest="warp",
        action="store_false",
        # SUPPRESS, not a default: argparse gives a dest's default to the
        # FIRST action registered for it, and `store_false` carries an
        # implicit `True`. Without this, `run.warp: false` would parse as
        # True.
        default=argparse.SUPPRESS,
        help="JAX rollouts, overriding run.warp.",
    )
    parser.add_argument(
        "--record",
        action="store_true",
        default=run.get("record", False),
        help="Write an mp4 to oim/recordings/ (needs ffmpeg). Defaults to "
        "run.record in the robot config.",
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
    parser.add_argument(
        "--show-contact-point",
        action="store_true",
        default=run.get("show_contact_point", False),
        help="Mark the agreed contact point on the object as a red dot "
        "at each planned pose. Requires --consensus contact_point; "
        "ignored (with a warning) under a wrench consensus, where there "
        "is no contact point to draw.",
    )
    parser.add_argument(
        "--no-show-contact-point",
        dest="show_contact_point",
        action="store_false",
        help="Do not mark the agreed contact point.",
    )


def _add_headless(parser: argparse.ArgumentParser, run: Dict[str, Any]) -> None:
    """`--headless`/`--no-headless`, defaulting to `run.headless`.

    Registered per subcommand rather than once at top level because the
    viewer is what a subcommand runs, and a flat baseline and ADMM reach
    it by different paths.

    Args:
        parser: The subparser to extend.
        run: The config's `run` block, for the default.
    """
    parser.add_argument(
        "--headless",
        action="store_true",
        default=run.get("headless", False),
        help="No viewer: run --steps steps and save a run file. "
        "Defaults to run.headless in the robot config.",
    )
    parser.add_argument(
        "--no-headless",
        dest="headless",
        action="store_false",
        # See `--no-warp`: the first action registered for a dest owns its
        # default, and `store_false`'s implicit one is True.
        default=argparse.SUPPRESS,
        help="Show the viewer, overriding run.headless.",
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
                default=run.get(kind) if object_only else None,
                help=f"{kind.capitalize()} pose key from "
                f"examples/poses/<task>.yaml, or 'random' "
                f"(default: random).",
            )
    parser.add_argument(
        "--no-plot",
        action="store_true",
        default=object_only and run.get("no_plot", False),
        help="Skip the summary figure.",
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

    parser.add_argument(
        "--gamma0-deg",
        type=float,
        default=None,
        help="Override costs.gamma0_deg (alignment cone half-angle) for "
        "this run only, unset keeps the config's. Added because a single "
        "xarm6.yaml value traded scenes against each other: 45 degrees "
        "took single_obstacle from 29%% to 57%% but dropped open_table "
        "from 71%% to 0%% (7-rep seed-0 measurements, see Tasks.md) -- "
        "no single global value serves every scene, and this codebase "
        "has one shared config per robot, not per scene.",
    )

    subparsers = parser.add_subparsers(dest="algorithm")
    if three_d:
        # Flat baselines have no consensus, so they do not take --n-admm,
        # --rho or --gamma. They used to accept and ignore them, which put
        # a value in the command line that never reached the algorithm.
        for name, helptext in (
            ("ps", "Predictive sampling"),
            ("mppi", "MPPI"),
            ("c3", "Contact-implicit MPC (Push Anything / C3+)"),
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
            _add_headless(sp, run)
            if name == "mppi":
                # xarm6 flat MPPI only -- oim.algs.mppi.MPPI's
                # `task_space_noise` mechanism (see Tasks.md, "Phase 14").
                # Not a `sampler.mppi:` config key: that block is shared
                # with ADMM's own robot-block MPPI, which never gets a
                # real per-step Jacobian (only oim.worlds.sim3d.run's
                # _run_plain computes one) -- defaulting this on there
                # would silently zero ADMM's exploration noise. CLI-only,
                # applied directly to the built controller in _run_3d,
                # same reason --gamma0-deg is a flag and not a config key.
                sp.add_argument(
                    "--task-space-noise",
                    type=float,
                    nargs=5,
                    default=None,
                    metavar=(
                        "SIGMA_X", "SIGMA_Y", "SIGMA_Z",
                        "SIGMA_TILT_X", "SIGMA_TILT_Y",
                    ),
                    help="Enable task-space exploration noise (tip x/y/z "
                    "velocity + 2 tilt-rate components, mapped to joint "
                    "velocities via the tip's damped-inverse Jacobian) "
                    "instead of the per-joint noise_level scheme. Unset "
                    "(default) leaves the per-joint scheme exactly as "
                    "configured. First-tested values: 0.15 0.15 0.02 "
                    "0.05 0.05 -- underperformed the per-joint baseline "
                    "on position tracking in that first round, see "
                    "Tasks.md before assuming these are good.",
                )
                sp.add_argument(
                    "--task-space-alpha",
                    type=float,
                    default=2.0,
                    help="Feedback gain (1/s) pulling tip height toward "
                    "tip_target_z and tilt toward vertical, under "
                    "--task-space-noise. Ignored otherwise.",
                )
                sp.add_argument(
                    "--task-space-damping",
                    type=float,
                    default=1e-4,
                    help="Damping term in the tip Jacobian's pseudo-"
                    "inverse, under --task-space-noise. Ignored "
                    "otherwise.",
                )

    admm = subparsers.add_parser(
        "admm", help="ADMM-coordinated object-informed MPPI"
    )
    if three_d:
        # On the admm subparser, not the shared 3D group: a flat baseline
        # has no object block to choose dynamics for.
        admm.add_argument(
            "--plant",
            choices=["analytic", "mujoco"],
            default="analytic",
            help="Which dynamics the object block plans against. This world "
            "always executes in MuJoCo, so unlike the object-only world "
            "there is no execution side to pick: 'analytic' (the default) "
            "is our formulation planning while MuJoCo grades. 'mujoco' "
            "runs the object block "
            "through MJX in parallel with the robot block, so both predict "
            "with the engine the run is executed in. Not free: the object "
            "block costs ~0.89 ms per horizon step per pass, linear in "
            "--horizon and --n-admm and flat in --object-samples. See "
            "oim/runtime/object_mjx.py.",
        )
        _add_object_substeps_argument(admm)
        # The robot block's counterpart. No default here: the shipped
        # value is per-config (`world3d.robot_substeps`), which this
        # overrides only when given, so the flag exists for A/B and
        # sweeps without editing a config.
        admm.add_argument(
            "--robot-substeps",
            type=int,
            default=None,
            help="MJX physics steps per planning step in the ROBOT "
            "rollout, overriding world3d.robot_substeps. The rollout "
            "still advances one planning_dt per step; only the contact "
            "integration inside it gets finer. 1 is the coarse single "
            "step, which under-predicts the object's motion.",
        )
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
            "--consensus",
            choices=["wrench", "contact_point", "object_pose"],
            default=adm.get("consensus", "wrench"),
            help="What the two blocks agree on: the contact wrench "
            "[f_x, f_y, tau] (paper eq. 24); the contact point "
            "[p_x, p_y, lambda] -- where on the object's boundary to "
            "push, in its body frame, and how hard along the inward "
            "normal; or the object pose [x, y, yaw], where the object "
            "block still samples wrenches and the robot block reads its "
            "A^r straight off the state with no force estimator.",
        )
        admm.add_argument(
            "--consensus-object-weight",
            type=float,
            default=adm.get("consensus_object_weight", 0.5),
            help="The object block's share w_o of the consensus update "
            "z <- w_o*(A^o + y_o) + (1 - w_o)*(A^r + y_r). 0.5 is the "
            "paper's plain average; above it, z tracks the object "
            "block's cleaner 3-DOF plan rather than a compromise with "
            "the robot's 5-DOF-sampled realization.",
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
    # 3D also drives the `local_goal` ghost marker from this.
    admm.add_argument(
        "--local-goal",
        action="store_true",
        default=adm.get("local_goal", False),
        help="Robot block tracks the object block's own plan instead of "
        "the global goal (ell_o and the terminal term only; the shaping "
        "fade stays on the global goal).",
    )
    admm.add_argument(
        "--local-goal-lookahead",
        type=float,
        default=adm.get("local_goal_lookahead", 0.0),
        help="With --local-goal: aim at the first planned pose this far "
        "[m] ahead of the object, re-picked every step, so the robot "
        "follows the plan's route and not only its endpoint. 0 keeps the "
        "endpoint.",
    )
    admm.add_argument("--seed", type=int, default=run["seed"])
    admm.add_argument("--steps", type=int, default=run["steps"])
    if three_d:
        _add_headless(admm, run)
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
    admm_fields = (
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
            consensus_object_weight=getattr(
                args, "consensus_object_weight", None
            ),
            consensus=getattr(args, "consensus", None),
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
            **admm_fields,
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
            consensus_object_weight=args.consensus_object_weight,
            rho_torque=args.rho_torque,
            consensus=args.consensus,
            plant=args.plant,
            object_substeps=args.object_substeps,
            robot_substeps=args.robot_substeps,
            local_goal=args.local_goal,
            local_goal_lookahead=args.local_goal_lookahead,
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
        # --task-space-noise: see build_parser's docstring on the flag
        # for why this is a post-build mutation rather than a
        # sampler_cfg key -- ctrl.use_task_space_noise is a plain `self.`
        # bool read inside sample_knots at trace time, so this must run
        # before jit_optimize's first call, which it does (build_flat_3d
        # just returned ctrl fresh, nothing has traced it yet).
        if (
            args.algorithm == "mppi"
            and args.robot == "xarm6"
            and getattr(args, "task_space_noise", None) is not None
        ):
            ctrl.use_task_space_noise = True
            ctrl.task_noise_level = jax.numpy.asarray(args.task_space_noise)
            ctrl.task_space_alpha = args.task_space_alpha
            ctrl.task_space_damping = args.task_space_damping
            print(
                f"task-space noise ON: sigma={args.task_space_noise} "
                f"alpha={args.task_space_alpha} "
                f"damping={args.task_space_damping}"
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
            show_contact_point=args.show_contact_point,
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
            # ADMM-only: a flat baseline has no consensus variable, so
            # there is no agreed contact point for it to draw.
            **(
                {"show_contact_point": args.show_contact_point}
                if is_admm
                else {}
            ),
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


def _mujoco_recording(
    args: argparse.Namespace,
    plant: Any,
    base_name: str,
    draw_contacts: bool = False,
    contact_height: float = CONTACT_POINT_HEIGHT,
) -> Tuple[Any, Any]:
    """A `(recorder, on_plan)` filming the MuJoCo plant, else `(None, None)`.

    Only a MuJoCo-executing plant owns a real scene to film; the analytic
    one is three numbers and records nothing at all (the caller warns).
    Frames are captured from the plant's own `MjData` at the physics rate,
    so playback is real time, and each control step's plans are handed to
    the overlay just before that step executes -- which is the window
    those frames fall in.

    Args:
        args: Parsed command line.
        plant: The plant, which must be a `MujocoPlant` to be filmed.
        base_name: Filename stem, shared with the run's plot and results.
        draw_contacts: Mark the contact point the block decided on, as a
            dot on the object at each pose of its plan. Already gated by
            the caller on the consensus variable being the contact point.
        contact_height: World z for those dots -- the task's
            `tip_target_z`, matching `run_3d_admm`, so the object world's
            mp4 and a 3D recording put them at the same height.

    Returns:
        The recorder (to close afterwards) and the per-step plan callback.
    """
    # Keyed on what *executes*, not on the mode, so this stays right if a
    # mode whose two halves differ is ever added back.
    if not (args.record and resolve_plant(args.plant)[1] == "mujoco"):
        return None, None

    os.makedirs(RECORDINGS_DIR, exist_ok=True)
    # The same overlay the 3D runners composite into their mp4s, with one
    # block instead of three, so candidates and the chosen plan read
    # identically across the two worlds. A contact-dot trace counts as a
    # block of its own (see `PlanOverlay`), hence the second slot.
    overlay = (
        PlanOverlay(
            horizon=args.horizon, max_blocks=2 if draw_contacts else 1
        )
        if (args.show_samples or args.show_optimal)
        else None
    )
    recorder = OffscreenRecorder(
        plant.mj_model,
        output_dir=RECORDINGS_DIR,
        base_name=base_name,
        target_fps=OBJECT_VIDEO_FPS,
        size=OBJECT_VIDEO_SIZE,
        camera=named_camera(plant.mj_model),
        overlay=overlay,
    )
    plant.attach_recorder(recorder)
    if overlay is None:
        return recorder, None

    def on_plan(plan, samples, contacts=None) -> None:  # noqa: ANN001
        """Hand this step's plans to the frames captured during it."""
        recorder.set_plans(
            traces_for(
                object_chosen=plan if args.show_optimal else None,
                object_samples=samples if args.show_samples else None,
                # Placed on the plan's own poses: p is body-frame, so it
                # only becomes a world point paired with the pose it was
                # decided for, which is what makes a dot track the same
                # material point as the object turns.
                contact_points=(
                    contact_points_world(plan, contacts, contact_height)
                    if draw_contacts and contacts is not None
                    else None
                ),
            )
        )

    return recorder, on_plan


def _fmt_pose(pose: Optional[Sequence[float]]) -> str:
    """An SE(2) pose as `x, y, theta` at fixed width, or the scene's own."""
    if pose is None:
        return "scene default"
    return "[" + ", ".join(f"{float(v):+.3f}" for v in pose) + "]"


def _print_object_header(
    args: argparse.Namespace,
    task: Any,
    block: Any,
    scene: str,
    start: Optional[Sequence[float]],
    goal: Optional[Sequence[float]],
) -> None:
    """One aligned block describing the run, before any of it happens.

    Everything the preamble used to print, in the order a reader wants it
    -- what problem, under what dynamics, with what block, at what tuning
    -- rather than in the order the code happened to construct it. The
    label column is what makes it scannable: the values move between runs,
    the labels do not.

    The config *file* is named rather than just the embodiment, because
    every default below came from it and `--robot point` quietly selects a
    file with none of the object block's own tuning in it.
    """
    title = f"object block alone -- {scene}, {args.config_name}.yaml"
    print(f"\n{title}\n{'-' * len(title)}")

    keys = ""
    if args.start_index is not None or args.goal_index is not None:
        keys = f"   (poses {args.start_index} -> {args.goal_index})"
    print(
        f"  problem   {_fmt_pose(start)}  ->  {_fmt_pose(goal)}{keys}"
    )

    predicts, executes = resolve_plant(args.plant)
    dynamics = f"{args.plant} -- predicts and executes with "
    dynamics += "eq. 5" if predicts == "analytic" else "MuJoCo"
    if executes == "mujoco":
        n = args.object_substeps
        dynamics += (
            f", friction {args.friction}, "
            f"{n} substep{'' if n == 1 else 's'}"
        )
    print(f"  dynamics  {dynamics}")

    print(
        f"  block     {args.object_opt} on {args.consensus}, H={args.horizon},"
        f" {args.samples} samples, {args.iterations} pass/step,"
        f" {args.steps} steps max"
    )
    # `check_action_budget` prints its own `budget` row here, plus any
    # reachability warning -- see `build_object_only(verbose=False)`.
    check_action_budget(task, plant=executes)

    rate = task.object_model.w_rate
    rate_name = "w_rate"
    if args.consensus == "contact_point":
        rate, rate_name = task._w_contact_rate, "w_contact_rate"
    # `float()` before formatting: these are float32 arrays, and printing
    # one raw gave `noise_level=0.20000000298023224`.
    sigma = np.atleast_1d(np.asarray(block.optimizer.noise_level))
    if args.consensus == "contact_point":
        # Per-channel and in real units after `object_noise_scale`, where
        # the mean of millimetres and newtons means nothing. Reported as
        # the reach it buys: sigma_p against the object's own radius.
        reach = sigma[0] / task._contact_reach
        noise = f"{1e3 * sigma[0]:.0f}mm ({reach:g} r_body)/{sigma[-1]:.1f}N"
    else:
        noise = f"{float(np.mean(sigma)):g}"
    temperature = getattr(block.optimizer, "temperature", None)
    print(
        f"  tuning    {rate_name} "
        f"{[round(float(v), 2) for v in rate]}, "
        f"noise {noise}, temperature {float(temperature):g}"
    )


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

    task, block, params, obj_state0 = build_object_only(
        scene,
        args.robot,
        args.cfg,
        horizon=args.horizon,
        samples=args.samples,
        seed=args.seed,
        object_opt=args.object_opt,
        iterations=args.iterations,
        consensus=args.consensus,
        plant=args.plant,
        object_substeps=args.object_substeps,
        wrench_fraction=args.wrench_fraction,
        contact_fraction=args.contact_fraction,
        w_rate=args.w_rate,
        w_contact_rate=args.w_contact_rate,
        noise_level=args.noise_level,
        temperature=args.temperature,
        goal=goal,
        start=start,
        # The budget line belongs inside the header below, not ahead of
        # it: it is a fact about this run, and printed from the builder it
        # arrived before anything had said which run it described.
        verbose=False,
    )
    _print_object_header(args, task, block, scene, start, goal)

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
    # Gated on the run, not just the flag: the dot is a place on the
    # object, which only the contact parameterization decides. Under a
    # wrench consensus the decision is [f_x, f_y, tau] and putting its
    # first two entries on the block would be a plausible-looking lie.
    # Warn rather than fail -- a sweep may set the flag once and vary
    # `consensus` across cells. Mirrors `run_3d_admm`'s own gate.
    draw_contacts = (
        args.show_contact_point and args.consensus == "contact_point"
    )
    if args.show_contact_point and not draw_contacts:
        print(
            f"  note      --show-contact-point ignored: needs --consensus "
            f"contact_point, got {args.consensus!r}"
        )
    if args.record and resolve_plant(args.plant)[1] != "mujoco":
        print(
            f"  note      --record has nothing to film under --plant "
            f"{args.plant}: it executes\n            with eq. 5, which has "
            f"no scene. Use --plant mujoco for an mp4."
        )
    recorder, on_plan = _mujoco_recording(
        args,
        plant,
        name(),
        draw_contacts=draw_contacts,
        # The height `w_z_tip` holds the tip at, so the dot marks a place
        # the tip is actually asked to reach. Same fallback `run_3d_admm`
        # uses for a scene without the attribute.
        contact_height=getattr(task, "tip_target_z", CONTACT_POINT_HEIGHT),
    )

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
                # Nothing reads the logged samples: the recording gets
                # each step's population through `on_plan`, while it is
                # still the current step's, and the run file excludes them
                # (`oim.utils.results._DYNAMIC_KEYS`). Keeping them cost
                # ~100 MB at 1000 steps / 128 samples / H=32 for a series
                # with no consumer.
                log_samples=False,
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
            # `_save` drops the ADMM consensus block for this world (one
            # block, nothing to coordinate), but this key is not about
            # coordination here -- it is the object block's own action
            # space, so a run file without it cannot tell two runs apart.
            consensus=args.consensus,
            wrench_fraction=args.wrench_fraction,
            contact_fraction=args.contact_fraction,
            # Resolved off the task, not off the flags: lambda's ceiling
            # in newtons, whichever of the two fractions (or neither, and
            # the config) ended up supplying it.
            contact_f_max=float(task._contact_f_max),
            w_rate=[float(v) for v in task.object_model.w_rate],
            # Resolved off the task, like `w_rate`: the run file records
            # what the run used, not whether a flag was passed.
            w_contact_rate=[
                float(v) for v in task._w_contact_rate
            ],
            noise_level=block.optimizer.noise_level,
            temperature=getattr(block.optimizer, "temperature", None),
        ),
    )

    if args.no_plot:
        return
    os.makedirs(RECORDINGS_DIR, exist_ok=True)
    # ~40 footprints regardless of run length: at --steps 1000 a fixed
    # stride of 5 draws 200 and they merge into one blob.
    steps_run = len(log["object_pose"]) - 1
    stride = max(1, steps_run // OBJECT_PLOT_FOOTPRINTS)
    plot_run_object(
        task,
        log,
        os.path.join(RECORDINGS_DIR, f"{name()}.png"),
        stride=stride,
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
    # --gamma0-deg only, not a general per-scene override mechanism: this
    # codebase has one costs: block per robot, shared by every scene that
    # robot has an MJCF for, and this is the one weight measured to need
    # different values per scene (see the flag's own help text). Copies
    # rather than mutates `cfg` -- `load_config` may cache/return a shared
    # dict, and mutating it in place would leak into any other caller that
    # holds the same reference within this process.
    if getattr(args, "gamma0_deg", None) is not None:
        args.cfg = {
            **cfg,
            "costs": {**cfg["costs"], "gamma0_deg": args.gamma0_deg},
        }
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
    _run_3d(experiment, args)
