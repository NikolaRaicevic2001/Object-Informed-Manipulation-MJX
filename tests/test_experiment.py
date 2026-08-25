"""Tests for the per-task scripts and the pipeline they share.

`examples/` scripts are declarations: what they are worth is that every one
gets the same runner, the same run-file fields, and a CLI that advertises
only what its world can honour. These check exactly that, without running a
simulation -- construction and the closed loop are covered elsewhere.
"""

import argparse
import ast
import importlib.util
import os
import pathlib
from typing import Any, Dict, List

import pytest

from oim.experiment import Experiment, build_parser, config_name, load_config
from oim.run_launch import build_command, expand, script_path
from oim.utils.scenes import SCENES

EXAMPLES = pathlib.Path(__file__).resolve().parents[1] / "examples"

# The scripts that declare an Experiment, i.e. the push tasks. They live
# in `examples/pusht/`; `examples/demos/` holds the unrelated inherited
# single-optimizer demos (cart_pole, walker, ...). Found by content rather
# than by directory, so a task script filed in the wrong place is still
# held to the contract below instead of quietly escaping it.
TASK_SCRIPTS = sorted(
    p.stem
    for p in EXAMPLES.glob("**/*.py")
    if "EXPERIMENT = Experiment(" in p.read_text()
)


def _is_docstring(node: ast.stmt) -> bool:
    """Is this top-level node the module docstring?"""
    return isinstance(node, ast.Expr) and isinstance(node.value, ast.Constant)


def _script(name: str) -> pathlib.Path:
    """The one `examples/**` script with this bare name."""
    (found,) = EXAMPLES.glob(f"**/{name}.py")
    return found


def _load(name: str) -> Any:
    """Import one `examples/` script by name."""
    spec = importlib.util.spec_from_file_location(
        f"_example_{name}", _script(name)
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _flags(parser: argparse.ArgumentParser) -> Dict[str, List[str]]:
    """`{"top": [...], "<algorithm>": [...]}` of every flag dest."""
    out: Dict[str, List[str]] = {"top": []}
    for action in parser._actions:
        if isinstance(action, argparse._SubParsersAction):
            for name, sub in action.choices.items():
                out[name] = [a.dest for a in sub._actions if a.dest != "help"]
        elif action.dest != "help":
            out["top"].append(action.dest)
    return out


def test_every_task_has_a_script() -> None:
    """Each 3D scene and 2D scenario is reachable as its own script."""
    declared = {}
    for name in TASK_SCRIPTS:
        exp = _load(name).EXPERIMENT
        declared[name] = exp.scene if exp.world == "3d" else exp.env
    missing = set(SCENES) - set(declared.values())
    assert not missing, f"scenes with no examples/ script: {missing}"


def test_scripts_are_declarations() -> None:
    """A task script stays a declaration, not a runner that drifted back.

    The whole point of `oim/experiment.py` is that recording, saving and
    the closed loop are written once. A script that grows its own would
    silently stop sharing them.
    """
    for name in TASK_SCRIPTS:
        tree = ast.parse(_script(name).read_text())
        body = [n for n in tree.body if not _is_docstring(n)]
        # An import, the declaration and the __main__ guard.
        assert len(body) <= 3, f"{name}.py has grown a body ({len(body)} nodes)"
        # Read off the syntax tree, not the source text: the docstring is
        # allowed to say `--plant mujoco`, and grepping the whole file for
        # "mujoco" cannot tell that from an import of it.
        code = ast.unparse(ast.Module(body=body, type_ignores=[]))
        for banned in ("save_run", "matplotlib", "argparse", "mujoco"):
            assert banned not in code, f"{name}.py should not touch {banned}"


@pytest.mark.parametrize("name", TASK_SCRIPTS)
def test_script_declares_a_valid_experiment(name: str) -> None:
    """`Experiment.__post_init__` accepted it, and it names real things."""
    exp = _load(name).EXPERIMENT
    assert exp.world in ("3d", "object")
    if exp.world == "3d":
        assert exp.scene in SCENES
        assert exp.robots == tuple(sorted(SCENES[exp.scene].mjcf_by_robot))
    elif exp.world == "object":
        # No MJCF of its own, so no scene until --scene supplies one; the
        # embodiment still selects the config and the scene variant.
        assert exp.scene is None
        assert exp.robots == ("xarm6", "point")
    else:
        assert exp.robots == ("disc",)


def test_experiment_rejects_a_scene_that_does_not_exist() -> None:
    """A typo in a script fails at import, not after the model loads."""
    with pytest.raises(ValueError, match="not in oim.utils.scenes"):
        Experiment(world="3d", scene="no_such_scene")
    with pytest.raises(ValueError, match="sets `scene`, not `env`"):
        Experiment(world="3d", env="gate")
    with pytest.raises(ValueError, match="world must be"):
        Experiment(world="1d")


def test_robot_choices_come_from_the_scene() -> None:
    """`--robot` offers exactly the embodiments with an MJCF, no more.

    Declaring them on the script instead would let it advertise an
    embodiment the scene has no model for, which fails only at load.
    """
    clutter = build_parser(Experiment(world="3d", scene="clutter"))
    robot = next(a for a in clutter._actions if a.dest == "robot")
    assert set(robot.choices) == {"point", "xarm6"}

    shelf = build_parser(Experiment(world="3d", scene="shelf_gap"))
    robot = next(a for a in shelf._actions if a.dest == "robot")
    # point-robot MJCFs were added for all 5 tabletop scenes (see
    # oim/models/xarm6_pusht_tabletop/*_point.xml and scenes.py's
    # mjcf_by_robot), so shelf_gap -- like clutter -- now supports both.
    assert set(robot.choices) == {"point", "xarm6"}


def test_no_flag_is_offered_that_its_world_would_ignore() -> None:
    """Each world advertises only what it can honour.

    `--no-plot` used to be accepted by 3D runs and disregarded, so a sweep
    asking for no figures still wrote one per cell.
    """
    three_d = _flags(build_parser(Experiment(world="3d", scene="shelf_gap")))
    obj = _flags(build_parser(Experiment(world="object")))

    # `--warp` picks an MJX backend the object world never steps;
    # `--object-samples` splits a budget it has only one block to spend.
    for flag in ("warp", "object_samples", "gamma0_deg"):
        assert flag in three_d["top"] and flag not in obj["top"]
    # The object block's own dynamics and tuning, which 3D takes on its
    # `admm` subcommand or not at all.
    for flag in ("plant", "friction", "consensus", "scene", "steps"):
        assert flag in obj["top"] and flag not in three_d["top"]
    # Shared, and honoured, in both.
    assert "record" in three_d["top"] and "record" in obj["top"]
    assert "no_plot" in three_d["top"] and "no_plot" in obj["top"]
    # One block, so no algorithm to choose: the object world has no
    # subcommands at all, where 3D carries `admm`'s consensus knobs.
    assert set(obj) == {"top"} and "headless" in three_d["admm"]


def test_flat_algorithms_take_no_consensus_knobs() -> None:
    """A flat baseline has no rho or n_admm to accept and then ignore."""
    flags = _flags(build_parser(Experiment(world="3d", scene="shelf_gap")))
    knobs = ("n_admm", "rho", "gamma", "robot_opt", "object_opt",
             "rho_torque")
    for algorithm in ("mppi", "ps"):
        for knob in knobs:
            assert knob not in flags[algorithm], f"{algorithm} takes {knob}"
    for knob in knobs:
        assert knob in flags["admm"]


def test_the_overlay_is_offered_to_every_3d_algorithm() -> None:
    """Every sampling-based method has samples and a chosen trajectory.

    The overlay used to be an ADMM-only subcommand flag, which left flat
    baselines with no way to draw the population they also sample.
    """
    parser = build_parser(Experiment(world="3d", scene="shelf_gap"))
    flags = _flags(parser)
    for flag in ("show_samples", "show_optimal"):
        assert flag in flags["top"], f"{flag} is not a top-level 3D flag"
        for algorithm in ("admm", "mppi", "ps"):
            assert flag not in flags[algorithm]
    for algorithm in ("admm", "mppi", "ps"):
        args = parser.parse_args(
            ["--show-samples", "--show-optimal", algorithm]
        )
        assert args.show_samples and args.show_optimal

    # The object world draws its own overlay, top level like 3D's.
    obj = _flags(build_parser(Experiment(world="object")))
    assert "show_samples" in obj["top"]


def test_there_is_no_config_flag() -> None:
    """The config is a property of the robot, not a separate choice."""
    flags = _flags(build_parser(Experiment(world="3d", scene="clutter")))
    assert "config" not in flags["top"]
    assert config_name("xarm6") == "xarm6"


def test_defaults_come_from_the_robots_own_config() -> None:
    """`--robot xarm6` picks up xarm6.yaml's step count, not the point's."""
    exp = Experiment(world="3d", scene="clutter")
    point = build_parser(exp, load_config("point")).parse_args(["admm"])
    xarm6 = build_parser(exp, load_config("xarm6")).parse_args(["admm"])
    assert point.steps == load_config("point")["run"]["steps"]
    assert xarm6.steps == load_config("xarm6")["run"]["steps"]
    assert point.steps != xarm6.steps


def test_task_id_is_what_run_eval_groups_on() -> None:
    """The identity string a run file carries, per world."""
    assert (
        Experiment(world="3d", scene="shelf_gap").task_id("xarm6")
        == "pusht3d_xarm6_shelf_gap"
    )
    assert (
        Experiment(world="object", scene="clutter").task_id("point")
        == "object_clutter"
    )


def test_filenames_name_the_task_that_produced_them() -> None:
    """A run's artifacts are stemmed with its own `task_id`.

    Naming them after the world alone left all four tabletop scenes
    writing `pusht3d_xarm6_admm_...`, distinguishable only by timestamp.
    """
    exp = Experiment(world="3d", scene="shelf_gap")
    admm = exp.run_name("xarm6", "admm", "mppi", "mppi")
    assert admm.stem == "pusht3d_xarm6_shelf_gap_admm_mppi_mppi"
    # A flat baseline has no sub-optimizers to name.
    assert exp.run_name("xarm6", "mppi").stem == "pusht3d_xarm6_shelf_gap_mppi"
    obj = Experiment(world="object").run_name("point", "mppi", scene="clutter")
    assert obj.stem == "object_clutter_mppi"


def test_every_task_stems_its_files_differently() -> None:
    """No two scripts write the same filename, run in the same second."""
    stems: Dict[str, str] = {}
    for name in TASK_SCRIPTS:
        exp = _load(name).EXPERIMENT
        # The object world takes its scene from --scene, so its stems are
        # only distinct once a scene is chosen -- which is the same claim,
        # made over the scenes it can be pointed at. Its identity omits the
        # embodiment (no robot is simulated), so it contributes one stem
        # per scene rather than one per scene and robot.
        object_only = exp.world == "object"
        scenes = sorted(SCENES) if object_only else [exp.scene]
        for scene in scenes:
            for robot in exp.robots[:1] if object_only else exp.robots:
                stem = exp.run_name(
                    robot, "admm", "mppi", "mppi", scene=scene
                ).stem
                assert stem.startswith(exp.task_id(robot, scene))
                assert stem not in stems, f"{name} collides with {stems[stem]}"
                stems[stem] = name


# ----------------------------------------------------------------------
# The sweep driver, which now addresses tasks by script
# ----------------------------------------------------------------------


def test_sweep_builds_a_command_for_the_named_script() -> None:
    """`{script: shelf_gap}` becomes `examples/pusht/shelf_gap.py ...`.

    The bare name is resolved against `examples/**`, so a sweep config
    does not track which subdirectory a script was filed in.
    """
    cell = {"task": {"script": "shelf_gap"}, "algorithm": "admm", "horizon": 16}
    cmd = build_command(cell, {"steps": 20, "headless": True})
    assert cmd[1].endswith(os.path.join("examples", "pusht", "shelf_gap.py"))
    assert "admm" in cmd
    # Top-level flags precede the algorithm, per-algorithm ones follow it.
    assert cmd.index("--horizon") < cmd.index("admm") < cmd.index("--steps")
    assert "--headless" in cmd


def test_sweep_extra_task_keys_become_flags() -> None:
    """`{script: clutter, robot: point}` passes `--robot point`."""
    cell = {
        "task": {"script": "clutter", "robot": "point"},
        "algorithm": "admm",
    }
    cmd = build_command(cell, {})
    assert cmd[cmd.index("--robot") + 1] == "point"


def test_sweep_drops_consensus_knobs_from_flat_cells() -> None:
    """`fixed:` applies to every cell, but a flat run has no rho.

    Without this the launcher would put `--rho` on an `mppi` command line,
    which the script now rejects rather than ignores. Same for `--plant`:
    it selects the object block's dynamics, and a flat baseline has none.
    """
    cell = {"task": {"script": "shelf_gap"}, "algorithm": "mppi"}
    cmd = build_command(
        cell,
        {"rho": 10.0, "n_admm": 8, "plant": "mujoco", "steps": 20},
    )
    assert "--rho" not in cmd and "--n-admm" not in cmd
    assert "--plant" not in cmd
    assert "--steps" in cmd
    admm = build_command(
        {"task": {"script": "shelf_gap"}, "algorithm": "admm"},
        {"plant": "mujoco"},
    )
    assert admm[admm.index("--plant") + 1] == "mujoco"


def test_sweep_can_turn_the_overlay_on_for_every_cell() -> None:
    """One `fixed:` block records trajectories for ADMM and flat alike.

    `--show-samples`/`--show-optimal` are top-level, so they land before
    the algorithm name and apply to whichever one the cell names.
    """
    fixed = {"show_samples": True, "show_optimal": True, "record": True}
    for algorithm in ("admm", "mppi"):
        cell = {"task": {"script": "shelf_gap"}, "algorithm": algorithm}
        cmd = build_command(cell, fixed)
        assert "--show-samples" in cmd and "--show-optimal" in cmd
        assert cmd.index("--show-samples") < cmd.index(algorithm)


def test_sweep_drops_admm_only_knobs_from_flat_cells() -> None:
    """`rho_torque` is an ADMM knob; a flat script rejects the flag.

    Every knob in `_ADMM_ONLY` reaches the flat command line unless it is
    dropped, which fails the cell at argparse rather than being ignored.
    """
    cell = {"task": {"script": "shelf_gap"}, "algorithm": "mppi"}
    cmd = build_command(cell, {"rho_torque": 0.2, "steps": 20})
    assert "--rho-torque" not in cmd
    admm = build_command(
        {"task": {"script": "shelf_gap"}, "algorithm": "admm"},
        {"rho_torque": 0.2},
    )
    assert admm[admm.index("--rho-torque") + 1] == "0.2"


def test_sweep_drops_flags_the_script_does_not_have() -> None:
    """One `fixed:` block can serve a mixed object/3D sweep.

    `object_only.py` has no `--headless`; dropping it is what lets both run
    from the same block. `oim/run_launch.py` prints which keys it dropped.
    """
    cell = {"task": {"script": "object_only"}, "algorithm": None}
    cmd = build_command(cell, {"headless": True, "steps": 20})
    assert "--headless" not in cmd
    assert "--steps" in cmd


def test_unknown_script_is_rejected_with_the_available_ones() -> None:
    """A typo names the alternatives rather than failing per cell."""
    with pytest.raises(ValueError, match=r"no examples/\*\*/nope.py"):
        script_path("nope")


def test_pose_flags_are_3d_only_and_default_to_random() -> None:
    """`--start`/`--goal` pick an initial condition, not a sampler knob.

    Unset means "draw one", so re-running a task varies the problem rather
    than only the noise. 2D scenarios carry their own start/goal, so the
    flags would have nothing to read.
    """
    parser = build_parser(Experiment(world="3d", scene="shelf_gap"))
    flags = _flags(parser)
    for flag in ("start", "goal"):
        assert flag in flags["top"]
    args = parser.parse_args(["admm"])
    assert args.start is None and args.goal is None
    args = parser.parse_args(["--start", "3", "--goal", "2", "admm"])
    assert (args.start, args.goal) == ("3", "2")

    obj = _flags(build_parser(Experiment(world="object")))
    assert "start" in obj["top"] and "goal" in obj["top"]


def test_poses_are_sweepable_axes() -> None:
    """A pose sweep expands; it used to collapse to one cell in silence."""
    combos = expand({
        "task": [{"script": "shelf_gap"}],
        "algorithm": ["admm"],
        "start": ["1", "2", "3"],
        "goal": ["1", "2"],
    })
    assert len(combos) == 6
    pairs = {(c["start"], c["goal"]) for c in combos}
    assert len(pairs) == 6


def test_an_unsweepable_axis_is_rejected() -> None:
    """`_AXES` was a silent whitelist: a typo ran as a single cell."""
    with pytest.raises(ValueError, match="not sweepable"):
        expand({"task": [{"script": "shelf_gap"}], "sed": [1, 2, 3]})
