"""Algorithm x task grid of the object trajectories a sweep actually drove.

One row per algorithm, one column per scene. A panel holds the measured
`object_pose` of every run in that cell, drawn as its trial number at
points along the path and coloured by time, over the scene's own backdrop
-- obstacles in yellow (the arm's base disc among them), the starts as
numbered green circles, and every goal the panel was aimed at as the
object's own outline, one colour per goal orientation.

A start is run once per goal orientation, so each green ring launches two
runs; the second is marked in Roman numerals so the two do not print the
same digit over each other.

Everything a panel draws is read off the runs and `oim.utils.scenes`,
never hardcoded, so the same call serves any set of scenes. A cell with no
runs is still drawn, backdrop and all: the grid is the experiment's shape,
not a report of which parts of it happen to have finished.

`eval_plots.py` is the other multi-run figure and stays separate -- it
plots aggregated step curves, where this plots raw paths in the plane.
matplotlib is imported inside the functions, as in `plotting.py`, so a
run with plotting off never pays for the import.
"""

from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from oim.objects import Box, Capsule, Circle, Polygon
from oim.utils.plotting import footprint_world, obstacle_outline
from oim.utils.scenes import SCENES

# Two physical starts are the SAME start when they are closer than this.
# The repeats measured on hardware land within ~1 cm of each other while
# distinct starts are ~10 cm apart, so 6 cm separates them with room on
# both sides.
_START_TOL = 0.06

# Radius of the green start ring, in metres. Big enough to hold a digit,
# small enough not to swallow the first few trajectory marks.
_START_RADIUS = 0.035

# At most this many marks per trajectory. A 300-step run drawn in full is
# an unreadable smear; this decimates it to a legible dotted path.
_MAX_MARKS = 55

_OBSTACLE_FACE = "#ffe14d"
_OBSTACLE_EDGE = "#d4a600"
# One per goal orientation, cycled. All in the red family so a goal
# still reads as a goal, far enough apart to tell the +90 and -90
# aimings of one scene from each other.
_GOAL_COLORS = ("#c62828", "#7b1f3a", "#e0574f", "#96402a")
_START_COLOR = "#1b7a2f"

# Two goal poses are the same goal when their yaws agree to this. The
# scenes here differ by 180 degrees, so it only has to beat the noise.
_YAW_TOL = 0.2

# Space between neighbouring panels, in inches -- the same distance
# between rows as between columns, whatever shape the cells come out.
_PANEL_GAP = 0.12

# Two recorded layouts are the SAME layout when every number in them
# agrees to this. Live calibration re-measures an obstacle to the
# millimetre at each launch; a scene parameter that was edited moves by
# centimetres, so 2 cm separates them.
_LAYOUT_TOL = 0.02


@dataclass(frozen=True)
class _Backdrop:
    """Everything a panel draws that is not a trajectory.

    Attributes:
        obstacles: Shapes to fill in yellow. The arm's own base keep-out
            disc is one of them -- it is as solid to the object as a
            cube is, and leaving it out drew a scene the object could
            have crossed.
        goals: Every goal pose the panel's runs were given, SE(2). More
            than one when a scene is run at several goal orientations
            (the T is aimed at +90 and -90 degrees), and all of them are
            drawn -- a panel showing one would be a panel half its runs
            were not aiming at.
        footprint: The pushed object's outline in its own body frame.
    """

    obstacles: Tuple[Any, ...]
    goals: Tuple[np.ndarray, ...]
    footprint: np.ndarray


def _shape_from_dict(spec: Dict[str, Any]) -> Any:
    """Rebuild an `oim.objects.sdf` shape from its serialized form.

    The inverse of `oim.utils.results._shape_to_dict`, so a run's recorded
    obstacles go back through the same `obstacle_outline` the live scenes
    use rather than growing a second drawing path.

    Args:
        spec: One entry of a run's `static.obstacles`.

    Returns:
        The shape.

    Raises:
        TypeError: If the payload names a shape this cannot rebuild.
    """
    kind = spec["type"]
    if kind == "circle":
        return Circle(np.asarray(spec["center"]), float(spec["radius"]))
    if kind == "box":
        return Box(
            np.asarray(spec["center"]),
            np.asarray(spec["half_extents"]),
            float(spec["angle"]),
        )
    if kind == "capsule":
        return Capsule(
            np.asarray(spec["a"]),
            np.asarray(spec["b"]),
            float(spec["radius"]),
        )
    if kind == "polygon":
        return Polygon(np.asarray(spec["vertices"]))
    raise TypeError(f"cannot rebuild shape {kind!r}")


def _layout(obstacles: Sequence[Dict[str, Any]]) -> Tuple[str, np.ndarray]:
    """A layout as `(shape signature, every number in it)`.

    The signature is what cannot be compared numerically -- two layouts
    holding different shapes are different, full stop. The numbers are
    compared with a tolerance instead, since a real scene re-measures its
    obstacles at every launch.
    """
    kinds = "|".join(str(o.get("type")) for o in obstacles)
    numbers: List[float] = []
    for shape in obstacles:
        for key in sorted(shape):
            value = shape[key]
            if key == "type":
                continue
            numbers.extend(
                np.ravel(np.asarray(value, dtype=float)).tolist()
            )
    return kinds, np.asarray(numbers, dtype=float)


def _majority_layout(
    task: str, runs: Sequence[Dict[str, Any]]
) -> Dict[str, Any]:
    """The run whose recorded obstacles the most runs of `task` shared.

    A panel draws ONE scene behind runs that need not have faced the same
    one: a scene parameter edited mid-session (the base keep-out went
    0.22 -> 0.12 m on 2026-09-13) splits a task's runs into layouts, and
    picking whichever file sorted first would draw a backdrop most of the
    panel never saw. Majority is at least the layout most of the panel
    ran against, and the disagreement is reported rather than hidden.

    Grouped by tolerance rather than by rounding, as `_trial_numbers`
    groups starts: a grid puts two runs 1 mm apart in different groups
    whenever they straddle one of its boundaries.
    """
    groups: List[Tuple[str, np.ndarray, List[Dict[str, Any]]]] = []
    for run in runs:
        kinds, numbers = _layout(run["static"].get("obstacles", ()))
        for group_kinds, group_numbers, members in groups:
            if kinds != group_kinds or numbers.shape != group_numbers.shape:
                continue
            # An obstacle-free scene compares as equal to another one:
            # `max` has no identity on an empty array to fall back on.
            gap = (
                0.0 if numbers.size == 0
                else float(np.max(np.abs(numbers - group_numbers)))
            )
            if gap <= _LAYOUT_TOL:
                members.append(run)
                break
        else:
            groups.append((kinds, numbers, [run]))

    ranked = sorted((g[2] for g in groups), key=len, reverse=True)
    if len(ranked) > 1:
        print(
            f"  [{task}] {len(ranked)} different obstacle layouts across "
            f"{len(runs)} runs ({'/'.join(str(len(g)) for g in ranked)}); "
            f"drawing the one {len(ranked[0])} of them used"
        )
    return ranked[0][0]


def _distinct_goals(
    runs: Sequence[Dict[str, Any]],
) -> Tuple[np.ndarray, ...]:
    """Every goal pose the runs were aimed at, one entry per distinct one.

    Ordered by yaw, descending, so the +90 degree goal comes before the
    -90 one and the drawing order does not depend on the file order.
    """
    goals: List[np.ndarray] = []
    for run in runs:
        goal = np.asarray(run["static"]["goal"], dtype=float)
        if not any(
            abs(goal[0] - g[0]) <= _LAYOUT_TOL
            and abs(goal[1] - g[1]) <= _LAYOUT_TOL
            and abs(goal[2] - g[2]) <= _YAW_TOL
            for g in goals
        ):
            goals.append(goal)
    return tuple(sorted(goals, key=lambda g: -g[2]))


def _backdrop(task: str, runs: Sequence[Dict[str, Any]]) -> _Backdrop:
    """The scene behind one panel.

    Read off a run when the panel has one -- a real scene calibrates its
    obstacles at launch, so what a run recorded is the layout that was
    actually pushed through, which the registry cannot know. Falls back to
    `oim.utils.scenes` so a panel with no runs is still a drawn scene
    rather than an empty box.

    Args:
        task: Scene name, a key of `SCENES`.
        runs: Every run of this task, across all algorithms. May be empty.

    Returns:
        The backdrop.

    Raises:
        KeyError: If `task` is neither in `SCENES` nor among the runs.
    """
    spec = SCENES.get(task)
    scene_outline = (
        np.asarray(
            spec.footprint_builder(**spec.footprint_kwargs).vertices,
            dtype=float,
        )
        if spec is not None
        else np.zeros((0, 2))
    )
    if runs:
        # Each field falls back to the registry independently: a run
        # written before that field existed still contributes its
        # trajectory instead of failing the whole figure.
        run = _majority_layout(task, runs)
        static = run["static"]
        outline = static.get("object_footprint_body")
        return _Backdrop(
            obstacles=tuple(
                _shape_from_dict(o) for o in static.get("obstacles", ())
            ),
            goals=_distinct_goals(runs),
            footprint=(
                scene_outline if outline is None
                else np.asarray(outline, dtype=float)
            ),
        )

    return _Backdrop(
        obstacles=tuple(spec.obstacles.shapes),
        goals=(np.asarray(spec.goal, dtype=float),),
        footprint=scene_outline,
    )


def _trial_numbers(
    runs: Sequence[Dict[str, Any]],
    labels: Optional[Sequence[str]] = None,
) -> Tuple[Dict[int, str], Dict[str, np.ndarray]]:
    """Label the distinct physical start poses.

    The trial a run belongs to is not in the run file -- `seed` is fixed
    across a hardware session -- so it is recovered from where the object
    started. Clustered over EVERY run in the figure, not per panel, so one
    physical start carries the same label in every cell, which is the only
    thing that makes two panels comparable by eye.

    The clusters are ORDERED by start position, so which start is which is
    a property of the layout and not of the order files were read in.
    `labels` then names them in that order -- the marks on the real table
    are not 1..N top to bottom, and a figure has to carry the numbering
    the bench actually uses or a reader cannot check a run against it.

    Args:
        runs: Every run the figure will draw.
        labels: One label per start, in position order. Falls back to
            "1".."N" when absent, or when it does not have one label per
            start found (which it says, rather than mislabelling).

    Returns:
        `(label_by_run_id, centre_by_label)`, the first keyed by
        `id(run)`.
    """
    starts = [
        np.asarray(r["dynamic"]["object_pose"][0][:2], dtype=float)
        for r in runs
    ]
    centres: List[np.ndarray] = []
    members: List[List[int]] = []
    for i, start in enumerate(starts):
        for k, centre in enumerate(centres):
            if float(np.linalg.norm(start - centre)) <= _START_TOL:
                members[k].append(i)
                centres[k] = np.mean([starts[j] for j in members[k]], axis=0)
                break
        else:
            centres.append(start)
            members.append([i])

    order = sorted(range(len(centres)), key=lambda k: tuple(centres[k]))
    if labels is not None and len(labels) != len(order):
        print(
            f"  [starts] {len(labels)} trial labels given for "
            f"{len(order)} starts found; numbering them 1..{len(order)}"
        )
        labels = None
    names = (
        [str(n) for n in range(1, len(order) + 1)] if labels is None
        else [str(x) for x in labels]
    )

    label_by_run: Dict[int, str] = {}
    centre_by_label: Dict[str, np.ndarray] = {}
    for name, k in zip(names, order, strict=True):
        centre_by_label[name] = centres[k]
        for i in members[k]:
            label_by_run[id(runs[i])] = name
    return label_by_run, centre_by_label


_ROMAN = (
    (1000, "M"), (900, "CM"), (500, "D"), (400, "CD"), (100, "C"),
    (90, "XC"), (50, "L"), (40, "XL"), (10, "X"), (9, "IX"), (5, "V"),
    (4, "IV"), (1, "I"),
)


def _roman(label: str) -> str:
    """`"4"` -> `"IV"`; anything not a plain number gets a prime instead.

    The point is only that the second pass over the starts is legible as
    the SAME trial and still tells itself apart from the first, which a
    numeral does and a repeated digit does not.
    """
    if not label.isdigit() or not 0 < int(label) < 4000:
        return f"{label}'"
    value, out = int(label), []
    for amount, numeral in _ROMAN:
        count, value = divmod(value, amount)
        out.append(numeral * count)
    return "".join(out)


def _goal_variants(
    runs: Sequence[Dict[str, Any]],
) -> Dict[int, int]:
    """Which goal orientation each run was aimed at, 0 for the first.

    Every start is run once per goal orientation, so a panel holds two
    runs from each green ring; giving both the same digit is what made
    them hard to tell apart. The variant index picks the numbering they
    are drawn with -- Arabic for the first goal, Roman for the second.

    Ordered by yaw, descending, so which orientation is "first" is a fact
    about the experiment and not about the order files were read in.
    """
    yaws: List[float] = []
    for run in runs:
        yaw = float(np.asarray(run["static"]["goal"], dtype=float)[2])
        if not any(abs(yaw - y) <= _YAW_TOL for y in yaws):
            yaws.append(yaw)
    order = sorted(yaws, reverse=True)

    variant: Dict[int, int] = {}
    for run in runs:
        yaw = float(np.asarray(run["static"]["goal"], dtype=float)[2])
        variant[id(run)] = min(
            range(len(order)), key=lambda k: abs(order[k] - yaw)
        )
    return variant


def _project(points: np.ndarray, frame: str) -> np.ndarray:
    """Map world (x, y) into the figure's axes.

    `"world"` draws x right and y up. `"paper"` draws y right and x down,
    which is the reading this hardware layout wants: the object travels
    along -y from start to goal, so it crosses the panel left to right
    instead of running off the bottom of a tall, narrow plot. The two
    swaps (axis order, then the inverted vertical) compose to a rotation,
    so nothing is mirrored -- an outline's handedness survives.
    """
    points = np.asarray(points, dtype=float)
    return points if frame == "world" else points[..., ::-1]


def _goal_yaws(backs: Sequence[_Backdrop]) -> List[float]:
    """Every distinct goal orientation in the figure, yaw descending.

    Figure-wide, not per panel, so one orientation keeps one colour
    across the grid and the legend written once holds everywhere.
    """
    yaws: List[float] = []
    for back in backs:
        for goal in back.goals:
            if not any(abs(goal[2] - y) <= _YAW_TOL for y in yaws):
                yaws.append(float(goal[2]))
    return sorted(yaws, reverse=True)


def _goal_color(yaw: float, yaws: Sequence[float]) -> str:
    """The colour of the goal at `yaw`, from its place in `yaws`."""
    index = min(range(len(yaws)), key=lambda k: abs(yaws[k] - yaw))
    return _GOAL_COLORS[index % len(_GOAL_COLORS)]


def _draw_backdrop(
    ax: Any, back: _Backdrop, frame: str, yaws: Sequence[float]
) -> None:
    """Obstacles, then every goal as the object's own outline.

    Neither success tolerance is drawn. The angular one sat almost on top
    of the goal at ~6 degrees and the positional one ringed it, and both
    read as clutter around a shape that is already exact: the outline IS
    the goal pose, and how close a run came to it is the trajectory's job
    to show.

    Args:
        ax: The panel.
        back: What to draw.
        frame: As `_project`.
        yaws: Every goal orientation in the figure, which fixes the
            colour each one is drawn in.
    """
    for shape in back.obstacles:
        xy = _project(obstacle_outline(shape), frame)
        ax.fill(
            xy[:, 0], xy[:, 1], facecolor=_OBSTACLE_FACE,
            edgecolor=_OBSTACLE_EDGE, linewidth=1.4, zorder=1.5,
        )

    if back.footprint.size == 0:
        return

    for goal in back.goals:
        color = _goal_color(float(goal[2]), yaws)
        poly = footprint_world(back.footprint, goal)
        xy = _project(np.vstack([poly, poly[:1]]), frame)
        ax.plot(
            xy[:, 0], xy[:, 1], "-", color=color, linewidth=2.2, zorder=3,
        )


def _draw_starts(
    ax: Any, centres: Dict[str, np.ndarray], frame: str
) -> None:
    """A labelled green ring at every start this panel was launched from."""
    from matplotlib.patches import Circle as _MplCircle  # noqa: PLC0415

    for label, centre in centres.items():
        xy = _project(centre, frame)
        ax.add_patch(
            _MplCircle(
                tuple(xy), _START_RADIUS, fill=False, edgecolor=_START_COLOR,
                linewidth=1.8, zorder=4,
            )
        )
        ax.text(
            xy[0], xy[1], label, color=_START_COLOR, fontsize=9,
            fontweight="bold", ha="center", va="center", zorder=4,
        )


def _draw_trajectory(
    ax: Any, run: Dict[str, Any], label: str, frame: str, cmap: Any
) -> None:
    """One run's measured path, as its trial label coloured by time."""
    poses = np.asarray(run["dynamic"]["object_pose"], dtype=float)
    if poses.size == 0:
        return
    last = len(poses) - 1
    idx = np.unique(
        np.linspace(0, last, min(len(poses), _MAX_MARKS)).round().astype(int)
    )
    xy = _project(poses[idx, :2], frame)
    for (px, py), step in zip(xy, idx, strict=True):
        ax.text(
            px, py, label, color=cmap(step / max(last, 1)), fontsize=7,
            ha="center", va="center", zorder=2,
        )


def _limits(
    backs: Sequence[_Backdrop],
    runs: Sequence[Dict[str, Any]],
    centres: Dict[str, np.ndarray],
    frame: str,
    margin: float = 0.06,
) -> Tuple[Tuple[float, float], Tuple[float, float]]:
    """One set of axis limits for the whole grid.

    Shared rather than per panel: two cells of the same row are two
    algorithms on one scene, and they are only comparable by eye if a
    centimetre is the same distance in both.
    """
    pts: List[np.ndarray] = []
    for back in backs:
        for shape in back.obstacles:
            pts.append(obstacle_outline(shape))
        for goal in back.goals:
            pts.append(goal[None, :2])
            if back.footprint.size:
                pts.append(footprint_world(back.footprint, goal))
    for run in runs:
        poses = np.asarray(run["dynamic"]["object_pose"], dtype=float)
        if poses.size:
            pts.append(poses[:, :2])
    for centre in centres.values():
        pts.append(centre[None, :] + _START_RADIUS * np.array(
            [[1.0, 1.0], [-1.0, -1.0]]
        ))

    xy = _project(np.vstack(pts), frame)
    lo, hi = xy.min(axis=0) - margin, xy.max(axis=0) + margin
    return (float(lo[0]), float(hi[0])), (float(lo[1]), float(hi[1]))


def _task_label(task: str) -> str:
    """`"single_obstacle_real"` -> `"Single Obstacle"`.

    The `_real` suffix distinguishes a hardware scene from its simulated
    twin in the registry; a figure has one or the other, never both, so
    it is noise on a panel title.
    """
    stem = task[: -len("_real")] if task.endswith("_real") else task
    return stem.replace("_", " ").title()


def _parse_algorithms(
    algorithms: Sequence[str],
) -> List[Tuple[str, str]]:
    """`"admm=CLOI"` -> `("admm", "CLOI")`; a bare name labels itself."""
    out = []
    for item in algorithms:
        key, _, label = item.partition("=")
        out.append((key, label or key.upper()))
    return out


def plot_trajectory_grid(
    runs: Sequence[Dict[str, Any]],
    path: str,
    tasks: Sequence[str],
    algorithms: Sequence[str],
    *,
    frame: str = "paper",
    colormap: str = "jet",
    task_labels: Optional[Dict[str, str]] = None,
    trial_labels: Optional[Sequence[str]] = None,
) -> str:
    """Draw the algorithm x task trajectory grid and save it.

    One row per algorithm, one column per scene: an algorithm's row reads
    as "this method, across the scenes", which is the comparison the
    figure exists to make.

    Args:
        runs: Loaded run payloads (`oim.run_eval.load_runs` output). Runs
            whose task or algorithm is not asked for are ignored, so the
            caller can hand over everything it loaded.
        path: Where to write the figure.
        tasks: Column order, as `run.task` values. A task with no runs is
            still given a column.
        algorithms: Row order, as `run.algorithm` values, optionally
            `key=Label` to label the row differently from the field
            (e.g. `admm=CLOI`).
        frame: `"paper"` (y right, x down) or `"world"` (x right, y up).
        colormap: Any matplotlib colormap; its cold end is the start of a
            trajectory and its warm end the most recent point.
        task_labels: Column titles, keyed by task. Defaults to
            `_task_label`.
        trial_labels: What the starts are called, in position order --
            the bench's own numbering, which need not run 1..N. See
            `_trial_numbers`.

    Returns:
        `path`.
    """
    import matplotlib  # noqa: PLC0415

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt  # noqa: PLC0415
    from matplotlib.ticker import MaxNLocator  # noqa: PLC0415

    rows = _parse_algorithms(algorithms)
    wanted: Dict[str, List[Dict[str, Any]]] = {t: [] for t in tasks}
    for run in runs:
        task = run["run"].get("task")
        if task in wanted and run["run"].get("algorithm") in dict(rows):
            wanted[task].append(run)

    drawn = [r for rs in wanted.values() for r in rs]
    numbers, centres = _trial_numbers(drawn, trial_labels)
    # Arabic for the first goal orientation, Roman for the second, so the
    # two runs launched from one ring do not print the same digit on top
    # of each other.
    variant = _goal_variants(drawn)
    marks = {
        key: (label if variant[key] == 0 else _roman(label))
        for key, label in numbers.items()
    }
    backs = {t: _backdrop(t, wanted[t]) for t in tasks}
    yaws = _goal_yaws(list(backs.values()))
    xlim, ylim = _limits(list(backs.values()), drawn, centres, frame)
    cmap = matplotlib.colormaps[colormap]

    # Panels sized to the DATA's aspect. `set_aspect("equal")` shrinks an
    # axes inside whatever box it is given, so any box of the wrong shape
    # comes back as padding INSIDE the grid, which is what opened the gap
    # between columns. Laying the figure out in inches instead makes each
    # cell exactly the shape of the data, and the only space left between
    # panels is `_PANEL_GAP`.
    #
    # That gap is set in INCHES, not in matplotlib's `wspace`/`hspace`,
    # which are fractions of the cell's own width and height -- equal
    # fractions of a 4.0 x 2.6 inch cell are not an equal gap. Dividing
    # each by its own dimension is what makes the rows and the columns
    # separate by the same distance.
    span_x, span_y = xlim[1] - xlim[0], ylim[1] - ylim[0]
    cell_w = 4.0
    cell_h = cell_w * span_y / span_x
    # Margins hold the tick labels and the row/column names, nothing
    # else: no colour bar, no legend, nothing under the panels.
    left, right, top, bottom = 0.62, 0.06, 0.34, 0.38     # inches
    fig_w = left + cell_w * len(tasks) + _PANEL_GAP * (len(tasks) - 1) + right
    fig_h = top + cell_h * len(rows) + _PANEL_GAP * (len(rows) - 1) + bottom
    fig, axes = plt.subplots(
        len(rows), len(tasks), squeeze=False, sharex=True, sharey=True,
        figsize=(fig_w, fig_h),
    )
    fig.subplots_adjust(
        left=left / fig_w, right=1.0 - right / fig_w,
        bottom=bottom / fig_h, top=1.0 - top / fig_h,
        wspace=_PANEL_GAP / cell_w, hspace=_PANEL_GAP / cell_h,
    )

    for col, task in enumerate(tasks):
        # Every panel shows every start, including a column with no runs
        # yet: the protocol launches the same physical starts at each
        # scene, so an empty column is a scene waiting for data rather
        # than a scene with one start.
        shown = {
            n: c for n, c in centres.items()
            if not wanted[task]
            or n in {numbers[id(r)] for r in wanted[task]}
        }
        for row, (key, label) in enumerate(rows):
            ax = axes[row][col]
            _draw_backdrop(ax, backs[task], frame, yaws)
            _draw_starts(ax, shown, frame)
            for run in wanted[task]:
                if run["run"].get("algorithm") == key:
                    _draw_trajectory(
                        ax, run, marks[id(run)], frame, cmap
                    )
            ax.set_xlim(*xlim)
            ax.set_ylim(*ylim)
            ax.set_aspect("equal")
            ax.grid(True, alpha=0.3, linestyle=":")
            ax.tick_params(labelsize=8)
            # Butted-up panels share an edge, so the end ticks of
            # neighbours would print on top of each other.
            ax.xaxis.set_major_locator(MaxNLocator(nbins=5, prune="both"))
            if row == 0:
                title = (task_labels or {}).get(task) or _task_label(task)
                ax.set_title(title, fontsize=13, fontweight="bold")
            if col == 0:
                ax.set_ylabel(label, fontsize=12, fontweight="bold")

    if frame != "world":
        axes[0][0].invert_yaxis()  # shared, so once does the whole grid

    fig.savefig(path, dpi=160)
    plt.close(fig)
    print(f"  runs per task: {ledger(wanted)}")
    return path


def ledger(wanted: Mapping[str, Sequence[Any]]) -> Dict[str, int]:
    """How many runs each column drew, for the caller to print."""
    return {task: len(runs) for task, runs in wanted.items()}
