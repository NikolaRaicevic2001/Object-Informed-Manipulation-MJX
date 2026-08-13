"""Figures from a finished run's log -- 2D and 3D share every primitive.

Nothing here knows how a run was produced: each function takes the task (or
its scenario), the log dict, and a path. That is what lets one plotting
routine serve every scene in `oim.utils.scenes` and every 2D scenario --
obstacles, goal and footprint are read off the task, never hardcoded.

matplotlib is imported inside the functions, not at module scope: the
backend has to be selected before `pyplot` loads, and a run with plotting
switched off should never pay for the import at all.
"""

from typing import Any, Dict

import numpy as np

from oim.objects import Box, Capsule, Circle, Polygon, rotate


def obstacle_outline(obs: object, n: int = 48) -> np.ndarray:
    """A closed polyline tracing an obstacle, for filling in matplotlib.

    Args:
        obs: Any `oim.objects.sdf` shape.
        n: Samples around curved shapes.

    Returns:
        Vertices of shape (m, 2).

    Raises:
        TypeError: If the shape has no outline defined here.
    """
    if isinstance(obs, Circle):
        ang = np.linspace(0, 2 * np.pi, n)
        return np.asarray(obs.center) + obs.radius * np.stack(
            [np.cos(ang), np.sin(ang)], axis=1
        )
    if isinstance(obs, Polygon):
        return np.asarray(obs.vertices)
    if isinstance(obs, Box):
        he = np.asarray(obs.half_extents)
        corners = (
            np.array([[-1, -1], [1, -1], [1, 1], [-1, 1]], dtype=float) * he
        )
        return np.asarray(obs.center) + np.asarray(rotate(obs.angle, corners))
    if isinstance(obs, Capsule):
        a, b = np.asarray(obs.a), np.asarray(obs.b)
        d = b - a
        ang = np.arctan2(d[1], d[0])
        t = np.linspace(-np.pi / 2, np.pi / 2, n // 2)
        cap_a = a + obs.radius * np.stack(
            [np.cos(t + ang + np.pi), np.sin(t + ang + np.pi)], axis=1
        )
        cap_b = b + obs.radius * np.stack(
            [np.cos(t + ang), np.sin(t + ang)], axis=1
        )
        return np.vstack([cap_a, cap_b])
    raise TypeError(f"no outline for {type(obs).__name__}")


def footprint_world(verts: np.ndarray, pose: np.ndarray) -> np.ndarray:
    """The object's footprint at a given SE(2) pose, in world coordinates."""
    return np.asarray(pose[:2]) + np.asarray(rotate(float(pose[2]), verts))


def _plot_plan_divergence(ax_e, log: Dict[str, Any]) -> None:  # noqa: ANN001
    """Mean separation between the two blocks' plans for the object.

    The numeric counterpart of the overlay's blue and magenta paths: both
    blocks predict a trajectory for the *same* object, so the distance
    between them is the consensus disagreement in metres rather than in
    whatever units the consensus variable happens to carry. Under a pose
    consensus variable it is the primal residual itself, unnormalized;
    under a wrench one it is the disagreement's spatial consequence, which
    the residual does not report at all.

    Only logged when a run was asked for the plans (`--show-optimal` or
    `--show-samples`), so this is a no-op otherwise.

    Args:
        ax_e: The axis carrying metres/radians, so the divergence is read
            against the goal errors rather than against the residuals.
        log: A run log from `oim.sim3d.run`.
    """
    object_plan = log.get("object_plan")
    robot_plan = log.get("robot_plan")
    if object_plan is None or robot_plan is None:
        return
    object_plan = np.asarray(object_plan)
    robot_plan = np.asarray(robot_plan)
    if object_plan.ndim != 3 or object_plan.shape != robot_plan.shape:
        return
    # Position only: the two plans' headings are also comparable, but in
    # radians, and mixing them into one curve would hide which of the two
    # the blocks are actually disagreeing about.
    separation = np.linalg.norm(
        object_plan[..., :2] - robot_plan[..., :2], axis=-1
    ).mean(axis=1)
    ax_e.plot(
        separation,
        label="plan divergence (m)",
        color="tab:pink",
        ls=":",
        lw=1.8,
    )


def _diagnostics_panel(ax_r, log: Dict[str, Any]) -> None:  # noqa: ANN001
    """ADMM residual/rho/wrench traces, plus the raw goal errors.

    The goal errors go on a twinned right-hand axis, in metres and radians,
    because nothing else in the figure reports them: the cost panel shows
    `q_pos * d^2` and `q_theta * dtheta^2`, which are cost units under two
    *different* weights -- 40 and 10 by default -- so a `goal_pos` of 26.9
    is a distance of sqrt(26.9/40) = 0.82 m, and the two curves cannot be
    read against each other as errors at all. They also share an axis with
    the residuals only by accident of magnitude, hence the second scale.

    A flat controller's log has no consensus quantities (`_init_log` with
    `admm=False` never allocates `primal_residual`/`dual_residual`/`rho`/
    `wrench`) -- plotting those unconditionally is what crashed every
    flat-baseline headless run with `KeyError: 'primal_residual'`. There
    the errors are the whole panel and keep the left axis to themselves.
    """
    errors = [
        ("pos_err", "position error (m)", "tab:purple"),
        ("theta_err", "orientation error (rad)", "tab:brown"),
    ]
    if "primal_residual" not in log:
        for key, label, colour in errors:
            if key in log:
                ax_r.plot(log[key], label=label, color=colour)
        ax_r.set_title("Convergence")
        ax_r.set_xlabel("control step")
        ax_r.legend()
        ax_r.grid(alpha=0.3)
        return

    ax_r.plot(log["primal_residual"], label="primal residual")
    ax_r.plot(log["dual_residual"], label="dual residual")
    ax_r.plot(log["rho"], label="rho")
    ax_r.plot(np.linalg.norm(log["wrench"], axis=1), label="|w_rob| (N)")
    ax_r.set_title("ADMM diagnostics")
    ax_r.set_xlabel("control step")
    ax_r.grid(alpha=0.3)

    ax_e = ax_r.twinx()
    for key, label, colour in errors:
        if key in log:
            ax_e.plot(log[key], label=label, color=colour, ls="--", lw=1.6)
    _plot_plan_divergence(ax_e, log)
    ax_e.set_ylabel("goal error / plan divergence (m, rad)")
    ax_e.set_ylim(bottom=0.0)
    # One legend for both axes: two boxes on a panel this dense read as
    # two unrelated plots.
    handles, labels = ax_r.get_legend_handles_labels()
    h2, l2 = ax_e.get_legend_handles_labels()
    ax_r.legend(handles + h2, labels + l2, fontsize=9, loc="best")


def _sweep_footprints(ax, verts: np.ndarray, poses, stride: int) -> None:  # noqa: ANN001
    """The object's footprint every `stride` steps, faded start to finish."""
    import matplotlib.pyplot as plt  # noqa: PLC0415

    n = len(poses)
    for i in range(0, n, stride):
        w = footprint_world(verts, poses[i])
        ax.fill(
            w[:, 0],
            w[:, 1],
            color=plt.cm.viridis(i / max(n - 1, 1)),
            alpha=0.35,
            zorder=2,
        )
    for pose, colour in ((poses[0], "tab:blue"), (poses[-1], "tab:red")):
        w = footprint_world(verts, pose)
        ax.fill(w[:, 0], w[:, 1], color=colour, alpha=0.9, zorder=3)


def _goal_and_obstacles(ax, obstacles, goal, verts: np.ndarray) -> None:  # noqa: ANN001
    """Everything in the scene that does not move."""
    for obs in obstacles:
        poly = obstacle_outline(obs)
        ax.fill(poly[:, 0], poly[:, 1], color="0.4", zorder=1)
    outline = footprint_world(verts, np.asarray(goal))
    closed = np.vstack([outline, outline[:1]])
    ax.plot(
        closed[:, 0], closed[:, 1], color="green", lw=2, label="goal", zorder=4
    )
    ax.set_aspect("equal")
    ax.grid(alpha=0.3)


def _rolling_mean(values: np.ndarray, window: int) -> np.ndarray:
    """Centered rolling mean of `values`, same length, edges held.

    Args:
        values: The series to smooth.
        window: Window width in samples; <= 1 returns the input unchanged.

    Returns:
        The smoothed series, same shape as the input.
    """
    if window <= 1 or len(values) < 2:
        return values
    pad = window // 2
    padded = np.pad(values, (pad, window - 1 - pad), mode="edge")
    return np.convolve(padded, np.ones(window) / window, mode="valid")


def _cost_panel(ax_c, task: Any, log: Dict[str, Any]) -> bool:  # noqa: ANN001
    """Each cost term's per-step value over the run, total in the legend.

    Per-step, not accumulated: the question this panel exists to answer is
    "is the run driving its costs down", and a running sum of a nonnegative
    series can never fall, so the accumulated view could show a term
    levelling off but never show one being *solved*. It answered a
    different question -- which term has cost the run the most so far --
    which the legend's totals still report.

    Two things make the per-step view legible where a plain log axis would
    not be:

    * **symlog.** The obstacle hinge and the alignment term are *exactly*
      zero whenever they are satisfied, which a log axis cannot place at
      all. `symlog` is linear through zero below `linthresh` and
      logarithmic above, so a term reaching zero lands on the axis instead
      of falling off it -- while the decades between the effort term (~0.1)
      and the obstacle hinge (weighted 60000) stay readable.
    * **Raw plus trend.** One control step's realized cost is noisy enough
      that a raw series hides its own trend, so every term is drawn twice:
      faint raw, solid rolling mean, one colour.

    Terms that stayed at zero the whole run are dropped rather than drawn
    flat along the bottom -- a term that never fired is not evidence about
    the run, and it costs a legend entry.

    Returns:
        Whether anything was drawn.
    """
    from oim.utils.costs import cost_totals, summarize  # noqa: PLC0415

    series = summarize(task, log)
    if series is None:
        return False
    totals = cost_totals(series)
    active = {k: v for k, v in series.items() if totals[k] > 0.0}
    if not active:
        return False

    n = len(next(iter(active.values())))
    window = max(1, min(n // 40, 25))

    for name, values in active.items():
        (line,) = ax_c.plot(values, lw=0.8, alpha=0.25)
        ax_c.plot(
            _rolling_mean(values, window),
            lw=1.6,
            color=line.get_color(),
            label=f"{name}  (Σ {totals[name]:.4g})",
        )
    total = np.sum(list(active.values()), axis=0)
    ax_c.plot(total, color="k", lw=0.8, alpha=0.25)
    ax_c.plot(
        _rolling_mean(total, window),
        color="k",
        lw=2.0,
        ls="--",
        label=f"total  (Σ {totals['total']:.4g})",
    )

    # Linear region set by the *smallest term that is actually on*, so the
    # cheapest live term (effort, ~0.1) still has vertical extent instead
    # of being flattened onto zero, and no term is pushed below the axis.
    # Floored so an all-zero-ish run cannot ask for linthresh = 0.
    positive = [v[v > 0.0] for v in active.values()]
    smallest = min(float(np.median(p)) for p in positive if p.size)
    ax_c.set_yscale("symlog", linthresh=max(smallest, 1e-3), linscale=0.5)
    ax_c.set_ylim(bottom=0.0)
    ax_c.set_xlabel("control step")
    ax_c.set_ylabel("cost per control step")
    ax_c.set_title("Cost terms (realized trajectory)")
    ax_c.legend(fontsize=9, loc="best", framealpha=0.9)
    ax_c.grid(alpha=0.3)
    return True


def _new_figure():  # noqa: ANN202
    """A trajectory panel, a diagnostics panel, and a cost panel.

    `layout="constrained"` rather than a closing `tight_layout()`: the
    trajectory panel is `set_aspect("equal")`, so its axes box cannot fill
    the slot the gridspec hands it and the leftover shows up as a band of
    white. `tight_layout` sizes slots without accounting for that, and the
    band survives; the constrained engine shrinks the slot to the box it
    actually needs, which is what closes the gaps between the panels.
    """
    import matplotlib  # noqa: PLC0415

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt  # noqa: PLC0415

    fig, axes = plt.subplots(
        1,
        3,
        figsize=(24, 6.8),
        gridspec_kw={"width_ratios": [1.5, 1, 1.1]},
        layout="constrained",
    )
    fig.get_layout_engine().set(w_pad=0.02, h_pad=0.02, wspace=0.015)
    return plt, (fig, axes)


def plot_run_3d(
    task: Any, log: Dict[str, Any], path: str, stride: int = 5
) -> None:
    """Draw the object's swept footprint, the pusher path, and diagnostics.

    Obstacles, goal and footprint come from `task.object_model` rather than
    a hardcoded constant, so this is correct for every scene.

    Args:
        task: A `PushT` with `object_model` populated (`clutter=True`).
        log: The run log from `oim.sim3d.run`.
        path: Where to write the PNG.
        stride: Draw the footprint every this many control steps.
    """
    plt, (fig, (ax, ax_r, ax_c)) = _new_figure()
    verts = np.asarray(task.object_model.footprint.vertices)
    _goal_and_obstacles(
        ax, task.object_model.obstacles.shapes, task.object_model.goal, verts
    )

    poses = log["object_pose"]
    _sweep_footprints(ax, verts, poses, stride)
    pusher = log["robot_pos"]
    ax.plot(
        pusher[:, 0],
        pusher[:, 1],
        "k.-",
        ms=4.5,
        lw=1.4,
        label="pusher",
        zorder=5,
    )
    ax.set_title(
        f"{'reached' if log['reached'] else 'not reached'} in "
        f"{len(poses) - 1} steps"
    )
    ax.legend(loc="upper left")
    _diagnostics_panel(ax_r, log)
    if not _cost_panel(ax_c, task, log):
        ax_c.set_visible(False)

    fig.savefig(path, dpi=130)
    plt.close(fig)
    print(f"saved plot to {path}")


def _draw_scene_2d(ax, scenario: Any, verts: np.ndarray) -> None:  # noqa: ANN001
    """Obstacles, goal and the scenario's own view window."""
    _goal_and_obstacles(ax, scenario.obstacles, scenario.goal, verts)
    ax.set_xlim(scenario.view[0], scenario.view[1])
    ax.set_ylim(scenario.view[2], scenario.view[3])


def plot_run_2d(
    task: Any,
    scenario: Any,
    log: Dict[str, Any],
    path: str,
    stride: int = 5,
) -> None:
    """Draw the object's swept footprint, the robot path, and diagnostics.

    Args:
        task: A `PushT2D`.
        scenario: The `Scenario` it was built from, for obstacles and view.
        log: The run log from `oim.sim2d.run_2d`.
        path: Where to write the PNG.
        stride: Draw the footprint every this many control steps.
    """
    plt, (fig, (ax, ax_r, ax_c)) = _new_figure()
    verts = np.asarray(task.footprint.vertices)
    _draw_scene_2d(ax, scenario, verts)

    poses = log["object_pose"]
    _sweep_footprints(ax, verts, poses, stride)
    robot = log["robot_pos"]
    ax.plot(
        robot[:, 0],
        robot[:, 1],
        "k.-",
        ms=4.5,
        lw=1.4,
        label="robot",
        zorder=5,
    )
    ax.set_title(
        f"{scenario.name}  |  "
        f"{'reached' if log['reached'] else 'not reached'} in "
        f"{len(poses) - 1} steps"
    )
    ax.legend(loc="upper left")
    _diagnostics_panel(ax_r, log)
    if not _cost_panel(ax_c, task, log):
        ax_c.set_visible(False)

    fig.savefig(path, dpi=130)
    plt.close(fig)
    print(f"saved plot to {path}")


def save_animation_2d(
    task: Any,
    scenario: Any,
    log: Dict[str, Any],
    path: str,
    fps: int = 15,
) -> None:
    """Write an animated gif of a 2D run.

    Worth having over the static plot: a swept-footprint figure shows
    *where* the object went but not *when* it stalled, reversed, or got
    shoved sideways by a contact that broke.

    Args:
        task: A `PushT2D`.
        scenario: The `Scenario` it was built from.
        log: The run log from `oim.sim2d.run_2d`.
        path: Where to write the gif.
        fps: Playback rate.
    """
    import matplotlib  # noqa: PLC0415

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt  # noqa: PLC0415
    from matplotlib import animation  # noqa: PLC0415

    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    verts = np.asarray(task.footprint.vertices)
    _draw_scene_2d(ax, scenario, verts)

    poses, robot = log["object_pose"], log["robot_pos"]
    (body,) = ax.fill([], [], color="tab:blue", alpha=0.85, zorder=3)
    (trail,) = ax.plot([], [], "k-", lw=1, alpha=0.6, zorder=5)
    (dot,) = ax.plot([], [], "ro", ms=5, zorder=6)
    title = ax.set_title("")

    def _update(i: int):  # noqa: ANN202
        body.set_xy(footprint_world(verts, poses[i]))
        trail.set_data(robot[: i + 1, 0], robot[: i + 1, 1])
        dot.set_data([robot[i, 0]], [robot[i, 1]])
        title.set_text(f"{scenario.name}  step {i}/{len(poses) - 1}")
        return body, trail, dot, title

    anim = animation.FuncAnimation(
        fig, _update, frames=len(poses), blit=False, interval=1000 // fps
    )
    anim.save(path, writer=animation.PillowWriter(fps=fps))
    plt.close(fig)
    print(f"saved animation to {path}")
