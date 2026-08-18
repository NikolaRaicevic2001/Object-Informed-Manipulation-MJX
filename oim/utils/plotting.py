"""Figures from a finished run's log -- 2D and 3D share every primitive.

Nothing here knows how a run was produced: each function takes the task (or
its scenario), the log dict, and a path. That is what lets one plotting
routine serve every scene in `oim.utils.scenes` and every 2D scenario --
obstacles, goal and footprint are read off the task, never hardcoded.

matplotlib is imported inside the functions, not at module scope: the
backend has to be selected before `pyplot` loads, and a run with plotting
switched off should never pay for the import at all.
"""

from typing import Any, Dict, Tuple

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
        log: A run log from `oim.worlds.sim3d.run`.
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
    """ADMM residual/rho traces, plus the raw goal errors.

    The goal errors go on a twinned right-hand axis, in metres and radians,
    because nothing else in the figure reports them: the cost panel shows
    `q_pos * d^2` and `q_theta * dtheta^2`, which are cost units under two
    *different* weights -- 40 and 10 by default -- so a `goal_pos` of 26.9
    is a distance of sqrt(26.9/40) = 0.82 m, and the two curves cannot be
    read against each other as errors at all. They also share an axis with
    the residuals only by accident of magnitude, hence the second scale.

    A flat controller's log has no consensus quantities (`_init_log` with
    `admm=False` never allocates `primal_residual`/`dual_residual`/`rho`)
    -- plotting those unconditionally is what crashed every flat-baseline
    headless run with `KeyError: 'primal_residual'`. There the errors are
    the whole panel and keep the left axis to themselves.
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
    # `|w_rob|` deliberately not drawn: its norm mixes N with N*m, so the
    # "(N)" label was wrong. Still recorded in `log["wrench"]`.
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


def _cost_panel(ax_c, series: Any) -> bool:  # noqa: ANN001
    """Each cost term's per-step value over the run, total in the legend.

    Takes the already-computed series rather than `(task, log)` so the same
    panel serves the robot-block decomposition (`costs.cost_series`) and
    the object-block one (`costs.object_cost_series`), which score
    different cost functions and cannot be told apart from the log alone.
    `None` (a log this decomposition does not fit) draws nothing.

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
    from oim.utils.costs import cost_totals  # noqa: PLC0415

    if not series:
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
        log: The run log from `oim.worlds.sim3d.run`.
        path: Where to write the PNG.
        stride: Draw the footprint every this many control steps.
    """
    from oim.utils.costs import summarize  # noqa: PLC0415

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
    if not _cost_panel(ax_c, summarize(task, log)):
        ax_c.set_visible(False)

    fig.savefig(path, dpi=130)
    plt.close(fig)
    print(f"saved plot to {path}")


def _object_diagnostics_panel(ax_r, task: Any, log: Dict[str, Any]) -> None:  # noqa: ANN001
    """Goal errors, and the applied wrench against its breakaway threshold.

    The generic `_diagnostics_panel` degrades to errors alone here (there
    are no residuals in a single-block run), which would leave out the one
    quantity this world exists to interrogate. `PlanarPushingObject.step`
    zeroes any wrench with `||w / w_limit|| < 1`, so that normalized
    magnitude is plotted with the threshold drawn on it: below the line the
    object does not move at all, however the cost panel looks. A run that
    stalls short of the goal with this curve pinned just under 1.0 is the
    documented near-goal deadzone, not a tuning problem.

    Args:
        ax_r: The panel axis.
        task: The `PushT`, for `object_model.wrench_limit`.
        log: The object-only run log.
    """
    ax_r.plot(log["pos_err"], label="position error (m)", color="tab:purple")
    ax_r.plot(
        log["theta_err"], label="orientation error (rad)", color="tab:brown"
    )
    # Only under a plant that is not the model -- it is identically zero
    # otherwise, and a flat line at zero reads as a broken series rather
    # than as the "no model error by construction" it actually is.
    pred_err = np.asarray(log.get("pred_pos_err", []))
    if len(pred_err) and float(np.max(pred_err)) > 0.0:
        ax_r.plot(
            pred_err,
            label="model error, per step (m)",
            color="tab:red",
            lw=1.0,
            alpha=0.7,
        )
    plant = log.get("plant", "analytic")
    ax_r.set_title(f"Object block: convergence and wrench ({plant} plant)")
    ax_r.set_xlabel("control step")
    ax_r.set_ylabel("goal error (m, rad)")
    ax_r.set_ylim(bottom=0.0)
    ax_r.grid(alpha=0.3)

    wrench = np.asarray(log["wrench"])
    ax_w = ax_r.twinx()
    if len(wrench):
        limit = np.asarray(task.object_model.wrench_limit)
        ax_w.plot(
            np.linalg.norm(wrench / limit, axis=1),
            label="|w| / friction-cone limit",
            color="tab:green",
            lw=1.4,
        )
        ax_w.axhline(
            1.0,
            color="tab:red",
            ls=":",
            lw=1.4,
            label="breakaway threshold",
        )
    ax_w.set_ylabel("normalized wrench magnitude")
    ax_w.set_ylim(bottom=0.0)
    handles, labels = ax_r.get_legend_handles_labels()
    h2, l2 = ax_w.get_legend_handles_labels()
    ax_r.legend(handles + h2, labels + l2, fontsize=9, loc="best")


def plot_run_object(
    task: Any,
    log: Dict[str, Any],
    path: str,
    stride: int = 5,
) -> None:
    """Diagnostics for an object-level-only run.

    See `oim.worlds.object_only.build`.

    The same three panels every other run gets, with the two differences a
    world containing no robot forces: the trajectory panel draws the
    object's own realized path where the others draw the pusher's, and the
    cost panel decomposes the *object* stage cost
    (`costs.object_cost_series`) rather than the robot one.

    The per-step plans and candidate rollouts are deliberately **not** here.
    One static frame carrying every step's horizon is unreadable -- hundreds
    of overlapping paths cover the scene and hide the one thing this panel
    is for, which is where the object actually went. They are a time-varying
    quantity and belong in the animation; see `save_animation_object`, which
    `--record` writes.

    Args:
        task: The `PushT` the run was built from, for goal/obstacles/
            footprint.
        log: The log from `oim.worlds.object_only.build.run_object`.
        path: Where to write the PNG.
        stride: Draw the object's footprint every this many control steps.
    """
    from oim.utils.costs import object_cost_series  # noqa: PLC0415

    plt, (fig, (ax, ax_r, ax_c)) = _new_figure()
    verts = np.asarray(task.object_model.footprint.vertices)
    _goal_and_obstacles(
        ax, task.object_model.obstacles.shapes, task.object_model.goal, verts
    )

    poses = np.asarray(log["object_pose"])
    _sweep_footprints(ax, verts, poses, stride)
    # The realized path of the object's own origin. The swept footprints
    # show where it went, but a run that barely moves renders them as one
    # blob -- this line makes the difference between "crawled" and "did not
    # move" visible at a glance, which is the first thing to check here.
    ax.plot(
        poses[:, 0],
        poses[:, 1],
        "k.-",
        ms=3.5,
        lw=1.2,
        label="object (realized)",
        zorder=5,
    )
    ax.set_title(
        f"object block alone  |  "
        f"{'reached' if log['reached'] else 'not reached'} in "
        f"{len(poses) - 1} steps"
    )
    ax.legend(loc="upper left")

    _object_diagnostics_panel(ax_r, task, log)
    if not _cost_panel(ax_c, object_cost_series(task, log)):
        ax_c.set_visible(False)

    fig.savefig(path, dpi=130)
    plt.close(fig)
    print(f"saved plot to {path}")


def _object_view_limits(
    task: Any, poses: np.ndarray, pad: float = 0.12
) -> Tuple[float, float, float, float]:
    """A view window covering the scene and everything the object did.

    A 2D `Scenario` carries its own `view`; a 3D `SceneSpec` does not, and
    the free-camera framing the MuJoCo runs use has no matplotlib
    equivalent -- so it is derived here from the obstacles, the goal and
    the realized path together. All three matter: framing on the path alone
    hides the obstacle the object failed to get around, and framing on the
    scene alone can crop a run that wandered out of it.

    Args:
        task: The `PushT`, for obstacles, goal and footprint.
        poses: The realized object poses, (n, 3).
        pad: Margin added on every side, in metres.

    Returns:
        `(xmin, xmax, ymin, ymax)`.
    """
    obj = task.object_model
    points = [np.asarray(poses)[:, :2], np.asarray(obj.goal)[None, :2]]
    for shape in obj.obstacles.shapes:
        points.append(obstacle_outline(shape))
    stacked = np.vstack(points)
    reach = float(obj.footprint.bounding_radius) + pad
    return (
        float(stacked[:, 0].min()) - reach,
        float(stacked[:, 0].max()) + reach,
        float(stacked[:, 1].min()) - reach,
        float(stacked[:, 1].max()) + reach,
    )


def save_animation_object(
    task: Any,
    log: Dict[str, Any],
    path: str,
    fps: int = 15,
    show_samples: bool = True,
    show_optimal: bool = True,
    max_frames: int = 240,
    max_sample_lines: int = 48,
) -> None:
    """Write an animated gif of an object-level-only run.

    This is where the trajectories belong. Each frame shows *one* control
    step's horizon -- the candidates the block sampled and the plan it
    settled on -- against the object where it actually was at that moment,
    so the plans stay legible and their evolution is the thing you watch.
    Collapsed onto a single static frame they are just clutter.

    Same colour language as `oim.runtime.overlay`, so an object-only gif
    and an ADMM recording read alike: pale cyan candidates, strong blue
    chosen plan, and its endpoint (x^{o*}_H, the local goal) marked.

    Args:
        task: The `PushT` the run was built from.
        log: The log from `oim.worlds.object_only.build.run_object`.
        path: Where to write the gif.
        fps: Playback rate.
        show_samples: Draw the candidate rollouts. Needs the run to have
            logged them (`run_object(log_samples=True)`).
        show_optimal: Draw the plan the block settled on.
        max_frames: Cap on frames; longer runs are strided down to it. A
            1000-step run at one frame per step is a 60 s gif of tens of
            megabytes, which nothing wants.
        max_sample_lines: Cap on candidates drawn per frame. 128 overlapping
            paths read as a solid wash; a subset shows the spread just as
            well and keeps the file small.
    """
    import matplotlib  # noqa: PLC0415

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt  # noqa: PLC0415
    from matplotlib import animation  # noqa: PLC0415

    verts = np.asarray(task.object_model.footprint.vertices)
    poses = np.asarray(log["object_pose"])
    plans = log.get("object_plan") if show_optimal else None
    samples = log.get("object_samples") if show_samples else None
    if samples is not None and not len(samples):
        samples = None

    frames = list(range(0, len(poses), max(1, len(poses) // max_frames)))

    fig, ax = plt.subplots(figsize=(7.0, 6.0), layout="constrained")
    _goal_and_obstacles(
        ax, task.object_model.obstacles.shapes, task.object_model.goal, verts
    )
    xmin, xmax, ymin, ymax = _object_view_limits(task, poses)
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)

    # Artists are created once and updated per frame; matplotlib redraws
    # far faster that way than by clearing and re-plotting, which matters
    # at a few hundred frames times `max_sample_lines` lines.
    n_lines = 0 if samples is None else min(samples.shape[1], max_sample_lines)
    sample_lines = [
        ax.plot([], [], color=(0.40, 0.82, 1.00), lw=0.6, alpha=0.35,
                zorder=1.5)[0]
        for _ in range(n_lines)
    ]
    (plan_line,) = ax.plot(
        [], [], color=(0.00, 0.30, 0.95), lw=2.0, zorder=4.5,
        label="chosen plan" if plans is not None else None,
    )
    (plan_end,) = ax.plot(
        [], [], "o", ms=6, color=(0.00, 0.30, 0.95), zorder=4.6,
        label="plan endpoint" if plans is not None else None,
    )
    (body,) = ax.fill([], [], color="tab:blue", alpha=0.85, zorder=3)
    (trail,) = ax.plot([], [], "k-", lw=1.2, alpha=0.7, zorder=5)
    if n_lines:
        sample_lines[0].set_label("candidates")
    title = ax.set_title("")
    ax.legend(loc="upper left", fontsize=9)

    def _update(i: int):  # noqa: ANN202
        body.set_xy(footprint_world(verts, poses[i]))
        trail.set_data(poses[: i + 1, 0], poses[: i + 1, 1])
        # The plan and candidates at index i were computed *from* pose i,
        # so they exist for every frame but the last (the state series runs
        # one longer than the input series -- see results._SCHEMA).
        if plans is not None and i < len(plans):
            plan = np.asarray(plans[i])
            plan_line.set_data(plan[:, 0], plan[:, 1])
            plan_end.set_data([plan[-1, 0]], [plan[-1, 1]])
        else:
            plan_line.set_data([], [])
            plan_end.set_data([], [])
        if samples is not None and i < len(samples):
            for line, cand in zip(
                sample_lines, np.asarray(samples[i]), strict=False
            ):
                line.set_data(cand[:, 0], cand[:, 1])
        else:
            for line in sample_lines:
                line.set_data([], [])
        title.set_text(f"object block alone  step {i}/{len(poses) - 1}")
        return (body, trail, plan_line, plan_end, title, *sample_lines)

    anim = animation.FuncAnimation(
        fig, _update, frames=frames, blit=False, interval=1000 // fps
    )
    anim.save(path, writer=animation.PillowWriter(fps=fps))
    plt.close(fig)
    print(f"saved animation to {path} ({len(frames)} frames)")


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
        log: The run log from `oim.worlds.sim2d.run_2d`.
        path: Where to write the PNG.
        stride: Draw the footprint every this many control steps.
    """
    from oim.utils.costs import summarize  # noqa: PLC0415

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
    if not _cost_panel(ax_c, summarize(task, log)):
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
        log: The run log from `oim.worlds.sim2d.run_2d`.
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
