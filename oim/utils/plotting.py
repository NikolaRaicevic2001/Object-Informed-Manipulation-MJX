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
        ax_r.set_title("Task diagnostics")
        ax_r.set_xlabel("control step")
        ax_r.legend()
        ax_r.grid(alpha=0.3)
        return

    ax_r.plot(log["primal_residual"], label="primal residual")
    ax_r.plot(log["dual_residual"], label="dual residual")
    ax_r.plot(log["rho"], label="rho")
    # `|w_rob|` deliberately not drawn: its norm mixes N with N*m, so the
    # "(N)" label was wrong. Still recorded in `log["wrench"]`.
    ax_r.set_title("Task diagnostics")
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


def _goal_and_obstacles(
    ax, obstacles, goal, verts: np.ndarray, support=None  # noqa: ANN001
) -> None:
    """Everything in the scene that does not move.

    `support` is the tabletop the object has to stay on -- the same `Box`
    `PlanarPushingObject.support_cost` charges against, so the dashed rim
    drawn here is exactly the boundary that cost is measured from, not a
    redrawing of it. None for a scene with no table geom (`clutter`), which
    simply omits it.
    """
    if support is not None:
        rim = obstacle_outline(support)
        closed_rim = np.vstack([rim, rim[:1]])
        ax.plot(
            closed_rim[:, 0],
            closed_rim[:, 1],
            color="0.55",
            lw=1.4,
            ls="--",
            label="table",
            zorder=0,
        )
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


def _start_centroid(ax, pose) -> None:  # noqa: ANN001
    """Mark the object's own frame origin at the START pose only.

    Every pose in this repo is that origin, not the footprint's area
    centroid: `q_pos` measures it against the goal, `w_align` builds its
    cone from it, and the contact-point consensus samples offsets around
    it. Where it sits inside the outline is a property of the FOOTPRINT
    (`t_shape_footprint` puts the T's at the crossbar/stem junction, not
    at the middle of the plan), so swapping in a different object moves it
    -- which is exactly what this marker is here to make visible.

    Start pose only: drawing it on every step would just retrace the path
    line that is already there.
    """
    pose = np.asarray(pose, dtype=float)
    # White face, black edge: it is drawn ON TOP of the start footprint,
    # which `_sweep_footprints` fills `tab:blue` (or `tab:red` where the
    # object barely moved and the last pose covers the first), so any
    # single-colour marker disappears into one of them.
    ax.plot(
        pose[0],
        pose[1],
        marker="P",
        mfc="white",
        mec="black",
        ms=10,
        mew=1.4,
        ls="none",
        label="centroid (start)",
        zorder=6,
    )


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


# One fixed colour per cost term, so a term that appears in both cost
# panels is the same colour in both. Matplotlib's default cycle assigns by
# draw order, which the two panels do not share: the object panel has no
# approach/align/tilt, so everything after `obstacle` -- `effort` and
# `admm_penalty` especially -- landed on a different colour there than in
# the robot panel, which is exactly the pair a reader wants to compare
# across the two.
#
# Every term gets its own fully distinct hue rather than hue-paired
# shades (2026-08-20, per Shahid): with up to 13 terms live on one panel,
# same-hue-different-shade pairs (e.g. two blues, two oranges, two
# purples) read as one smear at a glance, which defeats the panel's job
# of letting a reader see which term is doing what. Bright and maximally
# separated in hue instead, so e.g. `align` is checkable on sight against
# `top_contact`-suppression without hunting for the lighter of two
# similar oranges. Keys are `oim.utils.costs.TERM_ORDER`; an unlisted
# term falls back to the cycle rather than raising, so adding a term
# degrades to the old behaviour instead of breaking the figure. `total`
# stays black.
_TERM_COLOURS = {
    "goal_pos": "#0000cd",  # dark blue
    "goal_theta": "#00bfff",  # light/sky blue
    "obstacle": "#2ca02c",  # green
    "support": "#00734d",  # deep teal-green -- keep-IN, the mirror of
    # `obstacle`'s keep-out, so a related hue; dark enough not to be
    # mistaken for it when both fire in the same run.
    "rate": "#e31a1c",  # red
    "approach": "#6a3d9a",  # purple
    "align": "#ff1493",  # deep pink
    "tilt": "#9acd32",  # sap green / yellow-green
    "tip_z": "#ff7f00",  # orange
    "contact_z": "#8b4513",  # brown
    "robot_contact": "#ffd700",  # gold
    "effort": "#7f7f7f",  # grey
    "admm_penalty": "#8b008b",  # dark magenta
}


def _cost_panel(
    ax_c,  # noqa: ANN001
    series: Any,
    title: str = "Cost terms (realized trajectory)",
) -> bool:
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
        (line,) = ax_c.plot(
            values, lw=0.8, alpha=0.25, color=_TERM_COLOURS.get(name)
        )
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
    ax_c.set_title(title)
    ax_c.legend(fontsize=9, loc="best", framealpha=0.9)
    ax_c.grid(alpha=0.3)
    return True


def _new_figure(n_cost_panels: int = 1):  # noqa: ANN202
    """A trajectory panel, a diagnostics panel, and `n_cost_panels` cost panels.

    ADMM runs get two cost panels -- the robot block's decomposition and the
    object block's -- because the two score genuinely different cost
    functions and stacking them on one axis invites reading a robot term
    against an object term as if they traded off. Everything else gets one.

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

    widths = [1.5, 1] + [1.1] * n_cost_panels
    fig, axes = plt.subplots(
        1,
        2 + n_cost_panels,
        figsize=(24 + 7.5 * (n_cost_panels - 1), 6.8),
        gridspec_kw={"width_ratios": widths},
        layout="constrained",
    )
    fig.get_layout_engine().set(w_pad=0.02, h_pad=0.02, wspace=0.015)
    return plt, (fig, axes)


def _draw_cost_panels(
    axes,  # noqa: ANN001
    task: Any,
    log: Dict[str, Any],
) -> None:
    """Fill the one or two cost panels `_new_figure` allocated.

    One panel (robot block) for every run; a second (object block) only for
    ADMM, which is the only algorithm that *has* an object block. Detected
    by `primal_residual`, the same marker `_diagnostics_panel` keys on --
    `init_log` allocates it exactly when `admm=True`, so it is a fact about
    the controller rather than about the scene.

    A panel whose decomposition does not fit the log is hidden rather than
    left as an empty frame.
    """
    from oim.utils.costs import object_cost_series, summarize  # noqa: PLC0415

    if not _cost_panel(axes[0], summarize(task, log), "Robot block costs"):
        axes[0].set_visible(False)
    if len(axes) < 2:
        return
    try:
        object_series = object_cost_series(task, log)
    except (KeyError, IndexError, AttributeError):
        object_series = None
    if not _cost_panel(axes[1], object_series, "Object block costs"):
        axes[1].set_visible(False)


def _cost_panel_count(log: Dict[str, Any]) -> int:
    """2 for an ADMM run (robot + object blocks), 1 otherwise."""
    return 2 if "primal_residual" in log and "wrench" in log else 1


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
    plt, (fig, axes) = _new_figure(_cost_panel_count(log))
    ax, ax_r = axes[0], axes[1]
    verts = np.asarray(task.object_model.footprint.vertices)
    _goal_and_obstacles(
        ax,
        task.object_model.obstacles.shapes,
        task.object_model.goal,
        verts,
        support=task.object_model.support,
    )

    poses = log["object_pose"]
    _sweep_footprints(ax, verts, poses, stride)
    _start_centroid(ax, np.asarray(poses)[0])
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
    _draw_cost_panels(axes[2:], task, log)

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
    quantity and belong in the recording; see `--record`, which films the
    MuJoCo plant with `oim.runtime.overlay` compositing each step's plans
    into the frames captured during it.

    Args:
        task: The `PushT` the run was built from, for goal/obstacles/
            footprint.
        log: The log from `oim.worlds.object_only.build.run_object`.
        path: Where to write the PNG.
        stride: Draw the object's footprint every this many control steps.
    """
    from oim.utils.costs import object_cost_series  # noqa: PLC0415

    plt, (fig, (ax, ax_r, ax_c)) = _new_figure(1)
    verts = np.asarray(task.object_model.footprint.vertices)
    _goal_and_obstacles(
        ax,
        task.object_model.obstacles.shapes,
        task.object_model.goal,
        verts,
        support=task.object_model.support,
    )

    poses = np.asarray(log["object_pose"])
    _sweep_footprints(ax, verts, poses, stride)
    _start_centroid(ax, poses[0])
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
    if not _cost_panel(
        ax_c, object_cost_series(task, log), "Object block costs"
    ):
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
    lo = stacked.min(axis=0) - reach
    hi = stacked.max(axis=0) + reach
    # Union with the tabletop rim, so the dashed boundary the trajectory
    # panel draws is actually in frame. Padded by `pad` alone rather than
    # `reach`: `reach` exists so a footprint centred at an extreme pose is
    # not clipped, while the rim is already an absolute extent and needs
    # only enough margin to sit inside the axes rather than on them. This
    # does zoom out -- the lab table is 1.52 m long against a 0.65 m
    # corridor -- which is the trade for showing what the poses are
    # measured against.
    if obj.support is not None:
        rim = obstacle_outline(obj.support)
        lo = np.minimum(lo, rim.min(axis=0) - pad)
        hi = np.maximum(hi, rim.max(axis=0) + pad)
    return (float(lo[0]), float(hi[0]), float(lo[1]), float(hi[1]))
