"""Multi-run ablation figures for `oim.run_eval --plot`.

Takes the already-aggregated step curves from `evaluate_step_curves` and
draws a task × metric grid. Deliberately separate from `plotting.py`, which
serves a single finished run's trajectory / cost panels.
"""

from typing import Any, Dict, List, Mapping, Optional, Sequence, Set, Tuple

import numpy as np

# Columns 0-1 are single series; column 2 overlays primal and dual in the
# same method color, distinguished by linestyle + end-point markers.
_ERROR_COLS: Tuple[Tuple[str, str, str], ...] = (
    (r"Position error $\epsilon_d$ (m)", "pos_err_mean", "pos_err_std"),
    (
        r"Orientation error $\epsilon_\theta$ (rad)",
        "theta_err_mean",
        "theta_err_std",
    ),
)
_RESIDUAL_TITLE = r"ADMM residuals (○ primal, △ dual)"
_ERROR_LW = 1.6
_RESIDUAL_LW = 1.0
# Start/end markers: circle = primal, triangle = dual.
_PRIMAL_MARKER = "o"
_DUAL_MARKER = "^"
_MARKER_SIZE = 5.5


def _subtitle(
    ablate: Sequence[str],
    filters: Optional[Mapping[str, Sequence[str]]],
) -> str:
    """One-line caption of what is compared vs pinned."""
    parts: List[str] = []
    if ablate:
        parts.append("ablate: " + ", ".join(ablate))
    if filters:
        pinned = ", ".join(
            f"{k}={','.join(v)}" for k, v in sorted(filters.items())
        )
        parts.append("filter: " + pinned)
    return "  |  ".join(parts)


def _draw_series(
    ax: Any,
    series: Dict[str, Any],
    *,
    mean_key: str,
    std_key: str,
    color: Any,
    linestyle: str,
    label: Optional[str],
    lw: float = _ERROR_LW,
    marker: Optional[str] = None,
) -> None:
    """Plot one mean curve, with ±std band when n>1.

    When `marker` is set, the same symbol is drawn at the first and last
    finite sample so overlapping residual traces stay distinguishable.
    """
    if mean_key not in series:
        return
    x = np.asarray(series["steps"])
    y = np.asarray(series[mean_key])
    ax.plot(x, y, color=color, linestyle=linestyle, label=label, lw=lw)
    if marker is not None and len(x) > 0:
        finite = np.isfinite(y)
        if np.any(finite):
            idx = np.flatnonzero(finite)
            ends = np.unique([idx[0], idx[-1]])
            ax.plot(
                x[ends],
                y[ends],
                linestyle="None",
                marker=marker,
                markersize=_MARKER_SIZE,
                markerfacecolor=color,
                markeredgecolor="0.15",
                markeredgewidth=0.6,
                zorder=3,
            )
    n = int(series.get("n_trials", 1))
    if n > 1 and std_key in series:
        s = np.asarray(series[std_key])
        ax.fill_between(x, y - s, y + s, color=color, alpha=0.18, lw=0)


def _collect_legend(axes: Any) -> Tuple[List[Any], List[str]]:
    """Union of legend entries across axes, first occurrence wins."""
    handles: List[Any] = []
    labels: List[str] = []
    seen: Set[str] = set()
    for ax in axes.ravel():
        h, lab = ax.get_legend_handles_labels()
        for handle, label in zip(h, lab, strict=True):
            if label not in seen:
                handles.append(handle)
                labels.append(label)
                seen.add(label)
    return handles, labels


def _draw_row(
    axes_row: Any,
    group_curves: Dict[str, Dict[str, Any]],
    methods: Sequence[str],
    colors: Mapping[str, Any],
    row_label: str,
    *,
    show_titles: bool,
    show_xlabel: bool,
    show_residuals: bool = True,
) -> None:
    """Fill one task row: two error panels, and a residual overlay if any.

    `show_residuals` is False for a figure whose runs have no ADMM
    residuals at all -- an object-only sweep, or flat baselines alone.
    """
    for c, (title, mean_key, std_key) in enumerate(_ERROR_COLS):
        ax = axes_row[c]
        for method in methods:
            series = group_curves.get(method)
            if series is None:
                continue
            _draw_series(
                ax,
                series,
                mean_key=mean_key,
                std_key=std_key,
                color=colors[method],
                linestyle="-",
                label=method,
            )
        if show_titles:
            ax.set_title(title)
        if c == 0:
            ax.set_ylabel(row_label)
        if show_xlabel:
            ax.set_xlabel("control step")
        ax.grid(True, alpha=0.3)

    if not show_residuals:
        return

    ax = axes_row[2]
    for method in methods:
        series = group_curves.get(method)
        if series is None:
            continue
        # Same color per method; thinner strokes + end markers separate
        # primal (solid, ○) from dual (dashed, △).
        _draw_series(
            ax,
            series,
            mean_key="primal_residual_mean",
            std_key="primal_residual_std",
            color=colors[method],
            linestyle="-",
            label=method,
            lw=_RESIDUAL_LW,
            marker=_PRIMAL_MARKER,
        )
        _draw_series(
            ax,
            series,
            mean_key="dual_residual_mean",
            std_key="dual_residual_std",
            color=colors[method],
            linestyle="--",
            label=None,
            lw=_RESIDUAL_LW,
            marker=_DUAL_MARKER,
        )
    if show_titles:
        ax.set_title(_RESIDUAL_TITLE)
    if show_xlabel:
        ax.set_xlabel("control step")
    ax.grid(True, alpha=0.3)


def plot_step_curves(
    curves: Dict[str, Dict[str, Dict[str, Any]]],
    path: str,
    group_by: Sequence[str] = ("task",),
    ablate: Sequence[str] = (),
    filters: Optional[Mapping[str, Sequence[str]]] = None,
) -> str:
    """Write a rows=groups × cols=metrics figure comparing methods.

    Columns are position error, orientation error, and ADMM residuals
    (shared color per method; primal solid+○, dual dashed+△).

    Args:
        curves: `evaluate_step_curves` output (no Mean block expected).
        path: Destination PNG path.
        group_by: Unused; kept for call-site compatibility with `run_eval`.
        ablate: Ablated fields, for the figure subtitle.
        filters: Pinned fields, for the figure subtitle.

    Returns:
        The path written.
    """
    _ = group_by
    # matplotlib backend must be set before pyplot loads; Agg keeps this
    # headless-safe on the same machines that run the sweep.
    import matplotlib  # noqa: PLC0415

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt  # noqa: PLC0415
    from matplotlib.lines import Line2D  # noqa: PLC0415

    # Imported lazily so this module never pulls `run_eval` at import time
    # (run_eval loads us only under `--plot`).
    from oim.run_eval import MEAN_LABEL, _strip_common_prefix  # noqa: PLC0415

    groups = [g for g in curves if g != MEAN_LABEL]
    if not groups:
        raise ValueError("plot_step_curves needs at least one row group")

    short, _prefix = _strip_common_prefix(groups)
    labels = dict(zip(groups, short, strict=True))
    methods = sorted({m for g in groups for m in curves[g]})
    n_rows = len(groups)

    # Drop the residual column outright when nothing has residuals -- an
    # object-only sweep, or flat baselines alone. An always-present empty
    # panel spends a third of the figure saying nothing, and reads as a
    # plot that failed rather than a quantity that does not exist here.
    show_residuals = any(
        curves[g][m].get("primal_residual_mean") is not None
        for g in groups
        for m in curves[g]
    )
    n_cols = 3 if show_residuals else 2

    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(4.2 * n_cols, 2.8 * n_rows),
        sharex="col",
        squeeze=False,
    )
    cmap = plt.get_cmap("tab10")
    colors = {m: cmap(i % 10) for i, m in enumerate(methods)}

    for r, group in enumerate(groups):
        _draw_row(
            axes[r],
            curves[group],
            methods,
            colors,
            labels[group],
            show_titles=(r == 0),
            show_xlabel=(r == n_rows - 1),
            show_residuals=show_residuals,
        )

    handles, legend_labels = _collect_legend(axes)
    # Only when there is a residual panel for them to explain.
    style_handles = [] if not show_residuals else [
        Line2D(
            [0],
            [0],
            color="0.2",
            linestyle="-",
            lw=_RESIDUAL_LW,
            marker=_PRIMAL_MARKER,
            markersize=_MARKER_SIZE,
            label="primal",
        ),
        Line2D(
            [0],
            [0],
            color="0.2",
            linestyle="--",
            lw=_RESIDUAL_LW,
            marker=_DUAL_MARKER,
            markersize=_MARKER_SIZE,
            label="dual",
        ),
    ]
    if handles:
        extra = ["primal", "dual"] if show_residuals else []
        fig.legend(
            handles + style_handles,
            legend_labels + extra,
            loc="upper center",
            ncol=min(len(legend_labels) + len(extra), 5),
            frameon=False,
            bbox_to_anchor=(0.5, 1.02),
        )

    sub = _subtitle(ablate, filters)
    if sub:
        fig.suptitle(sub, y=1.04)
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path
