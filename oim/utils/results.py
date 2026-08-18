"""Persisting one closed-loop run to a single self-describing JSON file.

`results/runs/{stem}_{timestamp}.json` holds everything about a run that
was *observed* and cannot be recomputed later: the settings it ran under,
the scene it ran in, and the per-step trajectories, residuals and timings
it produced. Nothing derived is stored -- goal errors, success, execution
time and control frequency all follow from what is here, and are computed
on demand by `oim.utils.metrics` (see `oim/run_eval.py`).

That split is the point: re-deriving a metric must never require re-running
an experiment. A run file is therefore append-only evidence, and the
metrics over it are cheap and revisable.
"""

import json
import os
from datetime import datetime
from typing import Any, Dict, List, Optional

import numpy as np

from oim.objects import Box, Capsule, Circle, Polygon

# Per-step trajectory keys a run log may carry, in write order. Absent keys
# are skipped, so one writer serves both worlds: `qpos`/`qvel` exist only in
# MJX. The first six are defined identically in 2D and 3D, so the two
# worlds' files can be compared entry for entry.
_DYNAMIC_KEYS = (
    "time",
    "object_pose",  # [x, y, theta], world frame
    "object_velocity",  # [vx, vy, omega], world frame
    "robot_pos",  # world (x, y) of the robot's contact point
    "robot_vel",  # its world-frame velocity
    "robot_control",  # the command actually applied
    "wrench",  # A^r, the realized contact wrench
    "wrench_consensus",  # z_0, the agreed wrench for the executed step
    "qpos",  # full MuJoCo configuration (3D only)
    "qvel",  # full MuJoCo velocity (3D only)
    "object_plan",  # (H, 3) object block's predicted object trajectory
    "robot_plan",  # (H, 3) robot block's, same object -- only if requested
    "primal_residual",  # ADMM diagnostics, one scalar per control step
    "dual_residual",
    "rho",  # the adapted penalty weight, which drifts across a run
    "compute_time",  # wall-clock seconds spent planning that step
    "tip_z",         # stick-tip world z [m] -- FK read, for contact-height analysis
    "tip_tilt",      # tip tilt from vertical [rad] -- forearm-contact / horizontal-tip check
    "contact_normal_force_z",  # pusher-block contact normal force z-component [N],
                                # execution fidelity -- see PushT._contact_normal_force_z_mujoco
    "obstacle_contact_force",  # object-obstacle contact normal force [N], summed
                                # over contacts, execution fidelity -- see
                                # PushT._object_obstacle_force_mujoco
)

# Derived quantities that deliberately do *not* appear in a run file: they
# are functions of what does, so storing them would freeze one definition
# of a metric into the evidence. `oim.utils.metrics` computes them.
#   pos_err, theta_err  <- object_pose vs static.goal
#   reached             <- those two vs static.goal_pos_tol/goal_theta_tol
#   steps_run           <- len(dynamic.time) - 1
#   execution_time      <- steps_run * hyperparameters.control_dt
#   mean_frequency_hz   <- 1 / mean(compute_time)


# Written into every states file. State series carry the initial condition
# and so run one longer than the input series -- the kind of off-by-one that
# silently misaligns an analysis months later, so the file states it.
_SCHEMA = {
    "indexing": (
        "state[i] --robot_control[i]--> state[i+1]. State series "
        "(time, object_pose, object_velocity, robot_pos, robot_vel, "
        "qpos, qvel) have steps_run+1 entries, entry 0 being the initial "
        "condition. Input and wrench series (robot_control, wrench, "
        "wrench_consensus) have steps_run entries, entry i being what was "
        "applied over the step from state[i] to state[i+1]."
    ),
    "frames": (
        "object_pose is [x, y, theta] in the world frame; "
        "object_velocity is its time derivative [vx, vy, omega]. "
        "object_footprint_body is in the object's body frame -- rotate by "
        "theta and translate by [x, y] for world coordinates. Wrenches are "
        "[f_x, f_y, tau] in the world frame about the object's origin."
    ),
    "plans": (
        "object_plan and robot_plan, present only when a run was asked for "
        "them, are the two ADMM blocks' predictions of the *same* object's "
        "trajectory over the horizon, made at that step: what the object "
        "block intends, and what the robot block's controls would produce. "
        "Shape (steps_run, H, 3), aligned with the input series -- entry i "
        "was computed from state[i]. Their divergence is the primal "
        "residual, resolved per timestep instead of summed into a scalar."
    ),
    "velocities": (
        "In 2D both velocities are difference quotients of the logged "
        "positions, which is exact there (the engine integrates pose with "
        "forward Euler), so entry 0 is zero. In 3D object_velocity is "
        "MuJoCo's own qvel for the block, and robot_vel is a difference "
        "quotient of the contact point, which has no qvel entry."
    ),
}


class RunName:
    """Names every artifact of one run off a single shared timestamp.

    Stamping each file as it is written would scatter one run's outputs
    across several timestamps whenever a step takes more than a second,
    which is normal here -- so the timestamp is taken once, at construction.
    """

    def __init__(self, *parts: str) -> None:
        """Build the stem from `parts` and fix the timestamp.

        Args:
            parts: Name components, joined with underscores, e.g.
                ("pusht3d", "xarm6", "admm").
        """
        self.stem = "_".join(parts)
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    def __call__(self, kind: Optional[str] = None) -> str:
        """`{stem}_{kind}_{timestamp}`, or `{stem}_{timestamp}` if no kind.

        Args:
            kind: What the file holds ("metrics", "states"). Omit for
                artifacts that are unambiguous by extension (plot, video).

        Returns:
            A filename stem, with no extension.
        """
        return "_".join(p for p in (self.stem, kind, self.timestamp) if p)


def _jsonable(value: Any) -> Any:
    """Convert JAX/NumPy arrays and scalars to plain JSON types."""
    if value is None:
        return None
    array = np.asarray(value)
    if array.ndim == 0:
        return array.item()
    return array.tolist()


def _shape_to_dict(shape: Any) -> Dict[str, Any]:
    """Serialize an `oim.objects` primitive to a self-describing dict.

    Keeps the constructor's own parameters rather than a sampled outline,
    so the shape can be rebuilt exactly rather than approximated.

    Args:
        shape: A `Circle`, `Box`, `Capsule`, or `Polygon`.

    Returns:
        A dict with a `type` tag and that primitive's parameters.

    Raises:
        TypeError: If the shape is not one of the four primitives.
    """
    if isinstance(shape, Circle):
        return {
            "type": "circle",
            "center": _jsonable(shape.center),
            "radius": float(shape.radius),
        }
    if isinstance(shape, Box):
        return {
            "type": "box",
            "center": _jsonable(shape.center),
            "half_extents": _jsonable(shape.half_extents),
            "angle": float(shape.angle),
        }
    if isinstance(shape, Capsule):
        return {
            "type": "capsule",
            "a": _jsonable(shape.a),
            "b": _jsonable(shape.b),
            "radius": float(shape.radius),
        }
    if isinstance(shape, Polygon):
        return {"type": "polygon", "vertices": _jsonable(shape.vertices)}
    raise TypeError(f"cannot serialize shape {type(shape).__name__}")


def save_run(
    output_dir: str,
    name: RunName,
    run: Dict[str, Any],
    hyperparameters: Dict[str, Any],
    task: Any,
    log: Dict[str, Any],
    extra_static: Optional[Dict[str, Any]] = None,
) -> str:
    """Write one run to `{output_dir}/{name()}.json`.

    Args:
        output_dir: Directory to save into (created if missing).
        name: The run's `RunName`, shared with its plot and video.
        run: What identifies this run within a sweep -- world, task,
            algorithm, sub-optimizers, seed. `oim/run_eval.py` groups on
            these, so anything that should become a row or column of a
            results table belongs here.
        hyperparameters: What the run was configured with -- horizon,
            samples, n_admm, rho, gamma, steps, control_dt, tolerances.
            Must include `control_dt`, since execution time is derived
            from it.
        task: A `ConsensusTask` exposing `goal`, `dt` and `object_model`
            (`PushT` or `PushT2D`). Read for the static scene only.
        log: The run's log dict. Every key of `_DYNAMIC_KEYS` present is
            written and the rest ignored, so the two worlds' differing
            state representations both round-trip. Derived keys the
            runners happen to carry (`pos_err`, `reached`, ...) are
            dropped here by construction.
        extra_static: Additional fixed facts worth recording (embodiment,
            simulator timestep, qpos layout, scenario name, ...).

    Returns:
        The path written.
    """
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, f"{name()}.json")

    obj = task.object_model
    static: Dict[str, Any] = {
        "goal": _jsonable(task.goal),
        "object_footprint_body": _jsonable(obj.footprint.vertices),
        "object_limit_surface_d": _jsonable(obj.D),
        "object_wrench_limit": _jsonable(obj.wrench_limit),
        "obstacles": [_shape_to_dict(s) for s in obj.obstacles.shapes],
    }
    if extra_static:
        static.update({k: _jsonable(v) for k, v in extra_static.items()})

    dynamic: Dict[str, List[Any]] = {
        key: _jsonable(log[key]) for key in _DYNAMIC_KEYS if key in log
    }
    payload = {
        "schema": _SCHEMA,
        "run": run,
        "hyperparameters": {
            k: _jsonable(v) for k, v in hyperparameters.items()
        },
        "static": static,
        "dynamic": dynamic,
    }
    with open(path, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"saved run to {path}")
    return path


def load_run(path: str) -> Dict[str, Any]:
    """Load one run file written by `save_run`.

    Args:
        path: Path to the JSON file.

    Returns:
        The parsed payload: `schema`, `run`, `hyperparameters`, `static`,
        `dynamic`.
    """
    with open(path) as f:
        return json.load(f)
