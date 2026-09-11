"""CBF/CLF projections of robot control tapes, before physics rollout.

QPax solves two CBFs and a soft tilt CLF together. Constraints are prepared
at the observed state and shared across the tape, outside the physics scan.
"""

import argparse
from dataclasses import dataclass as config_dataclass
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
from flax.struct import dataclass
from mujoco import mjx

from oim.objects.sdf import rotate


@config_dataclass(frozen=True)
class ProjectionConfig:
    """Projection settings; heights and distances are in world metres.

    ``z_min`` and ``z_near`` default to offsets from the slider mid-height
    in ``configure_projection``. No config block means no projection.
    """

    mode: str = "off"
    z_min: float = 0.0
    z_near: float = 0.01
    z_far: float = 0.10
    distance_near: float = 0.02
    distance_far: float = 0.15
    footprint_margin: float = 0.01
    cbf_alpha: float = 2.0
    floor_alpha: float | None = None
    slider_alpha: float | None = None
    tilt_rate: float = 2.0
    tilt_weight: float = 10.0
    xy_weight: float = 100.0
    # The small QPs use local float64 arithmetic; physics remains float32.
    # Independently check the returned commands at feasibility_tol.
    solver_tol: float = 1e-6
    feasibility_tol: float = 1e-5
    max_iter: int = 30

    def __post_init__(self) -> None:
        """Reject malformed constraints before tracing the controller."""
        if self.mode not in {"off", "qpax"}:
            raise ValueError("projection.mode must be off or qpax")
        values = [v for v in vars(self).values() if isinstance(v, (int, float))]
        if not np.all(np.isfinite(values)):
            raise ValueError("projection settings must be finite")
        if not self.z_min < self.z_near <= self.z_far:
            raise ValueError("projection requires z_min < z_near <= z_far")
        if not 0 <= self.distance_near < self.distance_far:
            raise ValueError(
                "projection requires 0 <= distance_near < distance_far"
            )
        if self.footprint_margin < 0:
            raise ValueError("projection.footprint_margin must be nonnegative")
        if self.xy_weight < 0:
            raise ValueError("projection.xy_weight must be nonnegative")
        if (
            min(
                self.cbf_alpha,
                self.tilt_rate,
                self.tilt_weight,
                self.solver_tol,
                self.feasibility_tol,
            )
            <= 0
        ):
            raise ValueError(
                "projection gains, weights and tolerances must be positive"
            )
        if self.max_iter < 1 or int(self.max_iter) != self.max_iter:
            raise ValueError("projection.max_iter must be a positive integer")
        for gain in (self.floor_alpha, self.slider_alpha):
            if gain is not None and gain <= 0:
                raise ValueError("per-CBF gains must be positive")


_TUNING_FLAGS = {
    "cbf-alpha": ("cbf_alpha", "Shared floor/slider CBF response gain [1/s]."),
    "cbf-floor-alpha": (
        "floor_alpha", "Floor gain [1/s]; overrides shared gain."
    ),
    "cbf-slider-alpha": (
        "slider_alpha", "Slider gain [1/s]; overrides shared gain."
    ),
    "cbf-z-min": ("z_min", "Minimum tip height in world metres."),
    "cbf-z-near": ("z_near", "Tip height ceiling near the slider [world m]."),
    "cbf-z-far": ("z_far", "Tip height ceiling far from the slider [world m]."),
    "cbf-distance-near": ("distance_near", "Start of ceiling ramp [m]."),
    "cbf-distance-far": ("distance_far", "End of ceiling ramp [m]."),
    "cbf-margin": (
        "footprint_margin", "Clearance subtracted from footprint distance [m]."
    ),
    "cbf-xy-weight": (
        "xy_weight", "Penalty for changing tip x-y velocity, QPax only."
    ),
    "tilt-rate": ("tilt_rate", "Soft tilt CLF response gain [1/s], QPax only."),
    "tilt-weight": ("tilt_weight", "Soft tilt CLF slack penalty, QPax only."),
}


def add_projection_tuning_arguments(parser: argparse.ArgumentParser) -> None:
    """Expose the same physical tuning knobs in simulation and real drivers."""
    group = parser.add_argument_group("CBF / CLF tuning")
    for flag, (_, description) in _TUNING_FLAGS.items():
        group.add_argument(
            f"--{flag}", type=float, default=None,
            help=description + " Unset keeps the config/scene default.",
        )


def projection_settings(
    settings: dict | None, args: argparse.Namespace
) -> dict:
    """Merge only explicitly supplied CLI values into a fresh config block."""
    merged = dict(settings or {})
    if getattr(args, "control_projection", None) is not None:
        merged["mode"] = args.control_projection
    for flag, (key, _) in _TUNING_FLAGS.items():
        value = getattr(args, flag.replace("-", "_"), None)
        if value is not None:
            merged[key] = value
    return merged


@dataclass
class ProjectionConstraints:
    """Linear constraints and kinematics at the observed pose.

    CBFs use ``cbf_a @ u + cbf_b >= 0``; the CLF uses
    ``tilt_a @ u + tilt_b <= slack``. Rows are floor, slider ceiling.
    """

    cbf_a: jax.Array
    cbf_b: jax.Array
    tilt_a: jax.Array
    tilt_b: jax.Array
    u_min: jax.Array
    u_max: jax.Array
    xy_jacobian: jax.Array | None = None


@dataclass
class ProjectionDiagnostics:
    """Per-command diagnostics, with the same leading axes as the tape."""

    valid: jax.Array
    converged: jax.Array
    hard_violation: jax.Array
    correction_norm: jax.Array
    tilt_slack: jax.Array
    iterations: jax.Array


def height_ceiling(
    distance: jax.Array, config: ProjectionConfig
) -> tuple[jax.Array, jax.Array]:
    """Return a smooth distance-dependent ceiling and its derivative."""
    width = config.distance_far - config.distance_near
    t = jnp.clip((distance - config.distance_near) / width, 0.0, 1.0)
    height = config.z_near + (config.z_far - config.z_near) * t**2 * (3 - 2 * t)
    slope = (config.z_far - config.z_near) * 6 * t * (1 - t) / width
    return height, slope


def make_constraints(
    position: jax.Array,
    axis: jax.Array,
    jac_position: jax.Array,
    jac_rotation: jax.Array,
    distance: jax.Array,
    distance_gradient: jax.Array,
    distance_drift: jax.Array,
    u_min: jax.Array,
    u_max: jax.Array,
    config: ProjectionConfig,
) -> ProjectionConstraints:
    """Construct CBF/CLF rows from tip kinematics and footprint distance.

    ``distance_drift`` is distance rate due to the slider's motion alone.
    The stick site's local z axis points down in the upright xArm pose.
    """
    ceiling, slope = height_ceiling(distance, config)
    floor_a = jac_position[2]
    ceiling_a = slope * (distance_gradient @ jac_position[:2]) - floor_a
    cbf_b = jnp.stack(
        (
            (config.floor_alpha or config.cbf_alpha)
            * (position[2] - config.z_min),
            (config.slider_alpha or config.cbf_alpha)
            * (ceiling - position[2]) + slope * distance_drift,
        )
    )
    target = jnp.array([0.0, 0.0, -1.0])
    return ProjectionConstraints(
        cbf_a=jnp.stack((floor_a, ceiling_a)),
        cbf_b=cbf_b,
        tilt_a=jnp.cross(target, axis) @ jac_rotation,
        tilt_b=config.tilt_rate * (1.0 - target @ axis),
        u_min=u_min,
        u_max=u_max,
        xy_jacobian=jac_position[:2],
    )


class ControlProjector:
    """Batch projector with an optional QPax dependency."""

    def __init__(self, config: ProjectionConfig) -> None:
        """Load QPax only when its mode is selected."""
        self.config = config
        self.solve_qp = None
        if config.mode == "qpax":
            try:
                from qpax import solve_qp  # noqa: PLC0415
            except ImportError as exc:
                raise ImportError(
                    "QPax projection requires: uv sync --extra projection"
                ) from exc
            self.solve_qp = solve_qp

    def _qp_project(
        self, controls: jax.Array, c: ProjectionConstraints
    ) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
        """Use local float64 to avoid QPax's float32 slack/dual floor.

        Keep casts inside the precision context so nested jit/vmap/scan
        preserve the QP precision without changing the simulation dtypes.
        """
        dtype = controls.dtype
        with jax.enable_x64(True):
            out, slack, converged, iterations = self._qp_project_impl(
                controls.astype(jnp.float64),
                jax.tree.map(lambda x: x.astype(jnp.float64), c),
            )
            return (
                out.astype(dtype), slack.astype(dtype),
                converged > 0, iterations.astype(jnp.int32),
            )

    def _qp_project_impl(
        self, controls: jax.Array, c: ProjectionConstraints
    ) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
        """Batch standard QPs with shared matrices and CLF-only slack."""
        n = controls.shape[-1]
        # Penalize changes from the NOMINAL Cartesian velocity, not motion
        # itself: safe horizontal pushing should remain inexpensive.
        metric = jnp.eye(n)
        if c.xy_jacobian is not None:
            metric += self.config.xy_weight * (
                c.xy_jacobian.T @ c.xy_jacobian
            )
        q_mat = jnp.zeros((n + 1, n + 1)).at[:n, :n].set(metric)
        q_mat = q_mat.at[n, n].set(self.config.tilt_weight)
        clf_direction = jnp.linalg.solve(metric, c.tilt_a)
        cbf = jnp.concatenate(
            (-c.cbf_a, jnp.zeros((c.cbf_a.shape[0], 1))), axis=1
        )
        clf = jnp.concatenate((c.tilt_a, jnp.array([-1.0])))[None]
        box = jnp.concatenate((jnp.eye(n), jnp.zeros((n, 1))), axis=1)
        slack_row = jnp.concatenate((jnp.zeros(n), jnp.array([-1.0])))[None]
        g = jnp.concatenate((cbf, clf, box, -box, slack_row))
        h = jnp.concatenate(
            (c.cbf_b, -c.tilt_b[None], c.u_max, -c.u_min, jnp.zeros(1))
        )
        a, b = jnp.zeros((0, n + 1)), jnp.zeros(0)

        def solve(u: jax.Array) -> tuple:
            linear = jnp.concatenate((-metric @ u, jnp.zeros(1)))
            x, _, _, _, converged, iterations = self.solve_qp(
                q_mat,
                linear,
                a,
                b,
                g,
                h,
                solver_tol=self.config.solver_tol,
                max_iter=self.config.max_iter,
            )
            # Solve the soft CLF alone exactly. If no hard constraint is
            # active, this is also the full QP optimum. The interior-point
            # stopping tolerance otherwise introduces a substantial drift
            # even for a stationary, upright arm with inactive barriers.
            a_clf = c.tilt_a
            amount = self.config.tilt_weight * jnp.maximum(
                a_clf @ u + c.tilt_b, 0.0
            ) / (1.0 + self.config.tilt_weight * (a_clf @ clf_direction))
            free_u = u - amount * clf_direction
            free_slack = jnp.maximum(a_clf @ free_u + c.tilt_b, 0.0)
            free_valid = (
                jnp.all(c.cbf_a @ free_u + c.cbf_b >= 0.0)
                & jnp.all(free_u >= c.u_min)
                & jnp.all(free_u <= c.u_max)
            )
            return (
                jnp.where(free_valid, free_u, x[:n]),
                jnp.where(free_valid, free_slack, x[n]),
                jnp.where(free_valid, True, converged),
                jnp.where(free_valid, 0, iterations),
            )

        out, slack, converged, iterations = jax.vmap(solve)(
            controls.reshape(-1, n)
        )
        shape = controls.shape[:-1]
        return (
            out.reshape(controls.shape),
            slack.reshape(shape),
            converged.reshape(shape),
            iterations.reshape(shape),
        )

    def project(
        self, controls: jax.Array, constraints: ProjectionConstraints
    ) -> tuple[jax.Array, ProjectionDiagnostics]:
        """Project an arbitrary control batch and report unresolved constraints.

        Failed QPs use finite box-clipped nominal controls for simulation,
        but remain invalid. The rollout dispatcher rejects their tapes.
        """
        c = constraints
        shape = controls.shape[:-1]
        finite = jnp.all(jnp.isfinite(controls), axis=-1)
        nominal = jnp.where(jnp.isfinite(controls), controls, 0.0)
        iterations = jnp.zeros(shape, dtype=jnp.int32)
        slack = jnp.zeros(shape)
        if self.config.mode == "off":
            return controls, ProjectionDiagnostics(
                valid=finite,
                converged=jnp.ones(shape, dtype=bool),
                hard_violation=jnp.zeros(shape),
                correction_norm=jnp.zeros(shape),
                tilt_slack=slack,
                iterations=iterations,
            )
        candidate, slack, converged, iterations = self._qp_project(
            nominal, c
        )
        converged = (
            (converged > 0)
            & jnp.all(jnp.isfinite(candidate), axis=-1)
            & jnp.isfinite(slack)
        )
        out = jnp.where(
            converged[..., None], candidate,
            jnp.clip(nominal, c.u_min, c.u_max),
        )
        violation = jnp.maximum(
            0.0, jnp.max(-(
                jnp.sum(out[..., None, :] * c.cbf_a, axis=-1) + c.cbf_b
            ), axis=-1)
        )
        violation = jnp.maximum(
            violation,
            jnp.max(jnp.maximum(c.u_min - out, out - c.u_max), axis=-1),
        )
        valid = (
            finite
            & converged
            & jnp.isfinite(violation)
            & (violation <= self.config.feasibility_tol)
        )
        valid = valid & (slack >= -self.config.feasibility_tol)
        valid = valid & (
            jnp.sum(out * c.tilt_a, axis=-1) + c.tilt_b - slack
            <= self.config.feasibility_tol
        )
        return out, ProjectionDiagnostics(
            valid=valid,
            converged=converged,
            hard_violation=violation,
            correction_norm=jnp.linalg.norm(out - controls, axis=-1),
            tilt_slack=slack,
            iterations=iterations,
        )


class RobotControlProjector(ControlProjector):
    """Prepare xArm constraints from qpos, outside physics rollout scans."""

    def __init__(self, task: Any, config: ProjectionConfig) -> None:
        """Cache a JAX kinematics model, also usable with Warp rollouts."""
        super().__init__(config)
        self.task = task
        self.model = mjx.put_model(task.mj_model, impl="jax")
        self.data = mjx.make_data(self.model)
        self.site_id = task.tip_site_id
        self.body_id = int(task.mj_model.site_bodyid[self.site_id])
        self.dofs = jnp.asarray(task.robot_dof_adr)

    def prepare(self, state: mjx.Data) -> ProjectionConstraints:
        """Compute fresh kinematics from observations with stale site arrays."""
        data = self.data.replace(
            qpos=state.qpos,
            qvel=state.qvel,
            mocap_pos=state.mocap_pos,
            mocap_quat=state.mocap_quat,
        )
        data = mjx.com_pos(self.model, mjx.kinematics(self.model, data))
        position = data.site_xpos[self.site_id]
        jacp, jacr = mjx.jac(self.model, data, position, self.body_id)
        pose = self.task.object_state_from_robot(data)
        local = rotate(-pose[2], position[:2] - pose[:2])
        distance, grad_local = self.task.object_model.footprint.sdf_and_grad(
            local
        )
        gradient = rotate(pose[2], grad_local)
        # The task resolves its planar translation/yaw DOFs by joint name.
        pose_rate = data.qvel[self.task.block_dofs]
        relative = position[:2] - pose[:2]
        surface_velocity = pose_rate[:2] + pose_rate[2] * jnp.array(
            [-relative[1], relative[0]]
        )
        constraints = make_constraints(
            position,
            data.site_xmat[self.site_id, :, 2],
            jacp[self.dofs].T,
            jacr[self.dofs].T,
            distance - self.config.footprint_margin,
            gradient,
            -gradient @ surface_velocity,
            self.task.u_min,
            self.task.u_max,
            self.config,
        )
        return constraints


def configure_projection(task: Any, settings: dict | None) -> None:
    """Attach an opt-in xArm projector in both real/simulation builders."""
    if not settings or settings.get("mode", "off") == "off":
        return
    if getattr(task, "robot", None) != "xarm6":
        raise ValueError("control_projection currently supports only xarm6")
    if task.model.nu != len(task.robot_dof_adr):
        raise ValueError(
            "projection requires one velocity command per robot DOF"
        )
    if not np.all(np.isfinite(np.concatenate((task.u_min, task.u_max)))):
        raise ValueError("projection requires finite robot velocity limits")
    config = ProjectionConfig(
        **{
            "z_min": task.tip_target_z - 0.01,
            "z_near": task.tip_target_z,
            "z_far": task.tip_target_z + 0.10,
            **settings,
        }
    )
    task.control_projector = RobotControlProjector(task, config)


def prepare_projection(
    task: Any, state: mjx.Data
) -> ProjectionConstraints | None:
    """Prepare shared data once per solve, with a structural no-op default."""
    projector = getattr(task, "control_projector", None)
    return None if projector is None else projector.prepare(state)


def project_controls(
    task: Any,
    state: mjx.Data,
    controls: jax.Array,
    constraints: ProjectionConstraints | None = None,
) -> tuple[jax.Array, ProjectionDiagnostics | None]:
    """Project a tape before rollout dispatch, or return it unchanged."""
    projector = getattr(task, "control_projector", None)
    if projector is None:
        return controls, None
    if constraints is None:
        constraints = projector.prepare(state)
    return projector.project(controls, constraints)


def project_nominal(
    task: Any,
    params: Any,
    constraints: ProjectionConstraints | None,
) -> Any:
    """Project the knots of the plan that will actually be executed.

    `project_controls` projects the sampled control TAPES, but the knots
    stored on the resulting `Trajectory` are the nominal, unprojected ones
    (`tests/test_control_projection.py::
    test_flat_and_admm_use_projected_tapes_and_nominal_knots` pins that on
    purpose). The optimizers then average those unprojected knots into
    `params.mean`, and `mean` is what the real driver interpolates and
    publishes -- so before this, the projection shaped which samples looked
    good but never touched the command the arm received. On hardware
    2026-09-10 that let the tip climb to 0.72 m against a 0.13 m ceiling,
    because `w_tilt`/`w_z_tip_exp` were zeroed in favour of a CBF that the
    executed plan never saw. (Same gap as the "apply this to the final
    command as well" TODO in the Isaac projector.)

    Projecting the KNOTS is enough to make every published command
    feasible, and only under a linear spline: the hard constraints are an
    intersection of half-spaces, hence convex, so a linear interpolation
    between two feasible knots is feasible. Under a cubic/quintic robot
    spline that argument fails and the interpolated tape would have to be
    projected instead. `_clip_plan_to_joint_range` in the real driver runs
    after interpolation and is not part of this guarantee.

    Args:
        task: Carries `control_projector`, or nothing when disabled.
        params: Policy params exposing `mean`; `ADMMParams` delegates to
            its robot block, so the caller passes that block directly.
        constraints: Constraints already prepared for this solve. None
            means projection is disabled for this call and `params` is
            returned untouched -- there is no state here to prepare from.

    Returns:
        `params` with a projected `mean`, or unchanged when projection is
        off -- a structural no-op, so a disabled run traces identically.
    """
    projector = getattr(task, "control_projector", None)
    if projector is None or constraints is None:
        return params
    mean, _ = projector.project(params.mean, constraints)
    return params.replace(mean=mean)


def reject_invalid_tapes(
    costs: jax.Array, diagnostics: ProjectionDiagnostics | None
) -> jax.Array:
    """Reject tapes containing any unresolved projection."""
    if diagnostics is None:
        return costs
    valid = jnp.all(diagnostics.valid, axis=-1)
    return costs.at[..., -1].set(jnp.where(valid, costs[..., -1], jnp.inf))
